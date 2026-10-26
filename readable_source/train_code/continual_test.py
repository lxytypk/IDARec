import argparse
import copy
import inspect
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

try:
    from aletheia_probe import load_selected_layers, load_structural_layers
    _HAS_ALETHEIA = True
except ImportError:  # the upstream `peso/` directory predates the Aletheia probe
    load_selected_layers = load_structural_layers = None
    _HAS_ALETHEIA = False
from collator import TestCollator
from continual_data_v2 import ContinualSeqRecDataset_v2
from continual_lora import apply_continual_lora
from evaluate import get_metrics_results, get_topk_results
from safetensors.torch import load_file
from utils import *

# ---------------------------------------------------------------------------
# Unified evaluation harness.
#
# The decode protocol is a COMMAND-LINE SWITCH, not a hardcoded value, because
# two mutually incompatible protocols are in circulation and both must remain
# runnable from one file:
#
#   deterministic -> do_sample=False -> GenerationMode.BEAM_SEARCH
#                    Returns the num_beams highest-scoring sequences.  This is
#                    what Hit@K / NDCG@K are defined over.  Zero variance,
#                    seed-independent.
#
#   sampling      -> inherit the base model's generation_config ->
#                    GenerationMode.BEAM_SAMPLE for Llama-3.2-1B, which ships
#                    do_sample=true / temperature=0.6 / top_p=0.9.  Returns a
#                    stochastic draw of num_beams sequences.  This reproduces
#                    the official hsyoo32/peso released behaviour, since that
#                    repo never passes do_sample and pins transformers==4.52.4.
#
# Results files record which protocol produced them, and an existing results
# file written under the OTHER protocol is refused rather than overwritten.
# ---------------------------------------------------------------------------
EVAL_HARNESS_VERSION = "unified_v1_20260921"

DECODE_MODES = ("deterministic", "sampling")
EXPECTED_GENERATION_MODE = {
    "deterministic": "BEAM_SEARCH",
    "sampling": "BEAM_SAMPLE",
}


def decode_do_sample(args):
    """True only when the caller explicitly asked for the official sampling protocol."""
    return getattr(args, "eval_decode_mode", "deterministic") == "sampling"


def resolve_generation_mode(generation_config, num_beams):
    """Report the GenerationMode generate() will actually take, without mutating the config."""
    probe = copy.deepcopy(generation_config)
    probe.num_beams = num_beams
    probe.num_return_sequences = num_beams
    return probe.get_generation_mode(None).name


def assert_decode_protocol(args, generation_config, num_beams, local_rank, shipped=None):
    """Fail loudly if the resolved decode mode is not the one that was requested.

    This guard exists because the do_sample omission that caused a ~7% systematic
    metric offset was invisible for two months: BEAM_SAMPLE is fully reproducible
    under set_seed(), so a rerun matched bit-for-bit and looked healthy.
    """
    mode = resolve_generation_mode(generation_config, num_beams)
    expected = EXPECTED_GENERATION_MODE[args.eval_decode_mode]
    shipped = shipped or {}
    if local_rank == 0:
        print(
            f"[eval-protocol] harness={EVAL_HARNESS_VERSION} "
            f"decode_mode={args.eval_decode_mode} GenerationMode={mode} "
            f"do_sample={generation_config.do_sample} "
            f"temperature={generation_config.temperature} top_p={generation_config.top_p} "
            f"num_beams={num_beams} sampler=ExactDistributedEvalSampler"
        )
        if shipped:
            print(
                "[eval-protocol] base model shipped: do_sample="
                f"{shipped.get('do_sample')} temperature={shipped.get('temperature')} "
                f"top_p={shipped.get('top_p')}"
            )
    if mode != expected:
        raise RuntimeError(
            f"Decode protocol mismatch: --eval_decode_mode {args.eval_decode_mode} "
            f"should resolve to GenerationMode.{expected} but resolved to {mode}. "
            f"Check the base model's generation_config.json (shipped do_sample="
            f"{shipped.get('do_sample')}, temperature={shipped.get('temperature')}, "
            f"top_p={shipped.get('top_p')})."
        )
    return mode


def guard_results_file_protocol(args):
    """Refuse to overwrite a results file that was written under a different protocol.

    Mixing protocols inside one file is how the semantic-vs-full-vocab comparison
    became unreadable: the two halves of the table were produced by different
    decoders and nobody could tell from the numbers alone.  Checking before the
    eval starts means a mis-tagged run fails in one second instead of after
    several GPU-hours.
    """
    path = args.results_file
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    recorded = existing.get("eval_protocol")
    if recorded is None:
        print(
            f"[eval-protocol] WARNING: {path} already exists and records no protocol "
            f"(written by a pre-unification harness). It will be overwritten."
        )
        return
    if recorded.get("decode_mode") != args.eval_decode_mode:
        raise RuntimeError(
            f"{path} was written under decode_mode={recorded.get('decode_mode')!r} "
            f"but this run requested {args.eval_decode_mode!r}. Refusing to mix "
            f"protocols in one results file. Point --results_file at a "
            f"{args.eval_decode_mode}-suffixed path, or delete the old file first."
        )
    if recorded.get("num_beams") not in (None, args.num_beams):
        raise RuntimeError(
            f"{path} was written with num_beams={recorded.get('num_beams')} but this "
            f"run uses {args.num_beams}. Beam count changes the candidate pool size, "
            f"so Hit@K/NDCG@K are not comparable across it."
        )


class ExactDistributedEvalSampler(Sampler):
    """Partition evaluation rows without padding, duplication, or dropping."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def reconstruct_saved_parameter_lists(model, state_dict, local_rank):
    """Restore saved adapter history after loading a block checkpoint."""
    if local_rank == 0:
        print("Reconstructing ParameterList objects for saved matrices...")

    for name, module in model.named_modules():
        if hasattr(module, "A_matrices") and hasattr(module, "B_matrices"):
            module.A_matrices = nn.ParameterList([])
            module.B_matrices = nn.ParameterList([])

            module_prefix = name + "."
            a_keys = [k for k in state_dict if k.startswith(module_prefix + "A_matrices.")]
            b_keys = [k for k in state_dict if k.startswith(module_prefix + "B_matrices.")]

            a_keys.sort(key=lambda x: int(x.split(".")[-1]))
            b_keys.sort(key=lambda x: int(x.split(".")[-1]))

            for a_key, b_key in zip(a_keys, b_keys):
                module.A_matrices.append(nn.Parameter(state_dict[a_key]))
                module.B_matrices.append(nn.Parameter(state_dict[b_key]))

            if hasattr(module, "magnitudes"):
                module.magnitudes = nn.ParameterList([])
                mag_keys = [k for k in state_dict if k.startswith(module_prefix + "magnitudes.")]
                mag_keys.sort(key=lambda x: int(x.split(".")[-1]))
                for mag_key in mag_keys:
                    module.magnitudes.append(nn.Parameter(state_dict[mag_key]))

        elif hasattr(module, "A_dir_matrices") and hasattr(module, "B_dir_matrices") and hasattr(module, "magnitudes"):
            module.A_dir_matrices = nn.ParameterList([])
            module.B_dir_matrices = nn.ParameterList([])
            module.magnitudes = nn.ParameterList([])

            module_prefix = name + "."
            a_dir_keys = [k for k in state_dict if k.startswith(module_prefix + "A_dir_matrices.")]
            b_dir_keys = [k for k in state_dict if k.startswith(module_prefix + "B_dir_matrices.")]
            mag_keys = [k for k in state_dict if k.startswith(module_prefix + "magnitudes.")]

            a_dir_keys.sort(key=lambda x: int(x.split(".")[-1]))
            b_dir_keys.sort(key=lambda x: int(x.split(".")[-1]))
            mag_keys.sort(key=lambda x: int(x.split(".")[-1]))

            for a_key, b_key, mag_key in zip(a_dir_keys, b_dir_keys, mag_keys):
                module.A_dir_matrices.append(nn.Parameter(state_dict[a_key]))
                module.B_dir_matrices.append(nn.Parameter(state_dict[b_key]))
                module.magnitudes.append(nn.Parameter(state_dict[mag_key]))


def resolve_vocab_freeze_config(ckpt_path, local_rank=0):
    """Read vocab freeze configuration, returning (num_old_tokens, freeze_embed, freeze_head)."""
    config_candidates = [
        os.path.join(ckpt_path, "vocab_config.json"),
        os.path.join(os.path.dirname(os.path.realpath(ckpt_path)), "vocab_config.json"),
    ]
    config_path = next((p for p in config_candidates if os.path.exists(p)), None)
    if config_path is None:
        return None, False, False
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Support both old and new config formats
    if "freeze_old_embed" in config:
        freeze_embed = config["freeze_old_embed"]
        freeze_head = config["freeze_old_head"]
    else:
        freeze_embed = freeze_head = config.get("freeze_old_vocab", False)
    if not freeze_embed and not freeze_head:
        return None, False, False
    num_old_tokens = int(config["num_old_tokens"])
    if local_rank == 0:
        parts = []
        if freeze_embed:
            parts.append("embed_tokens")
        if freeze_head:
            parts.append("lm_head")
        print(
            f"[vocab] {config_path}: freezing {'+'.join(parts)} rows [0,{num_old_tokens}), "
            f"trainable rows={config.get('num_new_tokens')}"
        )
    return num_old_tokens, freeze_embed, freeze_head




def convert_partial_vocab_checkpoint(state_dict, num_old_tokens, freeze_embed, freeze_head, local_rank=0):
    """Convert partial-vocab checkpoint keys to full-vocab format where needed."""
    conversions = []
    for prefix, target_is_full in [
        ("model.embed_tokens", not freeze_embed),
        ("lm_head", not freeze_head),
    ]:
        old_key = prefix + ".original_module.weight"
        new_key = prefix + ".new_vocab_rows"
        full_key = prefix + ".modules_to_save.default.weight"
        if new_key in state_dict and old_key in state_dict and full_key not in state_dict:
            if not target_is_full:
                continue
            old_weight = state_dict[old_key]
            new_rows = state_dict[new_key]
            expected_new = old_weight.shape[0] - num_old_tokens
            if new_rows.shape[0] != expected_new:
                raise ValueError(
                    "Checkpoint conversion shape mismatch for " + prefix + ": "
                    "new_vocab_rows has " + str(new_rows.shape[0]) + " rows, expected " + str(expected_new)
                )
            state_dict[full_key] = torch.cat([old_weight[:num_old_tokens], new_rows], dim=0)
            del state_dict[new_key]  # Remove partial key to avoid unexpected-key error
            conversions.append(prefix)
    if conversions and local_rank == 0:
        print("[Checkpoint Conversion] Converted partial->full vocab for: " + str(conversions))
    return state_dict


def load_layer_selection(block_ckpt_path, local_rank):
    """Read Aletheia layer selection, degrading to None where the probe does not exist.

    Returns (selected_layers, structural_layers).  The upstream `peso/` directory
    has no aletheia_probe.py and no selected_layers support in continual_lora, so
    both come back None there and the full-layer PESO behaviour is preserved.
    """
    if not _HAS_ALETHEIA:
        if os.path.exists(os.path.join(block_ckpt_path, "selected_layers.json")):
            raise RuntimeError(
                f"{block_ckpt_path} carries a selected_layers.json but this directory has "
                f"no aletheia_probe.py, so the layer selection cannot be honoured. "
                f"Evaluating it here would silently run all 16 layers."
            )
        return None, None
    return load_selected_layers(block_ckpt_path), load_structural_layers(block_ckpt_path)


def call_apply_continual_lora(args, model, structural_layers, vocab_cfg, local_rank):
    """Call apply_continual_lora passing only the kwargs this directory's module declares.

    Three signatures are in circulation:
      peso/            (model, ..., num_blocks, option)
      Jul-08 lineage   + selected_layers, trainable_layers
      Sep-10 lineage   + num_old_tokens, freeze_embed, freeze_head
    An undeclared kwarg raises TypeError, so the call is adapted by signature.
    A semantic checkpoint in a directory that cannot rebuild the frozen-row
    structure is a hard error rather than a silent downgrade to full vocab.
    """
    num_old_tokens, freeze_embed, freeze_head = vocab_cfg
    params = inspect.signature(apply_continual_lora).parameters
    kwargs = {"num_blocks": args.num_blocks, "option": args.shift_flag}

    if "selected_layers" in params:
        kwargs["selected_layers"] = structural_layers
        if "trainable_layers" in params:
            kwargs["trainable_layers"] = [] if structural_layers is not None else None
    elif structural_layers is not None:
        raise RuntimeError(
            "This checkpoint uses Aletheia layer selection but the local "
            "continual_lora.apply_continual_lora has no selected_layers parameter, "
            "so the selection cannot be applied."
        )

    if num_old_tokens is not None:
        if "num_old_tokens" not in params:
            raise RuntimeError(
                "This is a semantic-rows (partial vocab) checkpoint but the local "
                "continual_lora.apply_continual_lora has no num_old_tokens parameter, "
                "so the frozen-row structure cannot be rebuilt. Evaluating here would "
                "silently drop new_vocab_rows and score an untrained semantic head."
            )
        kwargs["num_old_tokens"] = num_old_tokens
        if "freeze_embed" in params:
            kwargs["freeze_embed"] = freeze_embed
        if "freeze_head" in params:
            kwargs["freeze_head"] = freeze_head

    if local_rank == 0:
        print(f"[compat] apply_continual_lora kwargs: {sorted(kwargs)}")
    apply_continual_lora(
        model,
        args.lora_target_modules,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_modules_to_save,
        **kwargs,
    )


def load_continual_model(args, tokenizer, device_map, local_rank, block_ckpt_path, test_block):
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))

    selected_layers, structural_layers = load_layer_selection(block_ckpt_path, local_rank)
    if local_rank == 0 and selected_layers is not None:
        print(f"Loaded Aletheia selected layers from {block_ckpt_path}: {selected_layers}")
        print(f"Loaded Aletheia structural layers from {block_ckpt_path}: {structural_layers}")
        if args.use_final_model and test_block < args.num_blocks - 1:
            print(
                "Warning: --use_final_model with dynamic Aletheia layers uses the final "
                "checkpoint layer structure. Per-block checkpoint evaluation is the exact "
                "match for block-specific selected_layers.json files."
            )

    _vocab_num_old_tokens, _vocab_freeze_embed, _vocab_freeze_head = resolve_vocab_freeze_config(
        block_ckpt_path, local_rank
    )

    call_apply_continual_lora(
        args,
        model,
        structural_layers,
        (_vocab_num_old_tokens, _vocab_freeze_embed, _vocab_freeze_head),
        local_rank,
    )

    model_ckpt = os.path.join(block_ckpt_path, "model.safetensors")
    if not os.path.exists(model_ckpt):
        raise RuntimeError(f"Checkpoint for block {test_block} not found at {model_ckpt}!")

    state_dict = load_file(model_ckpt, device="cpu")
    # Convert partial-vocab checkpoint format if needed
    if _vocab_num_old_tokens is not None:
        state_dict = convert_partial_vocab_checkpoint(
            state_dict, _vocab_num_old_tokens,
            _vocab_freeze_embed, _vocab_freeze_head, local_rank
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Strict validation for vocab keys
    vocab_missing = [k for k in missing if "modules_to_save" in k or "new_vocab_rows" in k]
    vocab_unexpected = [k for k in unexpected if "modules_to_save" in k or "new_vocab_rows" in k]
    if vocab_missing or vocab_unexpected:
        raise RuntimeError(
            f"Vocabulary checkpoint mismatch at {model_ckpt}! "
            f"missing={vocab_missing}, unexpected={vocab_unexpected}"
        )
    if local_rank == 0:
        orphan_vocab_keys = [
            key for key in unexpected
            if "modules_to_save.default" in key or key.endswith("new_vocab_rows")
        ]
        if orphan_vocab_keys:
            print(
                f"Warning: {len(orphan_vocab_keys)} vocabulary-module keys were not consumed, "
                f"e.g. {orphan_vocab_keys[:3]}"
            )
    reconstruct_saved_parameter_lists(model, state_dict, local_rank)

    if args.use_final_model and test_block < args.num_blocks - 1:
        subset_file = os.path.join(args.ckpt_path, f"block_{test_block}", "modules_to_save.safetensors")
        if not os.path.exists(subset_file):
            raise FileNotFoundError(f"modules_to_save.safetensors not found in {subset_file}")

        subset_state = load_file(subset_file, device="cpu")
        missing, unexpected = model.load_state_dict(subset_state, strict=False)
        if local_rank == 0:
            print(f"Loaded per-block modules_to_save from: {subset_file}")
            if missing:
                print(f"Missing keys when loading subset: {len(missing)} (showing up to 5): {missing[:5]}")
            if unexpected:
                print(f"Unexpected keys when loading subset: {len(unexpected)} (showing up to 5): {unexpected[:5]}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # --- decode protocol -------------------------------------------------
    # Pin do_sample on the config as well as on the generate() call so the two
    # cannot drift apart.  "sampling" leaves temperature/top_p exactly as the
    # base model shipped them, which reproduces the official repo (it never
    # passes do_sample, so it inherits Llama-3.2-1B's do_sample=true /
    # temperature=0.6 / top_p=0.9).
    shipped = {
        "do_sample": bool(model.generation_config.do_sample),
        "temperature": model.generation_config.temperature,
        "top_p": model.generation_config.top_p,
    }
    do_sample = decode_do_sample(args)
    model.generation_config.do_sample = do_sample
    if not do_sample:
        # temperature/top_p are inert when do_sample=False.  Reset them to their
        # neutral values so GenerationConfig.validate() does not emit a warning
        # on every generate() call (17k+ lines of log noise per block).
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0

    generation_mode = assert_decode_protocol(
        args, model.generation_config, args.num_beams, local_rank, shipped
    )
    args.eval_protocol = {
        "harness": EVAL_HARNESS_VERSION,
        "decode_mode": args.eval_decode_mode,
        "generation_mode": generation_mode,
        "do_sample": do_sample,
        "shipped_do_sample": shipped["do_sample"],
        "shipped_temperature": shipped["temperature"],
        "shipped_top_p": shipped["top_p"],
        "effective_temperature": model.generation_config.temperature,
        "effective_top_p": model.generation_config.top_p,
        "num_beams": args.num_beams,
        "sampler": "ExactDistributedEvalSampler",
        "oom_policy": "raise (num_beams is never silently reduced)",
        "vocab_key_validation": "strict (raises on missing/unexpected vocab keys)",
    }

    return model


def continual_test_ddp(args):
    """Evaluate PESO checkpoints on continual test blocks."""
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))
        guard_results_file_protocol(args)

    if args.lora_modules_to_save == "":
        args.lora_modules_to_save = None
    else:
        args.lora_modules_to_save = args.lora_modules_to_save.split(",")
    if args.lora_target_modules == "":
        args.lora_target_modules = None
    else:
        args.lora_target_modules = args.lora_target_modules.split(",")

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_ids = [0]

    test_blocks = range(args.num_blocks) if args.test_all_blocks else [args.test_block]
    all_block_results = {}

    for test_block in test_blocks:
        if test_block == 0 and args.skip_block0:
            all_block_results[f"block_{test_block}"] = {
                "all_prompt_results": [{m: 0 for m in args.metrics.split(",")} for _ in prompt_ids]
            }
            continue

        if local_rank == 0:
            print(f"\n{'=' * 50}")
            print(f"Testing on Block {test_block}")
            print(f"{'=' * 50}")

        use_final_model = args.use_final_model
        if use_final_model:
            block_ckpt_path = os.path.join(args.ckpt_path, f"block_{args.num_blocks - 1}")
        else:
            block_ckpt_path = os.path.join(args.ckpt_path, f"block_{test_block}")

        if "pretrain" in args.shift_flag and test_block > 0:
            use_final_model = False
            block_ckpt_path = os.path.join(args.ckpt_path, "block_0")

        if local_rank == 0:
            print(f"Loading model from: {block_ckpt_path}")

        original_use_final_model = args.use_final_model
        args.use_final_model = use_final_model
        model = load_continual_model(args, tokenizer, device_map, local_rank, block_ckpt_path, test_block)
        args.use_final_model = original_use_final_model
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])

        for _, module in model.module.named_modules():
            if hasattr(module, "set_saved_matrices"):
                module.set_saved_matrices(test_block, use_final_model=use_final_model)

        test_data = ContinualSeqRecDataset_v2(
            args,
            mode="test",
            current_block=test_block,
            num_blocks=args.num_blocks,
        )

        if local_rank == 0:
            print(f"Test data size for block {test_block}: {len(test_data)}")

        ddp_sampler = ExactDistributedEvalSampler(
            test_data, num_replicas=world_size, rank=local_rank
        )
        collator = TestCollator(args, tokenizer)
        all_items = test_data.get_all_items()
        prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

        test_loader = DataLoader(
            test_data,
            batch_size=args.test_batch_size,
            collate_fn=collator,
            sampler=ddp_sampler,
            num_workers=2,
            pin_memory=True,
        )

        model.eval()
        metrics = args.metrics.split(",")
        all_prompt_results = []

        with torch.no_grad():
            for prompt_id in prompt_ids:
                if local_rank == 0:
                    print(f"Testing prompt {prompt_id} on block {test_block}")

                test_loader.dataset.set_prompt(prompt_id)
                local_topk_results = []

                for _, batch in enumerate(tqdm(test_loader, disable=local_rank != 0)):
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    num_beams = args.num_beams

                    try:
                        output = model.module.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            # Explicit, never inherited: this single omitted
                            # argument is what silently switched six directories
                            # to BEAM_SAMPLE and offset every metric by ~7%.
                            do_sample=decode_do_sample(args),
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                        )
                    except torch.cuda.OutOfMemoryError as exc:
                        raise RuntimeError(
                            "Deterministic evaluation must keep --num_beams fixed; "
                            "reduce the configured beam count and rerun every compared method."
                        ) from exc

                    output_ids = output["sequences"]
                    scores = output["sequences_scores"]
                    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                    topk_res = get_topk_results(
                        output,
                        scores,
                        targets,
                        num_beams,
                        all_items=all_items if args.filter_items else None,
                    )

                    local_topk_results.extend(topk_res)

                # Gather exactly once, after ranks with different local shard
                # sizes have finished.  The legacy per-step gather deadlocked
                # (or forced dropped rows) when a block had an odd test count.
                gathered_results = [None for _ in range(world_size)]
                dist.all_gather_object(obj=local_topk_results, object_list=gathered_results)

                if local_rank == 0:
                    all_device_topk_res = []
                    for rank_results in gathered_results:
                        all_device_topk_res.extend(rank_results)
                    total = len(all_device_topk_res)
                    if total != len(test_data):
                        raise RuntimeError(
                            f"Exact evaluation coverage failed for block {test_block}: "
                            f"got {total}, expected {len(test_data)}"
                        )
                    metrics_results = get_metrics_results(all_device_topk_res, metrics)
                    metrics_results = {
                        metric_name: value / total
                        for metric_name, value in metrics_results.items()
                    }

                    all_prompt_results.append(metrics_results)
                    print(f"Block {test_block}, Prompt {prompt_id} results:", metrics_results)

                dist.barrier()

        if local_rank == 0:
            print(f"\nBlock {test_block} Final Results:")
            all_block_results[f"block_{test_block}"] = {
                "all_prompt_results": all_prompt_results,
                "test_count": len(test_data),
            }

        dist.barrier()
        del model
        torch.cuda.empty_cache()

    if local_rank == 0:
        save_data = {
            "prompt_id": 0,
            "test_blocks": list(test_blocks),
            "eval_protocol": getattr(args, "eval_protocol", None),
            "all_block_results": all_block_results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Results saved to: {args.results_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual LLMRec Test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = parse_train_args(parser)

    parser.add_argument("--num_blocks", type=int, default=5, help="Number of data blocks for continual learning")
    parser.add_argument("--test_block", type=int, default=0, help="Specific block to test on")
    parser.add_argument("--test_all_blocks", action="store_true", default=False, help="Test on all blocks")
    parser.add_argument("--skip_block0", action="store_true", default=False, help="Skip block 0")

    parser.add_argument(
        "--eval_decode_mode",
        type=str,
        default="deterministic",
        choices=DECODE_MODES,
        help=(
            "Decode protocol used by generate(). "
            "'deterministic' passes do_sample=False -> GenerationMode.BEAM_SEARCH, "
            "returning the num_beams highest-scoring sequences, which is what "
            "Hit@K/NDCG@K are defined over; zero variance, seed-independent. "
            "'sampling' inherits the base model's generation_config -> "
            "GenerationMode.BEAM_SAMPLE for Llama-3.2-1B (do_sample=true, "
            "temperature=0.6, top_p=0.9), reproducing the official hsyoo32/peso "
            "released behaviour. Results files record which one produced them and "
            "refuse to be overwritten by the other. "
            "NEVER compare numbers produced under different modes."
        ),
    )

    args = parser.parse_args()

    if not args.test_all_blocks and args.test_block is None:
        args.test_block = 0

    continual_test_ddp(args)

