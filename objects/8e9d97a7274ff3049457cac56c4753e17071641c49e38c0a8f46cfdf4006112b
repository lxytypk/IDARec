import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from collator import TestCollator
from continual_data_v2 import ContinualSeqRecDataset_v2
from continual_lora import apply_continual_lora
from evaluate import get_metrics_results, get_topk_results
from safetensors.torch import load_file
from utils import *
from evaluation_audit import *


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


def load_continual_model(args, tokenizer, device_map, local_rank, block_ckpt_path, test_block):
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))

    apply_continual_lora(
        model,
        args.lora_target_modules,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_modules_to_save,
        num_blocks=args.num_blocks,
        option=args.shift_flag,
    )

    model_ckpt = os.path.join(block_ckpt_path, "model.safetensors")
    if not os.path.exists(model_ckpt):
        raise RuntimeError(f"Checkpoint for block {test_block} not found at {model_ckpt}!")

    state_dict = load_file(model_ckpt, device="cpu")
    missing_full, unexpected_full = model.load_state_dict(state_dict, strict=False)
    model._load_audit = dict(final_missing=missing_full, final_unexpected_before_history_reconstruction=unexpected_full)
    reconstruct_saved_parameter_lists(model, state_dict, local_rank)

    if args.use_final_model and test_block < args.num_blocks - 1:
        subset_file = os.path.join(args.ckpt_path, f"block_{test_block}", "modules_to_save.safetensors")
        if not os.path.exists(subset_file):
            raise FileNotFoundError(f"modules_to_save.safetensors not found in {subset_file}")

        subset_state = load_file(subset_file, device="cpu")
        missing, unexpected = model.load_state_dict(subset_state, strict=False)
        model._load_audit.update(subset_source=subset_file, subset_missing=missing, subset_unexpected=unexpected)
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

    pin_generation(model, args, test_block)
    return model


def continual_test_ddp(args):
    """Evaluate PESO checkpoints on continual test blocks."""
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == 2, world_size
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

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

    assert args.test_all_blocks and args.use_final_model and args.skip_block0
    test_blocks = [1, 2, 3, 4]
    all_block_results = {}
    if local_rank == 0: save_progress(args, all_block_results)

    for test_block in test_blocks:
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
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

        for _, module in model.module.named_modules():
            if hasattr(module, "set_saved_matrices"):
                module.set_saved_matrices(test_block, use_final_model=use_final_model)
        audit_stage(model.module, args, test_block)

        test_data = ContinualSeqRecDataset_v2(
            args,
            mode="test",
            current_block=test_block,
            num_blocks=args.num_blocks,
        )

        if local_rank == 0:
            print(f"Test data size for block {test_block}: {len(test_data)}")

        coverage_info = coverage_metadata(test_data, args, test_block, tokenizer)
        ddp_sampler = ExactSampler(test_data, num_replicas=world_size, rank=local_rank)
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
                metrics_results = {}
                total = 0
                seen_indices = []
                shard_indices = list(iter(ddp_sampler))

                for step, batch in enumerate(tqdm(test_loader)):
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    bs = len(targets)
                    seen_indices.extend(shard_indices[len(seen_indices):len(seen_indices)+bs])
                    num_beams = args.num_beams

                    assert not model.module.training
                    assert model.module.generation_config.get_generation_mode(None) == GenerationMode.BEAM_SEARCH
                    output = model.module.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        do_sample=False, num_beams=10, num_return_sequences=10,
                        max_new_tokens=10, early_stopping=True, synced_gpus=False,
                        output_scores=True, return_dict_in_generate=True,
                    )

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

                    total += bs
                    batch_metrics_res = get_metrics_results(topk_res, metrics)
                    for metric_name, value in batch_metrics_res.items():
                        metrics_results[metric_name] = metrics_results.get(metric_name, 0.) + value
                    if (step + 1) % 100 == 0:
                        print(f"Block {test_block}, rank {local_rank}, local batches {step+1}/{len(test_loader)}", flush=True)

                metrics_results, total = gather_complete(metrics_results, total, seen_indices, coverage_info, test_block, world_size)
                if local_rank == 0:
                    all_prompt_results.append(metrics_results)
                    print(f"Block {test_block}, Prompt {prompt_id} results:", metrics_results, flush=True)

                dist.barrier()

        if local_rank == 0:
            print(f"\nBlock {test_block} Final Results:")
            all_block_results[f"block_{test_block}"] = {
                "all_prompt_results": all_prompt_results,
                "original_users": coverage_info["original_users"],
                "eligible_users": coverage_info["eligible_users"],
                "evaluated_users": total,
                "unique_users": total,
            }

        dist.barrier()
        if local_rank == 0: save_progress(args, all_block_results)
        del model
        torch.cuda.empty_cache()

    if local_rank == 0: save_progress(args, all_block_results)

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

    args = parser.parse_args()

    if not args.test_all_blocks and args.test_block is None:
        args.test_block = 0

    continual_test_ddp(args)
