#!/usr/bin/env python3
"""One-shot static matrix selection for a full Block1--Block4 PESO chain.

The scoring location is the frozen base projection W.  The selected update unit
is its current LoRA A/B pair.  This deliberately differs from layer selection:
all 112 projection modules remain in the model; only the selected pairs update.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import transformers
import wandb
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

import matrix_static_pilot as core
from collator import Collator
from continual_data_v2 import load_continual_datasets_v2
from continual_train import ContinualLoRATrainer, apply_lora_for_block, load_backbone_model, reconstruct_saved_parameter_lists
from utils import parse_dataset_args, parse_global_args, parse_train_args, set_seed
from replay import ReplayDataset, fingerprint, score_contribution
from continual_data_v2 import ContinualSeqRecDataset_v2


def rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stage_dir(args):
    return os.path.join(args.root, "ckpt", "block_%d" % args.block_id)


def score_path(args):
    return os.path.join(args.root, "selection", "block_%d_matrix_scores.json" % args.block_id)


def set_stage(model, block_id):
    for _, module in model.named_modules():
        if hasattr(module, "set_current_block"):
            module.set_current_block(block_id)


def build_model(args, tokenizer, local_rank, selected=None):
    """Build all 16 LoRA structures, then load exactly one previous checkpoint."""
    model = load_backbone_model(args, tokenizer, {"": local_rank}, local_rank, args.block_id)
    apply_lora_for_block(args, model, selected_layers=list(range(16)), trainable_layers=list(range(16)))
    state = load_file(os.path.join(args.source_checkpoint, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    reconstruct_saved_parameter_lists(model, state, local_rank)
    del state
    set_stage(model, args.block_id)
    candidates = core.candidate_modules(model)
    core.set_matrix_trainability(candidates, selected or set())
    return model, candidates, {"missing": len(missing), "unexpected": len(unexpected)}


def tokenizer_and_data(args):
    tokenizer, _, added = core.build_tokenizer(args)
    train_data, valid_data = load_continual_datasets_v2(
        args, current_block=args.block_id, num_blocks=args.num_blocks
    )
    return tokenizer, train_data, valid_data, added


def trainable_vocab_count(model):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "modules_to_save.default" in name and parameter.requires_grad
    )


def score(args):
    """Rank-zero, no-step grad*weight scoring on the fixed current-block probe."""
    if not rank0():
        return
    start = time.time()
    tokenizer, train_data, _, added = tokenizer_and_data(args)
    n = min(len(train_data), max(args.probe_batches, int(math.ceil(len(train_data) * args.probe_ratio))))
    generator = torch.Generator().manual_seed(args.probe_seed + args.block_id)
    indices = torch.randperm(len(train_data), generator=generator)[:n].tolist()
    loader = DataLoader(Subset(train_data, indices), batch_size=args.probe_batch_size, shuffle=False,
                        collate_fn=Collator(args, tokenizer), num_workers=0, pin_memory=False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, candidates, load_info = build_model(args, tokenizer, 0, set())
    model.to(device)
    costs = core.candidate_costs(candidates)
    before = {key: core.tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    required = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    prior_mode, prior_cache = model.training, model.config.use_cache
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for row in candidates.values():
        row["module"].orig.weight.requires_grad_(True)
    values = {key: 0.0 for key in candidates}
    processed = 0
    try:
        for batch in loader:
            model.zero_grad(set_to_none=True)
            loss = model(**core.move_to_device(batch, device)).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Probe loss is missing or non-finite")
            loss.backward()
            for key, row in candidates.items():
                weight = row["module"].orig.weight
                if weight.grad is None:
                    raise RuntimeError("No base-weight gradient for %s" % key)
                value = score_contribution(weight, weight.grad, args.score_mode)
                if not torch.isfinite(value):
                    raise RuntimeError("Non-finite score for %s" % key)
                values[key] += float(value.item())
            processed += 1
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(required[name])
            parameter.grad = None
        model.train(prior_mode)
        model.config.use_cache = prior_cache
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)
    after = {key: core.tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    changed = [key for key in candidates if not core.signature_equal(before[key], after[key])]
    if changed or processed != n or sum(values.values()) <= 0:
        raise RuntimeError("Scoring validation failed; changed=%s processed=%d expected=%d" % (changed[:4], processed, n))
    if args.score_mode == "grad_norm":
        values = {key: value / processed for key, value in values.items()}
    rows = []
    for key, row in candidates.items():
        rows.append({"matrix": key, "model_module": row["name"], "layer": row["layer"],
                     "projection": row["projection"], "base_weight_shape": list(row["module"].orig.weight.shape),
                     "raw_score": values[key], "lora_cost": costs[key],
                     "score_per_cost": values[key] / costs[key]})
    rows.sort(key=lambda x: (-x["score_per_cost"], x["layer"], core.MODULE_INDEX[x["projection"]]))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    full = sum(costs.values())
    budget = int(math.floor(args.budget_ratio * full))
    if len(rows) != 112 or full <= 0 or budget <= 0:
        raise RuntimeError("Invalid candidate/budget statistics")
    report = {
        "method": "static_matrix_" + args.score_mode + "_global_budget",
        "adaptation": "Llama continual recommendation; one pre-training selection per block; projection matrices only",
        "block_id": args.block_id, "source_checkpoint": os.path.realpath(args.source_checkpoint),
        "probe": {"sampling": "fixed_seed_uniform_without_replacement", "probe_seed": args.probe_seed + args.block_id,
                  "probe_ratio": args.probe_ratio, "requested_probe_batches": args.probe_batches,
                  "effective_probe_microbatches": n, "probe_batch_size": args.probe_batch_size,
                  "sample_indices": indices, "model_mode": "eval", "loss": "original response prediction loss",
                  "score": ("sum_over_microbatches(sum(abs(W_m * grad_W_m_b)))" if args.score_mode == "grad_weight_abs"
                            else "mean_over_microbatches(sqrt(sum(grad_W_m_b ** 2)))"),
                  "score_mode": args.score_mode, "optimizer_steps": 0},
        "budget": {"budget_ratio": args.budget_ratio, "all_112_lora_AB_parameters": full,
                   "budget_P": budget, "vocabulary_parameters_excluded_from_P": trainable_vocab_count(model)},
        "matrix_candidates": rows,
        "verification": {"candidate_count": len(rows), "base_weight_signatures_unchanged": True,
                         "random_states_restored": True, "load_info": load_info},
        "timing": {"score_seconds": time.time() - start,
                   "peak_gpu_memory_bytes_rank0": int(torch.cuda.max_memory_allocated(device)), "added_tokens": added},
    }
    write_json(score_path(args), report)
    del model
    torch.cuda.empty_cache()


def prepare(args):
    if not rank0():
        return
    probe = read_json(score_path(args))
    budget = probe["budget"]["budget_P"]
    selected, used = core.greedy_selection(probe["matrix_candidates"], budget)
    if used > budget or not selected:
        raise RuntimeError("Budget selection failed")
    selected_set = set(selected)
    directory = stage_dir(args)
    os.makedirs(directory, exist_ok=True)
    matrix_rows = probe["matrix_candidates"]
    metadata = {
        "method": "matrix_static_budget", "score_mode": args.score_mode,
        "replay_ratio": args.replay_ratio, "block_id": args.block_id,
        "source_checkpoint": probe["source_checkpoint"], "selected_layers": sorted({int(x.split(".")[1]) for x in selected}),
        "structural_layers": list(range(16)), "matrix_selected": selected,
        "matrix_unselected": sorted({row["matrix"] for row in matrix_rows} - selected_set, key=core.matrix_sort_key),
        "matrix_budget_P": budget, "actual_selected_lora_AB_parameters": used,
        "remaining_lora_budget": budget - used, "budget_ratio": args.budget_ratio,
        "vocabulary_parameters_excluded_from_budget": probe["budget"]["vocabulary_parameters_excluded_from_P"],
        "selection_rule": "global descending raw_score/lora_cost; fixed layer/module ties; skip candidates that do not fit; no fallback",
        "score_file": os.path.realpath(score_path(args)),
    }
    write_json(os.path.join(directory, "selected_layers.json"), metadata)
    write_json(os.path.join(args.root, "selection", "block_%d_manifest.json" % args.block_id), metadata)


def one_batch(args, tokenizer, data):
    return next(iter(DataLoader(Subset(data, [0]), batch_size=1, shuffle=False, collate_fn=Collator(args, tokenizer))))


def logits(model, batch, device):
    mode, cache = model.training, model.config.use_cache
    model.eval(); model.config.use_cache = False
    with torch.no_grad():
        value = model(**core.move_to_device(batch, device)).logits.detach().float().cpu()
    model.train(mode); model.config.use_cache = cache
    return value


def preflight(args):
    if not rank0():
        return
    selection = read_json(os.path.join(stage_dir(args), "selected_layers.json"))
    selected = set(selection["matrix_selected"])
    tokenizer, train_data, _, _ = tokenizer_and_data(args)
    device = torch.device("cuda", 0); torch.cuda.set_device(device)
    model, candidates, info = build_model(args, tokenizer, 0, selected)
    model.to(device)
    batch = one_batch(args, tokenizer, train_data)
    if args.replay_ratio:
        mixed = training_data_with_replay(args, train_data)
        batch = Collator(args, tokenizer)([train_data[0], mixed.buffers[0][0]["example"]])
        write_json(os.path.join(args.root, "replay", "block_%d_preflight_schedule.json" % args.block_id), mixed.manifest)
    baseline = logits(model, batch, device)
    before = core.selected_reference(model, candidates)
    base = {key: core.tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    model.train(); model.zero_grad(set_to_none=True)
    loss = model(**core.move_to_device(batch, device)).loss
    loss.backward()
    got = {key for key, row in candidates.items() if row["module"].A.grad is not None or row["module"].B.grad is not None}
    if not got or not got.issubset(selected):
        raise RuntimeError("Selected/unselected gradient boundary failed")
    vocab_trainable = [name for name, parameter in model.named_parameters()
                       if "modules_to_save.default" in name and parameter.requires_grad]
    vocab_with_grad = [name for name, parameter in model.named_parameters()
                       if "modules_to_save.default" in name and parameter.grad is not None]
    if not any("embed_tokens" in name for name in vocab_trainable) or not any("lm_head" in name for name in vocab_trainable):
        raise RuntimeError("Full-vocabulary embed_tokens/lm_head update path is not trainable")
    if not any("embed_tokens" in name for name in vocab_with_grad) or not any("lm_head" in name for name in vocab_with_grad):
        raise RuntimeError("Full-vocabulary embed_tokens/lm_head did not receive the response-loss gradient")
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-7, weight_decay=0.0)
    opt.step()
    changes = core.update_report(candidates, before, selected)
    base_changed = [key for key in candidates if not core.signature_equal(base[key], core.tensor_signature(candidates[key]["module"].orig.weight))]
    if changes["selected_total_squared_change"] <= 0 or changes["unselected_total_squared_change"] != 0 or base_changed:
        raise RuntimeError("Small update freeze verification failed")
    write_json(os.path.join(args.root, "selection", "block_%d_preflight.json" % args.block_id), {
        "initial_logit_signature": core.tensor_signature(baseline), "selected_count": len(selected),
        "candidate_count": len(candidates), "small_update": changes,
        "base_weights_unchanged": True, "load_info": info,
        "full_vocab_trainable_parameter_names": vocab_trainable,
        "full_vocab_gradient_parameter_names": vocab_with_grad,
    })
    del model
    torch.cuda.empty_cache()


def tensor_hash(tensor):
    # NumPy cannot materialize bfloat16 on older builds.  Hash raw storage bytes
    # so BF16 checkpoints are checked exactly rather than after a lossy cast.
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def training_data_with_replay(args, current):
    if args.replay_ratio == 0:
        return current
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    buffers = {}
    try:
        for block in range(args.block_id):
            historical = ContinualSeqRecDataset_v2(args, mode="train", current_block=block,
                                                  num_blocks=args.num_blocks, prompt_sample_num=1)
            if len(historical) < args.history_samples:
                raise RuntimeError("Fewer historical examples than requested fixed memory")
            indices = random.Random(args.history_seed + block).sample(range(len(historical)), args.history_samples)
            records = [{"source_train_index": i, "example": historical[i]} for i in indices]
            source_file = os.path.join(args.data_path, args.dataset, "%s.all.block_%d.json" % (args.dataset, block))
            with open(source_file, "rb") as f:
                data_hash = hashlib.sha256(f.read()).hexdigest()
            metadata = {"block": block, "split": "train", "seed": args.history_seed + block,
                        "dataset_length": len(historical), "source_file": os.path.realpath(source_file),
                        "source_sha256": data_hash, "sample_indices": indices, "records": records}
            path = os.path.join(args.root, "history_memory", "block_%d.json" % block)
            if os.path.exists(path) and read_json(path) != metadata:
                raise RuntimeError("Historical training buffer changed: " + path)
            if rank0() and not os.path.exists(path):
                write_json(path, metadata)
            buffers[block] = records
        mixed = ReplayDataset(current, buffers, args.replay_ratio, args.history_seed, args.block_id)
        # Validate that replay examples take exactly the original response-loss route.
        if set(mixed[0]) != set(current[0]) or len(mixed) != len(current):
            raise RuntimeError("Replay changed dataset contract or step budget")
        return mixed
    finally:
        random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(cpu_state)


class ReplayEpochCallback(transformers.TrainerCallback):
    def __init__(self, dataset, experiment_args):
        self.dataset, self.experiment_args = dataset, experiment_args

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        self.dataset.set_epoch(epoch)
        digest = self.dataset.manifest["mapping_sha256"]
        if dist.is_initialized():
            digests = [None] * dist.get_world_size()
            dist.all_gather_object(digests, digest)
            if len(set(digests)) != 1:
                raise RuntimeError("Replay schedule differs across ranks")
        if rank0():
            write_json(os.path.join(self.experiment_args.root, "replay", "block_%d_epoch_%d.json" %
                       (self.experiment_args.block_id, epoch)), self.dataset.manifest)
            print("REPLAY epoch=%d current=%d historical=%d digest=%s" %
                  (epoch, self.dataset.manifest["current_count"], self.dataset.manifest["replay_count"], digest), flush=True)


def train(args):
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    selection = read_json(os.path.join(stage_dir(args), "selected_layers.json"))
    selected = set(selection["matrix_selected"])
    digest = hashlib.sha256(json.dumps(sorted(selected)).encode()).hexdigest()
    all_digests = [None] * dist.get_world_size(); dist.all_gather_object(all_digests, digest)
    if len(set(all_digests)) != 1:
        raise RuntimeError("Ranks disagree on selected matrices")
    local_rank = int(os.environ.get("LOCAL_RANK", "0")); device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    tokenizer, train_data, valid_data, added = tokenizer_and_data(args)
    original_train_length = len(train_data)
    train_data = training_data_with_replay(args, train_data)
    check_batch = one_batch(args, tokenizer, valid_data)
    model, candidates, load_info = build_model(args, tokenizer, local_rank, selected)
    before = core.selected_reference(model, candidates)
    if rank0():
        wandb.init(project="matrix_static_full50_v1", name="block%d" % args.block_id, config={**vars(args), "selected_count": len(selected)})
    learning_rate = args.learning_rate * 0.1
    out = stage_dir(args)
    if rank0():
        # Evaluation must reload the exact semantic-token vocabulary used in
        # training, rather than rebuilding it from whichever data path is live.
        os.makedirs(out, exist_ok=True)
        tokenizer.save_pretrained(out)
    trainer = ContinualLoRATrainer(
        model=model, train_dataset=train_data, eval_dataset=valid_data,
        args=transformers.TrainingArguments(seed=args.seed, per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio, num_train_epochs=args.epochs, learning_rate=learning_rate,
            weight_decay=args.weight_decay, lr_scheduler_type=args.lr_scheduler_type, report_to=["wandb"], fp16=args.fp16,
            bf16=args.bf16, logging_steps=args.logging_step, optim=args.optim, gradient_checkpointing=False,
            eval_strategy=args.save_and_eval_strategy, save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps, save_steps=args.save_and_eval_steps, output_dir=out, save_total_limit=1,
            load_best_model_at_end=True, deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False, eval_delay=1 if args.save_and_eval_strategy == "epoch" else 0,
            label_names=["labels"], dataloader_drop_last=False),
        processing_class=tokenizer, data_collator=Collator(args, tokenizer), continual_loss_weight=0.0)
    if isinstance(train_data, ReplayDataset):
        if trainer.args.dataloader_num_workers != 0:
            raise RuntimeError("Epoch replay schedule requires zero DataLoader workers")
        trainer.add_callback(ReplayEpochCallback(train_data, args))
    if rank0():
        write_json(os.path.join(out, "launch_config.json"), {
            "experiment_args": vars(args), "training_arguments": trainer.args.to_dict(),
            "original_train_length": original_train_length, "actual_train_length": len(train_data),
            "score_mode": args.score_mode, "replay_ratio": args.replay_ratio,
            "selected_digest": digest,
        })
    model.config.use_cache = False
    trainer.train()
    peak = int(torch.cuda.max_memory_allocated(device))
    peaks = [None] * dist.get_world_size()
    dist.all_gather_object(peaks, peak)
    core_model = getattr(trainer.model, "module", trainer.model)
    for _, module in core_model.named_modules():
        if hasattr(module, "save_current_matrices"):
            module.save_current_matrices()
    barrier()
    if rank0():
        pre_logits = logits(core_model, check_batch, device)
        pre_hashes = {key: tensor_hash(value) for key, value in core_model.state_dict().items()}
        trainer.save_state(); trainer.save_model(output_dir=out)
        for item in os.listdir(out):
            if item.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(out, item))
        updates = core.update_report(candidates, before, selected)
        write_json(os.path.join(out, "matrix_update_report.json"), updates)
        write_json(os.path.join(out, "training_metadata.json"), {
            "source_checkpoint": os.path.realpath(args.source_checkpoint), "block_id": args.block_id,
            "selected_digest": digest, "selected_count": len(selected), "effective_learning_rate": learning_rate,
            "added_tokens": added, "training_seconds": time.time() - start,
            "score_mode": args.score_mode, "replay_ratio": args.replay_ratio,
            "original_train_length": original_train_length, "actual_train_length": len(train_data),
            "global_steps": trainer.state.global_step, "peak_gpu_memory_bytes_by_rank": peaks,
            "peak_gpu_memory_bytes_rank0": int(torch.cuda.max_memory_allocated(device)), "load_info": load_info,
        })
        del trainer, model, core_model
        torch.cuda.empty_cache()
        verify_checkpoint(args, pre_hashes, pre_logits, check_batch, tokenizer, selected, device)
        if wandb.run is not None:
            wandb.finish()
    barrier()
    dist.destroy_process_group()


def verify_checkpoint(args, pre_hashes=None, pre_logits=None, check_batch=None, tokenizer=None, selected=None, device=None):
    """Prove serialization first, then prove two independent reloads agree."""
    if not rank0():
        return
    directory = stage_dir(args)
    selected = selected or set(read_json(os.path.join(directory, "selected_layers.json"))["matrix_selected"])
    tokenizer = tokenizer or tokenizer_and_data(args)[0]
    device = device or torch.device("cuda", 0)
    if check_batch is None:
        _, _, valid, _ = tokenizer_and_data(args); check_batch = one_batch(args, tokenizer, valid)
    state = load_file(os.path.join(directory, "model.safetensors"), device="cpu")
    saved_hashes = {key: tensor_hash(value) for key, value in state.items()}
    if pre_hashes is not None and pre_hashes != saved_hashes:
        raise RuntimeError("Saved tensors differ from the post-training in-memory model")
    reload_logits, reload_infos = [], []
    for pass_id in range(2):
        model, candidates, info = build_model(args, tokenizer, 0, selected)
        missing, unexpected = model.load_state_dict(state, strict=False)
        reconstruct_saved_parameter_lists(model, state, 0)
        restored = model.state_dict()
        mismatches = [key for key, value in state.items()
                      if key not in restored or not torch.equal(value.cpu(), restored[key].detach().cpu())]
        if mismatches:
            raise RuntimeError("Saved state failed exact reload: %s" % mismatches[:4])
        set_stage(model, args.block_id); core.set_matrix_trainability(candidates, selected)
        model.to(device)
        reload_logits.append(logits(model, check_batch, device))
        reload_infos.append({"pass": pass_id, "missing": len(missing), "unexpected": len(unexpected),
                             "build_load_info": info})
        del model
        torch.cuda.empty_cache()
    if not torch.equal(reload_logits[0], reload_logits[1]):
        max_delta = float((reload_logits[0] - reload_logits[1]).abs().max().item())
        raise RuntimeError("Two independent reloads disagree; max_abs_logit_delta=%g" % max_delta)
    in_memory_match = pre_logits is None or torch.equal(pre_logits, reload_logits[0])
    in_memory_max_delta = None if pre_logits is None else float((pre_logits - reload_logits[0]).abs().max().item())
    write_json(os.path.join(directory, "checkpoint_reload_validation.json"), {
        "saved_tensors_match_in_memory": pre_hashes is not None,
        "saved_tensors_reload_exactly": True,
        "two_independent_reloads_logits_exactly_equal": True,
        "in_memory_and_reloaded_logits_exactly_equal": in_memory_match,
        "in_memory_to_reload_max_abs_logit_delta": in_memory_max_delta,
        "logit_signature": core.tensor_signature(reload_logits[0]),
        "reload_passes": reload_infos,
        "note": "Exact tensor serialization plus identical fresh reloads is required. Any in-memory versus reload difference is retained as a diagnostic, not silently discarded.",
    })
    del state
    torch.cuda.empty_cache()


def parser():
    p = argparse.ArgumentParser(description="Full static 75% matrix LoRA selection")
    p = parse_global_args(p); p = parse_train_args(p); p = parse_dataset_args(p)
    p.add_argument("command", choices=("score", "prepare", "preflight", "train", "verify"))
    p.add_argument("--root", required=True); p.add_argument("--source_checkpoint", required=True)
    p.add_argument("--block_id", required=True, type=int, choices=(1, 2, 3, 4))
    p.add_argument("--num_blocks", type=int, default=5); p.add_argument("--probe_batches", type=int, default=8)
    p.add_argument("--probe_batch_size", type=int, default=1); p.add_argument("--probe_ratio", type=float, default=0.1)
    p.add_argument("--probe_seed", type=int, default=42); p.add_argument("--budget_ratio", type=float, default=0.75)
    p.add_argument("--score_mode", choices=("grad_weight_abs", "grad_norm"), default="grad_weight_abs")
    p.add_argument("--replay_ratio", type=float, default=0.0)
    p.add_argument("--history_samples", type=int, default=256)
    p.add_argument("--history_seed", type=int, default=42)
    return p


def main():
    args = core.normalize_args(parser().parse_args())
    if not 0 < args.budget_ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    if args.probe_batch_size != 1 or not 0 <= args.replay_ratio < 1:
        raise ValueError("This audited experiment requires probe batch 1 and replay ratio in [0,1)")
    set_seed(args.seed); os.makedirs(args.root, exist_ok=True)
    if args.command == "score": score(args)
    elif args.command == "prepare": prepare(args)
    elif args.command == "preflight": preflight(args)
    elif args.command == "train": train(args)
    else: verify_checkpoint(args)


if __name__ == "__main__":
    main()
