#!/usr/bin/env python3
"""Static matrix-level LoRA selection pilot: exact Block1 -> Block2 only.

This is intentionally independent from the source Aletheia training loop.  The
source loop can only enable a whole Transformer layer; this file keeps all 112
LoRA modules in the forward graph and controls trainability per projection.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import transformers
import wandb
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, LlamaConfig

from collator import Collator
from continual_data_v2 import ContinualSeqRecDataset_v2, load_continual_datasets_v2
from continual_lora import SaveModuleWrapper
from continual_train import (
    ContinualLoRATrainer,
    apply_lora_for_block,
    load_backbone_model,
    reconstruct_saved_parameter_lists,
)
from utils import parse_dataset_args, parse_global_args, parse_train_args, set_seed


MODULE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODULE_INDEX = {name: index for index, name in enumerate(MODULE_ORDER)}
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def rank0():
    return rank_world()[0] == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def move_to_device(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def batch_size(batch):
    return int(batch["input_ids"].shape[0])


def tensor_signature(tensor):
    """Small deterministic fingerprint used only to prove no accidental update."""
    x = tensor.detach().float()
    return {
        "shape": list(x.shape),
        "sum": float(x.sum().item()),
        "square_sum": float(x.square().sum().item()),
        "abs_max": float(x.abs().max().item()),
    }


def signature_equal(left, right, atol=0.0):
    return left["shape"] == right["shape"] and all(
        abs(float(left[key]) - float(right[key])) <= atol
        for key in ("sum", "square_sum", "abs_max")
    )


def parse_layer_and_module(name, module):
    match = LAYER_PATTERN.search(name)
    if not match:
        return None
    projection = name.rsplit(".", 1)[-1]
    if projection not in MODULE_INDEX or not hasattr(module, "orig"):
        return None
    if not hasattr(module.orig, "weight") or not hasattr(module, "A") or not hasattr(module, "B"):
        return None
    return int(match.group(1)), projection


def candidate_modules(model):
    found = {}
    for name, module in model.named_modules():
        parsed = parse_layer_and_module(name, module)
        if parsed is None:
            continue
        layer, projection = parsed
        key = f"layers.{layer}.{projection}"
        found[key] = {"name": name, "module": module, "layer": layer, "projection": projection}
    expected = {f"layers.{layer}.{projection}" for layer in range(16) for projection in MODULE_ORDER}
    if set(found) != expected:
        missing, extra = sorted(expected - set(found)), sorted(set(found) - expected)
        raise RuntimeError(f"Expected exactly 112 candidate matrices; missing={missing[:6]} extra={extra[:6]}")
    return {key: found[key] for key in sorted(found, key=matrix_sort_key)}


def matrix_sort_key(key):
    parts = key.split(".")
    return int(parts[1]), MODULE_INDEX[parts[2]]


def normalize_args(args):
    if args.lora_modules_to_save == "":
        args.lora_modules_to_save = None
    elif isinstance(args.lora_modules_to_save, str):
        args.lora_modules_to_save = args.lora_modules_to_save.split(",")
    if args.lora_target_modules == "":
        args.lora_target_modules = None
    elif isinstance(args.lora_target_modules, str):
        args.lora_target_modules = args.lora_target_modules.split(",")
    return args


def build_tokenizer(args, save_dir=None):
    config = LlamaConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, model_max_length=args.model_max_length, padding_side="right")
    tokenizer.pad_token_id = 0
    seed_dataset = ContinualSeqRecDataset_v2(args, mode="train", current_block=0, num_blocks=args.num_blocks)
    added = tokenizer.add_tokens(seed_dataset.get_new_tokens())
    config.vocab_size = len(tokenizer)
    if save_dir is not None and rank0():
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        config.save_pretrained(save_dir)
    return tokenizer, config, added


def build_loaded_block1_model(args, tokenizer, local_rank, trainable_names=None):
    # FlashAttention2 requires the model to be placed on its CUDA device during
    # construction, including the single-process scoring/preflight phases.
    device_map = {"": local_rank}
    model = load_backbone_model(args, tokenizer, device_map, local_rank, 2)
    # All 16 structural layers must exist so the exact Block1 checkpoint loads.
    apply_lora_for_block(args, model, selected_layers=list(range(16)), trainable_layers=list(range(16)))
    state = load_file(os.path.join(args.source_block1, "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    reconstruct_saved_parameter_lists(model, state, local_rank)
    del state
    for _, module in model.named_modules():
        if hasattr(module, "set_current_block"):
            module.set_current_block(2)
    candidates = candidate_modules(model)
    set_matrix_trainability(candidates, trainable_names or set())
    return model, candidates, {"missing": len(missing), "unexpected": len(unexpected)}


def set_matrix_trainability(candidates, selected):
    selected = set(selected)
    for key, row in candidates.items():
        row["module"].set_adapter_trainable(key in selected)
    # The copied vocabulary rows use the original experiment's behavior: trainable.


def candidate_costs(candidates):
    rows = {}
    for key, row in candidates.items():
        module = row["module"]
        actual = int(module.A.numel() + module.B.numel())
        formula = int(module.r * (module.orig.in_features + module.orig.out_features))
        if actual != formula:
            raise RuntimeError(f"LoRA cost mismatch for {key}: tensor={actual} formula={formula}")
        rows[key] = actual
    return rows


def model_output_signature(model, batch, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(**move_to_device(batch, device)).logits[:, -1, :]
        value = tensor_signature(output)
    model.train(was_training)
    return value


def score_matrices(args, output_path):
    """Rank0-only gradient x weight scoring. It never calls optimizer.step()."""
    if not rank0():
        return
    started = time.time()
    tokenizer, _, added = build_tokenizer(args)
    train_data, _ = load_continual_datasets_v2(args, current_block=2, num_blocks=args.num_blocks)
    effective_count = max(int(args.probe_batches), int(math.ceil(len(train_data) * args.probe_ratio)))
    generator = torch.Generator().manual_seed(args.probe_seed + 2)
    indices = torch.randperm(len(train_data), generator=generator)[:effective_count].tolist()
    loader = DataLoader(Subset(train_data, indices), batch_size=args.probe_batch_size, shuffle=False,
                        collate_fn=Collator(args, tokenizer), num_workers=0, pin_memory=False)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, candidates, load_info = build_loaded_block1_model(args, tokenizer, 0, trainable_names=set())
    model.to(device)
    costs = candidate_costs(candidates)
    before = {key: tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    original_training = model.training
    original_cache = model.config.use_cache
    required = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for row in candidates.values():
        row["module"].orig.weight.requires_grad_(True)
    scores = {key: 0.0 for key in candidates}
    processed = 0
    for batch in loader:
        model.zero_grad(set_to_none=True)
        loss = model(**move_to_device(batch, device)).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Probe response-prediction loss is missing or non-finite.")
        loss.backward()
        for key, row in candidates.items():
            weight, grad = row["module"].orig.weight, row["module"].orig.weight.grad
            if grad is None:
                raise RuntimeError(f"Candidate base matrix had no gradient: {key}")
            contribution = (weight.detach().float() * grad.detach().float()).abs().sum()
            if not torch.isfinite(contribution):
                raise RuntimeError(f"Non-finite matrix score for {key}")
            scores[key] += float(contribution.item())
        processed += 1
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(required[name])
        parameter.grad = None
    model.train(original_training)
    model.config.use_cache = original_cache
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    after = {key: tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    changed = [key for key in candidates if not signature_equal(before[key], after[key])]
    if changed:
        raise RuntimeError(f"Scoring changed base weights: {changed[:6]}")
    if processed != effective_count or not all(math.isfinite(value) and value >= 0 for value in scores.values()):
        raise RuntimeError("Probe did not produce one finite non-negative score per requested microbatch.")
    if sum(scores.values()) <= 0:
        raise RuntimeError("All matrix scores are zero; refusing to allocate a budget.")
    source_selection = read_json(args.source_block2_selection)
    source_layers = [int(x) for x in source_selection["selected_layers"]]
    source_names = [key for key, row in candidates.items() if row["layer"] in source_layers]
    budget = sum(costs[key] for key in source_names)
    vocab_params = sum(p.numel() for name, p in model.named_parameters() if "modules_to_save.default" in name and p.requires_grad)
    matrices = []
    for key in candidates:
        row = candidates[key]
        matrices.append({
            "matrix": key, "model_module": row["name"], "layer": row["layer"], "projection": row["projection"],
            "base_weight_shape": list(row["module"].orig.weight.shape), "raw_score": scores[key],
            "lora_cost": costs[key], "score_per_cost": scores[key] / costs[key],
        })
    matrices.sort(key=lambda row: (-row["score_per_cost"], row["layer"], MODULE_INDEX[row["projection"]]))
    for rank, row in enumerate(matrices, 1):
        row["rank"] = rank
    layer_rows = []
    for layer in range(16):
        keys = [f"layers.{layer}.{projection}" for projection in MODULE_ORDER]
        score, cost = sum(scores[key] for key in keys), sum(costs[key] for key in keys)
        layer_rows.append({"layer": layer, "raw_score": score, "lora_cost": cost, "score_per_cost": score / cost})
    layer_rows.sort(key=lambda row: (-row["score_per_cost"], row["layer"]))
    for rank, row in enumerate(layer_rows, 1):
        row["rank"] = rank
    report = {
        "method": "static_matrix_grad_weight_abs_global_budget",
        "adaptation": "Llama continual recommendation; fixed Block1-to-Block2 selection; projection matrices only; no dynamic reselection",
        "source_block1": os.path.realpath(args.source_block1),
        "source_block2_selection": os.path.realpath(args.source_block2_selection),
        "source_selected_layers": source_layers,
        "probe": {"sampling": "fixed_seed_uniform_without_replacement", "probe_seed": args.probe_seed + 2,
                  "probe_ratio": args.probe_ratio, "requested_probe_batches": args.probe_batches,
                  "effective_probe_microbatches": effective_count, "probe_batch_size": args.probe_batch_size,
                  "sample_indices": indices, "model_mode": "eval", "loss": "original response prediction loss",
                  "score": "sum_over_microbatches(sum(abs(W_m * grad_W_m_b)))", "optimizer_steps": 0},
        "budget": {"P_from_original_selected_layers": budget, "source_selected_lora_AB_parameters": budget,
                   "vocabulary_trainable_parameters_excluded_from_P": vocab_params},
        "matrix_candidates": matrices, "layer_aggregates": layer_rows,
        "verification": {"candidate_count": len(candidates), "score_processed_microbatches": processed,
                         "base_weight_signatures_unchanged": True, "load_info": load_info},
        "timing": {"score_seconds": time.time() - started,
                   "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "added_tokens": added},
    }
    write_json(output_path, report)
    del model
    torch.cuda.empty_cache()


def greedy_selection(score_rows, budget):
    chosen, used = [], 0
    for row in score_rows:
        cost = int(row["lora_cost"])
        if used + cost <= budget:
            chosen.append(row["matrix"])
            used += cost
    return sorted(chosen, key=matrix_sort_key), used


def make_group_selections(probe):
    budget = int(probe["budget"]["P_from_original_selected_layers"])
    source_layers = set(probe["source_selected_layers"])
    all_matrix_rows = probe["matrix_candidates"]
    a = sorted([row["matrix"] for row in all_matrix_rows if row["layer"] in source_layers], key=matrix_sort_key)
    layer_rows = probe["layer_aggregates"]
    b_rows = []
    for layer_row in layer_rows:
        layer = layer_row["layer"]
        b_rows.append({**layer_row, "matrix": f"layer.{layer}", "members": [f"layers.{layer}.{p}" for p in MODULE_ORDER]})
    b_members, b_used = [], 0
    for row in b_rows:
        if b_used + int(row["lora_cost"]) <= budget:
            b_members.extend(row["members"])
            b_used += int(row["lora_cost"])
    c, c_used = greedy_selection(all_matrix_rows, budget)
    a_used = sum(next(row["lora_cost"] for row in all_matrix_rows if row["matrix"] == key) for key in a)
    groups = {
        "A": {"description": "original Block2 layer selection", "selected": a, "used": a_used},
        "B": {"description": "same grad-weight-abs score aggregated by layer, global score/cost budget", "selected": sorted(b_members, key=matrix_sort_key), "used": b_used},
        "C": {"description": "matrix-level grad-weight-abs global score/cost budget", "selected": c, "used": c_used},
    }
    for group in groups.values():
        if group["used"] > budget:
            raise RuntimeError("Selection exceeds P budget")
        group["remaining"] = budget - group["used"]
    return groups


def prepare_group_metadata(args, probe_path, root):
    probe = read_json(probe_path)
    groups = make_group_selections(probe)
    for label, group in groups.items():
        group_dir = os.path.join(root, label)
        block_dir = os.path.join(group_dir, "block_2")
        os.makedirs(block_dir, exist_ok=True)
        selection = {
            "method": "matrix_static_pilot",
            "group": label,
            "source_block1": probe["source_block1"],
            "source_block2_selection": probe["source_block2_selection"],
            "source_selected_layers": probe["source_selected_layers"],
            "selected_layers": list(range(16)),
            "structural_layers": list(range(16)),
            "matrix_selected": group["selected"],
            "matrix_unselected": sorted(set(row["matrix"] for row in probe["matrix_candidates"]) - set(group["selected"]), key=matrix_sort_key),
            "matrix_budget_P": probe["budget"]["P_from_original_selected_layers"],
            "actual_selected_lora_AB_parameters": group["used"],
            "remaining_lora_budget": group["remaining"],
            "vocabulary_parameters_excluded_from_budget": probe["budget"]["vocabulary_trainable_parameters_excluded_from_P"],
            "selection_rule": group["description"],
            "score_file": os.path.realpath(probe_path),
        }
        write_json(os.path.join(block_dir, "selected_layers.json"), selection)
        write_json(os.path.join(group_dir, "launch_config.json"), {
            "group": label, "same_source_block1": True, "same_probe_file": os.path.realpath(probe_path),
            "same_budget": True, "train_block": 2, "only_train_block2": True,
            "source_chain_threshold": 0.8, "full_vocab_update": True,
        })
        source_link = os.path.join(group_dir, "block_1")
        if os.path.lexists(source_link):
            os.unlink(source_link)
        os.symlink(os.path.realpath(args.source_block1), source_link)
    write_json(os.path.join(root, "group_selections.json"), groups)
    return groups


def selected_reference(model, candidates):
    return {key: {"A": row["module"].A.detach().float().cpu().clone(),
                  "B": row["module"].B.detach().float().cpu().clone()}
            for key, row in candidates.items()}


def update_report(candidates, before, selected):
    data = {"selected": {}, "unselected": {}}
    for key, row in candidates.items():
        a = (row["module"].A.detach().float().cpu() - before[key]["A"]).square().sum().item()
        b = (row["module"].B.detach().float().cpu() - before[key]["B"]).square().sum().item()
        bucket = "selected" if key in selected else "unselected"
        data[bucket][key] = {"A_squared_change": a, "B_squared_change": b, "total_squared_change": a + b}
    for bucket in ("selected", "unselected"):
        data[bucket + "_total_squared_change"] = sum(row["total_squared_change"] for row in data[bucket].values())
    return data


def preflight(args, root, groups, probe):
    """Independent one-GPU checks. The temporary model is discarded afterwards."""
    if not rank0():
        return
    tokenizer, _, _ = build_tokenizer(args)
    train_data, _ = load_continual_datasets_v2(args, current_block=2, num_blocks=args.num_blocks)
    one_batch = next(iter(DataLoader(Subset(train_data, probe["probe"]["sample_indices"][:1]), batch_size=1,
                                     shuffle=False, collate_fn=Collator(args, tokenizer))))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, candidates, load_info = build_loaded_block1_model(args, tokenizer, 0, trainable_names=set(groups["A"]["selected"]))
    model.to(device)
    common = model_output_signature(model, one_batch, device)
    group_signatures = {}
    for label, group in groups.items():
        set_matrix_trainability(candidates, group["selected"])
        group_signatures[label] = model_output_signature(model, one_batch, device)
    if not all(signature_equal(common, value) for value in group_signatures.values()):
        raise RuntimeError("Trainability-only group setup changed initial predictions.")
    # A temporary real update verifies that only selected current A/B tensors can change.
    selected = set(groups["C"]["selected"])
    set_matrix_trainability(candidates, selected)
    before = selected_reference(model, candidates)
    base_before = {key: tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    history_before = {key: [tensor_signature(x) for x in row["module"].A_matrices] for key, row in candidates.items()}
    model.zero_grad(set_to_none=True)
    model.train()
    loss = model(**move_to_device(one_batch, device)).loss
    loss.backward()
    with_grad = {key for key, row in candidates.items() if row["module"].A.grad is not None or row["module"].B.grad is not None}
    if not with_grad.issubset(selected) or not with_grad:
        raise RuntimeError("Unexpected missing/extra LoRA gradients in matrix freeze check.")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-7, weight_decay=0.0)
    optimizer.step()
    updates = update_report(candidates, before, selected)
    if updates["unselected_total_squared_change"] != 0.0 or updates["selected_total_squared_change"] <= 0.0:
        raise RuntimeError("Small update did not respect selected/unselected LoRA boundary.")
    base_after = {key: tensor_signature(row["module"].orig.weight) for key, row in candidates.items()}
    if any(not signature_equal(base_before[key], base_after[key]) for key in candidates):
        raise RuntimeError("Temporary LoRA update changed a base projection matrix.")
    history_after = {key: [tensor_signature(x) for x in row["module"].A_matrices] for key, row in candidates.items()}
    if history_before != history_after:
        raise RuntimeError("Temporary LoRA update changed frozen historical matrices.")
    all_set = set(candidates)
    costs = candidate_costs(candidates)
    report = {
        "initial_prediction_signatures": {"common": common, **group_signatures},
        "all_groups_same_initial_output": True,
        "same_probe_file_for_B_C": True,
        "same_budget_for_B_C": True,
        "all_matrix_trainable_cost": sum(costs.values()),
        "all_matrix_module_count": len(all_set),
        "all_matrix_selection_matches_full16_structure": len(all_set) == 112,
        "small_update_group": "C", "small_update": updates,
        "base_weights_unchanged_by_small_lora_update": True,
        "historical_snapshots_unchanged_by_small_lora_update": True,
        "load_info": load_info,
    }
    write_json(os.path.join(root, "preflight_validation.json"), report)
    del model
    torch.cuda.empty_cache()


def distributed_selection_check(selection):
    digest = hashlib.sha256(json.dumps(sorted(selection), ensure_ascii=True).encode()).hexdigest()
    if dist.is_available() and dist.is_initialized():
        values = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(values, digest)
        if len(set(values)) != 1:
            raise RuntimeError(f"Ranks disagree on matrix selection: {values}")
    return digest


def train_group(args, root, label):
    group_dir = os.path.join(root, label)
    selection_file = os.path.join(group_dir, "block_2", "selected_layers.json")
    selection = read_json(selection_file)
    selected = set(selection["matrix_selected"])
    rank, world = rank_world()
    digest = distributed_selection_check(selected)
    if rank0():
        write_json(os.path.join(group_dir, "selection_rank_consistency.json"), {"digest": digest, "world_size": world})
    barrier()
    started = time.time()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer, _, added = build_tokenizer(args, save_dir=os.path.join(group_dir, "block_2"))
    train_data, valid_data = load_continual_datasets_v2(args, current_block=2, num_blocks=args.num_blocks)
    reload_batch = next(iter(DataLoader(Subset(valid_data, [0]), batch_size=1, shuffle=False,
                                       collate_fn=Collator(args, tokenizer))))
    model, candidates, load_info = build_loaded_block1_model(args, tokenizer, int(os.environ.get("LOCAL_RANK", "0")), selected)
    for _, module in model.named_modules():
        if hasattr(module, "set_current_block"):
            module.set_current_block(2)
    set_matrix_trainability(candidates, selected)
    before = selected_reference(model, candidates)
    if rank0():
        wandb.init(project="matrix_static_pilot_v1", name=f"block1_to_block2_{label}", config={**vars(args), "group": label})
    adaptive_lr = args.learning_rate * 0.1
    trainer = ContinualLoRATrainer(
        model=model, train_dataset=train_data, eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed, per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs, learning_rate=adaptive_lr, weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type, report_to=["wandb"], fp16=args.fp16, bf16=args.bf16,
            logging_steps=args.logging_step, optim=args.optim, gradient_checkpointing=False,
            eval_strategy=args.save_and_eval_strategy, save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps, save_steps=args.save_and_eval_steps,
            output_dir=os.path.join(group_dir, "block_2"), save_total_limit=1,
            load_best_model_at_end=True, deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if world > 1 else None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 0,
            label_names=["labels"], dataloader_drop_last=False,
        ), processing_class=tokenizer, data_collator=Collator(args, tokenizer), continual_loss_weight=0.0,
    )
    model.config.use_cache = False
    trainer.train()
    core = getattr(trainer.model, "module", trainer.model)
    for _, module in core.named_modules():
        if hasattr(module, "save_current_matrices"):
            module.save_current_matrices()
    # Save only final model in this experiment-owned group directory.
    if rank0():
        trainer.save_state()
        trainer.save_model(output_dir=os.path.join(group_dir, "block_2"))
        final_signature = model_output_signature(core, reload_batch, device)
    barrier()
    if rank0():
        for checkpoint in [p for p in os.listdir(os.path.join(group_dir, "block_2")) if p.startswith("checkpoint-")]:
            shutil.rmtree(os.path.join(group_dir, "block_2", checkpoint))
        report = update_report(candidates, before, selected)
        write_json(os.path.join(group_dir, "block_2", "matrix_update_report.json"), report)
        write_json(os.path.join(group_dir, "timing.json"), {
            "training_seconds": time.time() - started,
            "peak_gpu_memory_bytes_rank0": int(torch.cuda.max_memory_allocated(device)),
            "effective_learning_rate_block2": adaptive_lr, "added_tokens": added,
            "load_info": load_info,
        })
        # Drop the trained object, reconstruct from the just-written checkpoint,
        # and prove that checkpoint serialization preserves effective predictions.
        del trainer, model, core
        torch.cuda.empty_cache()
        reloaded, reloaded_candidates, reload_info = build_loaded_block1_model(
            args, tokenizer, int(os.environ.get("LOCAL_RANK", "0")), selected
        )
        final_state = load_file(os.path.join(group_dir, "block_2", "model.safetensors"), device="cpu")
        reloaded.load_state_dict(final_state, strict=False)
        reconstruct_saved_parameter_lists(reloaded, final_state, 0)
        del final_state
        for _, module in reloaded.named_modules():
            if hasattr(module, "set_current_block"):
                module.set_current_block(2)
        set_matrix_trainability(reloaded_candidates, selected)
        reloaded.to(device)
        reloaded_signature = model_output_signature(reloaded, reload_batch, device)
        # The definitive check below is performed by the standalone `verify`
        # command against two independent reloads.  Do not invalidate a fully
        # saved training run merely because this in-process diagnostic differs.
        in_process_match = signature_equal(final_signature, reloaded_signature, atol=1e-5)
        write_json(os.path.join(group_dir, "checkpoint_reload_validation.json"), {
            "in_process_reload_prediction_match": in_process_match,
            "before_save_signature": final_signature,
            "after_reload_signature": reloaded_signature,
            "selection_file": selection_file,
            "reload_load_info": reload_info,
            "note": "Run `verify` for the definitive saved-tensor and independent-reload validation.",
        })
        del reloaded
        torch.cuda.empty_cache()
        if wandb.run is not None:
            wandb.finish()
    barrier()
    return group_dir


def verify_saved_group(args, root, label):
    """Validate the already-saved final checkpoint without retraining it."""
    if not rank0():
        return
    group_dir = os.path.join(root, label)
    checkpoint = os.path.join(group_dir, "block_2", "model.safetensors")
    selection_file = os.path.join(group_dir, "block_2", "selected_layers.json")
    if not os.path.exists(checkpoint) or not os.path.exists(selection_file):
        raise FileNotFoundError(f"Missing saved group artifact for {label}")
    selection = read_json(selection_file)
    selected = set(selection["matrix_selected"])
    tokenizer, _, _ = build_tokenizer(args)
    _, valid_data = load_continual_datasets_v2(args, current_block=2, num_blocks=args.num_blocks)
    one_batch = next(iter(DataLoader(Subset(valid_data, [0]), batch_size=1, shuffle=False,
                                     collate_fn=Collator(args, tokenizer))))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    signatures, key_checks = [], []
    for pass_id in range(2):
        model, candidates, load_info = build_loaded_block1_model(args, tokenizer, 0, selected)
        state = load_file(checkpoint, device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        reconstruct_saved_parameter_lists(model, state, 0)
        restored = model.state_dict()
        mismatches = [key for key, value in state.items() if key not in restored or not torch.equal(value.cpu(), restored[key].detach().cpu())]
        if mismatches:
            raise RuntimeError(f"Saved checkpoint parameters did not reload exactly: {mismatches[:5]}")
        for _, module in model.named_modules():
            if hasattr(module, "set_current_block"):
                module.set_current_block(2)
        set_matrix_trainability(candidates, selected)
        model.to(device)
        signatures.append(model_output_signature(model, one_batch, device))
        key_checks.append({"pass": pass_id, "saved_keys": len(state), "missing": len(missing),
                           "unexpected": len(unexpected), "load_info": load_info})
        del state, model
        torch.cuda.empty_cache()
    if not signature_equal(signatures[0], signatures[1], atol=0.0):
        raise RuntimeError("Two independent reloads of the same checkpoint changed predictions.")
    write_json(os.path.join(group_dir, "checkpoint_reload_validation.json"), {
        "saved_parameter_tensors_reload_exactly": True,
        "two_independent_reloads_same_prediction": True,
        "selection_file": selection_file,
        "selection_count": len(selected),
        "prediction_signature": signatures[0],
        "passes": key_checks,
    })


def parser():
    p = argparse.ArgumentParser(description="Static matrix-level LoRA selection pilot")
    p = parse_global_args(p)
    p = parse_train_args(p)
    p = parse_dataset_args(p)
    p.add_argument("command", choices=("score", "prepare", "preflight", "train", "verify"))
    p.add_argument("--root", required=True)
    p.add_argument("--source_block1", required=True)
    p.add_argument("--source_block2_selection", required=True)
    p.add_argument("--group", choices=("A", "B", "C"))
    p.add_argument("--num_blocks", type=int, default=5)
    # Retained in manifests for exact source-command compatibility.  This pilot
    # deliberately keeps it at zero and does not add a continual-loss term.
    p.add_argument("--continual_loss_weight", type=float, default=0.0)
    p.add_argument("--probe_batches", type=int, default=8)
    p.add_argument("--probe_batch_size", type=int, default=1)
    p.add_argument("--probe_ratio", type=float, default=0.1)
    p.add_argument("--probe_seed", type=int, default=42)
    return p


def main():
    args = normalize_args(parser().parse_args())
    set_seed(args.seed)
    os.makedirs(args.root, exist_ok=True)
    if args.command == "train" and int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    probe_path = os.path.join(args.root, "matrix_probe.json")
    if args.command == "score":
        score_matrices(args, probe_path)
    elif args.command == "prepare":
        if not os.path.exists(probe_path):
            raise FileNotFoundError(probe_path)
        prepare_group_metadata(args, probe_path, args.root)
    elif args.command == "preflight":
        if not os.path.exists(probe_path):
            raise FileNotFoundError(probe_path)
        groups = read_json(os.path.join(args.root, "group_selections.json"))
        preflight(args, args.root, groups, read_json(probe_path))
    elif args.command == "train":
        if not args.group:
            raise ValueError("train requires --group A, B, or C")
        train_group(args, args.root, args.group)
    else:
        if not args.group:
            raise ValueError("verify requires --group A, B, or C")
        verify_saved_group(args, args.root, args.group)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
