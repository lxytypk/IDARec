import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset


SELECTED_LAYERS_FILENAME = "selected_layers.json"


def get_transformer_layers(model) -> List[torch.nn.Module]:
    """Return the decoder transformer layers for Llama-style causal LMs."""
    model = getattr(model, "module", model)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Aletheia probe currently expects a model.model.layers stack.")


def selected_layers_path(block_dir: str) -> str:
    return os.path.join(block_dir, SELECTED_LAYERS_FILENAME)


def load_selected_layers(block_dir: str) -> Optional[List[int]]:
    """Load Aletheia-selected layer ids for a block, or None for full-layer LoRA."""
    path = selected_layers_path(block_dir)
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    selected = payload.get("selected_layers")
    if selected is None:
        return None
    return sorted(int(layer_id) for layer_id in selected)


def load_structural_layers(block_dir: str) -> Optional[List[int]]:
    """Load all layer ids whose LoRA modules are present in a block checkpoint."""
    path = selected_layers_path(block_dir)
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    structural = payload.get("structural_layers")
    if structural is None:
        structural = payload.get("selected_layers")
    if structural is None:
        return None
    return sorted(int(layer_id) for layer_id in structural)


def save_selected_layers(
    block_dir: str,
    block_id: int,
    selected_layers: Sequence[int],
    layer_scores: Dict[int, float],
    selection_ratio: float,
    probe_batches: int,
    probe_chunk_size: int,
    target_modules: Optional[Sequence[str]],
    modules_to_save: Optional[Sequence[str]],
    previous_structural_layers: Optional[Sequence[int]] = None,
    structural_layers: Optional[Sequence[int]] = None,
    probe_metadata: Optional[Dict] = None,
) -> str:
    """Persist the layer-selection decision needed to reconstruct checkpoints."""
    os.makedirs(block_dir, exist_ok=True)
    ranked_layers = sorted(layer_scores, key=lambda layer_id: layer_scores[layer_id], reverse=True)
    selected_layer_ids = sorted(int(layer_id) for layer_id in selected_layers)
    previous_structural_layer_ids = (
        []
        if previous_structural_layers is None
        else sorted(int(layer_id) for layer_id in previous_structural_layers)
    )
    if structural_layers is None:
        structural_layer_ids = sorted(set(previous_structural_layer_ids) | set(selected_layer_ids))
    else:
        structural_layer_ids = sorted(int(layer_id) for layer_id in structural_layers)
    frozen_layer_ids = sorted(set(structural_layer_ids) - set(selected_layer_ids))
    inherited_trainable_layer_ids = sorted(set(previous_structural_layer_ids) & set(selected_layer_ids))
    new_trainable_layer_ids = sorted(set(selected_layer_ids) - set(previous_structural_layer_ids))
    metadata = probe_metadata or {}
    payload = {
        "method": (
            "taylor_lora_importance"
            if metadata.get("scoring_method") == "first_order_taylor_lora"
            else "aletheia_gradient_probe"
        ),
        "block_id": block_id,
        "selected_layers": selected_layer_ids,
        "structural_layers": structural_layer_ids,
        "previous_structural_layers": previous_structural_layer_ids,
        "frozen_layers": frozen_layer_ids,
        "inherited_trainable_layers": inherited_trainable_layer_ids,
        "new_trainable_layers": new_trainable_layer_ids,
        "ranked_layers": [int(layer_id) for layer_id in ranked_layers],
        "layer_scores": {str(k): float(v) for k, v in sorted(layer_scores.items())},
        "selection_ratio": float(selection_ratio),
        "probe_batches": int(probe_batches),
        "probe_chunk_size": int(probe_chunk_size),
        "target_modules": list(target_modules) if target_modules is not None else None,
        "modules_to_save": list(modules_to_save) if modules_to_save is not None else None,
        "probe_metadata": metadata,
    }
    path = selected_layers_path(block_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _move_batch_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    return batch.to(device) if torch.is_tensor(batch) else batch


def _freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None


def _active_lora_parameters(
    layer: torch.nn.Module,
    target_modules: Optional[Sequence[str]],
):
    """Yield the active A/B tensors of LoRA modules targeted by this run."""
    targets = tuple(target_modules or ())
    for module_name, module in layer.named_modules():
        if targets and not any(module_name.endswith(target) for target in targets):
            continue
        for parameter_name in ("A", "B"):
            parameter = getattr(module, parameter_name, None)
            if isinstance(parameter, torch.nn.Parameter):
                yield module_name, parameter_name, parameter


def _enable_lora_parameters(
    layers: Sequence[torch.nn.Module],
    target_modules: Optional[Sequence[str]],
) -> int:
    enabled = 0
    for layer in layers:
        for _, _, parameter in _active_lora_parameters(layer, target_modules):
            parameter.requires_grad_(True)
            enabled += parameter.numel()
    return enabled


def _layer_taylor_score(
    layer: torch.nn.Module,
    target_modules: Optional[Sequence[str]],
) -> Tuple[float, int]:
    """First-order LoRA saliency: mean(abs(theta * dL/dtheta))."""
    score_sum = 0.0
    parameter_count = 0
    for _, _, parameter in _active_lora_parameters(layer, target_modules):
        if parameter.grad is None:
            continue
        theta = parameter.detach().float()
        gradient = parameter.grad.detach().float()
        score_sum += (theta * gradient).abs().sum().item()
        parameter_count += parameter.numel()
    if parameter_count == 0:
        return 0.0, 0
    return score_sum / parameter_count, parameter_count


def compute_aletheia_layer_selection(
    model,
    train_dataset,
    collator,
    probe_batches: int,
    probe_batch_size: int,
    selection_ratio: float,
    probe_chunk_size: int,
    probe_ratio: float = 0.0,
    selection_threshold: float = 0.0,
    local_rank: int = 0,
    probe_seed: int = 42,
    block_id: int = 0,
    target_modules: Optional[Sequence[str]] = None,
) -> Tuple[List[int], Dict[int, float], Dict]:
    """
    Rank transformer layers by first-order Taylor importance of their active
    LoRA parameters and select the highest-scoring layers.

    The probe intentionally performs forward/backward passes without optimizer steps.
    The resulting gradients are discarded after producing layer scores.
    """
    if probe_batches <= 0:
        raise ValueError("--aletheia_probe_batches must be positive.")
    if probe_chunk_size <= 0:
        raise ValueError("--aletheia_probe_chunk_size must be positive.")
    if selection_ratio <= 0 or selection_ratio > 1:
        raise ValueError("--aletheia_selection_ratio must be in (0, 1].")

    # FIX 1: probe a fixed FRACTION of the block training set instead of a tiny
    # fixed count; probe_batches only acts as a lower bound now.
    if probe_ratio > 0:
        probe_batches = max(
            int(probe_batches), int(math.ceil(len(train_dataset) * probe_ratio))
        )

    layers = get_transformer_layers(model)
    num_layers = len(layers)
    selected_count = max(1, int(math.ceil(num_layers * selection_ratio)))
    selected_count = min(selected_count, num_layers)

    if len(train_dataset) < probe_batches:
        raise ValueError(
            f"Probe needs {probe_batches} samples, but block {block_id} has only {len(train_dataset)}."
        )

    # The original implementation always consumed the first probe_batches
    # records.  Keep every other probe behavior unchanged, but use a fixed,
    # reproducible uniform sample from the current block instead.
    generator = torch.Generator()
    generator.manual_seed(int(probe_seed))
    probe_indices = torch.randperm(len(train_dataset), generator=generator)[:probe_batches].tolist()
    probe_dataset = Subset(train_dataset, probe_indices)

    dataloader = DataLoader(
        probe_dataset,
        batch_size=probe_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    device = next(model.parameters()).device
    was_training = model.training
    # Disable dropout/stochastic training behavior while scoring layers.
    # Gradients still flow normally in eval mode.
    model.eval()
    model.config.use_cache = False

    layer_scores = {layer_id: 0.0 for layer_id in range(num_layers)}
    _freeze_all_parameters(model)
    batches_per_chunk = 0

    for start in range(0, num_layers, probe_chunk_size):
        end = min(start + probe_chunk_size, num_layers)
        chunk_layers = layers[start:end]

        _freeze_all_parameters(model)
        enabled_parameters = _enable_lora_parameters(chunk_layers, target_modules)
        if enabled_parameters == 0:
            raise RuntimeError(
                "Taylor probe found no active LoRA A/B parameters in layers "
                f"[{start}, {end}). The shared block-0 checkpoint must be loaded "
                "before scoring."
            )

        processed_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= probe_batches:
                break

            batch = _move_batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Aletheia probe expected model outputs to include a loss.")
            loss.backward()

            for offset, layer in enumerate(chunk_layers):
                score, parameter_count = _layer_taylor_score(layer, target_modules)
                if parameter_count == 0:
                    raise RuntimeError(
                        f"Taylor probe found no scored LoRA parameters in layer {start + offset}."
                    )
                layer_scores[start + offset] += score

            processed_batches += 1

        if processed_batches == 0:
            raise RuntimeError("Aletheia probe found no batches in the training dataset.")
        batches_per_chunk = processed_batches

        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if local_rank == 0:
            print(
                f"Taylor-LoRA probe chunk [{start}, {end}) processed "
                f"{processed_batches} batches ({enabled_parameters:,} LoRA parameters)"
            )

    # FIX 2: average over probe samples instead of a raw sum, so scores stay
    # comparable across blocks whose loss scales differ.
    if batches_per_chunk > 0:
        for layer_id in layer_scores:
            layer_scores[layer_id] /= batches_per_chunk

    _freeze_all_parameters(model)
    if was_training:
        model.train()
    else:
        model.eval()

    # FIX 3: normalize scores to sum to 1, then pick layers by cumulative
    # normalized importance (LoRA-drop style) instead of a hard layer ratio.
    total_score = sum(layer_scores.values())
    if not math.isfinite(total_score) or total_score <= 0:
        raise RuntimeError(
            "Taylor-LoRA probe produced degenerate layer scores (total=%s): every "
            "layer scored zero or non-finite. This means the probe batches carried "
            "no supervised signal (e.g. all labels == -100) or the backward pass did "
            "not run. Refusing to select layers from uninformative scores." % total_score
        )
    normalized_scores = {l: s / total_score for l, s in layer_scores.items()}
    ranked_layers = sorted(
        normalized_scores, key=lambda layer_id: normalized_scores[layer_id], reverse=True
    )
    if selection_threshold > 0:
        picked, acc = [], 0.0
        for layer_id in ranked_layers:
            picked.append(layer_id)
            acc += normalized_scores[layer_id]
            if acc >= selection_threshold:
                break
        selected_layers = sorted(picked)
        selection_rule = "cumulative_normalized_threshold_%s" % selection_threshold
        if local_rank == 0 and len(selected_layers) > selected_count:
            print(
                f"[Aletheia probe] note: threshold selection kept "
                f"{len(selected_layers)}/{num_layers} layers (hard ratio top-k would "
                f"keep {selected_count}); the importance distribution is flat."
            )
    else:
        selected_layers = sorted(ranked_layers[:selected_count])
        selection_rule = "top_ratio_%s" % selection_ratio
    probe_metadata = {
        "scoring_method": "first_order_taylor_lora",
        "score_formula": "mean_over_batches(mean_over_lora_params(abs(theta * dL_dtheta)))",
        "scored_parameter_scope": "active A/B tensors in configured LoRA target modules",
        "target_modules": list(target_modules) if target_modules is not None else None,
        "sampling": "fixed_seed_uniform_without_replacement",
        "probe_seed": int(probe_seed),
        "block_id": int(block_id),
        "probe_indices": [int(index) for index in probe_indices],
        "dropout_disabled": True,
        "per_sample_normalization": False,
        "score_aggregation": "mean_over_probe_samples",
        "batches_per_chunk": int(batches_per_chunk),
        "probe_ratio": float(probe_ratio),
        "effective_probe_batches": int(probe_batches),
        "selection": selection_rule,
        "requested_lora_budget_ratio": float(selection_ratio),
        "selected_layer_count": int(len(selected_layers)),
        "total_layer_count": int(num_layers),
        "actual_layer_budget_ratio": float(len(selected_layers) / num_layers),
        "normalized_scores": {str(k): round(v, 6) for k, v in sorted(normalized_scores.items())},
    }
    return selected_layers, layer_scores, probe_metadata
