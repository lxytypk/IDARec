"""Deterministic fixed-length rehearsal; no change to optimizer step budget.

Only train examples are stored. Each epoch replaces round(N * ratio) positions
with old examples; current examples are sampled without replacement. The old
blocks receive counts differing by at most one. DataLoader workers must be zero.
"""
import hashlib
import json
import math
import random


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def score_contribution(weight, gradient, mode):
    g = gradient.detach().float()
    if mode == "grad_weight_abs":
        return (weight.detach().float() * g).abs().sum()
    if mode == "grad_norm":
        return g.norm(p=2)
    raise ValueError(mode)


class ReplayDataset:
    def __init__(self, current, buffers, ratio, seed, stage):
        if not 0 < ratio < 1 or not current or not buffers:
            raise ValueError("Replay requires nonempty current/history and ratio in (0,1)")
        self.current, self.buffers = current, buffers
        self.ratio, self.seed, self.stage = ratio, seed, stage
        self.epoch = None
        self.set_epoch(0)

    def __len__(self):
        return len(self.current)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        rng = random.Random(self.seed + 10007 * self.stage + 1000003 * epoch)
        n = len(self.current)
        nr = int(math.floor(n * self.ratio + 0.5))
        if not 0 < nr < n:
            raise ValueError("Replay count rounds to zero or all current examples")
        cur = rng.sample(range(n), n - nr)
        old = sorted(self.buffers)
        # Rotate which old block receives any remainder across epochs.
        old = old[epoch % len(old):] + old[:epoch % len(old)]
        mapping = [(self.stage, i) for i in cur]
        chosen = {b: [] for b in old}
        for pos, b in enumerate(old):
            count = nr // len(old) + int(pos < nr % len(old))
            if not self.buffers[b]:
                raise ValueError("Empty historical memory")
            while len(chosen[b]) < count:
                cycle = list(range(len(self.buffers[b])))
                rng.shuffle(cycle)
                chosen[b].extend(cycle[:count - len(chosen[b])])
            mapping.extend((b, i) for i in chosen[b])
        rng.shuffle(mapping)
        self.mapping, self.epoch = mapping, epoch
        self.manifest = {
            "epoch": epoch, "stage": self.stage, "seed": self.seed,
            "original_current_dataset_length": n, "training_dataset_length": len(mapping),
            "current_count": n - nr, "replay_count": nr, "actual_replay_fraction": nr / n,
            "current_indices": cur, "history_buffer_positions": chosen,
            "position_to_source": mapping, "mapping_sha256": fingerprint(mapping),
            "note": "10% example positions, not 10% tokens; length/steps unchanged; current exposure reduced; distributed sampler padding follows original Trainer",
        }

    def __getitem__(self, position):
        block, index = self.mapping[position]
        if block == self.stage:
            return self.current[index]
        return dict(self.buffers[block][index]["example"])
