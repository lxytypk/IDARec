"""Squared gradient of mean probe loss in the current LoRA A/B coordinates.

With dropout disabled, G = dL/dW also equals dL/d(W + scaling * B @ A).
The chain rule gives dL/dA = scaling * B.T @ G and
dL/dB = scaling * G @ A.T. This leaves the existing frozen-A/B probe and
ZeRO optimizer hooks untouched, while measuring the requested A/B gradients.
"""
import torch


class LoRAABGradientAccumulator:
    def __init__(self, candidates):
        self.keys = list(candidates)
        self.sums = {}
        for key in self.keys:
            module = candidates[key]["module"]
            if module.training or module.dropout.training or module.merged:
                raise RuntimeError("A/B scoring requires unmerged eval-mode LoRA")
            self.sums[key] = (
                torch.zeros_like(module.A, dtype=torch.float32),
                torch.zeros_like(module.B, dtype=torch.float32),
            )
        self.count = 0

    @torch.no_grad()
    def add(self, candidates):
        for key in self.keys:
            module = candidates[key]["module"]
            gradient = module.orig.weight.grad
            if gradient is None:
                raise RuntimeError("Missing base gradient for " + key)
            gradient = gradient.detach().float()
            a, b = module.A.detach().float(), module.B.detach().float()
            ga, gb = self.sums[key]
            ga.add_(b.t().matmul(gradient), alpha=float(module.scaling))
            gb.add_(gradient.matmul(a.t()), alpha=float(module.scaling))
        self.count += 1

    @torch.no_grad()
    def finish(self, expected_count, distributed=False):
        if distributed:
            import torch.distributed as dist
            for ga, gb in self.sums.values():
                dist.all_reduce(ga)
                dist.all_reduce(gb)
            first = self.sums[self.keys[0]][0]
            count = torch.tensor(self.count, dtype=torch.long, device=first.device)
            dist.all_reduce(count)
            self.count = int(count.item())
        if self.count != expected_count or self.count <= 0:
            raise RuntimeError("Incorrect A/B probe sample count")
        scores = {}
        for key, (ga, gb) in self.sums.items():
            ga.div_(self.count)
            gb.div_(self.count)
            value = ga.square().sum(dtype=torch.float64) + gb.square().sum(dtype=torch.float64)
            if not torch.isfinite(value):
                raise RuntimeError("Non-finite A/B gradient score: " + key)
            scores[key] = float(value.item())
        return scores
