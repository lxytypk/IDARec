"""Tiny two-GPU check of mean-gradient aggregation, without loading a model."""
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from lora_ab_grad_score import LoRAABGradientAccumulator


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        device = torch.device("cuda", rank)
        a = torch.arange(10, dtype=torch.float32, device=device).reshape(2, 5) / 10
        b = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2) / 8
        g = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5) / 20
        weight = torch.nn.Parameter(torch.zeros_like(g))
        module = SimpleNamespace(A=a, B=b, orig=SimpleNamespace(weight=weight),
                                 scaling=2.0, training=False, merged=False,
                                 dropout=SimpleNamespace(training=False))
        candidates = {"projection": {"module": module}}
        accumulator = LoRAABGradientAccumulator(candidates)
        # Uneven shards: rank 0 has +G; rank 1 has -G and +2G.
        for multiplier in ([1.0] if rank == 0 else [-1.0, 2.0]):
            weight.grad = multiplier * g
            accumulator.add(candidates)
        result = accumulator.finish(3, distributed=True)["projection"]
        mean_g = (2.0 / 3.0) * g
        expected = ((2*b.t()@mean_g).square().sum(dtype=torch.float64)
                    + (2*mean_g@a.t()).square().sum(dtype=torch.float64)).item()
        assert abs(result-expected) <= 1e-5*max(1.0, expected), (result, expected)
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, result)
        assert len(set(values)) == 1
        if rank == 0:
            print("DISTRIBUTED_ABGRAD_CHECK_PASS", json.dumps({
                "world_size": 2, "uneven_probe_shards": True,
                "mean_before_square": True, "scores_agree": True,
                "relative_error": abs(result-expected)/expected,
            }), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
