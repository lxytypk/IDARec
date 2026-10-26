# IDARec

IDARec implements budget-constrained dynamic LoRA adaptation for continual
recommendation. This repository packages the supplied Books / Llama-3.2-1B
experiment snapshot, including training, evaluation, and budget configurations.
The internal scoring identifier remains `lora_delta_taylor` for compatibility.

## Method and included configurations

The backend accumulates per-example first-order Taylor importance of the LoRA
weight update, normalized by adapter parameter cost. It reselects trainable
adapter matrices every 100 optimizer steps under attention/MLP budgets, and
protects inactive matrices and their optimizer state. Budget 100% uses the
separate all-A/B profile.

The supplied configurations cover 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and
100% LoRA A/B parameter budgets. They use seed 42, an actual initial learning
rate of 1e-4, and two GPUs. For partial budgets the attention/MLP split is
40%/60%. The launcher preserves the original internal learning-rate multiplier.

## Layout

| Path | Purpose |
| --- | --- |
| `run.py` | Prepare an isolated experiment; optionally launch training and evaluation |
| `readable_source/train_code/` | Readable training and importance-scoring code |
| `readable_source/eval_code/` | Evaluation with historical stage restoration |
| `readable_source/official_baseline/` | Upstream evaluation reference and provenance |
| `objects/` | Immutable SHA-256-addressed backend files used by the launcher |
| `profiles/` | Backend file mappings for partial and full budgets |
| `configs/` | Configurations for new experiments |
| `historical_configs/`, `experiments.json` | Sanitized historical metadata; result files are not included |
| `queues/` | Historical queue scripts for reference; require local configuration |
| `audit/` | Source-snapshot records and release validation |

## Environment and external assets

Training targets Linux with Python 3.10 or later, CUDA, DeepSpeed, and two
compatible NVIDIA GPUs. The pipeline checks for more than 24 GiB free in
`/dev/shm` and approximately 22 GiB free persistent storage. These checks and
the original model/tokenizer/data compatibility checks are retained.

Install a CUDA-compatible PyTorch build, then the packages in
`requirements.txt`. Install FlashAttention separately for the selected
PyTorch/CUDA environment, for example `pip install flash-attn --no-build-isolation`.
The source pins Transformers 4.52.4; a full original environment lock was not
provided, so other package versions require validation in the target environment.

Provide the Books_cl_ori data files, the Llama-3.2-1B base model, and the shared
Block-0 checkpoint separately. They are not included in this code release.
`readable_source/block0_provenance.json` retains asset checksums needed by the
preflight; substitute assets require an appropriate provenance review.

Copy `resources.example.json` to `resources.local.json`, then set `data`,
`block0`, `base_model`, and `environment` to your absolute local paths. The
`environment` value is the virtual-environment directory containing `bin/python`.
The local configuration is ignored by Git. Default `/opt/idarec/` paths are
examples and do not refer to the original development machine.

## Prepare and launch

Run from this repository directory. Keep the output on the same filesystem
as the repository because the launcher uses hard links. Each output must be new.

```bash
cp resources.example.json resources.local.json
# Edit resources.local.json before launching.
mkdir -p runs
python run.py --config configs/budget20_lr1e4.json --output runs/books_budget20
```

The command above only prepares files and links. To prepare a new run and
start training followed by evaluation explicitly:

```bash
python run.py --config configs/budget20_lr1e4.json --output runs/books_budget20_training --launch
```

The readable source mirrors the grouped profile. The launcher loads code from
`objects/`, so editing only `readable_source/` does not change launched jobs.
Do not modify object files in place or edit prepared hard-linked code.

## Validation and sanitization

See `SANITIZATION.md` and `audit/release_validation.json`. Source paths,
internal network addresses, and personal filesystem identifiers were removed
from both readable files and immutable objects. Object hashes and profile
references were regenerated. No model weights, datasets, credentials, or live
run logs are included. Checks cover syntax, JSON, object integrity, and launcher
preparation/configuration; this release has not rerun GPU training or evaluation.

## Upstream reference

This snapshot includes code derived from [PESO](https://github.com/hsyoo32/peso)
at commit `edbec5f68ddc83e39f22a041f1f2eaf1c4ad172e`.
Upstream file provenance is preserved under `readable_source/official_baseline/`.
No new redistribution license is assigned by this packaging step.
