# IDARec

IDARec implements budget-constrained dynamic LoRA adaptation for continual
recommendation. It uses per-example Delta-W Taylor importance to select LoRA
matrices under attention/MLP parameter budgets. Selection is refreshed every
100 optimizer steps; inactive matrices and their optimizer state are protected.
The internal score-mode identifier is `lora_delta_taylor`.

## Code

| Directory | Purpose |
| --- | --- |
| `configs/` | Configurations for the supported LoRA parameter budgets |
| `data/` | Instruments, Movies, and Books interaction data, item metadata, and SID mappings |
| `objects/` | Executable backend files loaded by the experiment launcher |
| `profiles/` | Backend mappings for partial-budget and full-budget experiments |
| `readable_source/` | Readable training, selection, evaluation, and launch code |

Within `readable_source/`, `train_code/` implements training and matrix selection,
and `eval_code/` computes recommendation metrics using historical stage state.
`run_pipeline.py` runs the stages; `train_entry.py` starts distributed training.
The readable source mirrors the grouped backend. Editing only the readable
copy does not change launched runs because `run.py` loads files from `objects/`.

`configs/` includes 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and 100% LoRA A/B
budget configurations, seed 42, an actual initial learning rate of 1e-4, and two
GPUs. Partial budgets use a 40%/60% attention/MLP split. The 100% profile updates
all A/B matrices. The shared launcher is configured for Books/Llama-3.2-1B.

## Data

The datasets are directly available under `data/`:

| Directory | Items | SID mapping |
| --- | ---: | --- |
| `data/Instruments_cl_ori/` | 53,509 | Original PESO mapping |
| `data/Movies_and_TV_cl_ori/` | 46,074 | Unique four-token SIDs |
| `data/Books_cl_ori/` | 45,904 | Unique four-token SIDs |

Each dataset contains interaction splits `block_0` through `block_4`, the SID
index, and item metadata as `.item.part001.json`, `.item.part002.json`, etc.
These are ordinary JSON dictionaries; the data loader combines their records
automatically. Interaction splits and SID indices retain the original bytes;
item fields retain their original JSON values. Instruments retains its original
SID collisions. Movies and Books each use one distinct SID per item.

No model weights, Block-0 checkpoints, tokenizer files, or embedding arrays are
included. The dataset `block_0.json` files are included.

## Run

Training targets Linux, Python 3.10+, CUDA, DeepSpeed, and two NVIDIA GPUs. Install
a CUDA-compatible PyTorch build and the packages in `requirements.txt`.
FlashAttention must be installed for the selected CUDA/PyTorch environment.
The supplied source uses Transformers 4.52.4; the original complete environment
lock was not included.

Copy `resources.example.json` to `resources.local.json` and set the base-model,
Block-0 checkpoint, and Python environment paths. The default data resource
already points to `data/Books_cl_ori`. The local configuration is ignored by Git.

```bash
cp resources.example.json resources.local.json
mkdir -p runs
python run.py --config configs/budget20_lr1e4.json --output runs/books_budget20
```

The command above prepares a run. To prepare a new run and launch training
followed by recommendation evaluation:

```bash
python run.py --config configs/budget20_lr1e4.json --output runs/books_budget20_training --launch
```

Use a new output directory on the same filesystem as the repository because
prepared code uses hard links. Model checkpoints and training outputs are local
run artifacts.

## Acknowledgements

IDARec builds on [PESO](https://github.com/hsyoo32/peso) and uses its original
Instruments dataset. Movies and Books use the author-constructed temporal
datasets and unique SID mappings.
