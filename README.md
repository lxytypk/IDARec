# IDARec

Code for Not All Modules Need Updating: Importance-Guided Dynamic Adaptation for Continual Learning in Generative Recommendation.

## Code

| Path | Purpose |
| --- | --- |
| `configs/` | Configurations for the supported LoRA parameter budgets |
| `data/` | Instruments, Movies, and Books interaction data, item metadata, and SID mappings |
| `src/` | Editable source code used directly by the experiment launcher |
| `src/train_code/` | Training, importance scoring, and matrix selection |
| `src/eval_code/` | Recommendation evaluation with historical stage restoration |

`run.py` loads the chosen configuration and copies the current `src/` files into
an independent run directory. Edit the files in `src/` to change future runs;
existing run directories keep their own source copies. `src/run_config.py`
constructs the training and evaluation commands from configuration values.
`src/run_pipeline.py` runs the continual stages, and `src/train_entry.py`
starts distributed training.
The main training module is `src/train_code/dynamic_training.py`; shared static
selection helpers are in `static_training.py`, and replay support is in `replay.py`.
The parameter budget is controlled by the selected configuration.

`configs/` includes 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and 100% LoRA A/B
budget configurations, seed 42, an actual initial learning rate of 1e-4, and two
GPUs. Partial budgets use the configured attention/MLP share (40%/60% by default).
The 100% budget enables all A/B matrices and assigns the complete attention and
MLP parameter totals: 1,703,936 and 3,932,160, respectively. Learning rate,
reselection interval, GPU indices, and communication port are taken from the
chosen JSON configuration. The launcher targets Books/Llama-3.2-1B.

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

The default is a 40% total LoRA A/B parameter budget. Of that budget, 40% is
assigned to Attention and 60% to MLP. `attention_share=0.4` defines the split;
MLP receives the remaining share. The default configuration is
`configs/budget40_lr1e4.json`; use `--config` to select another supported budget.

Training targets Linux, Python 3.10+, CUDA, DeepSpeed, and two NVIDIA GPUs. Install
a CUDA-compatible PyTorch build and the packages in `requirements.txt`.
FlashAttention must be installed for the selected CUDA/PyTorch environment.
The supplied source uses Transformers 4.52.4; the original complete environment
lock was not included.

Copy `resources.json` to `resources.local.json` and set the base-model,
Block-0 checkpoint, and Python environment paths. The default data resource
already points to `data/Books_cl_ori`. The local configuration is ignored by Git.

```bash
cp resources.json resources.local.json
mkdir -p runs
python run.py --config configs/budget40_lr1e4.json --output runs/books_budget40
```

The command above prepares a run. To prepare a new run and launch training
followed by recommendation evaluation:

```bash
python run.py --config configs/budget40_lr1e4.json --output runs/books_budget40_training --launch
```

Use a new output directory for each experiment. Source files are copied, so the
run directory can be on a different filesystem from the repository. Model
checkpoints and training outputs remain local run artifacts.

## Acknowledgements

IDARec builds on [PESO](https://github.com/hsyoo32/peso) and uses its original
Instruments dataset. Movies and Books use the author-constructed temporal
datasets and unique SID mappings.
