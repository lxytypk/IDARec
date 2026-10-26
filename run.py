"""Prepare or launch an IDARec experiment from the editable source in src/."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from src.run_config import normalize_config, read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"


def load_resources():
    """Resolve resource paths, applying the ignored machine-local overrides."""
    resources = read_json(PROJECT_ROOT / "resources.json")
    local_file = PROJECT_ROOT / "resources.local.json"
    if local_file.is_file():
        resources.update(read_json(local_file))
    for name in ("data", "block0", "base_model", "environment"):
        path = Path(os.path.expandvars(resources[name])).expanduser()
        resources[name] = str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
    return resources


def prepare(config, output):
    """Copy current source into an independent run directory and link assets."""
    config = normalize_config(config)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory: " + str(output))
    if output.is_relative_to(SOURCE_ROOT.resolve()):
        raise ValueError("The output directory must be outside src/")
    resources = load_resources()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_ROOT, output,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"))
    for name in ("logs", "inputs", "ckpt", "datasets", "persistent_stages", "wandb"):
        (output / name).mkdir(exist_ok=True)
    for name, resource in (("shared_block0", "block0"), ("base_model", "base_model"), ("data", "data")):
        (output / "inputs" / name).symlink_to(resources[resource], target_is_directory=True)
    (output / "ckpt" / "block_0").symlink_to(resources["block0"], target_is_directory=True)
    (output / "datasets" / "Books_cl_ori").symlink_to(resources["data"], target_is_directory=True)
    (output / "env").symlink_to(resources["environment"], target_is_directory=True)
    config.update(method="IDARec", score_backend="lora_delta_taylor", generation_mode="beam-search",
                  evaluated_blocks=[1, 2, 3, 4], skipped_blocks=[0],
                  stage_restoration="historical_stage_state", source_directory=str(SOURCE_ROOT),
                  temporary_root=str(Path("/dev/shm") / ("idarec_" + output.name)))
    write_json(output / "requested_config.json", config)
    write_json(output / "status.json", dict(status="prepared_not_started", training_started=False))
    return output


def execute(output):
    """Start the prepared training/evaluation pipeline in its Python environment."""
    output = Path(output).resolve()
    config = read_json(output / "requested_config.json")
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, config["gpus"])), PYTHONUNBUFFERED="1")
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    python = str(output / "env" / "bin" / "python")
    with (output / "logs" / "launcher.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([python, str(output / "run_pipeline.py")], cwd=output,
                                   env=environment, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (output / "launcher.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), pipeline_pid=process.pid, training_then_evaluation=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the available budget configurations")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch", action="store_true", help="Start training and evaluation after preparation")
    args = parser.parse_args()
    if args.list:
        configs = [dict(name=path.stem, **read_json(path))
                   for path in sorted((PROJECT_ROOT / "configs").glob("*.json"))]
        print(json.dumps(configs, indent=2))
        return
    if args.config is None or args.output is None:
        parser.error("Provide --config and --output")
    output = prepare(read_json(args.config), args.output)
    if args.launch:
        execute(output)
    else:
        print(json.dumps(dict(output=str(output), status="prepared_not_started", no_gpu_job_started=True)))


if __name__ == "__main__":
    main()
