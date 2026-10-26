"""Train continual blocks and evaluate the historical stages of a prepared run."""

from contextlib import ExitStack
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import traceback

from run_config import build_evaluation_command, build_training_command, read_json, write_json


RUN_DIRECTORY = Path(__file__).resolve().parent
PYTHON = sys.executable


def timestamp():
    timezone = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(timezone).strftime("%Y%m%d_%H%M%S")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_rows():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    return [row.split(",") for row in output.splitlines()]


def gpu_applications():
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True)


def selected_gpu_uuids(config):
    selected = [row for row in gpu_rows() if int(row[0]) in config["gpus"]]
    assert len(selected) == 2
    assert all(int(row[4]) <= 64 and int(row[5]) == 0 for row in selected)
    uuids = {row[1].strip() for row in selected}
    assert not any(row.split(",")[0].strip() in uuids for row in gpu_applications().splitlines())
    return uuids


def run_phase(name, command, environment, selected_uuids, cwd):
    applications = gpu_applications()
    assert not any(row.split(",")[0].strip() in selected_uuids for row in applications.splitlines()), applications
    log_path = RUN_DIRECTORY / "logs" / (name + "_" + timestamp() + ".log")
    write_json(RUN_DIRECTORY / "status.json", dict(phase=name, status="running", log=str(log_path),
                                                  pid=os.getpid(), updated=timestamp()))
    record = dict(phase=name, argv=command, shell=shlex.join(command), cwd=str(cwd),
                  log=str(log_path), start=timestamp())
    print("PHASE", name, log_path, flush=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.run(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit=process.returncode, end=timestamp())
    with (RUN_DIRECTORY / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}; log={log_path}")


def seal_stage(block, config):
    """Record a complete model stage so interrupted runs can reuse it."""
    stage = RUN_DIRECTORY / "ckpt" / ("block_" + str(block))
    reload_result = read_json(stage / "checkpoint_reload_validation.json")
    assert reload_result["saved_tensors_reload_exactly"]
    assert reload_result["two_independent_reloads_logits_exactly_equal"]
    launch = read_json(stage / "launch_config.json")
    assert launch["experiment_args"]["attention_budget_share"] == config["attention_share"]
    assert launch["experiment_args"]["score_mode"] == "lora_delta_taylor"
    assert launch["training_arguments"]["learning_rate"] == config["actual_initial_lr"]
    assert launch["original_train_length"] == launch["actual_train_length"]
    files = {path.name: file_digest(path) for path in stage.iterdir()
             if path.is_file() and path.name != "stage_manifest.json"}
    write_json(stage / "stage_manifest.json", dict(block=block, files=files, complete=True,
               checkpoint_selection="minimum validation response CE", resume_semantics="model-only stage restart"))


def persistent_stage(block):
    return RUN_DIRECTORY / "persistent_stages" / ("block_" + str(block))


def persist_stage(block, temporary_root):
    """Move a completed stage from shared memory into persistent run storage."""
    stage = RUN_DIRECTORY / "ckpt" / ("block_" + str(block))
    source = stage.resolve()
    destination = persistent_stage(block)
    if source == destination.resolve() and destination.exists():
        return
    assert source.is_relative_to(temporary_root.resolve())
    assert (source / "stage_manifest.json").is_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + "_partial_" + timestamp())
    assert not destination.exists() and not partial.exists()
    shutil.copytree(source, partial, symlinks=False)
    for original in source.rglob("*"):
        if original.is_file():
            saved = partial / original.relative_to(source)
            assert saved.is_file() and not saved.is_symlink()
            assert file_digest(saved) == file_digest(original)
    partial.rename(destination)
    replacement = stage.with_name(stage.name + "_persistent_link")
    assert stage.is_symlink() and not replacement.exists()
    replacement.symlink_to(destination, target_is_directory=True)
    os.replace(replacement, stage)
    assert stage.resolve() == destination.resolve()
    assert source.resolve().is_relative_to(temporary_root.resolve())
    shutil.rmtree(source)


def make_evaluation_layout():
    from safetensors import safe_open
    from safetensors.torch import save_file

    layout = RUN_DIRECTORY / "evaluation_checkpoints"
    layout.mkdir(exist_ok=True)
    for name in ("config.json", "generation_config.json", "special_tokens_map.json",
                 "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(RUN_DIRECTORY / "ckpt/block_4" / name, layout / name)
    final = layout / "block_4"
    if not final.exists():
        final.symlink_to(RUN_DIRECTORY / "ckpt/block_4", target_is_directory=True)
    for block in range(1, 4):
        destination = layout / ("block_" + str(block))
        destination.mkdir(exist_ok=True)
        with safe_open(str(RUN_DIRECTORY / "ckpt" / ("block_" + str(block)) / "model.safetensors"),
                       framework="pt", device="cpu") as source:
            values = {key: source.get_tensor(key) for key in source.keys() if ".modules_to_save.default." in key}
        assert len(values) == 2
        save_file(values, str(destination / "modules_to_save.safetensors"))


def training_environment(config):
    environment = dict(os.environ)
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, config["gpus"])), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", CUDA_HOME="/usr/local/cuda",
        PATH="/usr/local/cuda/bin:" + environment.get("PATH", ""),
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:" + environment.get("LD_LIBRARY_PATH", ""),
        WANDB_MODE="offline", WANDB_DIR=str(RUN_DIRECTORY / "wandb"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(config["seed"]), OMP_NUM_THREADS="6",
        TORCH_EXTENSIONS_DIR=str(RUN_DIRECTORY / "torch_extensions"))
    return environment


def reserve_storage(config):
    remaining = sum(4.25e9 for block in (1, 2, 3, 4)
                    if not (persistent_stage(block) / "stage_manifest.json").exists())
    required = remaining + 4.5e9
    if config["all_AB_update"]:
        required += 12 * 1024**3
    assert shutil.disk_usage(RUN_DIRECTORY).free > required
    assert shutil.disk_usage(RUN_DIRECTORY).free > 512 * 1024**2
    assert shutil.disk_usage("/dev/shm").free > 24 * 1024**3


def train_blocks(config, environment, selected_uuids):
    temporary_root = Path(config["temporary_root"])
    for block in range(1, 5):
        stage = RUN_DIRECTORY / "ckpt" / ("block_" + str(block))
        if (stage / "stage_manifest.json").is_file():
            for name, expected in read_json(stage / "stage_manifest.json")["files"].items():
                assert file_digest(stage / name) == expected
            persist_stage(block, temporary_root)
            print("REUSE_STAGE", block, flush=True)
            continue
        if (stage / "checkpoint_reload_validation.json").is_file():
            seal_stage(block, config)
            persist_stage(block, temporary_root)
            continue
        if stage.exists():
            failed = stage.with_name(stage.name + "_failed_" + timestamp())
            stage.rename(failed)
            print("STAGE_RESTART", block, "preserved", failed, flush=True)
        temporary = temporary_root / ("block_" + str(block) + "_" + timestamp())
        temporary.mkdir(parents=True, exist_ok=False)
        stage.symlink_to(temporary, target_is_directory=True)
        for task in ("score", "prepare", "train"):
            command = build_training_command(config, RUN_DIRECTORY, block, task, PYTHON)
            run_phase(f"block{block}_{task}", command, environment, selected_uuids, RUN_DIRECTORY / "train_code")
        seal_stage(block, config)
        persist_stage(block, temporary_root)


def evaluate_blocks(config, environment, selected_uuids):
    make_evaluation_layout()
    budget_percent = round(config["budget_ratio"] * 100)
    result = RUN_DIRECTORY / f"Books_delta_budget{budget_percent:g}_{RUN_DIRECTORY.name}_beam_search.json"
    if result.exists():
        result = result.with_name(result.stem + "_" + timestamp() + ".json")
    assert not result.exists()
    command = build_evaluation_command(config, RUN_DIRECTORY, result, PYTHON)
    run_phase("evaluation_beam_search", command, environment, selected_uuids, RUN_DIRECTORY / "eval_code")
    data = read_json(result)
    assert data["completed_blocks"] == [1, 2, 3, 4]
    assert data["stage_restoration"] == "historical_stage_state"
    for block in range(1, 4):
        (RUN_DIRECTORY / "evaluation_checkpoints" / ("block_" + str(block)) / "modules_to_save.safetensors").unlink()
    return result


def main():
    import fcntl

    config = read_json(RUN_DIRECTORY / "requested_config.json")
    (RUN_DIRECTORY / "logs").mkdir(exist_ok=True)
    with ExitStack() as handles:
        lock = handles.enter_context((RUN_DIRECTORY / "pipeline.lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for gpu in config["gpus"]:
            handle = handles.enter_context(Path(f"/tmp/idarec_gpu_{gpu}.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (RUN_DIRECTORY / "pipeline.pid").write_text(str(os.getpid()), encoding="utf-8")
        if (RUN_DIRECTORY / "completed").exists():
            return
        reserve_storage(config)
        selected_uuids = selected_gpu_uuids(config)
        for port in (config["port"], config["port"] + 1, config["port"] + 2,
                     config["port"] + 3, config["port"] + 4, config["port"] + 10):
            with socket.socket() as connection:
                connection.bind(("127.0.0.1", port))
        environment = training_environment(config)
        train_blocks(config, environment, selected_uuids)
        result = evaluate_blocks(config, environment, selected_uuids)
        (RUN_DIRECTORY / "completed").write_text(timestamp(), encoding="utf-8")
        write_json(RUN_DIRECTORY / "status.json", dict(status="completed", result=str(result), updated=timestamp()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        write_json(RUN_DIRECTORY / "status.json", dict(status="failed", error=str(error), updated=timestamp()))
        sys.exit(1)
