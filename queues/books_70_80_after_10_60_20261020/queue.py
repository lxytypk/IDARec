"""Authorized Books Delta-Taylor dependency queue; never interrupts GPU jobs."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/opt/idarec/runs')
SHARED = BASE / 'lora_delta_taylor_shared'
GIB = 1024 ** 3


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(16 * 1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False


def verified(root):
    if not (root / 'completed').is_file():
        return False
    report = read(root / 'audit/final_verification.json')
    if not report.get('passed'):
        return False
    matches = [p for p in root.glob('*beam_search.json')
               if sha(p) == report['result_sha256']]
    assert len(matches) == 1, 'Verified result is missing or changed'
    result = read(matches[0])
    assert result['evaluated_blocks'] == [1, 2, 3, 4]
    assert result['skipped_blocks'] == [0]
    assert result['stage_restoration'] == 'historical_stage_state'
    status = read(root / 'status.json')
    assert status.get('status') in ('completed', 'complete', 'success'), status
    return dict(result_path=str(matches[0]), result_sha256=sha(matches[0]),
                macro_average=result['macro_average'],
                verification=str(root / 'audit/final_verification.json'))


def gpu_idle(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    selected = [r.split(',') for r in rows.splitlines() if int(r.split(',')[0]) in gpus]
    return len(selected) == 2 and all(int(r[2]) <= 64 and int(r[3]) == 0 and r[1].strip() not in apps for r in selected)


def cleanup(root, authorization):
    assert authorization, 'No weight deletion authorization'
    result = verified(root)
    assert result
    record_path = HERE / ('cleanup_' + root.name + '.json')
    if record_path.exists() and read(record_path).get('status') == 'complete':
        return
    files = []
    for b in range(1, 5):
        p = root / 'persistent_stages' / ('block_%d' % b) / 'model.safetensors'
        if not p.exists():
            continue
        assert p.is_file() and not p.is_symlink() and p.resolve() == p
        assert p.is_relative_to(root / 'persistent_stages')
        files.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    report = dict(status='verified_before_deletion', authorization=authorization,
                  predecessor_result=result, files=files)
    write(record_path, report)
    for row in files:
        Path(row['path']).unlink()
    assert sha(result['result_path']) == result['result_sha256']
    report.update(status='complete', freed_bytes=sum(r['bytes'] for r in files))
    write(record_path, report)


def future_disk(root):
    if (root / 'completed').exists():
        return 0
    missing = sum(not (root / 'persistent_stages' / ('block_%d' % b) / 'stage_manifest.json').exists()
                  for b in range(1, 5))
    return missing * 4.25e9 + 4.5e9


def main():
    lock = (HERE / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (HERE / 'queue.pid').write_text(str(os.getpid()))
    spec = importlib.util.spec_from_file_location('shared_delta', SHARED / 'run.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    while True:
        cfg = read(HERE / 'queue_config.json')
        states = []
        for item in cfg['chains']:
            row = dict(item)
            source, target = Path(item['after']), Path(item['target'])
            try:
                target_done = verified(target)
                if target_done:
                    row.update(status='completed_verified', result=target_done)
                    states.append(row)
                    continue
                status = read(target / 'status.json')
                if (target / 'launcher.pid').exists():
                    pid = int((target / 'launcher.pid').read_text())
                    row.update(status='target_running' if alive(pid) else 'target_needs_attention',
                               pipeline_pid=pid, target_status=status)
                    states.append(row)
                    continue
                predecessor = verified(source)
                if not predecessor:
                    row.update(status='waiting_predecessor', predecessor_status=read(source / 'status.json'))
                    states.append(row)
                    continue
                write(HERE / ('predecessor_result_%d.json' % item['budget_percent']), predecessor)
                source_pid = int((source / 'pipeline.pid').read_text())
                if alive(source_pid) or not gpu_idle(item['gpus']):
                    row.update(status='waiting_gpu_release')
                    states.append(row)
                    continue
                reserve = 0
                for other in cfg['chains']:
                    if other == item:
                        continue
                    for name in ['after', 'target']:
                        r = Path(other[name])
                        if (r / 'launcher.pid').exists() and not (r / 'completed').exists():
                            reserve += future_disk(r)
                needed = 22 * GIB + reserve
                if shutil.disk_usage(BASE).free <= needed and cfg.get('cleanup_authorization'):
                    cleanup(source, cfg['cleanup_authorization'])
                free = shutil.disk_usage(BASE).free
                if free <= needed:
                    row.update(status='waiting_disk', free_bytes=free, required_bytes=needed)
                    states.append(row)
                    continue
                assert status['status'] == 'prepared_not_started'
                launcher.execute(target)
                row.update(status='target_launched', pipeline_pid=int((target / 'launcher.pid').read_text()))
            except Exception:
                row.update(status='needs_attention', error=traceback.format_exc())
                print(row['error'], flush=True)
            states.append(row)
        done = all(r['status'] == 'completed_verified' for r in states)
        write(HERE / 'queue_status.json', dict(status='completed' if done else 'monitoring',
              updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), chains=states))
        if done:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
