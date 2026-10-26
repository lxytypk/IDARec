import json,pathlib,re,subprocess,sys,torch
ROOT=pathlib.Path(__file__).resolve().parent
dest=ROOT/'audit';dest.mkdir(exist_ok=True)
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
freeze=re.sub(r'(https?://)[^/\s@]+@',r'\1[redacted]@',freeze)
(dest/'dependency_versions.txt').write_text(freeze)
(dest/'gpu_hardware.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,pci.bus_id,memory.total','--format=csv'],text=True))
(dest/'gpu_topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
(dest/'runtime.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),system=subprocess.check_output(['uname','-a'],text=True).strip()),indent=2))
print('runtime, dependency versions, GPU identifiers and topology saved')
import datasets,pyarrow,pandas,flash_attn
(dest/'resolved_dependency_imports.json').write_text(json.dumps({name:dict(version=module.__version__,file=module.__file__)
    for name,module in [('datasets',datasets),('pyarrow',pyarrow),('pandas',pandas),('flash_attn',flash_attn)]},indent=2))
