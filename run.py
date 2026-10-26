"""IDARec launcher (Delta-W Taylor backend). Preparation is the default; launch is explicit."""
import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SHARED=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):Path(path).write_text(json.dumps(value,indent=2))


def digest(data):return hashlib.sha256(data).hexdigest()



def load_resources():
    """Keep machine-specific resource paths in an untracked local file."""
    resources = read(SHARED / 'resources.json')
    local = SHARED / 'resources.local.json'
    if local.is_file():
        resources.update(read(local))
    for name in ('data', 'block0', 'base_model', 'environment'):
        path = Path(os.path.expandvars(resources[name])).expanduser()
        resources[name] = str(path if path.is_absolute() else (SHARED / path).resolve())
    return resources


def materialize_blob(destination,data):
    key=digest(data);blob=SHARED/'objects'/key
    if not blob.exists():
        temp=blob.with_name(key+'.tmp.'+str(os.getpid()));temp.write_bytes(data)
        temp.chmod(0o444);temp.replace(blob)
    assert digest(blob.read_bytes())==key
    destination.parent.mkdir(parents=True,exist_ok=True)
    # A hard link retains the run-local __file__ path, unlike a code symlink.
    os.link(blob,destination)
    return key


def transform(name,text,cfg,profile,out):
    if '/' in name:return text
    budget=cfg['budget_ratio'];share=cfg['attention_share'];full=5636096
    total=math.floor(budget*full);att=math.floor(total*share);mlp=total-att
    if budget==1:
        assert profile['kind']=='allAB'
        total,att,mlp=full,1703936,3932160
    old=profile['values'];oldgpu=profile['gpus'];gpus=cfg['gpus']
    # Parameters, assertions and records change together; training/scoring core stays identical.
    text=text.replace(profile['temporary'],str(Path('/dev/shm')/('delta_'+out.name)))
    text=text.replace(profile['result_name'],f"Books_delta_budget{round(budget*100):g}_{out.name}_beam_search.json")
    text=text.replace('total_LoRA_AB_budget='+old['budget_literal'],'total_LoRA_AB_budget='+repr(budget))
    text=text.replace("['total_LoRA_AB_budget']=="+old['budget_literal'],"['total_LoRA_AB_budget']=="+repr(budget))
    text=text.replace('self.cfg.budget_ratio == '+old['budget_literal'],'self.cfg.budget_ratio == '+repr(budget))
    text=text.replace("'--budget_ratio','"+old['budget_literal']+"'","'--budget_ratio','"+repr(budget)+"'")
    effective_share=old['share_literal'] if budget==1 else repr(share)
    text=text.replace('attention_share='+old['share_literal'],'attention_share='+effective_share)
    text=text.replace('mlp_share='+old['mlp_literal'],'mlp_share='+repr(1-share if budget<1 else 1-float(effective_share)))
    text=text.replace("['attention_share']=="+old['share_literal'],"['attention_share']=="+effective_share)
    text=text.replace("['mlp_share']=="+old['mlp_literal'],"['mlp_share']=="+repr(1-share if budget<1 else 1-float(effective_share)))
    text=text.replace('self.cfg.attention_budget_share == '+old['share_literal'],'self.cfg.attention_budget_share == '+effective_share)
    text=text.replace("'--attention_budget_share','"+old['share_literal']+"'","'--attention_budget_share','"+effective_share+"'")
    for previous,current in [(old['total'],total),(old['attention'],att),(old['mlp'],mlp)]:
        text=re.sub(r'\b'+str(previous)+r'\b',str(current),text)
    newlr=repr(float(cfg['actual_initial_lr']));command_lr=str(Decimal(str(cfg['actual_initial_lr']))*10)
    text=text.replace("'--learning_rate','1e-3'","'--learning_rate','"+command_lr+"'")
    text=text.replace('command_learning_rate=1e-3','command_learning_rate='+command_lr)
    lines=[]
    for line in text.splitlines(True):
        if any(token in line for token in ['initial_lr','learning_rate']):
            line=line.replace('1e-4',newlr)
        if any(token in line for token in ['reselect_steps','reselect_optimizer_steps']):
            line=re.sub(r'\b100\b',str(cfg['reselect_steps']),line)
        if name=='final_verify.py' and 'for rank in [' in line:
            line=re.sub(r'for rank in \[[^\]]+\]', 'for rank in [0,1]',line)
        if 'for index in [' in line:
            line=re.sub(r'for index in \[[^\]]+\]',f'for index in {gpus}',line)
        if 'row.split' in line and ' in [' in line:
            line=re.sub(r' in \['+r'\s*,\s*'.join(map(str,oldgpu))+r'\]',f' in {gpus}',line)
        line=re.sub(r"CUDA_VISIBLE_DEVICES='[^']+'","CUDA_VISIBLE_DEVICES='"+','.join(map(str,gpus))+"'",line)
        line=re.sub(r'(selected_gpus|gpus)=\['+r'\s*,\s*'.join(map(str,oldgpu))+r'\]',lambda m:m[1]+'='+repr(gpus),line)
        line=re.sub(r'port='+str(profile['port'])+r'\b','port='+str(cfg['port']),line)
        lines.append(line)
    return ''.join(lines)


def prepare(cfg,output):
    cfg=dict(cfg)
    assert cfg.get('dataset','Books_cl_ori')=='Books_cl_ori'
    assert cfg.get('seed',42)==42,'This verified profile uses seed42'
    budget=float(cfg['budget_ratio']);assert 0<budget<=1
    share=float(cfg.get('attention_share',.4));assert 0<share<1
    assert float(cfg['actual_initial_lr'])>0
    gpus=cfg.get('gpus');assert isinstance(gpus,list) and len(gpus)==len(set(gpus))==2
    cfg.update(budget_ratio=budget,attention_share=share,gpus=gpus,reselect_steps=int(cfg.get('reselect_steps',100)),port=int(cfg.get('port',53412)))
    assert cfg['reselect_steps']>0 and 1024<cfg['port']<65525
    output=Path(output).absolute();assert not output.exists(), 'Use a new output directory'
    assert output.parent.is_dir() and output.parent.stat().st_dev==SHARED.stat().st_dev,'Output must be on the shared code filesystem'
    profile=read(SHARED/'profiles'/('allAB.json' if budget==1 else 'grouped.json'))
    if budget==1:
        cfg['requested_attention_share']=share
        cfg['attention_share']=float(profile['values']['share_literal'])
        cfg['all_AB_update']=True
    output.mkdir()
    files={}
    for name,key in profile['files'].items():
        raw=(SHARED/'objects'/key).read_bytes();assert digest(raw)==key
        if name.endswith('.py'):
            text=transform(name,raw.decode(),cfg,profile,output);compile(text,str(output/name),'exec');raw=text.encode()
        files[name]=materialize_blob(output/name,raw)
    for name in ['audit','logs','inputs','ckpt','datasets','persistent_stages','wandb']:(output/name).mkdir(exist_ok=True)
    resources=load_resources()
    for name,target in [('shared_block0',resources['block0']),('base_model',resources['base_model']),('data',resources['data'])]:
        (output/'inputs'/name).symlink_to(target,target_is_directory=True)
    (output/'ckpt/block_0').symlink_to(resources['block0'],target_is_directory=True)
    (output/'datasets/Books_cl_ori').symlink_to(resources['data'],target_is_directory=True)
    (output/'env').symlink_to(resources['environment'],target_is_directory=True)
    write(output/'requested_config.json',dict(cfg,method='IDARec',score_backend='lora_delta_taylor',generation_mode='beam-search',evaluated_blocks=[1,2,3,4],skipped_blocks=[0],stage_restoration='historical_stage_state',official_commit=resources['official_commit'],shared_code=str(SHARED),backend_profile=profile['kind']))
    write(output/'shared_code_manifest.json',dict(shared_objects=files,mode='read-only hardlinks; changed entry files receive new immutable objects',original_source=profile['source'],core_scoring_and_training_files_unchanged=True))
    write(output/'status.json',dict(status='prepared_not_started',training_started=False))
    return output


def execute(output):
    cfg=read(output/'requested_config.json');env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),PYTHONUNBUFFERED='1')
    env.pop('CUDA_LAUNCH_BLOCKING',None)
    python=str(output/'env/bin/python')
    with (output/'logs/initial_preflight.log').open('ab') as log:
        subprocess.run([python,str(output/'preflight.py')],cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (output/'logs/launcher.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([python,str(output/'run_pipeline.py')],cwd=output,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    (output/'launcher.pid').write_text(str(process.pid)+'\n')
    print(json.dumps(dict(output=str(output),pipeline_pid=process.pid,training_then_evaluation=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true')
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--launch',action='store_true',help='Explicitly run training and evaluation after preparing')
    args=p.parse_args()
    if args.list:
        print(json.dumps(read(SHARED/'experiments.json'),indent=2));return
    assert args.config and args.output,'Provide --config and --output'
    output=prepare(read(args.config),args.output)
    if args.launch:execute(output)
    else:print(json.dumps(dict(output=str(output),status='prepared_not_started',no_gpu_job_started=True)))


if __name__=='__main__':main()
