import datetime,hashlib,json,os,pathlib
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.utils.data import Sampler
from transformers import GenerationConfig
from transformers.generation.configuration_utils import GenerationMode

ROOT=pathlib.Path(__file__).resolve().parent.parent
METRICS=['hit@1','hit@5','hit@10','ndcg@5','ndcg@10']
GENERATION=dict(do_sample=False,num_beams=10,num_return_sequences=10,max_new_tokens=10,early_stopping=True)
def write(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
class ExactSampler(Sampler):
    def __init__(self,dataset,num_replicas,rank,**kwargs):self.n,self.world,self.rank=len(dataset),num_replicas,rank
    def __iter__(self):return iter(range(self.rank,self.n,self.world))
    def __len__(self):return len(range(self.rank,self.n,self.world))
def pin_generation(model,args,block):
    assert args.num_beams==10 and args.test_batch_size==1
    assert os.environ.get('CUDA_LAUNCH_BLOCKING') is None
    inherited=model.generation_config.to_dict()
    cfg=GenerationConfig.from_model_config(model.config)
    cfg.pad_token_id=model.config.pad_token_id;cfg.eos_token_id=model.config.eos_token_id
    for key,value in GENERATION.items():setattr(cfg,key,value)
    cfg.output_scores=True;cfg.return_dict_in_generate=True
    assert cfg.get_generation_mode(None)==GenerationMode.BEAM_SEARCH
    model.generation_config=cfg
    if dist.get_rank()==0:
        write(ROOT/'audit'/f'generation_block{block}.json',dict(inherited_configuration=inherited,
            effective_configuration=cfg.to_dict(),generate_kwargs=dict(GENERATION,output_scores=True,return_dict_in_generate=True,synced_gpus=False),
            prefix_allowed_tokens_fn='official ContinualSeqRecDataset_v2.get_prefix_allowed_tokens_fn',
            generation_mode='beam-search',enum_mode=str(cfg.get_generation_mode(None)),
            sampling_configuration='fresh GenerationConfig from model config; inherited sampling fields discarded',
            CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING'),seed=42))
def audit_stage(model,args,block):
    stage=ROOT/'ckpt'/f'block_{block}'/'model.safetensors'
    manifest=json.loads((stage.parent/'stage_manifest.json').read_text())
    expected_hash=manifest['files']['model.safetensors']
    assert digest(stage)==expected_hash
    actual=model.state_dict();checked={};missing=[];mismatch=[]
    with safe_open(str(stage),framework='pt',device='cpu') as saved:
        for key in saved.keys():
            expected=saved.get_tensor(key)
            if key not in actual:missing.append(key);continue
            got=actual[key].detach().cpu()
            equal=got.dtype==expected.dtype and got.shape==expected.shape and torch.equal(got,expected)
            if not equal:mismatch.append(key)
            checked[key]=dict(shape=list(got.shape),dtype=str(got.dtype),
                sha256=hashlib.sha256(got.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),equal=equal,
                source=str(stage))
    extras=sorted(set(actual)-set(checked))
    anomalies=[k for k in extras if '.A_matrices.' not in k and '.B_matrices.' not in k]
    adapters=[]
    for name,module in model.named_modules():
        if hasattr(module,'A_matrices') and hasattr(module,'A'):
            assert module.current_block==block and not module.merged
            assert module.option=='lora_kldiv_latest'
            assert len(module.A_matrices)==5 and len(module.B_matrices)==5
            assert torch.equal(module.A,module.A_matrices[block])
            assert torch.equal(module.B,module.B_matrices[block])
            adapters.append(name)
    assert len(adapters)==112 and not missing and not mismatch and not anomalies
    for name in ['lm_head.modules_to_save.default.weight','model.embed_tokens.modules_to_save.default.weight']:
        assert actual[name].shape[0]==model.config.vocab_size==129280
    if dist.get_rank()==0:
        write(ROOT/'audit'/f'stage_{block}_restoration.json',dict(block=block,source=str(stage),
              checkpoint_sha256=expected_hash,missing_keys=missing,unexpected_keys=anomalies,
              inactive_future_history_keys=extras,mismatched_keys=mismatch,tensors=checked,
              active_adapter_modules=adapters,current_snapshot=block,
              stage_restoration='historical_stage_state',all_saved_stage_tensors_exact=True,
              original_final_load=getattr(model,'_load_audit',{})))
    print('HISTORICAL_STAGE_EXACT',block,'rank',dist.get_rank(),'tensors',len(checked),flush=True)
def coverage_metadata(dataset,args,block,tokenizer):
    raw=dataset.block_data[block]
    users=[str(k) for k,v in raw.items() if len(v['current_block'])>=3]
    assert len(users)==len(set(users))==len(dataset) and not args.debug and not args.filter_items
    vocab=tokenizer.get_vocab()
    semantic=sorted({t for seq in dataset.indices.values() for t in seq})
    for t in semantic:
        assert t in vocab and tokenizer(t)['input_ids'][1]==vocab[t]
    for rows in dataset.block_data.values():
        for row in rows.values():
            for item in row['previous_block']+row['current_block']:assert str(item) in dataset.indices
    return dict(block=block,original_users=len(raw),eligible_users=len(users),row_to_user=users,
                tokenizer_size=len(vocab),special_tokens=tokenizer.special_tokens_map,
                special_token_ids={n:getattr(tokenizer,n) for n in ['bos_token_id','eos_token_id','pad_token_id','unk_token_id']},
                semantic_token_count=len(semantic),semantic_token_ids={t:vocab[t] for t in semantic},
                eligibility='len(current_block)>=3',all_interactions_have_semantic_ids=True)
def gather_complete(metrics_results,total,seen_indices,coverage,block,world):
    shards=[None]*world
    dist.all_gather_object(shards,dict(metrics=metrics_results,count=total,indices=seen_indices))
    total=sum(x['count'] for x in shards)
    merged=[i for x in shards for i in x['indices']]
    n=coverage['eligible_users']
    assert total==n and len(merged)==len(set(merged))==n and sorted(merged)==list(range(n))
    users=[coverage['row_to_user'][i] for i in merged]
    assert len(users)==len(set(users))==n
    values={m:sum(x['metrics'].get(m,0.) for x in shards)/total for m in METRICS}
    assert all(0<=v<=1 for v in values.values())
    coverage.update(evaluated_users=total,unique_users=len(set(users)),exactly_once=True,
                    evaluated_row_indices=merged,rank_shards=[x['indices'] for x in shards],
                    collective_policy='independent uneven local loops; one gather after all local batches')
    if dist.get_rank()==0:write(ROOT/'audit'/f'coverage_block{block}.json',coverage)
    return values,total
def save_progress(args,rows):
    done=sorted(int(k.split('_')[1]) for k in rows)
    result=dict(experiment=str(ROOT),generation_mode='beam-search',evaluated_blocks=[1,2,3,4],skipped_blocks=[0],
                stage_restoration='historical_stage_state',completed_blocks=done,
                status='completed' if done==[1,2,3,4] else 'incomplete',
                block_status={str(b):'completed' if b in done else '未完成' for b in range(1,5)},
                official_commit='edbec5f68ddc83e39f22a041f1f2eaf1c4ad172e',
                generation_configuration=GENERATION,all_block_results=rows,
                updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    if done==[1,2,3,4]:
        result['macro_average']={m:sum(rows[f'block_{b}']['all_prompt_results'][0][m] for b in range(1,5))/4 for m in METRICS}
    write(args.results_file,result)
    lines=['| 数据块 | 原始用户数 | 符合条件用户数 | 实际评测用户数 | hit@1 | hit@5 | hit@10 | ndcg@5 | ndcg@10 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in range(1,5):
        row=rows.get(f'block_{b}')
        if row:
            cells=[str(b),str(row['original_users']),str(row['eligible_users']),str(row['evaluated_users'])]
            cells += [f'{row["all_prompt_results"][0][m]:.4f}' for m in METRICS]
        else:cells=[str(b)]+['未完成']*8
        lines.append('| '+' | '.join(cells)+' |')
    if 'macro_average' in result:
        lines.append('| 四块等权宏平均 | — | — | — | '+' | '.join(f'{result["macro_average"][m]:.4f}' for m in METRICS)+' |')
    pathlib.Path(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
