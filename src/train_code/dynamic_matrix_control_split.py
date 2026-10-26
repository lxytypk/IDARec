"""Periodic base-W Taylor/GradNorm selection; persistent A/B and optimizer state.

All A/B tensors are registered once. Inactive gradients are masked before ZeRO
reduction/clipping. Inactive FP32 parameter and Adam moment slices are preserved
across the underlying optimizer step, before ZeRO copies them back to BF16.
requires_grad is not a count of the active update budget.
"""
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback
from continual_train import ContinualLoRATrainer
import static_training as base
from split_budget_utils import family_counts, greedy_selection_split, split_budget_limits
from lora_ab_grad_score import LoRAABGradientAccumulator
from score_variants import module_score
from regularization import relative_ab_anchor


class DynamicController(TrainerCallback):
    def __init__(self, args, model, candidates, dataset, tokenizer, selected):
        self.cfg, self.model, self.candidates = args, model, candidates
        self.dataset, self.tokenizer = dataset, tokenizer
        self.selected = set(selected)
        self.union = set(selected)
        self.costs = base.core.candidate_costs(candidates)
        self.budget = int(sum(self.costs.values()) * args.budget_ratio)
        self.attention_budget_share = float(args.attention_budget_share)
        self.family_budgets = split_budget_limits(self.budget, self.attention_budget_share)
        self.probe = base.read_json(base.score_path(args))["probe"]
        self.last_score_step = 0
        self.score_seconds = 0.0
        self.interval_before = self.snapshot()
        # DDP must know every parameter that might become active later.
        base.core.set_matrix_trainability(candidates, set(candidates))
        self.gradient_hooks=[]
        for key,row in candidates.items():
            for p in (row['module'].A,row['module'].B):
                self.gradient_hooks.append(p.register_hook(
                    lambda grad,k=key: grad if k in self.selected else torch.zeros_like(grad)))

    def on_train_begin(self,args,state,control,**kwargs):
        engine=self.trainer.model_wrapped
        resume=getattr(self.trainer,'dynamic_resume',None)
        if resume:
            self.restore(resume)
        if not hasattr(engine,'optimizer') or not hasattr(engine.optimizer,'single_partition_of_fp32_groups'):
            raise RuntimeError('Expected original DeepSpeed ZeRO-2 optimizer')
        zero=engine.optimizer
        optimizer=zero.optimizer
        if not isinstance(optimizer,torch.optim.AdamW):
            raise RuntimeError('Expected original torch AdamW')
        def before_step(opt,args,kwargs):
            self.saved_fragments=[]
            current={id(p) for group in opt.param_groups for p in group['params']}
            for key,row in self.candidates.items():
                if key in self.selected: continue
                for p in (row['module'].A,row['module'].B):
                    mapping=getattr(p,'_hp_mapping',None)
                    if mapping is None: continue  # Owned by the other rank.
                    master=zero.single_partition_of_fp32_groups[mapping.param_group_index]
                    if id(master) not in current: continue
                    address=mapping.hp_fragment_address
                    weight=master.data.narrow(0,address.start,address.numel)
                    states={}
                    for name in ('exp_avg','exp_avg_sq'):
                        value=opt.state[master].get(name)
                        states[name]=None if value is None else value.narrow(0,address.start,address.numel).clone()
                    self.saved_fragments.append((master,address,weight.clone(),states))
        def after_step(opt,args,kwargs):
            with torch.no_grad():
                for master,address,weight,states in self.saved_fragments:
                    master.narrow(0,address.start,address.numel).copy_(weight)
                    for name,old in states.items():
                        value=opt.state[master][name].narrow(0,address.start,address.numel)
                        if old is None: value.zero_()
                        else: value.copy_(old)
            self.saved_fragments=[]
        self.optimizer_hooks=[optimizer.register_step_pre_hook(before_step),optimizer.register_step_post_hook(after_step)]
        print('DYNAMIC_ZERO2_FREEZE installed: masked gradients; preserved FP32 weights and Adam moments',flush=True)

    def snapshot(self):
        return {k: (r['module'].A.detach().cpu().clone(), r['module'].B.detach().cpu().clone())
                for k, r in self.candidates.items()}

    def clear_inactive(self):
        for key, row in self.candidates.items():
            if key not in self.selected:
                row['module'].A.grad = None
                row['module'].B.grad = None

    def close(self):
        for handle in self.gradient_hooks + getattr(self,'optimizer_hooks',[]):
            handle.remove()
        self.gradient_hooks=[]
        self.optimizer_hooks=[]
        self.trainer=None

    def audit_interval(self, step):
        rows = {}
        for key, row in self.candidates.items():
            old = self.interval_before[key]
            now = (row['module'].A.detach().cpu(), row['module'].B.detach().cpu())
            delta = sum(float((a.float()-b.float()).square().sum()) for a,b in zip(now,old))
            rows[key] = {'active': key in self.selected, 'squared_change': delta}
            if key not in self.selected and any(not torch.equal(a,b) for a,b in zip(now,old)):
                raise RuntimeError('Inactive matrix changed: '+key)
        if base.rank0():
            base.write_json(os.path.join(self.cfg.root,'dynamic',f'block{self.cfg.block_id}',f'updates_{step}.json'), rows)
        self.interval_before = self.snapshot()

    def state_dict(self):
        rows = [dict(matrix=k, projection=r['projection']) for k, r in self.candidates.items()]
        family_used = {'attention': 0, 'mlp': 0}
        for key in self.selected:
            projection = self.candidates[key]['projection']
            family = 'attention' if projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
            family_used[family] += self.costs[key]
        return {'selected': sorted(self.selected), 'ever_selected': sorted(self.union),
                'last_score_step': self.last_score_step, 'score_seconds': self.score_seconds,
                'budget': self.budget, 'interval': self.cfg.reselect_steps,
                'attention_budget_share': self.attention_budget_share,
                'mlp_budget_share': 1.0 - self.attention_budget_share,
                'family_budget_P': self.family_budgets, 'family_used_P': family_used,
                'family_selected_matrix_count': family_counts(rows, self.selected),
                'selection_score_mode': self.cfg.selection_score_mode,
                'score_mode': self.cfg.score_mode}

    def restore(self, path):
        state = base.read_json(os.path.join(path,'dynamic_state.json'))
        if (state['budget'] != self.budget or state['interval'] != self.cfg.reselect_steps or
                state.get('attention_budget_share') != self.attention_budget_share or
                state.get('selection_score_mode', 'gradient_per_parameter') != self.cfg.selection_score_mode or
                state.get('score_mode', self.cfg.score_mode) != self.cfg.score_mode):
            raise RuntimeError('Resume dynamic configuration differs')
        self.selected = set(state['selected']); self.union = set(state['ever_selected'])
        self.last_score_step = state['last_score_step']; self.score_seconds = state['score_seconds']
        self.interval_before = self.snapshot()

    def save(self, path):
        if not base.rank0(): return
        base.write_json(os.path.join(path,'dynamic_state.json'),self.state_dict())
        meta = base.read_json(os.path.join(base.stage_dir(self.cfg),'selected_layers.json'))
        meta.update(method='dynamic_matrix_'+self.cfg.score_mode, score_mode=self.cfg.score_mode, selection_score_mode=self.cfg.selection_score_mode,
                    matrix_selected=sorted(self.selected),
                    matrix_unselected=sorted(set(self.candidates)-self.selected),
                    selected_layers=sorted({self.candidates[k]['layer'] for k in self.selected}),
                    actual_selected_lora_AB_parameters=sum(self.costs[k] for k in self.selected),
                    remaining_lora_budget=self.budget-sum(self.costs[k] for k in self.selected),
                    dynamic=self.state_dict(),
                    registered_optimizer_lora_parameters=sum(self.costs.values()),
                    freeze_mechanism='zero inactive gradients before reduction/clipping; preserve inactive ZeRO FP32 weights and Adam moments')
        base.write_json(os.path.join(path,'selected_layers.json'),meta)

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.cfg.reselect_steps == 0 and step != self.last_score_step:
            self.audit_interval(step)
            self.rescore(step)
        self.step_before = self.snapshot()

    def on_step_end(self, args, state, control, **kwargs):
        for key, row in self.candidates.items():
            if key in self.selected: continue
            for p,old in zip((row['module'].A,row['module'].B),self.step_before[key]):
                if not torch.equal(p.detach().cpu(),old):
                    raise RuntimeError(f'Optimizer changed inactive {key} at step {state.global_step}')
        del self.step_before

    def rescore(self, step):
        start = time.time()
        model = self.model
        device = next(model.parameters()).device
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        indices = self.probe['sample_indices']
        rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        modes = {m: m.training for m in model.modules()}
        required = {p: p.requires_grad for p in model.parameters()}
        versions = {n: p._version for n,p in model.named_parameters()}
        cache = model.config.use_cache
        values = torch.zeros(len(self.candidates),dtype=torch.float64,device=device)
        keys = list(self.candidates)
        try:
            model.zero_grad(set_to_none=True)
            model.eval(); model.config.use_cache=False
            for p in model.parameters(): p.requires_grad_(False)
            for r in self.candidates.values(): r['module'].orig.weight.requires_grad_(True)
            ab_accumulator = LoRAABGradientAccumulator(self.candidates) if self.cfg.score_mode == 'lora_ab_grad_sq' else None
            loader = DataLoader(Subset(self.dataset,indices[rank::world]),batch_size=1,shuffle=False,
                                num_workers=0,collate_fn=base.Collator(self.cfg,self.tokenizer))
            # Bypass DDP and Accelerate's mixed-precision forward wrapper, matching
            # the original standalone scoring path. Registered A/B are frozen here.
            wrapped_forward = model.forward
            if hasattr(model,'_original_forward'): model.forward = model._original_forward
            try:
                for i,batch in enumerate(loader):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda',enabled=False):
                        loss = model(**base.core.move_to_device(batch,device)).loss
                    if not torch.isfinite(loss): raise RuntimeError('Non-finite probe loss')
                    loss.backward()
                    if ab_accumulator is not None:
                        ab_accumulator.add(self.candidates)
                    for j,k in enumerate(keys):
                        g = self.candidates[k]['module'].orig.weight.grad
                        if g is None: raise RuntimeError('Missing candidate gradient: '+k)
                        if ab_accumulator is None:
                            values[j] += module_score(self.candidates[k]['module'], g, self.cfg.score_mode).double()
                    if rank == 0 and (i+1)%100 == 0:
                        print(f'DYNAMIC_PROBE step={step} local_samples={i+1}/{len(loader)}',flush=True)
            finally:
                model.forward = wrapped_forward
            if ab_accumulator is not None:
                ab_scores = ab_accumulator.finish(len(indices), distributed=dist.is_initialized())
                values.copy_(torch.tensor([ab_scores[k] for k in keys], dtype=torch.float64, device=device))
            elif dist.is_initialized():
                dist.all_reduce(values)
            if self.cfg.score_mode == 'grad_norm':
                values /= len(indices)
            if not torch.isfinite(values).all() or values.sum() <= 0:
                raise RuntimeError('Invalid dynamic scores')
            if any(p._version != versions[n] for n,p in model.named_parameters()):
                raise RuntimeError('Probe modified model weights')
        finally:
            model.zero_grad(set_to_none=True)
            for p,flag in required.items(): p.requires_grad_(flag)
            for m,mode in modes.items(): m.training=mode
            model.config.use_cache=cache
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        rows = [dict(matrix=k,layer=self.candidates[k]['layer'],projection=self.candidates[k]['projection'],
                     raw_score=float(values[j]),lora_cost=self.costs[k],score_per_cost=float(values[j])/self.costs[k],
                     selection_score=(float(values[j]) if self.cfg.selection_score_mode == 'pure_gradient'
                                      else float(values[j])/self.costs[k]))
                for j,k in enumerate(keys)]
        rows.sort(key=lambda r:(-r['selection_score'],r['layer'],base.core.MODULE_INDEX[r['projection']]))
        new,family_used,family_budgets = greedy_selection_split(
            rows, self.budget, self.attention_budget_share)
        used = sum(family_used.values())
        new=set(new)
        if (not new or used > self.budget or
                any(family_used[name] > family_budgets[name] for name in family_used)):
            raise RuntimeError('Invalid dynamic split budget')
        digest=hashlib.sha256(json.dumps(sorted(new)).encode()).hexdigest()
        if dist.is_initialized():
            digests=[None]*world; dist.all_gather_object(digests,digest)
            if len(set(digests))!=1: raise RuntimeError('Ranks disagree on selection')
        old=self.selected
        self.selected=new; self.union |= new; self.last_score_step=step
        elapsed=time.time()-start; self.score_seconds+=elapsed
        if base.rank0():
            base.write_json(os.path.join(self.cfg.root,'dynamic',f'block{self.cfg.block_id}',f'selection_{step}.json'),
                dict(step=step,probe=self.probe,score_mode=self.cfg.score_mode, selection_score_mode=self.cfg.selection_score_mode,
                     matrix_candidates=rows,selected=sorted(new),
                     added=sorted(new-old),removed=sorted(old-new),budget=self.budget,used=used,
                     attention_budget_share=self.attention_budget_share,
                     family_budget_P=family_budgets, family_used_P=family_used,
                     family_selected_matrix_count=family_counts(rows, new),
                     seconds=elapsed,weights_unchanged=True,rng_restored=True,rank_digest=digest))
            print(f'DYNAMIC_RESELECT step={step} selected={len(new)} added={len(new-old)} removed={len(old-new)} used={used}/{self.budget} seconds={elapsed:.1f}',flush=True)
        self.save(base.stage_dir(self.cfg))


class DynamicTrainer(ContinualLoRATrainer):
    def __init__(self,*args,dynamic_controller,**kwargs):
        self.dynamic_controller=dynamic_controller
        super().__init__(*args,**kwargs)
        self.add_callback(dynamic_controller)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        cfg = self.dynamic_controller.cfg
        mode = getattr(cfg, 'regularization_mode', 'none')
        weight = float(getattr(cfg, 'regularization_weight', 0.0))
        # Scoring and checkpoint selection remain response-CE only.  The anchor
        # is an explicitly enabled training objective, never an implicit default.
        if model.training and mode == 'ab_anchor' and weight > 0:
            penalties = []
            for key in sorted(self.dynamic_controller.selected):
                module = self.dynamic_controller.candidates[key]['module']
                penalties.append(relative_ab_anchor(module,cfg.block_id-1))
            penalty = torch.stack(penalties).mean()
            ce = loss
            loss = loss + weight * penalty
            step = self.state.global_step
            if step % 100 == 0 and getattr(self, '_anchor_logged_step', -1) != step:
                self.log({'anchor/relative_squared_distance': float(penalty.detach()),
                          'anchor/weighted_loss': float((weight*penalty).detach()),
                          'anchor/response_ce': float(ce.detach())})
                self._anchor_logged_step = step
        return (loss, outputs) if return_outputs else loss

    def training_step(self,*args,**kwargs):
        loss=super().training_step(*args,**kwargs)
        self.dynamic_controller.clear_inactive()
        return loss

    def _save_checkpoint(self,*args,**kwargs):
        super()._save_checkpoint(*args,**kwargs)
        path=os.path.join(self.args.output_dir,f'checkpoint-{self.state.global_step}')
        self.dynamic_controller.save(path)
        # Keep scheduler/RNG in data-only JSON, avoiding the old torch.load
        # scheduler format rejected by this server's Transformers version.
        if base.rank0(): base.write_json(os.path.join(path,'scheduler_state.json'),self.lr_scheduler.state_dict())
        ns=np.random.get_state()
        base.write_json(os.path.join(path,f'rng_state_{self.args.process_index}.json'),
            dict(python=random.getstate(),numpy=[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],
                 cpu=torch.get_rng_state().tolist(),cuda=[s.tolist() for s in torch.cuda.get_rng_state_all()]))

    def _load_optimizer_and_scheduler(self,checkpoint):
        if checkpoint is None: return
        if not self.is_deepspeed_enabled: raise RuntimeError('Expected original DeepSpeed configuration')
        # DeepSpeed has already restored its model and optimizer state in Trainer.
        self.lr_scheduler.load_state_dict(base.read_json(os.path.join(checkpoint,'scheduler_state.json')))

    def _load_rng_state(self,checkpoint):
        if checkpoint is None: return
        s=base.read_json(os.path.join(checkpoint,f'rng_state_{self.args.process_index}.json'))
        def tuples(x): return tuple(tuples(v) for v in x) if isinstance(x,list) else x
        random.setstate(tuples(s['python']))
        n=s['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        torch.set_rng_state(torch.tensor(s['cpu'],dtype=torch.uint8))
        torch.cuda.set_rng_state_all([torch.tensor(v,dtype=torch.uint8) for v in s['cuda']])

    def _load_best_model(self):
        self.dynamic_controller.audit_interval(self.state.global_step)
        super()._load_best_model()
        if self.state.best_model_checkpoint:
            self.dynamic_controller.restore(self.state.best_model_checkpoint)

    def _load_from_checkpoint(self,resume_from_checkpoint,*args,**kwargs):
        super()._load_from_checkpoint(resume_from_checkpoint,*args,**kwargs)
        self.dynamic_controller.restore(resume_from_checkpoint)
