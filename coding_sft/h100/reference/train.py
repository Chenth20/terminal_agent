import sys,math,json
from pathlib import Path
import torch,yaml
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp
import selective

w=Path(__file__).parent
mode=sys.argv[1]
losses=[1.3434,1.3483,1.3894,1.3508,1.3658,1.3448,1.3009,1.3173,1.2971,1.2830]
tokens=[8348608,16327384,24341360,33227784,41692920,49913696,57817352,64952816,73580416,81642520]
class FinishedProbe(Exception):pass
class Validate(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==3:
            control.should_save=True
    def on_log(self,args,state,control,logs=None,**kwargs):
        if not logs or 'loss' not in logs or not state.global_step:return
        i=state.global_step-1
        d=dict(step=i+1,tokens=state.num_input_tokens_seen,loss=logs['loss'],lr=logs['learning_rate'],layers=sorted(selective.moved),rank=torch.distributed.get_rank(),max_allocated_GiB=round(torch.cuda.max_memory_allocated()/2**30,2),max_reserved_GiB=round(torch.cuda.max_memory_reserved()/2**30,2))
        ok=d['layers']==sorted(selective.OFFLOAD_LAYERS) and math.isfinite(d['loss'])
        if mode=='check10':
            ok=ok and abs(d['loss']-losses[i])<=0.01 and d['tokens']==tokens[i] and math.isclose(d['lr'],i*1e-5/31,abs_tol=1e-10)
        else:ok=ok and d['tokens']==json.loads((w/'probe_tokens.json').read_text())
        d['passed']=ok
        with (w/f'{mode}-rank{d["rank"]}.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        if not ok:raise RuntimeError(f'Validation failed: {d}')
        if mode=='probe':raise FinishedProbe()
        if i==9:
            control.should_training_stop=True
            control.should_save=True
try:run_exp(yaml.safe_load((w/(mode+'.yaml')).read_text()),callbacks=[Validate()])
except FinishedProbe:pass
