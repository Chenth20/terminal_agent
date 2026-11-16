from functools import partial, wraps
from pathlib import Path
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
from llamafactory.model.model_utils import checkpointing

OFFLOAD_LAYERS=frozenset(i for i in range(32) if i % 8 != 7)
moved=set()
original_init=Qwen3_5DecoderLayer.__init__
@wraps(original_init)
def tag_layer(self,config,layer_idx):
    original_init(self,config,layer_idx)
    self.offload_index=layer_idx
Qwen3_5DecoderLayer.__init__=tag_layer

def check_memory():
    cg=Path('/sys/fs/cgroup')
    used=int((cg/'memory.current').read_text())
    limit=int((cg/'memory.max').read_text())
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if min(limit-used,available)<40*1024**3:
        raise RuntimeError('Host memory reserve below 40 GiB')

def unpack(obj):
    if isinstance(obj,tuple):
        cpu,device=obj
        return cpu.to(device,non_blocking=False)
    return obj

base=checkpointing.get_custom_gradient_checkpointing_func
def selective_wrapper(fn):
    original=base(fn)
    def wrapped(func,*args,**kwargs):
        mod=(func.func if isinstance(func,partial) else func).__self__
        idx=getattr(mod,'offload_index',None)
        if idx not in OFFLOAD_LAYERS or not torch.is_grad_enabled():
            return original(func,*args,**kwargs)
        ptr=args[0].data_ptr()
        size=args[0].numel()*args[0].element_size()
        count=0
        def pack(t):
            nonlocal count
            if t.device.type=='cuda' and t.data_ptr()==ptr and t.numel()*t.element_size()==size:
                check_memory()
                cpu=torch.empty_like(t,device='cpu',pin_memory=False)
                cpu.copy_(t.detach(),non_blocking=False)
                count+=1
                moved.add(idx)
                return cpu,t.device
            return t.detach()
        with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            result=original(func,*args,**kwargs)
        if count!=1:raise RuntimeError(f'Layer {idx}: expected 1 checkpoint activation, saw {count}')
        return result
    return wrapped
checkpointing.get_custom_gradient_checkpointing_func=selective_wrapper
