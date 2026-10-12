"""Opaque runtime Post+Pre+Norm; original fallback and six outputs preserved."""
import torch

import os,json,pathlib,time
_ROUTE_SEEN=set()
def _route_probe(kind,eligible,rows):
    directory=os.environ.get('V41_RUNTIME_ROUTE_PROBE')
    if not directory or (kind,eligible) in _ROUTE_SEEN:return
    import torch.distributed as dist
    rank=dist.get_rank() if dist.is_initialized() else 0
    path=pathlib.Path(directory)/('rank-'+str(rank)+'.json')
    value=json.loads(path.read_text()) if path.exists() else {'rank':rank,'routes':{},'scope':'CPU implementation returned after private operator submission/capture; not NPU start or graph replay count. Completion requires separate model smoke and driver gates.'}
    value['routes'][kind+(':eligible' if eligible else ':fallback')]={'rows':int(rows),'cpu_wall_time':time.time(),'private_operator_returned':bool(eligible)}
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path);_ROUTE_SEEN.add((kind,eligible))
@torch.library.custom_op('v41_joint_runtime::dispatch',mutates_args=())
def dispatch(residual:torch.Tensor,fn:torch.Tensor,scale:torch.Tensor,base:torch.Tensor,mix:torch.Tensor,x:torch.Tensor,post:torch.Tensor,comb:torch.Tensor,gamma:torch.Tensor,iters:int,norm_eps:float,hc_eps:float,variance_eps:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    eligible=(residual.ndim==3 and residual.shape[1:]==(4,5120) and 0<residual.shape[0]<=192 and residual.dtype==torch.bfloat16 and residual.is_contiguous() and iters==20)
    if eligible:
        out=torch.ops.v41_postprenorm_native.post_pre_norm(residual,fn,scale,base,mix,x,post,comb,gamma,iters,norm_eps,hc_eps,variance_eps)
        _route_probe('joint',True,residual.shape[0]);return out
    h=torch.ops._C_ascend.npu_hc_post(x.unsqueeze(0),residual.unsqueeze(0),post.unsqueeze(0),comb.unsqueeze(0)).squeeze(0)
    y,p,c,pre=torch.ops._C_ascend.npu_hc_pre_v2(h,fn,scale,base,mix,hc_mult=4,hc_sinkhorn_iters=iters,norm_eps=norm_eps,hc_eps=hc_eps)
    norm,norm32=torch.ops._C_ascend.npu_rms_norm_cast(y,gamma,variance_eps)
    _route_probe('joint',False,residual.shape[0]);return norm,p,c,pre,h,norm32
@dispatch.register_fake
def fake(residual,fn,scale,base,mix,x,post,comb,gamma,iters,norm_eps,hc_eps,variance_eps):
    n=residual.shape[0]
    return (residual.new_empty((n,5120)),residual.new_empty((n,4),dtype=torch.float32),residual.new_empty((n,4,4),dtype=torch.float32),residual.new_empty((n,4),dtype=torch.float32),residual.new_empty(residual.shape),residual.new_empty((n,5120),dtype=torch.float32))
