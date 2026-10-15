"""Strict independent native correctness; no performance claim or service deployment."""
import argparse,json,pathlib,time,traceback
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--library',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if not a.execute:print('Prepared only. Requires compiled isolated vendor and --execute in an idle NPU window.');raise SystemExit(0)
assert not a.output.exists()
import torch,torch_npu
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.device.device_op import DeviceOperator
torch.set_num_threads(1);torch.npu.set_device(0);assert enable_custom_op();torch.ops.load_library(a.library)
result={'started':time.time(),'passed':False,'checks':[],'scope':'Attn Post + FFN Pre + Norm. Full residual, BF16 rounding, all6 combined outputs, replay changed inputs; no model/TP collective/timing validation.'}
def save():a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.output)
def baseline(args):
 residual,fn,scale,base,mix,x,post,comb,gamma=args
 h=torch.ops._C_ascend.npu_hc_post(x.unsqueeze(0),residual.unsqueeze(0),post.unsqueeze(0),comb.unsqueeze(0)).squeeze(0)
 outputs=torch.ops._C_ascend.npu_hc_pre_v3(h,fn,scale,base,mix,hc_mult=4,hc_sinkhorn_iters=20,norm_eps=1e-6,hc_eps=1e-6)
 y,post2,comb2,pre2=outputs
 y,y32=DeviceOperator.rms_norm_cast(y,gamma,1e-6)
 return (y,post2,comb2,pre2,h,y32)
from vllm_ascend.ops import v41_joint_runtime as runtime_dispatch
def candidate(args):return torch.ops.v41_joint_runtime.dispatch(*args,20,1e-6,1e-6,1e-6)
def make(n,seed,pattern):
 torch.manual_seed(seed);r=torch.randn(n,4,5120);x=torch.randn(n,5120)
 if pattern=='cancellation':r[:,1]=-r[:,0];r[:,3]=-r[:,2];x*=.001
 if pattern=='wide_scale':r*=128;x*=.0009765625
 r=r.to(torch.bfloat16);x=x.to(torch.bfloat16);fn=torch.randn(24,20480)/20480;scale=torch.tensor([.7,.9,.6]);base=torch.randn(24)*.1;mix=torch.sigmoid(torch.randn(n,4));post=2*torch.sigmoid(torch.randn(n,4));comb=torch.softmax(torch.randn(n,4,4),-1)
 for _ in range(20):comb=comb/(comb.sum(-2,keepdim=True)+1e-6);comb=comb/(comb.sum(-1,keepdim=True)+1e-6)
 return [r,fn,scale,base,mix,x,post,comb,(1+torch.randn(5120)*.1).to(torch.bfloat16)]
def oracle(host):
 r,fn,scale,base,mix,x,post,comb,gamma=host;acc=x.float().unsqueeze(-2)*post.unsqueeze(-1)
 for source in range(4):acc=acc+r[:,source].float().unsqueeze(-2)*comb[:,source,:].unsqueeze(-1)
 return acc.to(torch.bfloat16)
def check(ref,out,host):
 checks=[]
 for name,x,y in zip(['y','post','comb','pre','residual','y32'],ref,out):
  x=x.detach().cpu();y=y.detach().cpu();equal=torch.equal(x,y);diff=(x.float()-y.float()).abs();checks.append({'name':name,'bitexact':equal,'max_abs':diff.max().item(),'mismatches':int((x!=y).sum())})
 assert all(x['bitexact'] for x in checks),checks
 assert torch.equal(out[4].cpu(),oracle(host)),'Post RINT/operation-order differs from explicit FP32 CPU oracle'
 assert torch.equal(out[0].float().cpu(),out[5].cpu()), "router FP32 must widen rounded BF16"
 return checks
try:
 from symbolic_meta_check import run_symbolic_meta_checks
 result['symbolic_meta']=run_symbolic_meta_checks('v41_postprenorm_native')
 for n in [1,2,3,4,5,6,12,24,48,96,192,193,8192]:
  for pattern in (['random','cancellation','wide_scale'] if n<=192 else ['random']):
   host=make(n,1024,pattern);args=[t.npu() for t in host];original=[x.clone() for x in args]
   for replay in range(2):
    host=make(n,1024+replay,pattern)
    for dst,src in zip(args,host):dst.copy_(src)
    ref=baseline(args);out=candidate(args);torch.npu.synchronize();checks=check(ref,out,host);assert out[4].data_ptr()!=args[0].data_ptr(),'residual alias'
   stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream())
   with torch.npu.stream(stream):
    for _ in range(2):baseline(args);candidate(args)
   torch.npu.current_stream().wait_stream(stream);torch.npu.synchronize();gr=torch.npu.NPUGraph();gc=torch.npu.NPUGraph()
   with torch.npu.graph(gr,stream=stream):ref=baseline(args)
   with torch.npu.graph(gc,stream=stream):out=candidate(args)
   for replay in range(3):
    host=make(n,2048+replay,pattern)
    for dst,src in zip(args,host):dst.copy_(src)
    torch.npu.synchronize();gr.replay();gc.replay();torch.npu.synchronize();checks=check(ref,out,host)
    assert all(torch.equal(dst.cpu(),src) for dst,src in zip(args,host)),'input mutated'
   result['checks'].append({'rows':n,'pattern':pattern,'eager_checks':2,'graph_replays':3,'strict_outputs':checks});save();del gr,gc,ref,out,args,original
 # Empty rows are unsupported by baseline HcPre as well; record explicit reject.
 empty=[t.npu() for t in make(0,0,'random')]
 rejects=[]
 for name,fn in [('baseline',baseline),('candidate',candidate)]:
  try:fn(empty);rejects.append(False)
  except (RuntimeError,ValueError):rejects.append(True)
 assert all(rejects),rejects
 result.update(passed=True,empty_rejected_by_both=True,finished=time.time(),peak_allocated_bytes=torch.npu.max_memory_allocated(),loaded_vendor_maps=[l for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'v41hcpostprenorm' in l or 'v41_hc_post_pre_norm' in l]);save()
except BaseException:result.update(error=traceback.format_exc(),finished=time.time());save();raise
print(json.dumps({'passed':True,'cases':len(result['checks']),'output':str(a.output)}))
