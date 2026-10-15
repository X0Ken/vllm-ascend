import ast,json,pathlib,types,typing,collections.abc
import torch,torch_npu
import torch.nn as nn
import torch.nn.functional as F
torch.npu.set_device(0)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
ROOT=pathlib.Path(__file__).resolve().parent
def pick(path,names,cls=None):
 tree=ast.parse(path.read_text());nodes=tree.body
 if cls:nodes=next(x for x in nodes if isinstance(x,ast.ClassDef) and x.name==cls).body
 return compile(ast.Module(body=[x for x in nodes if getattr(x,'name',None) in names],type_ignores=[]),str(path),'exec')
def load(rope,ds):
 g={'torch':torch,'torch_npu':torch_npu,'nn':nn,'F':F,'Any':typing.Any,'Iterable':collections.abc.Iterable}
 exec(pick(rope,['RopeGlobalState','RopeDataProxy','get_cos_and_sin_dsa']),g);g['_ROPE_STATE']=g['RopeGlobalState']()
 exec(pick(rope,['forward'],'ComplexExpRotaryEmbedding'),g);fwd=g['forward']
 class Rotary:
  layername='mtp.0.self_attn.attn'
  def __call__(self,x,cos,sin):return fwd(self,x,cos,sin)
 g['Rotary']=Rotary
 exec(pick(ds,['_apply_dsv4_rope']),g);exec(pick(ds,['_project_shared_kv','precompute_and_store_context_kv'],'DeepseekV4DSparkModel'),g)
 return g
A=load(ROOT/'reference_rope.py',ROOT/'reference_dspark.py');B=load(pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/rope_dsv4.py'),pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py'))
positions_all=torch.arange(4096,device='npu',dtype=torch.float32).reshape(-1,1,1,1);freq=torch.arange(64,device='npu',dtype=torch.float32).reshape(1,1,1,-1)/64
cos=torch.cos(positions_all*freq).to(torch.bfloat16);sin=torch.sin(positions_all*freq).to(torch.bfloat16)
for g in [A,B]:
 state=g['_ROPE_STATE'];state.full_rope_cache={'main':(cos,sin),'unused':(cos+1,sin+1)};state.registry_summary={'main':{'default'},'unused':{'default'}};state.layer_info={f'mtp.{i}.self_attn.attn':('main',['default']) for i in range(3)}
results=[]
for n in [1,2,3,4,5,6,7,24,48,96,192]:
 torch.manual_seed(17+n);h=torch.randn(n,512,device='npu',dtype=torch.bfloat16);pos=(torch.arange(n,device='npu')*3+5).long()
 def calculate(g,x):
  last=[]
  def projection(v):out=v.clone();last.append(out);return out
  attn=types.SimpleNamespace(wkv=projection,kv_norm=lambda v:v,rotary_emb=g['Rotary'](),nope_head_dim=448,rope_head_dim=64,head_dim=512)
  out=g['_project_shared_kv'](None,x,pos,attn);return out,last[-1]
 ya,pa=calculate(A,h);yb,pb=calculate(B,h);torch.npu.synchronize();assert torch.equal(ya,yb);assert yb.untyped_storage().data_ptr()==pb.untyped_storage().data_ptr();assert yb.is_contiguous();assert torch.equal(yb[:,0,:448],h[:,:448]);assert not torch.equal(yb[:,0,448:],h[:,448:])
 # Actual V4.1 cache scatter op, including a non-contiguous page plane and invalid sentinel.
 ca=torch.zeros(4,3,128,512,device='npu',dtype=torch.bfloat16)[:,1]
 cb=torch.zeros_like(ca)
 ix=torch.arange(n,device='npu',dtype=torch.int32);slots=torch.stack((ix//128,ix%128),dim=-1);slots[-1].fill_(-1)
 torch.ops._C_ascend.npu_scatter_nd_update_sk(ca,slots,ya.squeeze(1).contiguous())
 torch.ops._C_ascend.npu_scatter_nd_update_sk(cb,slots,yb.squeeze(1).contiguous())
 torch.npu.synchronize();assert torch.equal(ca,cb)
 ga=torch.npu.NPUGraph();gb=torch.npu.NPUGraph()
 for _ in range(3):calculate(A,h);calculate(B,h)
 torch.npu.synchronize()
 with torch.npu.graph(ga):oa,_=calculate(A,h)
 with torch.npu.graph(gb):ob,_=calculate(B,h)
 for j in range(3):
  h.copy_(torch.randn_like(h));ga.replay();gb.replay();torch.npu.synchronize();assert torch.equal(oa,ob)
 results.append({'tokens':n,'dtype':'bfloat16','head_dim':512,'rope_dim':64,'eager_exact':True,'alias':True,'non_rope_preserved':True,'writeback':True,'native_scatter_exact':True,'graph_replays_exact':3})
# Same and mixed per-layer RoPE configurations, including unused filtering.
for mixed in [False,True]:
 seen=[];g=B;state=g['_ROPE_STATE'];state.full_rope_cache['second']=(cos+2,sin+2);state.registry_summary['second']={'default'}
 names=[f'mtp.{i}.self_attn.attn' for i in range(3)]
 for i,name in enumerate(names):state.layer_info[name]=('second' if mixed and i==1 else 'main',['default'])
 pos=torch.tensor([1,3,5,7],device='npu');cs=torch.zeros(4,512,device='npu',dtype=torch.bfloat16)
 layers={name:types.SimpleNamespace(self_attn=types.SimpleNamespace(rotary_emb=types.SimpleNamespace(layername=name))) for name in names}
 def project(h,p,a,rope):
  seen.append(rope);c,s=rope;key=state.layer_info[a.rotary_emb.layername][0];assert torch.equal(c[a.rotary_emb.layername],state.full_rope_cache[key][0][p]);return h
 model=types.SimpleNamespace(layers=layers,_project_shared_kv=project,_store_standard_swa_kv=lambda *args:None)
 g['precompute_and_store_context_kv'](model,cs,pos,[pos,pos,pos]);assert len(seen)==3 and all(v is seen[0] for v in seen);assert set(seen[0][0]._data)==({'main','second'} if mixed else {'main'})
print(json.dumps({'status':'passed','projection_cases':results,'lookup_shared_and_mixed_passed':True,'scope':'Extracted actual A/B source methods; native NPU rotary_mul, alias/writeback and torch.npu.NPUGraph; projection isolated as clone/identity, full model tested separately.'}))
