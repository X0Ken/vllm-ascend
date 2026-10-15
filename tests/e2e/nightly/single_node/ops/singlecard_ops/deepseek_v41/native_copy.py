"""Standalone exact actual AsyncGPUModelRunnerOutput constructor on NPU; no latency claim."""
import argparse,ast,hashlib,json,pathlib,time,types,traceback
import torch,torch_npu
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
r={'started':time.time(),'ok':False,'scope':'actual constructor AST; CUDA namespace maps to NPU as Ascend runtime; synthetic integer samples and subsequent draft-like reads; no full-model or performance claim','eager':[],'graph':[]}
def save():out.write_text(json.dumps(r,indent=2))
try:
 raw=pathlib.Path(a.source).read_bytes();tree=ast.parse(raw)
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AsyncGPUModelRunnerOutput');ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 isolated=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='ActualConstructor',bases=[],keywords=[],body=[ctor],decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(isolated)
 scope={'torch':types.SimpleNamespace(cuda=torch.npu)};exec(compile(isolated,'actual-AsyncGPUModelRunnerOutput-constructor','exec'),scope);Actual=scope['ActualConstructor'];torch.npu.set_device(0);copy_stream=torch.npu.Stream()
 for dtype in [torch.int32,torch.int64]:
  for n in [1,4,8,24,48,96,192]:
   for cols in [1,6]:
    x=torch.randint(0,129280,(n,cols),device='npu',dtype=dtype);stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
     for _ in range(3):sample=x.clone()
     torch.npu.synchronize();graph=torch.npu.NPUGraph()
     with torch.npu.graph(graph,stream=stream):sample=x.clone()
    torch.npu.current_stream().wait_stream(stream)
    for mode in ['eager','graph']:
     for replay in range(3):
      x.copy_(torch.randint_like(x,0,129280))
      if mode=='graph':graph.replay()
      else:sample=x.clone()
      meta=object();early=Actual(meta,sample,None,[],copy_stream,129280)
      # Mimic confirmed proposer ownership: clone before mutation, consume derived seed.
      derived=sample.clone();derived.masked_fill_(derived<0,0);derived.clone()
      late=Actual(meta,sample,None,[],copy_stream,129280)
      early.async_copy_ready_event.synchronize();late.async_copy_ready_event.synchronize();expected=sample.to('cpu')
      record={'dtype':str(dtype),'rows':n,'cols':cols,'replay':replay,'exact_early_late_reference':bool(torch.equal(early.sampled_token_ids_cpu,late.sampled_token_ids_cpu) and torch.equal(expected,early.sampled_token_ids_cpu)),'metadata_identity':early._model_runner_output is meta}
      r[mode].append(record);save();assert record['exact_early_late_reference'] and record['metadata_identity'],record
 r['ok']=True;r['source_sha256']=hashlib.sha256(raw).hexdigest();r['actual_graph_replays']=len(r['graph'])
except BaseException:r['error']=traceback.format_exc();raise
finally:r['ended']=time.time();save()
