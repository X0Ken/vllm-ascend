"""Run on an idle 910B3 only; no service patching or model requests.

python native_kv.py --module MODULE --out JSON [--mode full|rope|both]
Strict cache equality is the admission gate, not merely a loose tolerance.
Failure diagnostics retain exact mismatch counts and max absolute errors.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("full", "rope", "both"), default="both")
    args = p.parse_args()
    torch.npu.set_device(0)
    assert enable_custom_op()
    spec = importlib.util.spec_from_file_location("candidate_kv", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    result = {"status": "running", "cases": rows}
    try:
        for mode in (("full", "rope") if args.mode == "both" else (args.mode,)):
            for n in (1, 6, 20, 24, 48, 96, 192):
                for trig_dtype in (torch.float32, torch.bfloat16):
                    for use_bias in (False, True):
                        torch.manual_seed(731 + n)
                        x = torch.randn(n, 512, device="npu", dtype=torch.bfloat16)
                        gamma = (1 + torch.randn(512, device="npu") * .1).bfloat16()
                        bias = (torch.randn(512, device="npu") * .01).bfloat16() if use_bias else None
                        norm = SimpleNamespace(weight=gamma, bias=bias, bias_loaded=use_bias, variance_epsilon=1e-6)
                        phase = torch.randn(n, 1, 1, 32, device="npu").repeat_interleave(2, -1)
                        cos, sin = phase.cos().to(trig_dtype), phase.sin().to(trig_dtype)
                        # Model-compatible holes between adjacent physical pages.
                        ref_base = torch.full((8, 3, 128, 512), 17, device="npu", dtype=torch.bfloat16)
                        out_base = ref_base.clone()
                        ref, out = ref_base[:, 1], out_base[:, 1]
                        seq = torch.arange(n, device="npu", dtype=torch.int32) + 127
                        slots = torch.stack((seq // 128, seq % 128), -1)
                        if n > 1:
                            slots[-1].fill_(-1)

                        def normalize():
                            value, _ = torch_npu.npu_rms_norm(x, gamma, 1e-6)
                            if use_bias:
                                value.add_(bias)
                            return value

                        def baseline():
                            value = normalize()
                            torch.ops._C_ascend.inplace_partial_rotary_mul(
                                value.view(n, 1, 1, 512), cos, sin,
                                rotary_mode="interleave", partial_slice=[448, 512],
                            )
                            torch.ops._C_ascend.npu_scatter_nd_update_sk(ref, slots, value)

                        def candidate():
                            module.kv_norm_rope_cache(
                                x if mode == "full" else normalize(), norm, cos, sin, slots, out,
                                normalize=mode == "full",
                            )

                        baseline()
                        candidate()
                        torch.npu.synchronize()
                        diff = (ref_base.float() - out_base.float()).abs()
                        row = {"mode": mode, "tokens": n, "trig": str(trig_dtype), "bias": use_bias,
                               "exact": torch.equal(ref_base, out_base),
                               "mismatches": int((diff != 0).sum()), "max_abs": float(diff.max())}
                        rows.append(row)
                        # Graph-captured cache writes must follow changing input and device slots.
                        for _ in range(2):
                            baseline()
                            candidate()
                        ga, gb = torch.npu.NPUGraph(), torch.npu.NPUGraph()
                        with torch.npu.graph(ga):
                            baseline()
                        with torch.npu.graph(gb):
                            candidate()
                        graph_exact = True
                        for step in range(3):
                            x.copy_(torch.randn_like(x))
                            cos.copy_((phase + step).cos().to(trig_dtype))
                            sin.copy_((phase + step).sin().to(trig_dtype))
                            slots[:, 0].copy_(((seq + step * 128) // 128))
                            if n > 1:
                                slots[-1].fill_(-1)
                            ga.replay()
                            gb.replay()
                            torch.npu.synchronize()
                            graph_exact &= torch.equal(ref_base, out_base)
                        row["graph_replays_exact"] = graph_exact
                        row["unrelated_planes_unchanged"] = bool(torch.all(out_base[:, 0] == 17) and torch.all(out_base[:, 2] == 17))
        # Runtime launch metadata must not create kernels for each row count/stride.
        # Original strict 28 eager/graph cases above remain the admission gate.
        if args.mode == "rope" and all(r["exact"] and r["graph_replays_exact"] and r["unrelated_planes_unchanged"] for r in rows):
            cache_count = lambda: sum(len(v) for v in module._kv_store.cache.values())
            before = cache_count()
            checks = []
            lengths = [1,3,6,7,13,20,23,24,31,47,48,63,95,96,127,159,160,191,192]
            layouts = [(512,64,2,196608),(1024,128,4,65536),(512,1024,2,73856),(1024,65536,4,196608)]
            for dtype in [torch.float32, torch.bfloat16]:
                for n in lengths:
                    for xrow,crow,srow,pstride in layouts:
                        x = torch.empty_strided((n,512),(xrow,1),device="npu",dtype=torch.bfloat16)
                        x.copy_(torch.randn_like(x))
                        cs = torch.empty_strided((n,1,1,64),(crow,64,64,1),device="npu",dtype=dtype)
                        # Allocate sin independently with the same explicit row stride.
                        sn = torch.empty_strided((n,1,1,64),(crow,64,64,1),device="npu",dtype=dtype)
                        phi = torch.randn(n,1,1,32,device="npu").repeat_interleave(2,-1)
                        cs.copy_(phi.cos().to(dtype));sn.copy_(phi.sin().to(dtype))
                        sl = torch.empty_strided((n,2),(srow,1),device="npu",dtype=torch.int32)
                        seq = torch.arange(n,device="npu",dtype=torch.int32)+127
                        sl.copy_(torch.stack((seq//128,seq%128),-1))
                        if n>1:sl[-1].fill_(-1)
                        size=7*pstride+128*512
                        ab=torch.full((size,),17,device="npu",dtype=torch.bfloat16);bb=ab.clone()
                        ac=torch.as_strided(ab,(8,128,512),(pstride,512,1));bc=torch.as_strided(bb,(8,128,512),(pstride,512,1))
                        norm=SimpleNamespace(weight=torch.ones(512,device="npu",dtype=torch.bfloat16),bias=None,bias_loaded=False,variance_epsilon=1e-6)
                        value=x.contiguous().clone()
                        # Baseline CANN consumes identical logical values after explicit
                        # materialization; candidate directly consumes each strided view.
                        torch.ops._C_ascend.inplace_partial_rotary_mul(value.view(n,1,1,512),cs.contiguous(),sn.contiguous(),rotary_mode="interleave",partial_slice=[448,512])
                        torch.ops._C_ascend.npu_scatter_nd_update_sk(ac,sl.contiguous(),value)
                        module.kv_norm_rope_cache(x,norm,cs,sn,sl,bc,normalize=False)
                        torch.npu.synchronize()
                        assert torch.equal(ab,bb),("runtime-metadata-output",n,xrow,crow,srow,pstride,str(dtype))
                        checks.append({"tokens":n,"x_row":xrow,"cos_row":crow,"slot_row":srow,"page_stride":pstride,"trig":str(dtype),"whole_storage_exact":True})
            after=cache_count()
            assert before==after,("runtime metadata compiled new kernel",before,after)
            result["runtime_metadata_cache"]={"before":before,"after":after,"cases":checks,"checks":len(checks),"new_kernels":0,"baseline_layout_materialization_explicit":True}
        result["status"] = "passed" if all(r["exact"] and r["graph_replays_exact"] and r["unrelated_planes_unchanged"] for r in rows) else "failed-numerical"
    except Exception as exc:
        result.update(status="failed-runtime", error=repr(exc))
        raise
    finally:
        Path(args.out).write_text(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
