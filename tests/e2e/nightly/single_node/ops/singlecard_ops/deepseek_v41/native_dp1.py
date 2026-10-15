"""One-NPU slot/scatter/draft-index validation of compact graph boundary cases.
Only --execute initializes NPU. Does not claim full-model/TP8 graph correctness.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import pathlib
import time
import traceback
from types import SimpleNamespace as NS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--source-root", type=pathlib.Path, default=pathlib.Path("/vllm-workspace/vllm-ascend"))
    a = p.parse_args()
    if not a.execute:
        print(
            json.dumps(
                {
                    "prepared_only": True,
                    "shapes": [6, 12, 18, 24, 30, 36, 42, 48],
                    "empty_and_padding": True,
                    "dtype": "bfloat16",
                    "head_dim": 512,
                    "storage_block": 128,
                    "cache_page_stride": "two interleaved planes",
                    "graph_replays": 3,
                    "need_one_npu": True,
                    "full_model_tp8_still_required": True,
                }
            )
        )
        return
    if a.output.exists():
        raise FileExistsError(a.output)
    paths = [
        "vllm_ascend/attention/deepseek_v41_slots.py",
        "vllm_ascend/attention/dsa_v41.py",
        "vllm_ascend/attention/dsa_v1.py",
    ]
    manifest = {name: hashlib.sha256((a.source_root / name).read_bytes()).hexdigest() for name in paths}
    import torch

    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(1)
    torch.npu.set_device(a.device)
    assert enable_custom_op()
    result = {
        "started": time.time(),
        "source_hashes": {n: manifest[n] for n in paths},
        "checks": [],
        "passed": False,
        "scope": (
            "Actual pinned slot_coordinates Triton, scatter_cache_sk custom op, and build_dspark_swa_in"
            "dices; eager+ACLGraph on one NPU. Model graph/collectives/capture memory/output quality no"
            "t covered."
        ),
    }

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))

    def fn(name, path, ns):
        n = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.FunctionDef) and x.name == name)
        exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), "exec"), ns)
        return ns[name]

    spec = importlib.util.spec_from_file_location("dp1_real_slots", a.source_root / paths[0])
    slotmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slotmod)
    ns = {"torch": torch}
    scatter = fn("scatter_cache_sk", a.source_root / paths[1], ns)
    ns = {"torch": torch, "build_dspark_swa_indices_triton": None}
    fn("_aligned_dspark_index_width", a.source_root / paths[2], ns)
    indices_fn = fn("build_dspark_swa_indices", a.source_root / paths[2], ns)

    def equal(x, y):
        return bool(torch.equal(x.detach().cpu(), y.detach().cpu()))

    # DP1 six-row buckets: real, empty and padded; independently executed results required.
    cases = [
        (physical, actual) for physical in [6, 12, 18, 24, 30, 36, 42, 48] for actual in sorted({0, 1, physical // 6})
    ]
    try:
        for physical, actual in cases:
            padded = (physical + 5) // 6
            actual_tokens = actual * 6
            ql_cpu = torch.tensor([min(i * 6, physical) for i in range(padded + 1)], dtype=torch.int32)
            ql = ql_cpu.to("npu")
            raw = torch.empty(physical, dtype=torch.int64, device="npu")
            pos = torch.empty(physical, dtype=torch.int32, device="npu")
            values = torch.empty((physical, 512), dtype=torch.bfloat16, device="npu")
            # Non-contiguous page stride models a plane view inside a layer-outermost allocation.
            backing = torch.empty((64, 2, 128, 1, 512), dtype=torch.bfloat16, device="npu")
            cache = backing[:, 1]
            builder = NS(
                _flat_slot_mapping=torch.empty(physical, dtype=torch.int64, device="npu"),
                _slot_mapping_2d=torch.empty((physical, 2), dtype=torch.int32, device="npu"),
                kv_cache_spec=NS(storage_block_size=128),
            )
            common = NS(slot_mapping=raw, query_start_loc=ql)
            draft_q = 5
            draft_rows = padded * draft_q
            draft_ql = torch.arange(padded + 1, dtype=torch.int32, device="npu") * draft_q
            seq = torch.empty(padded, dtype=torch.int32, device="npu")
            block_table = torch.arange(padded * 8, dtype=torch.int32, device="npu").reshape(padded, 8)
            index_buffer = torch.empty((draft_rows + 3, 1, 256), dtype=torch.int32, device="npu")

            def inputs(
                replay,
                actual=actual,
                actual_tokens=actual_tokens,
                backing=backing,
                index_buffer=index_buffer,
                padded=padded,
                physical=physical,
                pos=pos,
                raw=raw,
                seq=seq,
                values=values,
            ):
                r = torch.full((physical,), -1, dtype=torch.int64)
                positions = torch.zeros(physical, dtype=torch.int32)
                for req in range(actual):
                    positions[req * 6 : req * 6 + 6] = torch.arange(6, dtype=torch.int32) + 125 + replay * 128
                    r[req * 6 : req * 6 + 6] = req * 512 + positions[req * 6 : req * 6 + 6]
                if actual_tokens > 2:
                    r[2] = -1  # invalid slot within a live request must also be skipped.
                v = (torch.arange(physical * 512, dtype=torch.float32).reshape(physical, 512) % 257 + replay).to(
                    torch.bfloat16
                )
                s = torch.zeros(padded, dtype=torch.int32)
                s[:actual] = 130 + replay * 128
                raw.copy_(r)
                pos.copy_(positions)
                values.copy_(v)
                seq.copy_(s)
                backing.fill_(-7)
                index_buffer.fill_(-777)
                return r, positions, v, s

            for ratio, compressed, skip in [(1, False, False), (2, True, False), (2, True, True)]:

                def call(
                    actual=actual,
                    actual_tokens=actual_tokens,
                    block_table=block_table,
                    builder=builder,
                    cache=cache,
                    common=common,
                    compressed=compressed,
                    draft_ql=draft_ql,
                    draft_rows=draft_rows,
                    index_buffer=index_buffer,
                    physical=physical,
                    pos=pos,
                    ratio=ratio,
                    seq=seq,
                    skip=skip,
                    values=values,
                ):
                    coords = slotmod.slot_coordinates(
                        builder, common, pos, physical, actual, actual_tokens, ratio, compressed, skip
                    )
                    scatter(cache, coords, values)
                    idx, lens = indices_fn(
                        block_table,
                        5,
                        128,
                        128,
                        draft_ql,
                        seq,
                        draft_rows,
                        index_width=256,
                        buffer=index_buffer,
                        use_logical_indices=True,
                    )
                    return coords, idx, lens

                def check(
                    outputs,
                    host,
                    actual=actual,
                    actual_tokens=actual_tokens,
                    backing=backing,
                    builder=builder,
                    compressed=compressed,
                    draft_q=draft_q,
                    draft_rows=draft_rows,
                    index_buffer=index_buffer,
                    padded=padded,
                    physical=physical,
                    ratio=ratio,
                    skip=skip,
                ):
                    coords, idx, lens = outputs
                    r, positions, v, s = host
                    want = torch.full((physical, 2), -1, dtype=torch.int32)
                    target = torch.full(backing.shape, -7, dtype=torch.bfloat16)
                    for row in range(actual_tokens):
                        val = int(r[row])
                        valid = val >= 0
                        if compressed:
                            valid = valid and (val + 1) % 2 == 0 and int(positions[row]) % 2 == 1 and not skip
                            val //= 2
                        if valid:
                            want[row] = torch.tensor([val // 128, val % 128])
                            target[val // 128, 1, val % 128, 0] = v[row]
                    assert equal(coords, want), ("slot mismatch", physical, actual, ratio, skip)
                    flat = torch.where(want[:, 0] >= 0, want[:, 0].to(torch.int64) * 128 + want[:, 1], -1)
                    assert equal(builder._flat_slot_mapping, flat), (
                        "flat slots mismatch",
                        physical,
                        actual,
                        ratio,
                        skip,
                    )
                    assert equal(backing, target), (
                        "cache touched wrong slot/plane or wrong value",
                        physical,
                        actual,
                        ratio,
                        skip,
                    )
                    wi = torch.full((draft_rows, 1, 256), -1, dtype=torch.int32)
                    wl = torch.zeros(draft_rows, dtype=torch.int32)
                    for req in range(padded):
                        start = max(0, int(s[req]) - draft_q - 128)
                        visible = int(s[req]) - start
                        if visible:
                            wi[req * draft_q : (req + 1) * draft_q, 0, :visible] = torch.arange(
                                start, int(s[req]), dtype=torch.int32
                            )
                            wl[req * draft_q : (req + 1) * draft_q] = visible
                    assert equal(idx, wi) and equal(lens, wl), ("draft index mismatch", physical, actual)
                    assert equal(index_buffer[draft_rows:], torch.full((3, 1, 256), -777, dtype=torch.int32)), (
                        "draft index overflow"
                    )
                    assert idx.data_ptr() == index_buffer.data_ptr(), "lost persistent index buffer"

                for replay in range(2):
                    h = inputs(replay)
                    out = call()
                    torch.npu.synchronize()
                    check(out, h)
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(2):
                        _warm = call()
                torch.npu.current_stream().wait_stream(stream)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    captured = call()
                for replay in range(3):
                    h = inputs(replay)
                    torch.npu.synchronize()
                    graph.replay()
                    torch.npu.synchronize()
                    check(captured, h)
                result["checks"].append(
                    {
                        "physical_tokens": physical,
                        "actual_requests": actual,
                        "metadata_requests": padded,
                        "draft_rows": draft_rows,
                        "compressed_ratio": ratio,
                        "skip_ring_write": skip,
                        "eager_checks": 2,
                        "graph_replays": 3,
                        "cache_plane_bitexact": True,
                        "dummy_slots_skipped": True,
                        "draft_mask_and_buffer_guard": True,
                    }
                )
                save()
        result.update(passed=True, finished=time.time(), peak_allocated_bytes=torch.npu.max_memory_allocated())
        save()
        print(json.dumps({"passed": True, "cases": len(result["checks"]), "output": str(a.output)}))
    except BaseException:
        result.update(passed=False, finished=time.time(), error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
