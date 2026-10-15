"""Execute in an idle window. Strict NPU semantics precede micro A/B/A."""
import argparse
import importlib.util
import json
import pathlib
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location('markov_candidate', args.module)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    result = {'status': 'running', 'cases': [], 'micro_aba': []}
    path = pathlib.Path(args.output)
    try:
        for rows in [1, 2, 4, 8, 16, 32]:
            # Baseline DSpark slices have row stride K*V; output is a K+1 column.
            raw = torch.empty((rows, 5, 129280), device='npu', dtype=torch.bfloat16)
            bias = torch.empty((rows, 129280), device='npu', dtype=torch.bfloat16)
            out = torch.full((rows, 6), -777, device='npu', dtype=torch.int64)
            x = raw[:, 2]
            destination = out[:, 3]
            def inputs(kind, seed):
                g = torch.Generator().manual_seed(seed + rows)
                xx = torch.randn(rows, 129280, generator=g).bfloat16()
                yy = torch.randn(rows, 129280, generator=g).bfloat16()
                if kind == 'tie':
                    xx.fill_(-2); yy.fill_(0)
                    for i in [0, 1023, 1024, 64512, 129279]: xx[:, i] = 1
                elif kind == 'rounding':
                    xx.fill_(-2); yy.fill_(0)
                    # BF16 produces a tie at 1 -> ID1023. An incorrect FP32
                    # comparison would choose ID1024, so this detects removal
                    # of the original intermediate BF16 storage boundary.
                    xx[:, 1023] = 1; yy[:, 1023] = .001953125
                    xx[:, 1024] = 1; yy[:, 1024] = .00390625
                elif kind == 'nan':
                    xx[:, 1023] = float('nan'); yy[:, 1024] = float('nan')
                    xx[-1, 0] = float('nan')
                elif kind == 'infs':
                    xx[:, 1023] = float('inf'); xx[:, 129279] = float('inf')
                    xx[:, 1024] = -float('inf')
                elif kind == 'all-neginf': xx.fill_(-float('inf')); yy.fill_(0)
                elif kind == 'inf-cancellation':
                    xx[:, 1023] = float('inf'); yy[:, 1023] = -float('inf')
                elif kind == 'signed-zero': xx.fill_(-0.); yy.fill_(0.)
                raw.fill_(-999); x.copy_(xx); bias.copy_(yy); out.fill_(-777)
            for kind in ['random', 'tie', 'rounding', 'nan', 'infs', 'all-neginf',
                         'inf-cancellation', 'signed-zero']:
                inputs(kind, 1807)
                before = x.cpu()
                reference = (x + bias).argmax(dim=-1)
                if kind == 'rounding': assert torch.all(reference == 1023).item()
                op.add_argmax_out(x, bias, destination)
                torch.npu.synchronize()
                assert torch.equal(reference, destination), (rows, kind, 'eager', reference.cpu().tolist(), destination.cpu().tolist())
                torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0, equal_nan=True)
                assert torch.all(out[:, [0, 1, 2, 4, 5]] == -777).item()
                assert torch.all(raw[:, [0, 1, 3, 4]] == -999).item()
                result['cases'].append({'rows': rows, 'pattern': kind, 'eager_exact': True})
            inputs('random', 2000)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph): op.add_argmax_out(x, bias, destination)
            for n, kind in enumerate(['random', 'tie', 'nan', 'rounding']):
                inputs(kind, 2112 + n)
                graph.replay()
                torch.npu.synchronize()
                assert torch.equal(destination, (x + bias).argmax(dim=-1)), (rows, kind, 'graph')
            result['cases'].append({'rows': rows, 'graph_exact': True, 'replays': 4})
            # Whole replaced boundary: BF16 correction, argmax, and strided copy.
            # Clone cost intentionally omitted: baseline in-place Add and fused
            # candidate each require one BF16 read; stable tests use out-of-place
            # Add to avoid changing logits across repetitions.
            inputs('random', 3881)
            baseline_graph = torch.npu.NPUGraph()
            with torch.npu.graph(baseline_graph): destination.copy_((x + bias).argmax(dim=-1))
            timings = []
            for label, selected in [('A1', baseline_graph), ('B', graph), ('A2', baseline_graph)]:
                for _ in range(20): selected.replay()
                torch.npu.synchronize()
                begin = time.perf_counter()
                for _ in range(200): selected.replay()
                torch.npu.synchronize()
                timings.append({'phase': label, 'wall_us_per_replay': (time.perf_counter() - begin) * 1e6 / 200})
            result['micro_aba'].append({'rows': rows, 'timings': timings, 'scope': 'correction+argmax+strided-copy graph; excludes Markov GEMM; host replay included'})
            del graph, baseline_graph, raw, bias, out, x, destination, reference
        result['status'] = 'passed'
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        path.write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'cases': len(result['cases'])}))


if __name__ == '__main__': main()
