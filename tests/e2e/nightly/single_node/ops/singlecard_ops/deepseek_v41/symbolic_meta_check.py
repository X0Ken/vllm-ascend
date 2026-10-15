"""Compile actual registered C++ Meta operators with symbolic rows; no NPU tensors."""


def run_symbolic_meta_checks(namespace):
    import torch

    def t(shape, dtype=torch.float32):
        return torch.empty(shape, device="meta", dtype=dtype)

    bf = torch.bfloat16

    def shared(n):
        return [
            t((n, 4, 5120), bf),
            t((24, 20480)),
            t((3,)),
            t((24,)),
            t((n, 4)),
            t((n, 5120), bf),
            t((n, 4)),
            t((n, 4, 4)),
        ]

    ops = getattr(torch.ops, namespace)
    if namespace == "v41_postpre_native":
        specs = [
            (
                "post_pre",
                lambda n: shared(n) + [20, 1e-6, 1e-6],
                [(5120, bf), (4, torch.float32), ((4, 4), torch.float32), (4, torch.float32), ((4, 5120), bf)],
            )
        ]
    elif namespace == "v41_postprenorm_native":
        specs = [
            (
                "post_pre_norm",
                lambda n: shared(n) + [t((5120,), bf), 20, 1e-6, 1e-6, 1e-6],
                [
                    (5120, bf),
                    (4, torch.float32),
                    ((4, 4), torch.float32),
                    (4, torch.float32),
                    ((4, 5120), bf),
                    (5120, torch.float32),
                ],
            )
        ]
    elif namespace == "v41_fusion_native":
        specs = [
            (
                "hc_pre_norm",
                lambda n: [
                    t((n, 4, 5120), bf),
                    t((24, 20480)),
                    t((3,)),
                    t((24,)),
                    t((n, 4)),
                    t((5120,), bf),
                    20,
                    1e-6,
                    1e-6,
                    1e-6,
                ],
                [(5120, bf), (4, torch.float32), ((4, 4), torch.float32), (4, torch.float32), (5120, torch.float32)],
            )
        ]
    elif namespace == "v41_delayed_native":
        specs = [
            (
                "coeff",
                lambda n: [t((n, 4, 5120), bf), t((24, 20480)), t((3,)), t((24,)), 20, 1e-6, 1e-6],
                [(4, torch.float32), ((4, 4), torch.float32), (4, torch.float32)],
            ),
            (
                "collapse_norm",
                lambda n: [t((n, 4, 5120), bf), t((n, 4)), t((5120,), bf), 1e-6],
                [(5120, bf), (5120, torch.float32)],
            ),
            (
                "post_collapse_norm",
                lambda n: [
                    t((n, 4, 5120), bf),
                    t((n, 4)),
                    t((5120,), bf),
                    t((n, 5120), bf),
                    t((n, 4)),
                    t((n, 4, 4)),
                    1e-6,
                ],
                [(5120, bf), (5120, torch.float32), ((4, 5120), bf)],
            ),
        ]
    else:
        raise ValueError(namespace)
    result = []
    for name, factory, outputs in specs:
        graphs = []
        op = getattr(ops, name)

        def call(*args, op=op):
            return op(*args)

        def backend(gm, inputs, graphs=graphs):
            graphs.append(str(gm.graph))
            return gm.forward

        compiled = torch.compile(call, backend=backend, fullgraph=True, dynamic=True)
        for n in [160, 24, 48, 192, 6]:
            args = factory(n)
            for i, arg in enumerate(args):
                # Weights are constant; batch tensors share the residual's row count.
                if isinstance(arg, torch.Tensor) and i == 0:
                    torch._dynamo.mark_dynamic(arg, 0)
            out = compiled(*args)
            assert len(out) == len(outputs)
            for tensor, (tail, dtype) in zip(out, outputs):
                tail = tail if isinstance(tail, tuple) else (tail,)
                assert tuple(tensor.shape) == (n, *tail) and tensor.dtype == dtype and tensor.device.type == "meta", (
                    name,
                    n,
                    tensor.shape,
                    tensor.dtype,
                )
        assert len(graphs) == 1, (namespace, name, "row specialized", len(graphs))
        assert namespace + "." + name in graphs[0], (namespace, name, "opaque operator absent")
        result.append(
            {
                "op": namespace + "." + name,
                "rows": [160, 24, 48, 192, 6],
                "graphs": len(graphs),
                "symbolic_fullgraph": True,
                "execution": "actual C++ Meta dispatch only; no NPU allocation or math proof",
            }
        )
    return {"passed": True, "ops": result, "actual_model_B_compile_still_required": True}
