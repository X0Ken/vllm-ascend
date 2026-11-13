# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch one A3 TP8/EP8 PD role with the qualified single-node settings."""

import argparse
import json
import os
import subprocess

DEFAULT_MODE = "ced"


def main():
    """Keep the tested P/D settings together; inherit credentials and network identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["ordinary", "ced"], default=DEFAULT_MODE)
    parser.add_argument("--decode-spec", type=int, choices=[0, 5], default=5)
    parser.add_argument("--served-model-name", default="dsv41")
    parser.add_argument("--port", type=int)
    parser.add_argument("--kv-port", type=int)
    parser.add_argument("--devices", help="Default: P uses 0-7, D uses 8-15 on one 16-chip A3 node")
    parser.add_argument("--dry-run", action="store_true", help="Print argv and explicit environment defaults")
    args = parser.parse_args()
    producer = args.role == "prefill"
    spec = 0 if producer else args.decode_spec
    additional = {
        "enable_dsa_cp": False,
        "enable_cpu_binding": True,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": True,
            "enable_super_kernel": False,
        },
        "multistream_overlap_shared_expert": False,
        "multistream_dsv4_dsa_overlap": False,
        "enable_flashcomm1": True,
        "enable_dsv41_draft_graph": bool(spec),
        "enable_dsv41_async_metadata": False,
        "enable_dsv41_prefill_draft_graph": bool(spec),
        "enable_dsv41_inline_metadata": False,
        "enable_dsv41_deferred_corrections": False,
        "enable_dsv41_decode_first": True,
        "enable_dsv41_admission_gate": False,
        "enable_dsv41_engram_wkv_tp": True,
        "dspark_draft_graph_capture_sizes": [6, 12, 18, 24, 30, 36, 42, 48, 60, 96, 120, 160, 192],
        "dsv41_ced_role": "prefill" if producer and args.mode == "ced" else ("off" if producer else "decode"),
        "dsv41_ced_trace": False,
    }
    transfer = {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "kv_producer" if producer else "kv_consumer",
        "kv_port": args.kv_port or (27000 if producer else 28000),
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "named_cache_transfer": True,
            "prefill": {"tp_size": 8, "dp_size": 1},
            "decode": {"tp_size": 8, "dp_size": 1},
        },
    }
    capture = [6, 12, 18, 24, 30, 36, 42, 48, 72, 96, 144, 192] if spec else [1, 2, 4, 8, 16, 24, 32]
    command = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port or (18990 if producer else 18991)),
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "1",
        "--data-parallel-size-local",
        "1",
        "--enable-expert-parallel",
        "--distributed-executor-backend",
        "mp",
        "--safetensors-load-strategy",
        "lazy",
        "--quantization",
        "ascend",
        "--dtype",
        "bfloat16",
        "--block-size",
        "128",
        "--max-model-len",
        "150000",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        "--tokenizer-mode",
        "deepseek_v41",
        "--reasoning-parser",
        "deepseek_v41",
        "--tool-call-parser",
        "deepseek_v41",
        "--enable-auto-tool-choice",
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--async-scheduling",
        "--no-disable-hybrid-kv-cache-manager",
        "--swa-bounded-replay",
        "--additional-config",
        json.dumps(additional),
        "--kv-transfer-config",
        json.dumps(transfer),
        "--engram-config",
        json.dumps({"cpu_offload": True, "dp_shared_memory": False}),
        "--compilation-config",
        json.dumps({"cudagraph_mode": "NONE" if producer else "FULL_DECODE_ONLY", "cudagraph_capture_sizes": capture}),
    ]
    if producer:
        command += ["--enforce-eager"]
    if spec:
        command += [
            "--speculative-config",
            json.dumps({"method": "dspark", "num_speculative_tokens": spec, "enforce_eager": False}),
        ]
    defaults = {
        "ASCEND_RT_VISIBLE_DEVICES": args.devices or ",".join(map(str, range(8) if producer else range(8, 16))),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_BUFFSIZE": "200",
        "HCCL_DETERMINISTIC": "false",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "VLLM_ASCEND_TP_BF16_ALLREDUCE_FP32": "1",
        "VLLM_ASCEND_QLI_CANDIDATE_SKIP": "1",
        "TASK_QUEUE_ENABLE": "1",
        "LOCAL_WORLD_SIZE": "8",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "36000",
    }
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "argv": command, "environment_defaults": defaults}, indent=2))
        return
    if not os.environ.get("VLLM_HOST_IP"):
        parser.error("Set VLLM_HOST_IP to this node's transfer address before launching")
    if not os.environ.get("VLLM_API_KEY"):
        parser.error("Set VLLM_API_KEY from the shared proxy secret file before launching")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") != "1":
        parser.error("This PD entrypoint requires the v2 model runner")
    env = dict(os.environ)
    for key, value in defaults.items():
        env.setdefault(key, value)
    env["ASCEND_RT_VISIBLE_DEVICES"] = defaults["ASCEND_RT_VISIBLE_DEVICES"]
    env.setdefault("HCCL_IF_IP", env["VLLM_HOST_IP"])
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
