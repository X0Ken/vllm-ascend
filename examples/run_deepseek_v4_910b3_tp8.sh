#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Validated: 8x910B3, V4 W8A8 Sept-7 weights, TP8/DP1, EP off.
# Enhanced K7: compressor metadata reuse, compressor/Q tail overlap, and
# asynchronous target DSA metadata. The library keeps async metadata opt-in.
# Requires p6-20260921 or the patched p5-20260916 runtime (vLLM 752a3a504) and a configured
# AscendStoreConnector/Mooncake master. Run inside that container after
# installing this branch; provide node-specific HCCL/VLLM host settings.
# Defaults bind localhost:8901. Additional arguments are passed to vllm.
set -euo pipefail

model_path=${DSV4_MODEL_PATH:-/model}
if [[ ! -f "${model_path}/config.json" ]]; then
  echo "Set DSV4_MODEL_PATH to the local V4 W8A8 model directory." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_INTRA_ROCE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Keep any caller-provided preload libraries after jemalloc.
export LD_PRELOAD="libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"

exec vllm serve "${model_path}" \
  --host 127.0.0.1 \
  --port 8901 \
  --served-model-name dsv4 \
  --default-repetition-detection-config '{"min_pattern_size":8,"max_pattern_size":64,"min_count":8}' \
  --max-model-len 409600 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-prefix-caching \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 1.0 \
  --enable-per-request-metrics \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --quantization ascend \
  --block-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": true, "enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "dspark_main_proj_tp": true, "async_dsv4_metadata": true}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}' \
  --kv-transfer-config '{"kv_connector": "AscendStoreConnector", "kv_role": "kv_both", "kv_load_failure_policy": "fail", "kv_connector_extra_config": {"lookup_rpc_port": "19015", "backend": "mooncake", "use_layerwise": false, "load_async": false}}' \
  "$@"
