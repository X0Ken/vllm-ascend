# Experimental DeepSeek V4.1 CED-PD

CED-PD uses explicit P/D roles on the A3 BF16 KV ABI. The core configuration
keeps `additional_config.dsv41_ced_role="off"`. The single-node A3 PD entrypoint
selects CED with D-side DSpark K5 by default after the C32 validation below.
Each engine still receives an explicit role; existing generic engine invocations
retain the off default.

The producer constructs backbone layers 0–20 instead of all 40 layers. It runs
layers 0–19 and uses layer 20's native global-source writer after its mHC and
normalization input path. It does not run layer 20's full attention/FFN or later
layers. The boundary layer's unused parameters are currently retained. The
producer's single output token is an internal transfer marker, not an answer.

The decoder retains the full target model. Its replay runs all target layers
to reconstruct missing SWA; upper-decoder compaction is disabled for this role.
The current upstream scheduler rounds remote hits to a complete KV block, so
count the replay and newly computed prompt tail separately. Do not assume the
decoder processes exactly 128 replay tokens plus one new token.

The split follows the [upstream CED-PD design](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/tree/ab88ff84efa7970875f3f8cc38dfcb06c491b154/deploy/a3-ced-pd).
This port adapts it to the current MRV2 cache layout, block-aligned remote hits,
and native A3 source writer.

## Configuration

Use separate P and D engines, each DP1 without PP, PCP, or DCP. The currently
supported split has 40 target layers, a 128-token SWA window, the last global KV
source at layer 20 with ratio 1, and no Engram layers above the split. P must
disable speculative decoding and sequence-parallel MoE. D can independently
configure DSpark after correctness validation. Both roles require SWA bounded
replay and `MooncakeHybridConnector` with `named_cache_transfer=true`.

P additional configuration:

```json
{"dsv41_ced_role": "prefill", "dsv41_ced_trace": false}
```

D additional configuration:

```json
{"dsv41_ced_role": "decode", "dsv41_ced_trace": false}
```

Merge these fields with the deployment's existing additional configuration.
For initial A3 correctness validation, disable
`multistream_overlap_shared_expert` and `multistream_dsv4_dsa_overlap` on both
engines and use eager execution. Configure the connector with distinct engine
IDs and non-overlapping KV ports, a producer/consumer role, and matching TP
topologies. For P and D each TP8:

```json
{
  "kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_producer",
  "kv_port": 27000,
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "named_cache_transfer": true,
    "prefill": {"tp_size": 8, "dp_size": 1},
    "decode": {"tp_size": 8, "dp_size": 1}
  }
}
```

D uses `kv_consumer` and a different KV port. The frontend first sends P the
original prompt with one output token and
`kv_transfer_params={"do_remote_decode":true,"do_remote_prefill":false}`.
It then passes the actual returned `kv_transfer_params` to D with the original
prompt and desired output length. Keep P's marker endpoint internal. Do not
fabricate successful transfer metadata or count P's marker as a generated answer.

## Transfer and replay contract

Named transfer exchanges semantic cache names, tensor planes, actual page
strides, and view segments. Each peer resolves its own cache-group indices.
The receiver's unhashed suffix has an explicit logical block offset: a shorter
receive is not aligned to the end of the producer's block list. This preserves
both complete-prefix loads and loads after a local prefix hit when P has a
partial last block.

Only compatible required planes may be transferred. Missing, incompatible,
null, or out-of-range required pages fail before use. The receive waits for the
compute stream's queued page initialization on a transfer worker, and a failed
named DMA prevents completion from being accepted as valid KV. SWA is rebuilt
locally. Received compressor state is preserved only for requests that actually
completed a receive; an unrelated local prefix hit still rebuilds its state.

The decoder uses a compact physical SWA view during replay rather than exposing
retired leading pages to the attention kernel. Global compression/index cache
positions retain their original coordinates. Device cache planning still
enforces existing per-slot address limits; 64-bit transfer address arithmetic
does not remove those kernel limits.

This experimental connector pulls registered device KV directly. It does not
establish compatibility with the CPU Store key namespace or its publication
contract. Compare ordinary PD and CED-PD using the same transport, and label
historical CPU Store results separately.

## Acceptance and measurement

Enable `dsv41_ced_trace` only for correctness probes to record CPU-side request
coordinates and replay counts without device-to-host reads. Cover long-context
retrieval, block-boundary lengths, repeated prompts, mixed hit/miss batches,
prefix resets, and D with/without speculation. A short arithmetic answer alone
does not qualify long-context correctness.

After acceptance, disable tracing and compare ordinary P with CED P while
holding the D path and other settings fixed. Measure both P and D: weights,
activation/graph/non-torch memory, KV allocation, queueing, prompt work, transfer,
request latency, TPOT, decode iteration time, steady decode, and whole-wave
throughput. A weight reduction may expand the KV pool instead of lowering total
allocated HBM, and it does not by itself establish a performance improvement.

## Single-node A3 entrypoint

`examples/deepseek_v41/a3_pd.py` keeps the tested TP8/DP1/EP8 settings for each
role together. It supports `--mode ced` and the matched `--mode ordinary`
comparison. CED is the entrypoint default, with D independently selecting `--decode-spec 5` or
`--decode-spec 0`. Both modes use the experimental named transport and the same
full-layer replay D path. Ordinary mode retains all P weights and the existing
native P decoder replay optimization. It is a controlled comparison of P pruning,
not the historical CPU Store PD deployment.

Run P and D in separate processes on an otherwise available 16-chip A3 node.
The entrypoint selects devices 0–7 for P and 8–15 for D; `--devices` overrides
that selection. Configure the CANN environment, `VLLM_HOST_IP`, `HCCL_IF_IP`, and
the appropriate HCCL/Gloo network interfaces for the actual node first. Supply
the same `VLLM_API_KEY` to both roles through the existing secret-file mechanism.
The script inherits authentication and network settings; `--dry-run` prints
only argv and explicit non-secret environment defaults.

```bash
python examples/deepseek_v41/a3_pd.py --role prefill --mode ced --model /models/dsv41
python examples/deepseek_v41/a3_pd.py --role decode --mode ced --model /models/dsv41
```

The defaults are P eager, D FULL_DECODE_ONLY, max batch 8192, max sequences 32,
block size 128, max model length 150000, memory utilization 0.90, bounded SWA
replay, and Engram CPU offload. D defaults to DSpark K5. The experiment keeps
DSA CP and attention/shared-expert multistream overlap disabled, with tracing
off. Other layouts, KV ABIs, concurrency limits, and optimizations require
separate validation.

The companion `examples/deepseek_v41/pd_proxy.py` forwards P's real transfer
metadata to D and preserves both roles' usage and phase timings. Its
`QC_PD_PROXY_CONFIG` points to a JSON configuration such as:

```json
{
  "key_file": "/run/secrets/api-key",
  "prefill_url": "http://127.0.0.1:18990",
  "decode_url": "http://127.0.0.1:18991"
}
```

Use a restricted configuration/secret file and start the proxy in the same
host network as the two engines:

```bash
QC_PD_PROXY_CONFIG=/path/to/pd-proxy.json python -m uvicorn pd_proxy:app \
  --app-dir examples/deepseek_v41 --host 127.0.0.1 --port 18992
```

The measured workload uses `/v1/completions` with exact input and output token
IDs. `prefill_usage` describes the P cache hit; D usage describes the original
prompt plus its own external/local restore accounting. Proxy phase timings and
end-to-end TTFT have different origins and must be reported separately.

## Validated scope

On 2026-11-13, one 16-chip A3 node ran P and D each TP8/DP1/EP8 with the
DeepSeek V4.1 Flash 1017 W4A8 weights, BF16 KV, and Engram CPU offload. The
four-way comparison used ordinary/CED P and D K5/K0. Each configuration passed
31 cold/hot, block-boundary and mixed-hit PD probes against 14 full-model
references. Each of the two workloads then ran two warm C32 waves and five
formal C32 waves, with additional pre/post semantic checks. All 1280 formal
responses had 1024 output tokens, zero preemption, stable runtime identity,
and no observed compilation or compiler-cache writes in the complete HTTP
windows. The public proxy also passed a real streaming transfer check.

For 102400 input tokens with 97280 P-side prefix hits, and for 8192 inputs
with zero hits, CED/K5 whole-wave output throughput was 905.21 and
984.42 token/s: +9.25% and +12.17% against the matched ordinary/K5 path.
P weights fell from 36.76 to 20.47 GiB per chip, while its measured active
tensor peak fell from 49.60 to 39.27 GiB and its KV allocation increased
from 8.67 to 14.65 GiB. D retains the full model. These observations support
the PD entrypoint default for the measured workloads.

Whole-wave throughput, per-request TPOT and fully resident decode throughput
measure different behavior. Report steady decode and iteration time only when
all five waves have a valid common decode window; the 8K/K5 pipeline did not
provide those windows. The recorded semantic probes qualify the tested replay
and transfer cases, rather than a complete model-accuracy suite. Additional
concurrency, context, hardware and cache formats need their own measurements.
