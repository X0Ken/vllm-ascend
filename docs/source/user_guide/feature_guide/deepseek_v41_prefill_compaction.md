# Experimental non-PD terminal FFN compaction

The additional configuration `dsv41_prefill_compaction="terminal"` selects only
requested output rows for the routed experts in the final target FFN. The default is `off`. This
experiment targets A3 BF16 KV with TP8/DP2/EP16, auxiliary-only DSpark and ordinary
prefix caching, with FlashComm sequence parallelism enabled and SWA bounded replay disabled. It retains every target layer
and all attention, KV, index and DSpark auxiliary-feature rows.

All DP participants agree on admission and output-row capacity using CPU-side
scheduling metadata. Decode-only and idle peers participate when another peer
prefills. Each TP group keeps equal shard lengths; an AllToAllV decode/idle DP peer may
keep its dense, shorter local FFN while another DP compacts its prefill.
padding rows never overwrite real outputs. FlashComm sequence-parallel ranks
retain their existing token ownership; only the owning rank restores each
requested row, while other ranks contribute zero for that position. Prompt scoring and unsupported
parallel/cache layouts keep the full path. FULL decode graphs keep their
ordinary FFN execution.

An opaque full-shape terminal operator selects the FFN rows at runtime. The
compiled backbone is preserved. The full Router GEMM and shared-expert calculation/TP collectives are retained
at their original row shapes. Only routed-expert dispatch and GMM rows shrink,
reducing additional numerical changes from Router selection or shared reductions.
The compact call retains the original
AllToAll/AllGather backend, expert ownership and quantization path rather than
switching communication policy along with the row-count optimization.

Broader token trimming must include all KV/index/Engram sources and DSpark
feature dependencies. Each independent local-attention layer expands the
required history by its window minus one. Reusable prefix caches additionally
need all state claimed by their publication contract. The current 40-layer
1017 model has late DSpark features at 37/38/39; the conservative boundary leaves
only its terminal consumer layer. Attention suffix trimming is therefore not
activated for prefix-cached workloads by this option.

Validate deterministic output IDs and logprobs against an untrimmed reference,
including cold and hot 100K inputs, both DPs, long generations, mixed batches,
and prefix reuse before recording performance. Retain the original configuration
unless the measured tradeoff supports this experimental path.

Diagnostic verification may run the full FFN twice, then compare compact FFN
rows and the actual HC-collapse/norm/LM-head greedy decisions. Only logit rows
consumed by sampling or speculative verification enter the decision check.
Different batch and collective sizes can change BF16 rounding; record those
differences and the full-FFN replay control instead of claiming bitwise equality.
The finite probe set does not establish general model quality. Disable all
verification before warmup and performance measurement.

For the validated single-node TP8/DP2/EP16 setup, use MRV2 and set the HCCL
option before starting the service:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_DETERMINISTIC=true
```

Merge `{"dsv41_prefill_compaction": "terminal"}` into the existing additional
configuration. Preserve the remaining baseline settings. The tested workload is
C32, 102400 input tokens, 1024 forced output tokens and 97280 actual HBM-prefix
hit tokens, with K5 and FlashComm SP. Compare against `off` with the same HCCL
setting; changing HCCL determinism can affect performance independently. Retain
the original deployment as an additional reference.

Without deterministic collectives, even full-FFN self-replay can differ around
a masked-EOS tie. Verification must check both ordinary and ignore-EOS greedy
choices, and record the full-FFN control. Successful finite-input verification
does not justify enabling this experiment by default or claim broad quality
equivalence. This option does not remove model weights or KV allocations.
