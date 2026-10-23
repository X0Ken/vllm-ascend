# QC DeepSeek V4.1 image

`Dockerfile.qc-dsv41` builds the QC branch against the pinned vLLM 0.31.0
commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`. `install.py` applies the
source patches during the image build; they are not runtime source overlays.

## Decode before prefill

Opt in with the additional config:

```json
{"enable_dsv41_decode_first": true, "enable_dsv41_admission_gate": false}
```

Keep `--max-num-batched-tokens 8192`. Each scheduling round visits both running
and waiting/resumed decode requests before visiting running and new prefills.
Decode target rows and DSpark's per-request drafting workspace consume the
normal input budget first. All prefills then share the remaining capacity,
without a separate prefill token cap. When no decode is runnable, prefills can
use the full capacity, subject to the usual drafting, KV, encoder, context,
chunk-alignment, pause and configured long-prefill constraints.

The FCFS or priority order is preserved within each phase. Under KV pressure,
decode allocation first considers prefill victims; prefill allocation cannot
preempt decode requests. Requests protected by an unresolved async prefill
barrier cannot be scheduled or selected as preemption victims. New prefill
submissions wait for that barrier to drain, while unrelated decode requests
can continue. In particular, advancing async `num_computed_tokens` past the
prompt end does not make that request's first decode safe before its prefill
result has been consumed. Structured speculative draft validation is retained.

This option defaults to false and requires the standard V1 Scheduler or its
AsyncScheduler subclass. It is incompatible with the older admission gate,
which deliberately separates P-only and D-only rounds. DP balance scheduling
must also remain disabled. Other custom scheduler
implementations must implement equivalent phase and dependency handling before
opting in. This patch is scoped to the pinned core source; contribute phase and
request-dependency hooks upstream before dropping the downstream patch.

CPU integration coverage is in `tests/ut/core/test_dsv41_decode_first.py` and
`tests/ut/patch/worker/test_v41_prefill_policy.py`. Model execution, numerical
qualification and workload timing still require Ascend hardware.
