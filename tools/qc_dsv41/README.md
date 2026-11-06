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

## A2 performance candidates

`VLLM_ASCEND_QLI_CANDIDATE_SKIP=1` enables QLI consumer mode 4 on qualified
A2 profiles. It compacts scores and positions from the existing candidate
blocks before sorting; candidate-source mode 1 and full-search mode 3 are
unchanged. The default is `0`; valid values are `0` and `1`, and the variable
contains no sensitive data. Other hardware profiles retain mode 2. This is
separate from `enable_dsv41_qli_no_candidate`, which disables candidate
filtering and therefore bypasses this optimization. Rebuild the native
operators with the Python sources before enabling mode 4.

The logical DSpark SWA builder can write directly into persistent indices and
length outputs without changing their graph addresses. The CPU reference keeps
its existing one-dimensional metadata-length contract. Engram lookups accept
strided hash slices and fully write their outputs, including zeroes for invalid
IDs, so they do not require a copy or an initial zero fill.

For the pinned vLLM 0.31 core, `v41_staged_writes.patch` backports the admission
flush policy from upstream vLLM PR 62748 and tracks dirty block counts. Sampler
state still flushes after request and model state on admissions. Block counts
remain dirty after a failed copy. Remove this source patch when the pinned core
includes both guards; a model-runner override of the complete upstream method
is unnecessary.

The hardware benchmark records for 193 are maintained in the notes repository.
Keep model, precision, parallelism, graph, cache and request parameters with
reported results. Per-operator measurements do not imply the same percentage
improvement in serving TPOT or throughput.
