# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig, DynamicSpecConfig


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v41_text")),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_sample_method="greedy",
            num_speculative_tokens=5,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(sample_from_anchor=True)),
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=512),
        lora_config=None,
    )


def test_disabled_graph_does_not_restrict_other_configs():
    AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))._validate_dsv41_draft_graph(SimpleNamespace())


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text", "deepseek_v4.1_text"])
def test_supported_draft_graph_config(model_type):
    vc = config()
    vc.model_config.hf_text_config.model_type = model_type
    AscendConfig(
        enable_dsv41_draft_graph=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
    )._validate_dsv41_draft_graph(vc)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model_config.hf_text_config.model_type", "deepseek_v4", "V4.1"),
        ("speculative_config", None, "DSpark"),
        ("speculative_config.method", "mtp", "DSpark"),
        ("speculative_config.draft_sample_method", "probabilistic", "greedy"),
        ("speculative_config.draft_model_config.hf_config.sample_from_anchor", False, "sample_from_anchor"),
        ("parallel_config.prefill_context_parallel_size", 2, "context parallelism"),
        ("parallel_config.decode_context_parallel_size", 2, "context parallelism"),
        ("lora_config", object(), "LoRA"),
        ("scheduler_config.max_num_batched_tokens", 128, "batched tokens"),
    ],
)
def test_reject_unsupported_draft_graph_config(path, value, message):
    vc = config()
    parts = path.split(".")
    parent = vc
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)
    with pytest.raises(ValueError, match=message):
        AscendConfig(
            enable_dsv41_draft_graph=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
        )._validate_dsv41_draft_graph(vc)


@pytest.mark.parametrize("method", ["dspark", "dflash"])
@pytest.mark.parametrize("enabled", [False, True])
def test_dynamic_budget_requires_eager_draft(method, enabled):
    ascend = AscendConfig(
        enable_dsv41_draft_graph=enabled,
        dynamic_spec_config=DynamicSpecConfig(method=method),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    if enabled:
        with pytest.raises(ValueError, match="dynamic speculative budgets"):
            ascend._validate_dsv41_draft_graph(config())
    else:
        ascend._validate_dsv41_draft_graph(config())


def test_prefill_graph_requires_decode_graph_opt_in():
    with pytest.raises(ValueError, match="requires enable_dsv41_draft_graph"):
        AscendConfig(
            enable_dsv41_prefill_draft_graph=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
        )._validate_dsv41_draft_graph(config())


def async_metadata_config():
    vc = config()
    vc.use_v2_model_runner = False
    vc.scheduler_config.async_scheduling = True
    vc.parallel_config.data_parallel_size = 1
    vc.speculative_config.disable_padded_drafter_batch = False
    return vc


def test_async_metadata_bounds_supported_config():
    AscendConfig(
        enable_dsv41_async_metadata=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
    )._validate_dsv41_async_metadata(async_metadata_config())


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("model_config.hf_text_config.model_type", "qwen3", "DeepSeek V4.1"),
        ("parallel_config.data_parallel_size", 2, "DP1"),
        ("parallel_config.prefill_context_parallel_size", 2, "context parallelism"),
        ("parallel_config.decode_context_parallel_size", 2, "context parallelism"),
        ("use_v2_model_runner", True, "Model Runner V1"),
        ("lora_config", object(), "LoRA"),
        ("scheduler_config.async_scheduling", False, "async scheduling"),
        ("speculative_config", None, "padded DSpark"),
        ("speculative_config.method", "mtp", "padded DSpark"),
        ("speculative_config.disable_padded_drafter_batch", True, "padded DSpark"),
    ],
)
def test_async_metadata_bounds_reject_exact_host_consumers(path, value, message):
    vc = async_metadata_config()
    parts = path.split(".")
    parent = vc
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)
    with pytest.raises(ValueError, match=message):
        AscendConfig(
            enable_dsv41_async_metadata=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
        )._validate_dsv41_async_metadata(vc)


@pytest.mark.parametrize("window,sparse,cp", [(128, False, False), (None, True, False), (None, False, True)])
def test_async_metadata_bounds_reject_optional_host_consumers(window, sparse, cp):
    with pytest.raises(ValueError):
        AscendConfig(
            enable_dsv41_async_metadata=True,
            enable_dsa_cp=cp,
            draft_window_size=window,
            sparse_kv_offload_config=SimpleNamespace(enabled=sparse),
        )._validate_dsv41_async_metadata(async_metadata_config())


def test_inline_metadata_requires_async_bounds_validation():
    with pytest.raises(ValueError, match="requires enable_dsv41_async_metadata"):
        AscendConfig(
            enable_dsv41_inline_metadata=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
        )._validate_dsv41_async_metadata(SimpleNamespace())
