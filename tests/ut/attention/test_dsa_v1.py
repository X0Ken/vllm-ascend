from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder, build_compressor_metadata_out
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


def _make_decode_builder(compressor_ratio: int, enabled: bool):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": query_start_loc,
        "input_positions": torch.arange(2),
        "cos": torch.ones((2, 1)),
        "sin": torch.zeros((2, 1)),
        "query_start_loc_cpu": query_start_loc,
        "max_seq_lens": 9,
        "seq_lens_list": [8, 9],
        "max_seqlen_kv": 9,
        "max_seqlen_q": 1,
        "start_pos_decode": torch.tensor([7, 8], dtype=torch.int32),
    }
    builder.compressor_ratio = compressor_ratio
    builder.num_decodes = 2
    builder.num_decode_tokens = 2
    builder.seq_lens = torch.tensor([8, 9], dtype=torch.int32)
    builder.start_pos_decode = torch.zeros(2, dtype=torch.int32)
    builder.block_table = torch.zeros((2, 2), dtype=torch.int32)
    builder.slot_mapping = torch.zeros((2, 2), dtype=torch.int32)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_attention_heads=8,
            index_topk=512,
            index_n_heads=64,
            index_head_dim=128,
            sliding_window=4096,
        ),
        get_head_size=lambda: 192,
    )
    builder.seqused_q = torch.empty(0)
    builder.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.empty(0, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.empty(0, dtype=torch.int32)
    builder._device_metadata_enabled = enabled
    builder._device_metadata_tasks = ()
    builder.prefill_compressor_metadata_buffers = None
    builder.decode_compressor_metadata_buffers = None
    if enabled and compressor_ratio > 1:
        builder.prefill_compressor_metadata_buffers = tuple(torch.empty((8, 2), dtype=torch.int32) for _ in range(3))
        builder.decode_compressor_metadata_buffers = tuple(torch.empty((8, 2), dtype=torch.int32) for _ in range(3))
    builder.block_size = 128
    builder.cache_group_key = "group"
    builder.get_block_table_size = MagicMock(return_value=2)
    builder._num_compressor_metadata_rows = MagicMock(return_value=2)
    return builder


@pytest.mark.parametrize("compressor_ratio", [1, 4, 128])
@pytest.mark.parametrize("enabled", [False, True])
def test_decode_metadata_defers_device_work(
    compressor_ratio: int,
    enabled: bool,
):
    builder = _make_decode_builder(compressor_ratio, enabled)
    sas_output = torch.full((1024,), 3, dtype=torch.int32)
    qli_output = torch.full((1024,), 4, dtype=torch.int32)
    sas_op = MagicMock(return_value=sas_output)

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(torch.ones(1), torch.zeros(1)),
        ),
        patch.object(
            DeviceOperator,
            "pad_dsa_decode_slot_mapping",
            return_value=torch.zeros((2, 2), dtype=torch.int32),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=torch.tensor([0, 8, 17], dtype=torch.int32),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_cmp_kv",
            return_value=None,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=sas_op,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_kwargs",
            return_value={},
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            create=True,
            return_value=qli_output,
        ) as qli_op,
    ):
        metadata = builder.build_decode_metadata(0, SimpleNamespace(), 2)
        tasks = builder.take_device_metadata_tasks()
        assert builder.take_device_metadata_tasks() == ()

        if enabled:
            expected = (
                list(DeviceMetadataStage)
                if compressor_ratio == 4
                else [DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.ATTENTION]
                if compressor_ratio > 1
                else [DeviceMetadataStage.ATTENTION]
            )
            assert [task.stage for task in tasks] == expected
            expected_groups = []
            if compressor_ratio > 1:
                assert builder.decode_compressor_metadata_buffers is not None
                expected_groups.append(id(builder.decode_compressor_metadata_buffers[0]))
            if compressor_ratio == 4:
                expected_groups.append(id(builder.decode_qli_metadata))
            expected_groups.append(id(builder.decode_sas_metadata))
            assert [task.group_id for task in tasks] == expected_groups
            sas_op.assert_not_called()
            qli_op.assert_not_called()
            with patch("vllm_ascend.attention.dsa_v1.build_compressor_metadata_out"):
                for task in tasks:
                    task.run()
        else:
            assert tasks == ()

        sas_op.assert_called_once()
        assert sas_op.call_args.kwargs["cmp_ratio"] == compressor_ratio
        if compressor_ratio == 4 or not enabled:
            qli_op.assert_called_once()
            assert qli_op.call_args.kwargs["max_seqlen_q"] == 1
            assert qli_op.call_args.kwargs["max_seqlen_k"] == 9
        else:
            qli_op.assert_not_called()
        assert metadata.sas_metadata is builder.decode_sas_metadata
        assert (metadata.qli_metadata is builder.decode_qli_metadata) is (compressor_ratio == 4 or not enabled)
        assert torch.equal(builder.decode_sas_metadata, sas_output)
        if compressor_ratio == 4:
            assert torch.equal(builder.decode_qli_metadata, qli_output)


def _make_prefill_builder(compressor_ratio: int, enabled: bool):
    builder = _make_decode_builder(compressor_ratio, enabled)
    builder.prefill_ratio_to_sas_metadata = {
        "input_positions": torch.arange(3),
        "max_query_len": 2,
        "max_seq_lens": 3,
        "prefill_input_positions": torch.tensor([1, 2]),
        "prefill_query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "cos": torch.ones((2, 1)),
        "sin": torch.zeros((2, 1)),
        "prefill_seq_lens": torch.tensor([3], dtype=torch.int32),
        "num_prefill": 1,
    }
    builder.decode_ratio_to_sas_metadata = {}
    builder.num_decodes = 1
    builder.num_decode_tokens = 1
    builder.num_prefill_tokens = 2
    builder.num_actual_tokens = 3
    builder.query_lens = torch.tensor([1, 2], dtype=torch.int32)
    builder.seq_lens = torch.tensor([1, 3], dtype=torch.int32)
    builder.start_pos_prefill = torch.zeros(2, dtype=torch.int32)
    builder.block_table = torch.zeros((2, 2), dtype=torch.int32)
    builder.slot_mapping = torch.zeros((3, 2), dtype=torch.int32)
    builder.prefill_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.prefill_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    return builder


@pytest.mark.parametrize("compressor_ratio", [1, 4, 128])
@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_metadata_defers_device_work(
    compressor_ratio: int,
    enabled: bool,
):
    builder = _make_prefill_builder(compressor_ratio, enabled)
    sas_output = torch.full((1024,), 5, dtype=torch.int32)
    qli_output = torch.full((1024,), 6, dtype=torch.int32)
    sas_op = MagicMock(return_value=sas_output)
    common_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(torch.ones(1), torch.zeros(1)),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=sas_op,
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_kwargs",
            return_value={},
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            create=True,
            return_value=qli_output,
        ) as qli_op,
    ):
        metadata = builder.build_prefill_metadata(0, common_metadata, 2)
        tasks = builder.take_device_metadata_tasks()

        if enabled:
            expected_stages = (
                list(DeviceMetadataStage)
                if compressor_ratio == 4
                else [DeviceMetadataStage.COMPRESSOR, DeviceMetadataStage.ATTENTION]
                if compressor_ratio > 1
                else [DeviceMetadataStage.ATTENTION]
            )
            expected_groups = []
            if compressor_ratio > 1:
                assert builder.prefill_compressor_metadata_buffers is not None
                expected_groups.append(id(builder.prefill_compressor_metadata_buffers[0]))
            if compressor_ratio == 4:
                expected_groups.append(id(builder.prefill_qli_metadata))
            expected_groups.append(id(builder.prefill_sas_metadata))
            assert [task.stage for task in tasks] == expected_stages
            assert [task.group_id for task in tasks] == expected_groups
            sas_op.assert_not_called()
            qli_op.assert_not_called()
            with patch("vllm_ascend.attention.dsa_v1.build_compressor_metadata_out"):
                for task in tasks:
                    task.run()
        else:
            assert tasks == ()

        sas_op.assert_called_once()
        assert sas_op.call_args.kwargs["cmp_ratio"] == compressor_ratio
        assert ("cmp_mask_mode" in sas_op.call_args.kwargs) == (compressor_ratio > 1)
        assert ("cmp_topk" in sas_op.call_args.kwargs) == (compressor_ratio == 4)
        if compressor_ratio == 4 or not enabled:
            qli_op.assert_called_once()
            assert qli_op.call_args.kwargs["max_seqlen_q"] == 2
            assert qli_op.call_args.kwargs["max_seqlen_k"] == 3
        else:
            qli_op.assert_not_called()
        if enabled:
            assert metadata.sas_metadata is builder.prefill_sas_metadata
            assert (metadata.qli_metadata is builder.prefill_qli_metadata) is (compressor_ratio == 4)
            assert torch.equal(builder.prefill_sas_metadata, sas_output)
            if compressor_ratio == 4:
                assert torch.equal(builder.prefill_qli_metadata, qli_output)
        else:
            assert metadata.sas_metadata is sas_output
            assert metadata.qli_metadata is qli_output


def test_build_compressor_metadata_out_uses_fixed_outputs():
    metadata = SimpleNamespace(
        full_compress_cos=torch.ones((8, 1, 1, 4)),
        full_compress_sin=torch.zeros((8, 1, 1, 4)),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        start_pos=torch.tensor([1], dtype=torch.int32),
        block_table=torch.tensor([[3]], dtype=torch.int32),
        block_size=128,
        num_reqs_actual=1,
    )
    outputs = (
        torch.empty((2, 1, 1, 4)),
        torch.empty((2, 1, 1, 4)),
        torch.empty((2, 2), dtype=torch.int32),
    )

    with (
        patch.object(DeviceOperator, "get_dsa_compressor_slot_mapping_format", return_value=2),
        patch.object(torch.ops._C_ascend, "compressor_metadata_out", create=True) as metadata_out,
    ):
        build_compressor_metadata_out(metadata, 4, outputs)

    assert metadata_out.call_args.args[-3:] == outputs


@pytest.mark.parametrize(
    ("mode", "allocates_buffers"),
    [
        (CUDAGraphMode.NONE, True),
        (CUDAGraphMode.FULL_AND_PIECEWISE, True),
        (CUDAGraphMode.FULL, False),
    ],
)
def test_enable_device_metadata_keeps_pure_full_compressor_legacy(
    mode: CUDAGraphMode,
    allocates_buffers: bool,
):
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder._device_metadata_enabled = False
    builder.compressor_ratio = 4
    builder.device = torch.device("cpu")
    builder.slot_mapping_shape = (8, 2)
    builder.model_config = SimpleNamespace(hf_config=SimpleNamespace(qk_rope_head_dim=4))
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    builder.prefill_compressor_metadata_buffers = None
    builder.decode_compressor_metadata_buffers = None

    builder.enable_device_metadata()

    assert builder._device_metadata_enabled
    assert (builder.prefill_compressor_metadata_buffers is not None) is allocates_buffers
    assert (builder.decode_compressor_metadata_buffers is not None) is allocates_buffers
    if allocates_buffers:
        assert builder.prefill_compressor_metadata_buffers is not None
        assert builder.decode_compressor_metadata_buffers is not None
        for buffers in (
            builder.prefill_compressor_metadata_buffers,
            builder.decode_compressor_metadata_buffers,
        ):
            assert buffers[0].shape == (8, 1, 1, 4)
            assert buffers[1].shape == (8, 1, 1, 4)
            assert buffers[2].shape == (8, 2)
            assert buffers[0].dtype == buffers[1].dtype == torch.float32
            assert buffers[2].dtype == torch.int32


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_full_graph_compressor_uses_stable_padded_extent(phase: str):
    builder = _make_prefill_builder(4, True)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": torch.tensor([0, 1, 2], dtype=torch.int32),
        "input_positions": torch.arange(2),
        "cos": torch.ones((2, 1)),
        "sin": torch.zeros((2, 1)),
        "query_start_loc_cpu": torch.tensor([0, 1, 2], dtype=torch.int32),
        "max_seq_lens": 9,
        "seq_lens_list": [8, 9],
        "max_seqlen_kv": 9,
        "max_seqlen_q": 1,
        "start_pos_decode": torch.tensor([7, 8], dtype=torch.int32),
    }
    common = SimpleNamespace(
        num_input_tokens=8,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )

    with (
        patch(
            "vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa",
            return_value=(torch.ones(1), torch.zeros(1)),
        ),
        patch.object(
            DeviceOperator,
            "get_dsa_sparse_attn_metadata_op",
            return_value=MagicMock(return_value=torch.ones(1024, dtype=torch.int32)),
        ),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        patch.object(
            DeviceOperator,
            "get_dsa_decode_cu_seqlens_ori_kv",
            return_value=torch.tensor([0, 8, 17], dtype=torch.int32),
        ),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", return_value=None),
        patch.object(
            torch.ops._C_ascend,
            "npu_vllm_quant_lightning_indexer_metadata",
            create=True,
            return_value=torch.ones(1024, dtype=torch.int32),
        ),
    ):
        if phase == "prefill":
            capture_metadata = builder.build_prefill_metadata(0, common, 2, full_graph_mode=True)
            metadata = builder.build_prefill_metadata(0, common, 1, full_graph_mode=True)
            buffers = builder.prefill_compressor_metadata_buffers
            expected_rows = 3
            expected_reqs = 1
        else:
            builder.num_decodes = 2
            builder.num_decode_tokens = 2
            capture_metadata = builder.build_decode_metadata(0, common, 2, full_graph_mode=True)
            metadata = builder.build_decode_metadata(0, common, 1, full_graph_mode=True)
            buffers = builder.decode_compressor_metadata_buffers
            expected_rows = 4
            expected_reqs = 2

    assert buffers is not None
    assert metadata.num_compressed_tokens == expected_rows
    assert metadata.num_reqs_actual == expected_reqs
    assert metadata.compressor_metadata is not None
    assert metadata.compressor_metadata[0].shape[0] == expected_rows
    assert metadata.compressor_metadata[0].data_ptr() == buffers[0].data_ptr()
    assert capture_metadata.compressor_metadata is not None
    assert capture_metadata.compressor_metadata[0].shape == metadata.compressor_metadata[0].shape
    assert capture_metadata.compressor_metadata[0].data_ptr() == metadata.compressor_metadata[0].data_ptr()


def test_mixed_prefill_decode_keeps_independent_persistent_outputs():
    builder = _make_prefill_builder(4, True)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "input_positions": torch.arange(1),
        "cos": torch.ones((1, 1)),
        "sin": torch.zeros((1, 1)),
        "query_start_loc_cpu": torch.tensor([0, 1], dtype=torch.int32),
        "max_seq_lens": 1,
        "seq_lens_list": [1],
        "max_seqlen_kv": 1,
        "max_seqlen_q": 1,
        "start_pos_decode": torch.tensor([0], dtype=torch.int32),
    }
    common = SimpleNamespace(query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32))
    sas_op = MagicMock(side_effect=lambda **kwargs: torch.full((1024,), int(kwargs["max_seqlen_kv"])))
    qli_op = MagicMock(side_effect=lambda **kwargs: torch.full((1024,), int(kwargs["max_seqlen_k"])))
    with (
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm_ascend.attention.dsa_v1.get_full_cos_and_sin_dsa", return_value=(torch.ones(1), torch.zeros(1))),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_op", return_value=sas_op),
        patch.object(DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", return_value={}),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_ori_kv", return_value=torch.tensor([0, 1])),
        patch.object(DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", return_value=None),
        patch.object(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer_metadata", qli_op, create=True),
        patch("vllm_ascend.attention.dsa_v1.build_compressor_metadata_out"),
    ):
        prefill = builder.build_prefill_metadata(0, common, 2)
        decode = builder.build_decode_metadata(0, common, 2)
        tasks = builder.take_device_metadata_tasks()
        sas_op.assert_not_called()
        qli_op.assert_not_called()
        for task in sorted(tasks, key=lambda task: task.stage):
            task.run()

    assert len(tasks) == 6
    assert prefill.sas_metadata.data_ptr() != decode.sas_metadata.data_ptr()
    assert prefill.qli_metadata.data_ptr() != decode.qli_metadata.data_ptr()
    assert prefill.compressor_metadata[0].data_ptr() != decode.compressor_metadata[0].data_ptr()
    assert torch.all(prefill.sas_metadata == 3)
    assert torch.all(prefill.qli_metadata == 3)
    assert torch.all(decode.sas_metadata == 1)
    assert torch.all(decode.qli_metadata == 1)
