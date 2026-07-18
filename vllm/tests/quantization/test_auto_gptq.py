# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the auto_gptq quantization method works correctly.

Run `pytest tests/quantization/test_auto_gptq.py -v -s`.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.scalar_type import scalar_types

PROMPT = "On the surface of Mars, we found"

MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
]


@pytest.mark.skipif(
    not is_quant_method_supported("auto_gptq"),
    reason="auto_gptq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
def test_auto_gptq_quantization_method(vllm_runner, model_id: str, monkeypatch):
    """Test that quantization='auto_gptq' loads and runs correctly."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_gptq",
        max_model_len=2048,
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            for name, submodule in model.named_modules():
                if name == "model.layers.0.self_attn.qkv_proj":
                    assert isinstance(submodule.quant_method, AutoGPTQLinearMethod)
                    break

        llm.apply_model(check_model)

        outputs = llm.generate_greedy([PROMPT], max_tokens=8)
        assert outputs
        assert len(outputs[0][1]) > 0


def test_auto_gptq_config_get_name():
    """Test that AutoGPTQConfig.get_name() returns 'auto_gptq'."""
    assert AutoGPTQConfig.get_name() == "auto_gptq"


def test_auto_gptq_2bit_uses_uint2b2_quant_type():
    config = AutoGPTQConfig(
        weight_bits=2,
        group_size=32,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={"bits": 2, "sym": True},
    )

    assert config.quant_type == scalar_types.uint2b2
    assert config.pack_factor == 16


def test_auto_gptq_linear_uses_gptq_v1_zero_point_offset(monkeypatch):
    captured = {}

    class DummyKernel:

        __name__ = "DummyKernel"

        def __init__(self, config, **_kwargs):
            captured["config"] = config

        def process_weights_after_loading(self, _layer):
            raise NotImplementedError

        def apply_weights(self, _layer, _x, _bias=None):
            raise NotImplementedError

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_gptq.choose_mp_linear_kernel",
        lambda _config: DummyKernel,
    )

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    config = AutoGPTQConfig(
        weight_bits=2,
        group_size=32,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={"bits": 2, "sym": True},
    )
    method = AutoGPTQLinearMethod(config)
    method.create_weights(
        torch.nn.Module(),
        input_size_per_partition=512,
        output_partition_sizes=[1024],
        input_size=512,
        output_size=1024,
        params_dtype=torch.float16,
    )

    assert captured["config"].zero_point_offset == 1
