# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.platforms import rocm
from vllm.platforms.rocm import RocmPlatform


@pytest.fixture(autouse=True)
def clear_device_name_caches():
    RocmPlatform.get_device_name.cache_clear()
    rocm._query_device_name_from_amdsmi.cache_clear()
    yield
    RocmPlatform.get_device_name.cache_clear()
    rocm._query_device_name_from_amdsmi.cache_clear()


def test_get_device_name_maps_known_amdsmi_device_id(monkeypatch):
    monkeypatch.setattr(rocm, "amdsmi_init", lambda: None)
    monkeypatch.setattr(rocm, "amdsmi_shut_down", lambda: None)
    monkeypatch.setattr(rocm, "amdsmi_get_processor_handles", lambda: [object()])
    monkeypatch.setattr(
        rocm,
        "amdsmi_get_gpu_asic_info",
        lambda _handle: {
            "device_id": "0x1586",
            "market_name": "AMD Radeon Graphics",
        },
    )

    assert RocmPlatform.get_device_name() == "AMD_Radeon_8060S"


def test_get_device_name_falls_back_to_torch_when_amdsmi_unavailable(monkeypatch):
    amdsmi_query = Mock(side_effect=RuntimeError("driver not loaded"))
    monkeypatch.setattr(rocm, "_query_device_name_from_amdsmi", amdsmi_query)
    monkeypatch.setattr(
        rocm.torch.cuda,
        "get_device_name",
        lambda device_id: "AMD Radeon 8060S Graphics",
    )

    assert RocmPlatform.get_device_name() == "AMD_Radeon_8060S"
    amdsmi_query.assert_called_once_with(0)
