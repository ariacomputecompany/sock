# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.kernels.linear.mixed_precision.conch as conch


class _FakeLauncher:
    def __init__(self):
        self.__globals__ = {"_get_tuning_parameters": lambda: {"old": True}}


def test_conch_dummy_zeros_are_cached_by_device(monkeypatch):
    monkeypatch.setattr(conch, "_CONCH_DUMMY_ZEROS", {})

    first = conch._get_conch_dummy_zeros(torch.device("cpu"))
    second = conch._get_conch_dummy_zeros(torch.device("cpu"))

    assert first is second
    assert first.shape == (0, 0)
    assert first.dtype == torch.int32


def test_conch_gfx1151_installs_strix_halo_tuning(monkeypatch):
    launcher = _FakeLauncher()
    monkeypatch.setattr(conch, "_CONCH_TUNING_INSTALLED", False)
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1151", lambda: True)

    conch._install_conch_tuning_policy(launcher)

    assert launcher.__globals__["_get_tuning_parameters"]() == conch._CONCH_GFX1151_TUNING
    assert conch._CONCH_TUNING_INSTALLED is True


def test_conch_non_rocm_leaves_tuning_unchanged(monkeypatch):
    launcher = _FakeLauncher()
    monkeypatch.setattr(conch, "_CONCH_TUNING_INSTALLED", False)
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: False)

    conch._install_conch_tuning_policy(launcher)

    assert launcher.__globals__["_get_tuning_parameters"]() == {"old": True}
    assert conch._CONCH_TUNING_INSTALLED is True
