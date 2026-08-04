# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError


class HealthyEngineCore:

    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.calls = 0

    async def get_supported_tasks_async(self):
        self.calls += 1
        return ("generate", )

    def shutdown(self, timeout=None):
        pass


def make_llm(engine_core):
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine_core = engine_core
    llm.output_handler = None
    return llm


@pytest.mark.asyncio
async def test_check_health_round_trips_to_engine_core():
    engine_core = HealthyEngineCore()

    await AsyncLLM.check_health(make_llm(engine_core))

    assert engine_core.calls == 1


@pytest.mark.asyncio
async def test_check_health_raises_when_engine_core_probe_fails():

    class FailingEngineCore(HealthyEngineCore):

        async def get_supported_tasks_async(self):
            raise RuntimeError("utility channel is unavailable")

    with pytest.raises(EngineDeadError):
        await AsyncLLM.check_health(make_llm(FailingEngineCore()))


@pytest.mark.asyncio
async def test_check_health_raises_when_engine_core_probe_times_out(
        monkeypatch):
    monkeypatch.setenv("VLLM_ENGINE_HEALTH_TIMEOUT_S", "0.1")

    class WedgedEngineCore(HealthyEngineCore):

        async def get_supported_tasks_async(self):
            await asyncio.sleep(1)

    with pytest.raises(EngineDeadError):
        await AsyncLLM.check_health(make_llm(WedgedEngineCore()))


@pytest.mark.asyncio
async def test_check_health_falls_back_for_invalid_timeout(monkeypatch):
    monkeypatch.setenv("VLLM_ENGINE_HEALTH_TIMEOUT_S", "not-a-number")
    engine_core = HealthyEngineCore()

    await AsyncLLM.check_health(make_llm(engine_core))

    assert engine_core.calls == 1
