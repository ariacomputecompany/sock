# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.serve.instrumentator.health import health
from vllm.v1.engine.exceptions import EngineDeadError


class Client:

    def __init__(self, error=None):
        self.error = error

    async def check_health(self):
        if self.error is not None:
            raise self.error


def request_with_client(client):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=client)))


@pytest.mark.asyncio
async def test_health_returns_200_when_engine_probe_succeeds():
    response = await health(request_with_client(Client()))

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_200_for_render_only_server():
    response = await health(request_with_client(None))

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_503_for_engine_dead():
    response = await health(request_with_client(Client(EngineDeadError())))

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_health_returns_503_for_probe_exception():
    response = await health(request_with_client(Client(RuntimeError("boom"))))

    assert response.status_code == 503
