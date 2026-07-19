# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.tmh_policy import TMHKVRuntimePolicy

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@torch.inference_mode()
def warmup_tmh_physical_kernels(
    model_runner: "GPUModelRunner",
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Compile physical TMH kernels through the canonical scheduler path.

    This warmup is intentionally specific to the physical TMH legacy GPU runner
    path. It does not emulate attention metadata and does not use a generic dummy
    forward; it sends normal scheduler outputs with TMH physical descriptor
    events so the worker exercises the same raw/warm descriptor runtime that
    request serving uses.
    """

    if model_runner.is_pooling_model:
        return

    kv_cache_config = model_runner.kv_cache_config
    if kv_cache_config.tmh_kv_policy != "physical":
        raise RuntimeError(
            "warmup_tmh_physical_kernels requires physical TMH KV policy."
        )
    if not kv_cache_config.kv_cache_groups:
        return

    block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    decode_query_len = model_runner.uniform_decode_query_len
    prompt_len = max(block_size, decode_query_len + 1)
    decode_len = prompt_len + decode_query_len
    max_num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        max(1, model_runner.max_num_tokens // max(prompt_len, decode_query_len)),
    )

    group_block_sizes = [
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    ]
    prefill_block_counts = [
        cdiv(prompt_len, block_size) for block_size in group_block_sizes
    ]
    decode_block_counts = [
        cdiv(decode_len, block_size) for block_size in group_block_sizes
    ]
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, prefill_block_counts)
    ]
    blocks_per_req = sum(decode_block_counts)
    max_num_reqs = min(
        max_num_reqs,
        max(1, (kv_cache_config.num_blocks - 1) // max(1, blocks_per_req)),
    )
    if max_num_reqs <= 0:
        logger.warning("Skipping physical TMH warmup because no KV blocks are available.")
        return

    warmup_num_reqs = [1]
    if max_num_reqs > 1:
        warmup_num_reqs.append(max_num_reqs)

    for shape_index, num_reqs in enumerate(warmup_num_reqs):
        logger.info(
            "Warming physical TMH kernels with num_reqs=%d prompt_len=%d "
            "decode_query_len=%d block_size=%d.",
            num_reqs,
            prompt_len,
            decode_query_len,
            block_size,
        )

        tmh_policy = TMHKVRuntimePolicy.from_kv_cache_config(
            kv_cache_config,
            block_size,
        )
        req_ids = [
            f"_tmh_physical_warmup_{shape_index}_{i}_"
            for i in range(num_reqs)
        ]
        prompt_token_ids = list(range(prompt_len))
        sampling_params = SamplingParams(max_tokens=2, temperature=0.0)
        next_block_id = 1

        def alloc_blocks(num_blocks: int) -> list[int]:
            nonlocal next_block_id
            block_ids = list(range(next_block_id, next_block_id + num_blocks))
            next_block_id += num_blocks
            return block_ids

        prefill_blocks_by_req: dict[str, tuple[list[int], ...]] = {}
        new_reqs: list[NewRequestData] = []
        for req_id in req_ids:
            block_ids = tuple(alloc_blocks(n) for n in prefill_block_counts)
            prefill_blocks_by_req[req_id] = block_ids
            new_reqs.append(
                NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=prompt_token_ids,
                    mm_features=[],
                    sampling_params=sampling_params,
                    pooling_params=None,
                    block_ids=block_ids,
                    num_computed_tokens=0,
                    lora_request=None,
                    prefill_token_ids=prompt_token_ids,
                )
            )

        prefill_output = SchedulerOutput.make_empty()
        prefill_output.scheduled_new_reqs = new_reqs
        prefill_output.num_scheduled_tokens = {
            req_id: prompt_len for req_id in req_ids
        }
        prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
        prefill_output.num_common_prefix_blocks = [
            0
        ] * len(kv_cache_config.kv_cache_groups)
        _attach_tmh_events(
            tmh_policy,
            prefill_output,
            total_tokens_by_req={req_id: prompt_len for req_id in req_ids},
            block_ids_by_req=prefill_blocks_by_req,
        )
        worker_execute_model(prefill_output)
        worker_sample_tokens(None)

        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_block_deltas)
        decode_blocks_by_req: dict[str, tuple[list[int], ...]] = {}
        new_block_ids: list[tuple[list[int], ...] | None] = []
        for req_id in req_ids:
            if new_block:
                delta_blocks = tuple(alloc_blocks(n) for n in decode_block_deltas)
                new_block_ids.append(delta_blocks)
                decode_blocks_by_req[req_id] = tuple(
                    list(existing) + list(delta)
                    for existing, delta in zip(
                        prefill_blocks_by_req[req_id],
                        delta_blocks,
                    )
                )
            else:
                new_block_ids.append(None)
                decode_blocks_by_req[req_id] = prefill_blocks_by_req[req_id]
        cached_req_data.new_block_ids = new_block_ids

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: decode_query_len for req_id in req_ids
        }
        decode_output.total_num_scheduled_tokens = decode_query_len * num_reqs
        decode_output.num_common_prefix_blocks = [
            0
        ] * len(kv_cache_config.kv_cache_groups)
        if model_runner.num_spec_tokens > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * model_runner.num_spec_tokens for req_id in req_ids
            }
        _attach_tmh_events(
            tmh_policy,
            decode_output,
            total_tokens_by_req={req_id: decode_len for req_id in req_ids},
            block_ids_by_req=decode_blocks_by_req,
        )
        worker_execute_model(decode_output)
        worker_sample_tokens(None)

        cleanup_output = SchedulerOutput.make_empty()
        cleanup_output.finished_req_ids = set(req_ids)
        for req_id in cleanup_output.finished_req_ids:
            tmh_policy.forget_request(req_id)
        cleanup_output.tmh_physical_events = tmh_policy.take_physical_events() or None
        worker_execute_model(cleanup_output)
        torch.accelerator.synchronize()


def _attach_tmh_events(
    policy: TMHKVRuntimePolicy,
    scheduler_output: SchedulerOutput,
    *,
    total_tokens_by_req: dict[str, int],
    block_ids_by_req: dict[str, tuple[list[int], ...]],
) -> None:
    for req_id, total_tokens in total_tokens_by_req.items():
        policy.record_physical_descriptors_from_block_ids(
            request_id=req_id,
            total_tokens=total_tokens,
            logical_block_ids=tuple(block_ids_by_req[req_id][0]),
        )
    scheduler_output.tmh_physical_events = policy.take_physical_events() or None
