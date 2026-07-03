# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code generation for split_gm stitching graph execution.

Generates a plain Python function that replaces the FX GraphModule's
interpreter-based execution of the stitching graph, eliminating
nn.Module.__call__ overhead and __getattr__ dispatch.
"""

import importlib
import json
import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import torch.fx
from torch._dynamo.utils import dynamo_timed
from torch._logging import trace_structured
from torch.fx.node import _get_qualified_name


def generate_execution_code_with_name(
    split_gm: torch.fx.GraphModule,
    fn_name: str,
    with_submod: bool,
    consts: list[Any] | None = None,
    const_index: dict[int, int] | None = None,
) -> tuple[str, list[str], list[Any]]:
    lines: list[str] = []
    param_names: list[str] = []
    submod_names: list[str] = []
    submod_index: dict[str, int] = {}
    if consts is None:
        consts = []
    if const_index is None:
        const_index = {}

    # Build node ordering for liveness analysis.
    nodes = list(split_gm.graph.nodes)
    node_order = {node: i for i, node in enumerate(nodes)}
    inlined_submods: list[str] = []

    # For each value-producing node, find the position of its last consumer.
    # If the last consumer is the output node, skip (return handles cleanup).
    # Otherwise, schedule a del after that consumer to free memory early.
    del_after: dict[int, list[str]] = {}  # position -> names to delete
    for node in nodes:
        if node.op == "output":
            continue
        users = list(node.users.keys())
        if not users:
            continue
        last_user = max(users, key=lambda u: node_order[u])
        if last_user.op == "output":
            continue
        del_after.setdefault(node_order[last_user], []).append(node.name)

    def ref(arg: Any) -> str:
        return _node_ref(arg, consts, const_index)

    for i, node in enumerate(nodes):
        if node.op == "placeholder":
            param_names.append(node.name)

        elif node.op == "call_module":
            target = node.target
            if not with_submod:
                raise RuntimeError(
                    f"call_module is not allowed for codegen target {target}."
                )
            if target not in submod_index:
                submod_index[target] = len(submod_names)
                submod_names.append(target)
            idx = submod_index[target]
            args_str = ", ".join(ref(a) for a in node.args)
            kwargs_str = ", ".join(f"{k}={ref(v)}" for k, v in node.kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            submod = getattr(split_gm, target)
            if isinstance(submod, torch.fx.GraphModule):
                callable_name = f"__vllm_inlined_submods__{idx}"
                inlined_code, _, _ = generate_execution_code_with_name(
                    submod,
                    callable_name,
                    with_submod=False,
                    consts=consts,
                    const_index=const_index,
                )
                inlined_submods.append(inlined_code)
            else:
                callable_name = f"__vllm_submods__[{idx}]"
            lines.append(f"    {node.name} = {callable_name}({all_args})")

        elif node.op == "call_function":
            if node.target is operator.getitem:
                source = ref(node.args[0])
                index = node.args[1]
                assert isinstance(index, int)
                lines.append(f"    {node.name} = {source}[{index}]")
            else:
                args_str = ", ".join(ref(a) for a in node.args)
                kwargs_str = ", ".join(f"{k}={ref(v)}" for k, v in node.kwargs.items())
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                lines.append(
                    f"    {node.name} = {_get_qualified_name(node.target)}({all_args})"
                )

        elif node.op == "output":
            assert len(node.args) == 1
            ret = ref(node.args[0])
            lines.append(f"    return {ret}")

        else:
            raise RuntimeError(f"Unsupported node from codegen: {node.format_node()}")

        # Emit del for variables whose last use was this node.
        if i in del_after and i < len(nodes) - 2:
            names = sorted(del_after[i])
            lines.append(f"    del {', '.join(names)}")

    assert len(param_names) > 0
    params = ", ".join(param_names)
    kw_params = ", *, __vllm_submods__" if with_submod else ""
    header = f"\ndef {fn_name}({params}{kw_params}):"
    return (
        "".join(inlined_submods) + "\n".join([header] + lines) + "\n",
        submod_names,
        consts,
    )


@dynamo_timed("vllm.generate_execution_code")
def generate_execution_code(
    split_gm: torch.fx.GraphModule,
) -> tuple[str, list[str], list[Any]]:
    """Generate Python source code from a split_gm's stitching graph.

    Walks split_gm.graph.nodes and produces a function that calls
    submodules via a __vllm_submods__ list, avoiding FX GraphModule overhead
    and dict lookup cost.

    Non-primitive constant arguments (e.g. torch.device, DTensor placement
    types) are collected into a constants list and referenced by index
    in the generated code, avoiding reliance on repr() being eval-able.

    If a submodule is a plain torch.fx.GraphModule, it is inlined directly
    in the generated code and we do not need to serialize it in the artifact.

    Args:
        split_gm: The split graph module produced by split_graph().

    Returns:
        A tuple of (code, submod_names, consts) where code is the Python
        source, submod_names is the ordered list of submodule target names
        corresponding to list indices used in the generated code, and
        consts is a list of non-primitive constant objects referenced
        by the generated code via __vllm_consts__. These objects are
        kept alive for the lifetime of the compiled function.
    """
    code, submod_names, consts = generate_execution_code_with_name(
        split_gm, "execution_fn", with_submod=True
    )
    return "import torch\nimport operator\n" + code, submod_names, consts


def _plan_ref(
    arg: Any,
    consts: list[Any],
    const_index: dict[int, int],
) -> dict[str, Any]:
    if isinstance(arg, torch.fx.Node):
        return {"kind": "node", "name": arg.name}
    if isinstance(arg, list):
        return {
            "kind": "list",
            "items": [_plan_ref(x, consts, const_index) for x in arg],
        }
    if isinstance(arg, tuple):
        return {
            "kind": "tuple",
            "items": [_plan_ref(x, consts, const_index) for x in arg],
        }
    if isinstance(arg, dict):
        return {
            "kind": "dict",
            "items": [
                [
                    _plan_ref(key, consts, const_index),
                    _plan_ref(value, consts, const_index),
                ]
                for key, value in arg.items()
            ],
        }
    if isinstance(arg, (int, float, bool, str, bytes, type(None))):
        return {"kind": "literal", "value": arg}
    key = id(arg)
    if key not in const_index:
        const_index[key] = len(consts)
        consts.append(arg)
    return {"kind": "const", "index": const_index[key]}


def generate_execution_plan_with_name(
    split_gm: torch.fx.GraphModule,
    fn_name: str,
    with_submod: bool,
    consts: list[Any] | None = None,
    const_index: dict[int, int] | None = None,
) -> tuple[dict[str, Any], list[str], list[Any]]:
    param_names: list[str] = []
    submod_names: list[str] = []
    submod_index: dict[str, int] = {}
    ops: list[dict[str, Any]] = []
    if consts is None:
        consts = []
    if const_index is None:
        const_index = {}

    nodes = list(split_gm.graph.nodes)
    node_order = {node: i for i, node in enumerate(nodes)}
    del_after: dict[int, list[str]] = {}
    for node in nodes:
        if node.op == "output":
            continue
        users = list(node.users.keys())
        if not users:
            continue
        last_user = max(users, key=lambda u: node_order[u])
        if last_user.op == "output":
            continue
        del_after.setdefault(node_order[last_user], []).append(node.name)

    def ref(arg: Any) -> dict[str, Any]:
        return _plan_ref(arg, consts, const_index)

    for i, node in enumerate(nodes):
        if node.op == "placeholder":
            param_names.append(node.name)
        elif node.op == "call_module":
            target = node.target
            if not with_submod:
                raise RuntimeError(
                    f"call_module is not allowed for codegen target {target}."
                )
            if target not in submod_index:
                submod_index[target] = len(submod_names)
                submod_names.append(target)
            submod = getattr(split_gm, target)
            op: dict[str, Any] = {
                "kind": "call_submod",
                "out": node.name,
                "submod_index": submod_index[target],
                "args": [ref(arg) for arg in node.args],
                "kwargs": {key: ref(value) for key, value in node.kwargs.items()},
            }
            if isinstance(submod, torch.fx.GraphModule):
                nested_plan, _, _ = generate_execution_plan_with_name(
                    submod,
                    f"__vllm_inlined_submods__{submod_index[target]}",
                    with_submod=False,
                    consts=consts,
                    const_index=const_index,
                )
                op["kind"] = "call_inlined_subplan"
                op["plan"] = nested_plan
            ops.append(op)
        elif node.op == "call_function":
            if node.target is operator.getitem:
                assert isinstance(node.args[1], int)
                ops.append(
                    {
                        "kind": "getitem",
                        "out": node.name,
                        "source": ref(node.args[0]),
                        "index": node.args[1],
                    }
                )
            else:
                ops.append(
                    {
                        "kind": "call_function",
                        "out": node.name,
                        "target": _get_qualified_name(node.target),
                        "args": [ref(arg) for arg in node.args],
                        "kwargs": {
                            key: ref(value) for key, value in node.kwargs.items()
                        },
                    }
                )
        elif node.op == "output":
            assert len(node.args) == 1
            ops.append({"kind": "return", "value": ref(node.args[0])})
        else:
            raise RuntimeError(f"Unsupported node from codegen: {node.format_node()}")

        if i in del_after and i < len(nodes) - 2:
            ops.append({"kind": "del", "names": sorted(del_after[i])})

    if not param_names:
        raise RuntimeError("Expected at least one placeholder in stitching graph.")

    return (
        {
            "schema_version": 1,
            "name": fn_name,
            "with_submods": with_submod,
            "params": param_names,
            "ops": ops,
        },
        submod_names,
        consts,
    )


@dynamo_timed("vllm.generate_execution_plan")
def generate_execution_plan(
    split_gm: torch.fx.GraphModule,
) -> tuple[dict[str, Any], list[str], list[Any]]:
    return generate_execution_plan_with_name(
        split_gm, "execution_fn", with_submod=True
    )


def _resolve_qualified_target(name: str) -> Any:
    module_name, _, attr_path = name.partition(".")
    obj = importlib.import_module(module_name)
    if not attr_path:
        return obj
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _eval_plan_ref(ref: dict[str, Any], env: dict[str, Any], consts: list[Any]) -> Any:
    kind = ref["kind"]
    if kind == "literal":
        return ref["value"]
    if kind == "node":
        return env[ref["name"]]
    if kind == "const":
        return consts[ref["index"]]
    if kind == "list":
        return [_eval_plan_ref(item, env, consts) for item in ref["items"]]
    if kind == "tuple":
        return tuple(_eval_plan_ref(item, env, consts) for item in ref["items"])
    if kind == "dict":
        return {
            _eval_plan_ref(key, env, consts): _eval_plan_ref(value, env, consts)
            for key, value in ref["items"]
        }
    raise RuntimeError(f"Unsupported plan ref kind: {kind}")


def _execute_execution_plan(
    plan: dict[str, Any],
    args: tuple[Any, ...],
    *,
    submods: list[Callable[..., Any] | None],
    consts: list[Any],
) -> Any:
    env = dict(zip(plan["params"], args, strict=True))
    for op in plan["ops"]:
        kind = op["kind"]
        if kind == "call_submod":
            submod = submods[op["submod_index"]]
            if submod is None:
                raise RuntimeError(
                    f"Missing submodule binding for index {op['submod_index']}"
                )
            env[op["out"]] = submod(
                *[_eval_plan_ref(arg, env, consts) for arg in op["args"]],
                **{
                    key: _eval_plan_ref(value, env, consts)
                    for key, value in op["kwargs"].items()
                },
            )
        elif kind == "call_inlined_subplan":
            env[op["out"]] = _execute_execution_plan(
                op["plan"],
                tuple(_eval_plan_ref(arg, env, consts) for arg in op["args"]),
                submods=submods,
                consts=consts,
            )
        elif kind == "call_function":
            target = _resolve_qualified_target(op["target"])
            env[op["out"]] = target(
                *[_eval_plan_ref(arg, env, consts) for arg in op["args"]],
                **{
                    key: _eval_plan_ref(value, env, consts)
                    for key, value in op["kwargs"].items()
                },
            )
        elif kind == "getitem":
            env[op["out"]] = _eval_plan_ref(op["source"], env, consts)[op["index"]]
        elif kind == "del":
            for name in op["names"]:
                env.pop(name, None)
        elif kind == "return":
            return _eval_plan_ref(op["value"], env, consts)
        else:
            raise RuntimeError(f"Unsupported execution plan op kind: {kind}")
    raise RuntimeError("Execution plan terminated without return op.")


@dynamo_timed("vllm.compile_execution_plan_fn")
def compile_execution_plan_fn(
    plan: dict[str, Any],
    submod_callables: dict[str, Callable[..., Any]],
    submod_names: list[str],
    consts: list[Any] | None = None,
) -> Callable[..., Any]:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "vllm_execution_plan",
            "encoding": "json",
        },
        payload_fn=lambda: json.dumps(plan, sort_keys=True, separators=(",", ":")),
    )
    bound_consts = consts or []
    submods_list = [submod_callables.get(name) for name in submod_names]

    def execution_fn(*args: Any) -> Any:
        return _execute_execution_plan(
            plan,
            args,
            submods=submods_list,
            consts=bound_consts,
        )

    return execution_fn


@dynamo_timed("vllm.compile_execution_fn")
def compile_execution_fn(
    code: str,
    submod_callables: dict[str, Callable[..., Any]],
    submod_names: list[str],
    consts: list[Any] | None = None,
) -> Callable[..., Any]:
    """Compile execution code and bind submodule callables.

    Args:
        code: Python source from generate_execution_code().
        submod_callables: Mapping of submodule names to their callables.
        submod_names: Ordered list of submodule names matching the indices
            used in the generated code.
        consts: List of non-primitive constant objects referenced by the
            generated code via __vllm_consts__. None for legacy cached
            code that predates this feature.

    Returns:
        A callable that executes the stitching logic.
    """
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "vllm_execution_code",
            "encoding": "string",
        },
        payload_fn=lambda: code,
    )
    namespace: dict[str, Any] = {}
    if consts is not None:
        namespace["__vllm_consts__"] = consts
    exec(code, namespace)  # noqa: S102
    fn = namespace["execution_fn"]
    # Using .get() is intentional here because only piecewise backend will
    # be stored in submod_callables. The other submodules are inlined and
    # we don't need to bind them to the execution function. Instead, we
    # should use None as placeholder to ensure the list indices are preserved
    # for better debuggability.
    submods_list = [submod_callables.get(name) for name in submod_names]
    return partial(fn, __vllm_submods__=submods_list)


def _node_ref(arg: Any, consts: list[Any], const_index: dict[int, int]) -> str:
    """Convert an FX node argument to a source code reference."""
    if isinstance(arg, torch.fx.Node):
        return arg.name
    if isinstance(arg, list):
        return f"[{', '.join(_node_ref(x, consts, const_index) for x in arg)}]"
    if isinstance(arg, tuple):
        items = ", ".join(_node_ref(x, consts, const_index) for x in arg)
        return f"({items},)" if len(arg) == 1 else f"({items})"
    if isinstance(arg, dict):
        return (
            "{"
            + ", ".join(
                f"{_node_ref(k, consts, const_index)}: "
                f"{_node_ref(v, consts, const_index)}"
                for k, v in arg.items()
            )
            + "}"
        )
    if isinstance(arg, (int, float, bool, str, bytes, type(None))):
        return repr(arg)
    # Dedup by identity, not equality: safe because FX graph args
    # are live for the entire code-generation pass. Objects stored
    # here must be picklable (for compile-artifact caching).
    key = id(arg)
    if key not in const_index:
        const_index[key] = len(consts)
        consts.append(arg)
    return f"__vllm_consts__[{const_index[key]}]"
