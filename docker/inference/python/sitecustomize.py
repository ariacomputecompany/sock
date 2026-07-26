"""Runtime Python customizations for the SOCK text inference image."""

from __future__ import annotations

import importlib
import os
import sys


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _env_disabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _vision_deps_enabled() -> bool:
    """Return whether the text inference image may import optional VL stacks."""
    if "SOCK_ENABLE_VISION_DEPS" in os.environ:
        return _env_enabled("SOCK_ENABLE_VISION_DEPS")
    return not _env_disabled("SOCK_DISABLE_TORCHVISION", "1")


def _disable_transformers_torchvision() -> None:
    def unavailable() -> bool:
        return False

    def not_greater_or_equal(_version: str) -> bool:
        return False

    try:
        import_utils = importlib.import_module("transformers.utils.import_utils")
    except Exception:
        return

    import_utils.is_torchvision_available = unavailable
    import_utils.is_torchvision_v2_available = unavailable
    import_utils.is_torchvision_greater_or_equal = not_greater_or_equal

    utils = sys.modules.get("transformers.utils")
    if utils is not None:
        utils.is_torchvision_available = unavailable
        utils.is_torchvision_v2_available = unavailable
        utils.is_torchvision_greater_or_equal = not_greater_or_equal


if not _vision_deps_enabled():
    _disable_transformers_torchvision()
