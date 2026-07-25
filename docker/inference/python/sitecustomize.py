"""Runtime Python customizations for the SOCK text inference image."""

from __future__ import annotations

import importlib
import os
import sys


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


if os.environ.get("SOCK_DISABLE_TORCHVISION", "1").lower() not in {"0", "false", "no"}:
    _disable_transformers_torchvision()
