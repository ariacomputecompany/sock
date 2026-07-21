import sys

import pytest

from scripts.runtime_cli import apply_sock_modality_cli_defaults


def test_sock_text_modality_adds_language_only_flags(monkeypatch):
    monkeypatch.setenv("SOCK_INFERENCE_MODALITY", "text")
    monkeypatch.setattr(sys, "argv", ["runtime_cli.py", "serve", "model"] )

    apply_sock_modality_cli_defaults()

    assert "--language-model-only" in sys.argv
    assert "--skip-mm-profiling" in sys.argv


def test_sock_auto_modality_leaves_cli_unchanged(monkeypatch):
    monkeypatch.setenv("SOCK_INFERENCE_MODALITY", "auto")
    monkeypatch.setattr(sys, "argv", ["runtime_cli.py", "serve", "model"] )

    apply_sock_modality_cli_defaults()

    assert sys.argv == ["runtime_cli.py", "serve", "model"]


def test_sock_modality_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("SOCK_INFERENCE_MODALITY", "vision-but-not-really")
    monkeypatch.setattr(sys, "argv", ["runtime_cli.py", "serve", "model"] )

    with pytest.raises(ValueError):
        apply_sock_modality_cli_defaults()
