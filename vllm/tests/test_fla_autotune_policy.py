from vllm.model_executor.layers.fla.ops import utils


def test_rocm_platform_policy_uses_bounded_configs(monkeypatch):
    monkeypatch.setattr(utils, "FLA_AUTOTUNE_POLICY", "platform")
    monkeypatch.setattr(utils, "is_amd", True)

    assert utils.platform_autotune_configs(["full"], rocm=["rocm"]) == ["rocm"]


def test_cuda_platform_policy_preserves_full_configs(monkeypatch):
    monkeypatch.setattr(utils, "FLA_AUTOTUNE_POLICY", "platform")
    monkeypatch.setattr(utils, "is_amd", False)

    assert utils.platform_autotune_configs(["a", "b"], rocm=["rocm"]) == ["a", "b"]


def test_explicit_full_policy_preserves_full_configs(monkeypatch):
    monkeypatch.setattr(utils, "FLA_AUTOTUNE_POLICY", "full")
    monkeypatch.setattr(utils, "is_amd", True)

    assert utils.platform_autotune_configs(["a", "b"], rocm=["rocm"]) == ["a", "b"]


def test_explicit_bounded_policy_uses_first_config_without_platform_override(monkeypatch):
    monkeypatch.setattr(utils, "FLA_AUTOTUNE_POLICY", "bounded")
    monkeypatch.setattr(utils, "is_amd", False)

    assert utils.platform_autotune_configs(["a", "b"]) == ["a"]
