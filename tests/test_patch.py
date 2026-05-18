from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from thriftattention.config import AttentionConfig
from thriftattention.patch import patch_model, unpatch_model


def test_patch_model_rejects_old_selection_name_as_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        patch_model(object(), mode="topk")


def test_patch_model_validates_fraction_and_top_k():
    with pytest.raises(ValueError, match="fp16_fraction"):
        patch_model(object(), fp16_fraction=1.5)
    with pytest.raises(ValueError, match="top_k"):
        patch_model(object(), top_k=-1)
    with pytest.raises(ValueError, match="fallback"):
        patch_model(object(), fallback="eager")


def test_patch_model_sets_noncausal_config(monkeypatch):
    from thriftattention.integrations import transformers as hf

    model = SimpleNamespace()
    monkeypatch.setattr(hf, "patch_hf_model", lambda patched_model, config: config)

    config = patch_model(model, causal=False)

    assert config.causal is False


def test_hf_patch_sets_backend_and_tags_modules(monkeypatch):
    from thriftattention.integrations import transformers as hf

    registered = {}

    class Registry:
        @staticmethod
        def register(name, fn):
            registered[name] = fn

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", lambda name=None: None)

    child = SimpleNamespace()

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.calls = []

        def modules(self):
            return [self, child]

        def set_attn_implementation(self, name):
            self.calls.append(name)
            self.config._attn_implementation = name

    model = Model()
    patched = patch_model(model)

    assert patched is model
    assert "thriftattention" in registered
    assert model.config._attn_implementation == "thriftattention"
    assert model.calls == ["thriftattention"]
    assert child._thriftattention_config.fallback == "error"

    unpatch_model(model)
    assert model.config._attn_implementation == "sdpa"
    assert not hasattr(child, "_thriftattention_config")


def test_hf_patch_requires_public_set_attn_implementation(monkeypatch):
    from thriftattention.integrations import transformers as hf

    class Registry:
        @staticmethod
        def register(name, fn):
            pass

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", lambda name=None: None)

    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))

    with pytest.raises(TypeError, match="set_attn_implementation"):
        patch_model(model)

    assert not hasattr(model, "_thriftattention_original_attn_implementation")
    assert not hasattr(model, "_thriftattention_config")


def test_register_transformers_attention_registers_custom_name(monkeypatch):
    from thriftattention.integrations import transformers as hf

    registered = {}
    masks = {}

    class Registry:
        @staticmethod
        def register(name, fn):
            registered[name] = fn

    def register_mask(name=None):
        masks[name] = True

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", register_mask)

    name = hf.register_transformers_attention(
        hf.TransformersAttentionConfig(name="thrift_attention", mode="fp4")
    )

    assert name == "thrift_attention"
    assert registered["thrift_attention"] is hf.thriftattention_forward
    assert masks["thrift_attention"] is True
    assert hf.get_registered_transformers_attention_config("thrift_attention").mode == "fp4"


def test_registered_attention_config_is_used_without_patch(monkeypatch):
    from thriftattention.integrations import transformers as hf

    class Registry:
        @staticmethod
        def register(name, fn):
            pass

    monkeypatch.setattr(hf, "_get_attention_registry", lambda: Registry)
    monkeypatch.setattr(hf, "_register_attention_mask", lambda name=None: None)
    hf.register_transformers_attention(hf.TransformersAttentionConfig(name="unit_thrift"))

    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="unit_thrift"),
        training=False,
        is_causal=True,
        num_key_value_groups=1,
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_prepare_transformers_generation_cache_pads_and_tags_modules():
    from thriftattention.integrations.transformers import prepare_transformers_generation_cache

    child = SimpleNamespace()

    class Model:
        config = SimpleNamespace(max_position_embeddings=1024)

        def modules(self):
            return [self, child]

    model = Model()
    config = AttentionConfig(backend="hf", fp16_fraction=0.05)

    prepared = prepare_transformers_generation_cache(
        model,
        [1, 2, 3],
        config=config,
        max_new_tokens=5,
        device="cpu",
    )

    assert prepared.input_ids.shape == (1, 64)
    assert prepared.cache_position.tolist() == list(range(64))
    assert prepared.prompt_length == 3
    assert prepared.padding == 61
    assert prepared.past_key_values.max_cache_len == 69
    assert prepared.past_key_values.prefill_real_seq_len == 3
    assert child._thriftattention_config.top_k == 1


def test_cache_crop_zeroes_rounded_physical_tail():
    from thriftattention.integrations.transformers_cache import ThriftAttentionCacheLayer

    layer = ThriftAttentionCacheLayer()
    layer.seq_len = 128
    layer.capacity = 128
    layer.k_fp16 = torch.ones(1, 1, 128, 64)
    layer.v_fp16 = torch.ones(1, 1, 128, 64)
    layer.k_packed = torch.empty(1)
    layer.k_scale = torch.empty(1)
    layer.v_packed_t = torch.empty(1)
    layer.v_scale_t = torch.empty(1)
    calls = []

    def quantize_k(key_states, start, end):
        calls.append(("k", tuple(key_states.shape), start, end))

    def quantize_v(start, end):
        calls.append(("v", start, end))

    layer._quantize_k_range = quantize_k
    layer._quantize_v_range = quantize_v

    layer.crop(65)

    assert layer.seq_len == 65
    assert torch.all(layer.k_fp16[:, :, :65] == 1)
    assert torch.all(layer.v_fp16[:, :, :65] == 1)
    assert torch.all(layer.k_fp16[:, :, 65:128] == 0)
    assert torch.all(layer.v_fp16[:, :, 65:128] == 0)
    assert calls == [("k", (1, 1, 63, 64), 65, 128), ("v", 65, 128)]


def test_generate_cache_injection_sets_prompt_specific_config():
    from thriftattention.integrations import transformers as hf

    child = SimpleNamespace()

    class Model:
        training = False
        config = SimpleNamespace(is_encoder_decoder=False, max_position_embeddings=128)

        def modules(self):
            return [self, child]

    model = Model()
    input_ids = torch.ones(1, 65, dtype=torch.long)
    seen = {}

    def original_generate(*args, **kwargs):
        cache = kwargs["past_key_values"]
        seen["cache"] = cache
        assert hf.get_active_thriftattention_cache() is cache
        return "generated"

    result = hf._generate_with_thriftattention_cache(
        model,
        original_generate,
        AttentionConfig(backend="hf", fp16_fraction=0.05),
        input_ids,
        max_new_tokens=3,
    )

    assert result == "generated"
    assert seen["cache"].max_cache_len == 68
    assert seen["cache"].prefill_real_seq_len == 65
    assert child._thriftattention_config.top_k == 1


def test_adapter_requires_patched_module_config():
    from thriftattention.integrations.transformers import thriftattention_forward

    module = SimpleNamespace(training=False, is_causal=True)
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="unpatched module"):
        thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_adapter_error_fallback_reports_rejection_reason():
    from thriftattention.integrations.transformers import thriftattention_forward

    module = SimpleNamespace(
        training=False,
        is_causal=True,
        num_key_value_groups=1,
        _thriftattention_config=AttentionConfig(backend="hf", fallback="error"),
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)


def test_adapter_fast_path_transposes_thrift_output(monkeypatch):
    from thriftattention.integrations import transformers as hf

    module = SimpleNamespace(
        training=False,
        is_causal=True,
        _thriftattention_config=AttentionConfig(backend="hf", fallback="error"),
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    def fake_attention(q, k, v, **kwargs):
        assert kwargs["selector"] == "block_mean"
        return torch.zeros(1, 2, 64, 64)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert output.shape == (1, 64, 2, 64)
    assert weights is None


def test_adapter_fast_path_passes_noncausal_mode(monkeypatch):
    from thriftattention.integrations import transformers as hf

    module = SimpleNamespace(
        training=False,
        is_causal=False,
        _thriftattention_config=AttentionConfig(backend="hf", causal=False, fallback="error"),
    )
    query = torch.randn(1, 2, 64, 64)
    key = torch.randn(1, 2, 64, 64)
    value = torch.randn(1, 2, 64, 64)

    def fake_attention(q, k, v, **kwargs):
        assert kwargs["causal"] is False
        return torch.zeros(1, 2, 64, 64)

    monkeypatch.setattr(hf, "_fast_path_rejection_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf, "thrift_attention", fake_attention)

    output, weights = hf.thriftattention_forward(module, query, key, value, None, scaling=64**-0.5)

    assert output.shape == (1, 64, 2, 64)
    assert weights is None
