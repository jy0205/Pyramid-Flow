"""Lightweight regression tests for pre-computed text feature handling."""

import importlib.util
from pathlib import Path

import pytest
import torch


_MODULE_PATH = Path(__file__).parents[1] / "pyramid_dit" / "text_features.py"
_SPEC = importlib.util.spec_from_file_location("pyramid_text_features", _MODULE_PATH)
_TEXT_FEATURES = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TEXT_FEATURES)


def _batched_features(batch_size=2):
    return {
        "prompt_embed": torch.arange(batch_size * 3 * 4, dtype=torch.float32).reshape(batch_size, 3, 4),
        "prompt_attention_mask": torch.ones(batch_size, 3),
        "pooled_prompt_embed": torch.arange(batch_size * 5, dtype=torch.float32).reshape(batch_size, 5),
    }


def test_normalize_accepts_bucketeer_and_legacy_layouts():
    features = _batched_features()

    nested = {"video": torch.zeros(2, 1), "text": features, "identifier": ["video", "video"]}
    normalized_nested = _TEXT_FEATURES.normalize_text_features(nested, batch_size=2)
    normalized_legacy = _TEXT_FEATURES.normalize_text_features(features, batch_size=2)

    assert set(normalized_nested) == set(_TEXT_FEATURES.TEXT_FEATURE_KEYS)
    for key in _TEXT_FEATURES.TEXT_FEATURE_KEYS:
        assert torch.equal(normalized_nested[key], normalized_legacy[key])


def test_move_text_features_to_device_moves_all_tensors():
    normalized = _TEXT_FEATURES.move_text_features_to_device(
        {"text": _batched_features()},
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert all(value.device.type == "cpu" for value in normalized.values())
    assert normalized["prompt_embeds"].shape == (2, 3, 4)
    assert normalized["prompt_attention_mask"].shape == (2, 3)
    assert normalized["pooled_prompt_embeds"].shape == (2, 5)


def test_null_features_remove_only_singleton_batch():
    null_features = _TEXT_FEATURES.normalize_null_text_features(
        {
            "prompt_embed": torch.zeros(1, 3, 4),
            "prompt_attention_mask": torch.ones(1, 3),
            "pooled_prompt_embed": torch.zeros(1, 5),
        }
    )

    assert null_features["prompt_embeds"].shape == (3, 4)
    assert null_features["prompt_attention_mask"].shape == (3,)
    assert null_features["pooled_prompt_embeds"].shape == (5,)


def test_null_features_reject_non_singleton_batch():
    with pytest.raises(ValueError, match="singleton"):
        _TEXT_FEATURES.normalize_null_text_features(
            {
                "prompt_embeds": torch.zeros(2, 3, 4),
                "prompt_attention_mask": torch.ones(1, 3),
                "pooled_prompt_embeds": torch.zeros(1, 5),
            }
        )


def test_load_null_features_validates_path_and_schema(tmp_path):
    path = tmp_path / "null_prompt.pt"
    torch.save(
        {
            "prompt_embed": torch.zeros(1, 3, 4),
            "prompt_attention_mask": torch.ones(1, 3),
            "pooled_prompt_embed": torch.zeros(1, 5),
        },
        path,
    )

    loaded = _TEXT_FEATURES.load_null_text_features(path)
    assert loaded["prompt_embeds"].shape == (3, 4)
    with pytest.raises(FileNotFoundError):
        _TEXT_FEATURES.load_null_text_features(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="required"):
        _TEXT_FEATURES.validate_null_text_feature_path("")
