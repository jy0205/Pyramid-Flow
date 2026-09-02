"""Helpers for loading and preparing pre-computed text features.

The text-feature extraction script historically wrote singular keys
(``prompt_embed`` and ``pooled_prompt_embed``), while the pipeline consumes
batched dictionaries with plural keys.  These helpers keep both formats
usable and make the conversion and device placement easy to test without
loading a model checkpoint.
"""

import os
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch


TEXT_FEATURE_KEYS = (
    "prompt_embeds",
    "prompt_attention_mask",
    "pooled_prompt_embeds",
)

_FEATURE_KEY_ALIASES = {
    "prompt_embeds": ("prompt_embeds", "prompt_embed"),
    "prompt_attention_mask": (
        "prompt_attention_mask",
        "prompt_attention_masks",
    ),
    "pooled_prompt_embeds": (
        "pooled_prompt_embeds",
        "pooled_prompt_embed",
    ),
}

_NULL_FEATURE_RANKS = {
    "prompt_embeds": 2,
    "prompt_attention_mask": 1,
    "pooled_prompt_embeds": 1,
}


def _contains_feature_key(features: Mapping) -> bool:
    return any(
        alias in features
        for aliases in _FEATURE_KEY_ALIASES.values()
        for alias in aliases
    )


def _select_feature_mapping(features: Mapping) -> Mapping:
    """Select the feature dictionary from a batch or a saved artifact."""

    nested = features.get("text")
    if isinstance(nested, Mapping):
        # TemporalLengthBucketeer emits ``{"text": {...}, "video": ...}``.
        # Prefer the nested value when it carries the feature keys.  Falling
        # back to the outer mapping keeps malformed/legacy wrappers readable
        # and lets the normal missing-key error explain what is wrong.
        if _contains_feature_key(nested) or not _contains_feature_key(features):
            return nested
    return features


def normalize_text_features(
    text_features: Mapping,
    batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Return a canonical, plural-keyed text-feature dictionary.

    ``text_features`` may be the nested dictionary produced by
    :class:`TemporalLengthBucketeer`, a legacy top-level dataset dictionary,
    or a saved feature artifact.  The returned mapping contains only
    ``prompt_embeds``, ``prompt_attention_mask`` and
    ``pooled_prompt_embeds``.  Tensor values are not copied here; callers can
    choose whether copying is needed for their mutation pattern.

    Args:
        text_features: Mapping containing the three feature tensors.  A
            mapping under the optional ``text`` key is unwrapped first.
        batch_size: If supplied, verify that every tensor has this leading
            batch dimension.

    Raises:
        TypeError: If the input is not a mapping or a value is not a tensor.
        KeyError: If one of the required feature keys is missing.
        ValueError: If tensor batch dimensions do not agree with
            ``batch_size``.
    """

    if not isinstance(text_features, Mapping):
        raise TypeError(
            "Pre-computed text features must be a mapping, got "
            f"{type(text_features).__name__}"
        )

    features = _select_feature_mapping(text_features)
    normalized: Dict[str, torch.Tensor] = {}
    missing = []

    for canonical_key, aliases in _FEATURE_KEY_ALIASES.items():
        value: Any = None
        found = False
        for alias in aliases:
            if alias in features:
                value = features[alias]
                found = True
                break
        if not found:
            missing.append(canonical_key)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Text feature '{canonical_key}' must be a torch.Tensor, "
                f"got {type(value).__name__}"
            )
        normalized[canonical_key] = value

    if missing:
        raise KeyError(
            "Missing required pre-computed text feature(s): "
            + ", ".join(missing)
        )

    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 0:
            raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}")
        for key, value in normalized.items():
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    f"Text feature '{key}' has shape {tuple(value.shape)}; "
                    f"expected leading batch dimension {batch_size}"
                )

    return normalized


def move_text_features_to_device(
    text_features: Mapping,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize text features and move each tensor to ``device``."""

    normalized = normalize_text_features(text_features, batch_size=batch_size)
    return {key: value.to(device) for key, value in normalized.items()}


def prepare_text_features(
    text_features: Mapping,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Convenience alias for the normalization/device-placement operation."""

    return move_text_features_to_device(
        text_features,
        device=device,
        batch_size=batch_size,
    )


def normalize_null_text_features(text_features: Mapping) -> Dict[str, torch.Tensor]:
    """Normalize a saved null-prompt artifact to unbatched tensors.

    Feature extraction saves a singleton batch (for example ``[1,  T, C]``),
    whereas CFG dropout replaces one row of a training batch and therefore
    needs ``[T, C]``.  A singleton leading dimension is removed for each
    tensor.  Any other batch size is rejected so a mismatched artifact cannot
    silently alter the training algorithm.
    """

    normalized = normalize_text_features(text_features)
    unbatched: Dict[str, torch.Tensor] = {}

    for key, value in normalized.items():
        expected_rank = _NULL_FEATURE_RANKS[key]
        if value.ndim == expected_rank + 1:
            if value.shape[0] != 1:
                raise ValueError(
                    f"Null text feature '{key}' must be unbatched or have a "
                    f"singleton leading dimension; got shape {tuple(value.shape)}"
                )
            value = value[0]
        if value.ndim != expected_rank:
            raise ValueError(
                f"Null text feature '{key}' has shape {tuple(value.shape)}; "
                f"expected rank {expected_rank} (optionally with a leading 1)"
            )
        # Keep the loaded artifact independent from a caller's dictionary and
        # ensure CFG replacement never tracks gradients.
        unbatched[key] = value.detach().clone()

    prompt_embeds = unbatched["prompt_embeds"]
    prompt_attention_mask = unbatched["prompt_attention_mask"]
    if prompt_embeds.shape[0] != prompt_attention_mask.shape[0]:
        raise ValueError(
            "Null text feature sequence lengths do not match: "
            f"prompt_embeds={tuple(prompt_embeds.shape)}, "
            f"prompt_attention_mask={tuple(prompt_attention_mask.shape)}"
        )

    return unbatched


def validate_null_text_feature_path(path: Any) -> str:
    """Validate and return a path to a null text-feature artifact."""

    if path is None:
        raise ValueError(
            "null_text_feature_path is required when load_text_encoder=False"
        )
    try:
        path_string = os.fspath(path)
    except TypeError as exc:
        raise TypeError(
            "null_text_feature_path must be a filesystem path"
        ) from exc

    if not path_string or not str(path_string).strip():
        raise ValueError(
            "null_text_feature_path is required when load_text_encoder=False"
        )
    if not os.path.isfile(path_string):
        raise FileNotFoundError(
            f"Null text feature file does not exist: {path_string}"
        )
    return path_string


def load_null_text_features(path: Any) -> Dict[str, torch.Tensor]:
    """Load, validate and normalize a null text-feature artifact on CPU."""

    path_string = validate_null_text_feature_path(path)
    try:
        loaded = torch.load(path_string, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load null text feature file '{path_string}': {exc}"
        ) from exc

    try:
        return normalize_null_text_features(loaded)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid null text feature file '{path_string}': {exc}"
        ) from exc


# Explicit aliases make the small public API convenient for callers that use
# either the noun-oriented or action-oriented spelling.
canonicalize_text_features = normalize_text_features
move_text_features = move_text_features_to_device
