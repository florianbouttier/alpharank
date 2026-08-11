"""Tests for SHAP row coverage controls."""

from __future__ import annotations

from alpharank.multihorizon.explain import shap_row_indexes


def test_zero_shap_sample_size_keeps_every_row() -> None:
    indexes = shap_row_indexes(row_count=497, sample_size=0, seed=42)

    assert indexes.tolist() == list(range(497))


def test_positive_shap_sample_size_remains_deterministic() -> None:
    first = shap_row_indexes(row_count=497, sample_size=80, seed=42)
    second = shap_row_indexes(row_count=497, sample_size=80, seed=42)

    assert first.tolist() == second.tolist()
    assert len(first) == 80
