from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.explainability import (
    _interaction_ranking,
    _interaction_strength_matrix,
    _ordered_feature_indices,
)


def test_ordered_feature_indices_follow_descending_mean_abs_shap() -> None:
    mean_abs_shap = np.array([0.20, 0.70, 0.10, 0.40], dtype=float)

    order = _ordered_feature_indices(mean_abs_shap)

    assert order.tolist() == [1, 3, 0, 2]


def test_interaction_strength_matrix_keeps_diagonal_but_ranking_excludes_it() -> None:
    interaction_values = np.array(
        [
            [
                [0.90, 0.20, 0.10],
                [0.20, 0.50, 0.40],
                [0.10, 0.40, 0.30],
            ],
            [
                [1.10, 0.40, 0.20],
                [0.40, 0.70, 0.60],
                [0.20, 0.60, 0.20],
            ],
        ],
        dtype=float,
    )

    strength = _interaction_strength_matrix(interaction_values, include_diagonal=True)
    ranking = _interaction_ranking(interaction_values)

    assert strength.shape == (3, 3)
    assert strength[0, 0] == np.mean(np.abs(interaction_values[:, 0, 0]))
    assert strength[1, 1] == np.mean(np.abs(interaction_values[:, 1, 1]))
    assert ranking[0]["i"] == 1.0
    assert ranking[0]["j"] == 2.0
    assert ranking[0]["strength"] == np.mean(np.abs(interaction_values[:, 1, 2]))
