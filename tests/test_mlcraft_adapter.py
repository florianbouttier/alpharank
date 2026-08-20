from __future__ import annotations

import sys

from alpharank.backtest.mlcraft_adapter import (
    ensure_mlcraft_importable,
    to_mlcraft_model_and_fit_params,
    to_mlcraft_search_space,
)


def test_translates_legacy_xgboost_params_to_mlcraft_contract() -> None:
    model_params, fit_params = to_mlcraft_model_and_fit_params(
        {
            "n_estimators": 750,
            "n_jobs": -1,
            "random_state": 42,
            "objective": "binary:logistic",
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "max_depth": 4,
        }
    )

    assert fit_params == {"num_boost_round": 750}
    assert model_params["nthread"] == -1
    assert model_params["alpha"] == 0.3
    assert model_params["lambda"] == 2.0
    assert model_params["max_depth"] == 4
    assert "objective" not in model_params
    assert "random_state" not in model_params


def test_translates_legacy_optuna_space_to_mlcraft_space() -> None:
    converted = to_mlcraft_search_space(
        {
            "n_estimators": ("int", 100, 500),
            "learning_rate": ("loguniform", 0.001, 0.1),
            "reg_lambda": ("float", 0.0, 5.0),
        }
    )

    assert converted["num_boost_round"] == {"type": "int", "low": 100, "high": 500, "target": "fit"}
    assert converted["learning_rate"] == {"type": "float", "low": 0.001, "high": 0.1, "log": True}
    assert converted["lambda"] == {"type": "float", "low": 0.0, "high": 5.0}


def test_loads_local_mlcraft_without_mutating_import_path(tmp_path, monkeypatch) -> None:
    package = tmp_path / "src" / "mlcraft"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("LOCAL_FIXTURE = True\n", encoding="utf-8")
    monkeypatch.setenv("MLCRAFT_SRC", str(tmp_path / "src"))
    monkeypatch.delitem(sys.modules, "mlcraft", raising=False)
    original_path = list(sys.path)

    loaded_from = ensure_mlcraft_importable()

    assert loaded_from == tmp_path / "src"
    assert sys.path == original_path
    assert sys.modules["mlcraft"].LOCAL_FIXTURE is True
    sys.modules.pop("mlcraft", None)
