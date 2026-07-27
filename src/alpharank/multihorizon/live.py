from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from alpharank.multihorizon.data import build_research_frame
from alpharank.multihorizon.legacy_ema import (
    legacy_winning_pairs,
    point_in_time_fold_features,
)
from alpharank.multihorizon.modeling import fit_booster
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor


@dataclass(frozen=True)
class LiveAlphaConfig:
    data_dir: Path
    legacy_detailed_returns_path: Path
    output_dir: Path
    decision_month: date
    horizon: int = 6
    validation_months: int = 6
    start_month: str = "2005-01"
    positive_quantile: float = 0.9
    missing_feature_threshold: float = 0.35
    num_boost_round: int = 100
    random_seed: int = 42
    excluded_tickers: tuple[str, ...] = ()
    minimum_monthly_price_observations: int = 10
    minimum_monthly_median_dollar_volume: float = 1_000_000.0
    maximum_monthly_ohlc_violation_rate: float = 0.05
    top_n_values: tuple[int, ...] = (5, 10)


def run_live_alpha(config: LiveAlphaConfig) -> Path:
    decision_month = config.decision_month.replace(day=1)
    holding_month = _offset_months(decision_month, 1)
    label_cutoff = _offset_months(decision_month, -config.horizon)
    exact_pairs = legacy_winning_pairs(config.legacy_detailed_returns_path)
    research = build_research_frame(
        data_dir=config.data_dir,
        legacy_detailed_returns_path=config.legacy_detailed_returns_path,
        horizons=(1, config.horizon),
        start_month=config.start_month,
        excluded_tickers=config.excluded_tickers,
        relative_ema_pairs=exact_pairs,
        minimum_monthly_price_observations=config.minimum_monthly_price_observations,
        minimum_monthly_median_dollar_volume=config.minimum_monthly_median_dollar_volume,
        maximum_monthly_ohlc_violation_rate=config.maximum_monthly_ohlc_violation_rate,
    )
    frame = research.frame
    score_frame = frame.filter(pl.col("decision_month") == decision_month)
    if score_frame.is_empty():
        available = frame.select(pl.col("decision_month").max()).item()
        raise ValueError(
            f"No eligible live rows exist for decision_month={decision_month}; "
            f"latest available month is {available}."
        )
    score_frame, holding_universe_audit = _filter_holding_month_universe(
        score_frame,
        constituents_path=config.data_dir / "SP500_Constituents.csv",
        holding_month=holding_month,
    )

    labeled = (
        frame.filter(
            (pl.col("decision_month") <= label_cutoff)
            & pl.col(f"future_excess_return_{config.horizon}m").is_not_null()
        )
        .sort(["decision_month", "ticker"])
    )
    labeled_months = labeled["decision_month"].unique().sort().to_list()
    if len(labeled_months) <= config.validation_months:
        raise ValueError("Not enough mature labeled months for live training and validation.")
    validation_months = labeled_months[-config.validation_months :]
    train_months = labeled_months[: -config.validation_months]
    train = labeled.filter(pl.col("decision_month").is_in(train_months))
    validation = labeled.filter(pl.col("decision_month").is_in(validation_months))
    train_cutoff = max(train_months)

    fold_features, fold_pairs = point_in_time_fold_features(
        all_features=research.feature_columns,
        legacy_path=config.legacy_detailed_returns_path,
        train_decision_cutoff=train_cutoff,
        include_non_relative_features=False,
    )
    preprocessor = fit_fold_preprocessor(
        train,
        fold_features,
        max_missing_ratio=config.missing_feature_threshold,
    )
    _, X_train = preprocessor.transform(train)
    _, X_validation = preprocessor.transform(validation)
    score_transformed, X_score = preprocessor.transform(score_frame)
    fitted = fit_booster(
        method="classification",
        horizon=config.horizon,
        train_frame=train,
        validation_frame=validation,
        X_train=X_train,
        X_validation=X_validation,
        features=preprocessor.features,
        positive_quantile=config.positive_quantile,
        seed=config.random_seed,
        num_boost_round=config.num_boost_round,
        params=None,
    )
    validation_raw = fitted.predict_raw_score(X_validation)
    validation_probability = fitted.predict(X_validation)
    live_raw = fitted.predict_raw_score(X_score)
    live_probability = fitted.predict(X_score)
    live_scores = (
        score_transformed.select(
            "decision_month",
            "holding_month",
            "decision_asof_date",
            "ticker",
            "last_close",
        )
        .with_columns(
            pl.Series("raw_score", live_raw),
            pl.Series("calibrated_probability", live_probability),
        )
        .sort(["raw_score", "ticker"], descending=[True, False])
        .with_row_index("rank", offset=1)
        .with_columns(
            *[
                (pl.col("rank") <= top_n).alias(f"selected_top{top_n}")
                for top_n in config.top_n_values
            ]
        )
    )
    validation_metrics = _validation_metrics(
        validation,
        horizon=config.horizon,
        positive_quantile=config.positive_quantile,
        raw_score=validation_raw,
        calibrated_probability=validation_probability,
    )
    legacy_live = _legacy_basket(
        config.legacy_detailed_returns_path,
        holding_month=holding_month,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = config.output_dir / "html"
    html_dir.mkdir(exist_ok=True)
    live_scores.write_parquet(config.output_dir / "live_scores.parquet")
    for top_n in config.top_n_values:
        (
            live_scores.filter(pl.col(f"selected_top{top_n}"))
            .with_columns(pl.lit(1.0 / top_n).alias("equal_weight"))
            .write_csv(config.output_dir / f"portfolio_top{top_n}.csv")
        )
    legacy_live.write_csv(config.output_dir / "legacy_portfolio_same_holding_month.csv")
    (config.output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "features": preprocessor.features,
                "global_medians": preprocessor.global_medians,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    calibration = {
        "method": "isotonic" if fitted.calibrator is not None else "none",
        "x_thresholds": (
            fitted.calibrator.X_thresholds_.tolist()
            if fitted.calibrator is not None
            else []
        ),
        "y_thresholds": (
            fitted.calibrator.y_thresholds_.tolist()
            if fitted.calibrator is not None
            else []
        ),
    }
    (config.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )
    fitted.model.model_.save_model(config.output_dir / "xgboost_model.ubj")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "production_candidate",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "decision_month": decision_month.isoformat(),
        "holding_month": holding_month.isoformat(),
        "label_maturity_cutoff": label_cutoff.isoformat(),
        "train_start": min(train_months).isoformat(),
        "train_end": max(train_months).isoformat(),
        "validation_start": min(validation_months).isoformat(),
        "validation_end": max(validation_months).isoformat(),
        "train_rows": train.height,
        "validation_rows": validation.height,
        "live_universe_rows": score_frame.height,
        "holding_month_universe_filter": holding_universe_audit,
        "winner_pair_count": len(fold_pairs),
        "winner_pairs": fold_pairs,
        "feature_count": len(preprocessor.features),
        "features": preprocessor.features,
        "validation_metrics": validation_metrics,
        "legacy_same_month_available": not legacy_live.is_empty(),
        "input_paths": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in research.input_paths.items()
        },
        "legacy_detailed_returns": {
            "path": str(config.legacy_detailed_returns_path),
            "sha256": _sha256(config.legacy_detailed_returns_path),
        },
        "protocol": {
            "features": "exact Legacy-winning relative EMA pairs only",
            "target": (
                f"top {int((1.0 - config.positive_quantile) * 100)}% cross-sectional "
                f"S&P 500 excess return over {config.horizon} months"
            ),
            "label_maturity": (
                "training labels end at decision_month - horizon; the partial current "
                "calendar month is never used as a mature target"
            ),
            "validation": (
                "last mature labeled months, strictly after training; used for early "
                "stopping and isotonic calibration, therefore not reported as sealed test"
            ),
            "ranking": "descending raw XGBoost score; calibrated probability is explanatory only",
            "portfolio": "top 5 is the retained champion; top 10 remains a diagnostic because its promotion gates failed",
            "live_universe": (
                "decision-month eligible rows intersected with the known holding-month "
                "S&P 500 membership when that snapshot is available"
            ),
        },
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_live_html(
        html_dir / "live_alpha_portfolio.html",
        manifest=manifest,
        live_scores=live_scores,
        legacy_live=legacy_live,
    )
    return config.output_dir


def previous_completed_month(as_of_date: date) -> date:
    return _offset_months(as_of_date.replace(day=1), -1)


def _validation_metrics(
    frame: pl.DataFrame,
    *,
    horizon: int,
    positive_quantile: float,
    raw_score: np.ndarray,
    calibrated_probability: np.ndarray,
) -> dict[str, float]:
    rank = frame[f"future_excess_rank_{horizon}m"].to_numpy()
    target = (rank >= positive_quantile).astype(np.int8)
    prevalence = float(target.mean())
    return {
        "rows": float(len(target)),
        "positive_rate": prevalence,
        "roc_auc": float(roc_auc_score(target, raw_score)),
        "pr_auc_average_precision": float(average_precision_score(target, raw_score)),
        "pr_auc_lift_vs_prevalence": float(
            average_precision_score(target, raw_score) / prevalence
        ),
        "brier": float(brier_score_loss(target, calibrated_probability)),
        "log_loss": float(
            log_loss(
                target,
                np.clip(calibrated_probability, 1e-8, 1.0 - 1e-8),
            )
        ),
    }


def _legacy_basket(path: Path, *, holding_month: date) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    return (
        frame.filter(
            (pl.col("portfolio_model") == "Combined_Frequency")
            & (pl.col("year_month").cast(pl.Date) == holding_month)
        )
        .select(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            "ticker",
            "n_models",
            "weight_normalized",
        )
        .unique(["holding_month", "ticker"])
        .sort(["weight_normalized", "ticker"], descending=[True, False])
    )


def _filter_holding_month_universe(
    frame: pl.DataFrame,
    *,
    constituents_path: Path,
    holding_month: date,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if not constituents_path.exists():
        return frame, {
            "applied": False,
            "reason": "constituent file missing",
            "holding_month": holding_month.isoformat(),
            "rows_before": frame.height,
            "rows_after": frame.height,
            "removed_tickers": [],
        }
    constituents = (
        pl.read_csv(constituents_path, try_parse_dates=True)
        .filter(pl.col("Date").cast(pl.Date) == holding_month)
        .select(
            pl.col("Ticker").cast(pl.String).alias("ticker")
        )
        .unique()
    )
    if constituents.is_empty():
        return frame, {
            "applied": False,
            "reason": "holding-month snapshot unavailable",
            "holding_month": holding_month.isoformat(),
            "rows_before": frame.height,
            "rows_after": frame.height,
            "removed_tickers": [],
        }
    source_tickers = constituents["ticker"].to_list()
    eligible = sorted(
        {
            f"{ticker}.US"
            for source in source_tickers
            for ticker in (source, source.replace(".", "-"), source.replace("-", "."))
        }
    )
    removed = (
        frame.filter(~pl.col("ticker").is_in(eligible))
        .select("ticker")
        .unique()
        .sort("ticker")
        .to_series()
        .to_list()
    )
    filtered = frame.filter(pl.col("ticker").is_in(eligible))
    return filtered, {
        "applied": True,
        "reason": "known holding-month membership",
        "holding_month": holding_month.isoformat(),
        "rows_before": frame.height,
        "rows_after": filtered.height,
        "removed_tickers": removed,
    }


def _offset_months(value: date, months: int) -> date:
    absolute = value.year * 12 + value.month - 1 + months
    return date(absolute // 12, absolute % 12 + 1, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_live_html(
    path: Path,
    *,
    manifest: dict[str, Any],
    live_scores: pl.DataFrame,
    legacy_live: pl.DataFrame,
) -> None:
    top_rows = "".join(
        "<tr>"
        f"<td>{int(row['rank'])}</td><td>{html.escape(str(row['ticker']))}</td>"
        f"<td>{html.escape(str(row['decision_asof_date']))}</td>"
        f"<td>{float(row['last_close']):.2f}</td><td>{float(row['raw_score']):.4f}</td>"
        f"<td>{100.0 * float(row['calibrated_probability']):.1f}%</td>"
        f"<td>{'oui' if row.get('selected_top5') else ''}</td>"
        f"<td>{'oui' if row.get('selected_top10') else ''}</td>"
        "</tr>"
        for row in live_scores.head(20).to_dicts()
    )
    legacy_rows = (
        "".join(
            "<tr>"
            f"<td>{html.escape(str(row['ticker']))}</td>"
            f"<td>{int(row['n_models'])}</td>"
            f"<td>{100.0 * float(row['weight_normalized']):.1f}%</td>"
            "</tr>"
            for row in legacy_live.to_dicts()
        )
        or '<tr><td colspan="3">Aucun portefeuille Legacy pour ce même mois dans le fichier fourni.</td></tr>'
    )
    metrics = manifest["validation_metrics"]
    path.write_text(
        f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portefeuille Alpha live</title><style>
body{{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:24px}}
main{{max-width:1180px;margin:auto}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #dde3ec;border-radius:14px;padding:18px;margin:14px 0}}
.metric{{font-size:28px;font-weight:700}} table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:9px;border-bottom:1px solid #e7ebf1;text-align:left}} .warn{{background:#fff4d8;color:#704600;padding:12px;border-radius:8px}}
code{{word-break:break-all}}</style></head><body><main>
<h1>Portefeuille Alpha live — décision {html.escape(str(manifest['decision_month']))}</h1>
<p>Portefeuille détenu en {html.escape(str(manifest['holding_month']))}. Prix de décision : dernière clôture ajustée disponible du mois de décision.</p>
<div class="grid">
<div class="card"><div>Univers éligible</div><div class="metric">{manifest['live_universe_rows']}</div></div>
<div class="card"><div>ROC AUC validation</div><div class="metric">{metrics['roc_auc']:.3f}</div></div>
<div class="card"><div>PR AUC validation</div><div class="metric">{metrics['pr_auc_average_precision']:.3f}</div></div>
<div class="card"><div>Lift PR / prévalence</div><div class="metric">{metrics['pr_auc_lift_vs_prevalence']:.2f}×</div></div>
</div>
<div class="card"><h2>Classement Alpha</h2>
<p class="warn">Top 5 = portefeuille champion retenu. Top 10 = diagnostic, non promu lors du test historique. La probabilité isotone aide à interpréter le score ; l'ordre est déterminé par le score brut.</p>
<table><thead><tr><th>Rang</th><th>Ticker</th><th>Date du prix</th><th>Prix ajusté</th><th>Score</th><th>Probabilité</th><th>Top 5</th><th>Top 10</th></tr></thead><tbody>{top_rows}</tbody></table></div>
<div class="card"><h2>Legacy — même mois de détention</h2>
<table><thead><tr><th>Ticker</th><th>Votes</th><th>Poids</th></tr></thead><tbody>{legacy_rows}</tbody></table></div>
<div class="card"><h2>Méthode et anti-fuite</h2>
<ul><li>Classification XGBoost, horizon 6 mois, cible = décile supérieur de surperformance face au S&amp;P 500.</li>
<li>Entrées : uniquement les ratios EMA relatifs action / S&amp;P 500 qui ont historiquement gagné dans Legacy, avec rangs et z-scores mensuels.</li>
<li>Entraînement : {manifest['train_start']} → {manifest['train_end']}; validation/calibration : {manifest['validation_start']} → {manifest['validation_end']}.</li>
<li>Les labels s'arrêtent à {manifest['label_maturity_cutoff']}; juillet partiel ne devient jamais une vérité future.</li>
<li>L'univers éligible du mois de décision est intersecté avec les membres connus du mois de détention ; les sorties d'indice connues ne peuvent pas entrer dans le panier live.</li>
<li>La validation sert à l'arrêt anticipé et à la calibration : ce n'est pas un test scellé. Les statistiques historiques hors échantillon restent celles du rapport de recherche.</li></ul>
<p><strong>Features :</strong> {manifest['feature_count']} issues de {manifest['winner_pair_count']} couples EMA.</p></div>
</main></body></html>""",
        encoding="utf-8",
    )
