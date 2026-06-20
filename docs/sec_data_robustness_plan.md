# SEC Data Robustness Plan

This document captures the June 2026 incident around non-replayable monthly
legacy runs and large fundamental data drift. It is intentionally operational:
an unknown agent should be able to understand what failed, what is now fixed,
and what still needs to be made robust.

## Incident Summary

The 2026-05 monthly portfolio could not be perfectly replayed from the artifacts
available before this fix. The issue is not acceptable for production research:
if the same calculation date cannot be replayed with the exact historical data
package, historical backtests and month-end decisions are not auditable.

Observed portfolio drift:

- 2026-05-31 historical monthly portfolio included `KEYS.US`.
- The 2026-06-07 rerun at the same viewed month dropped `KEYS.US`.
- Direct selection reason: `KEYS.US` was selected only by `Legacy_Optuna_11`.
  The selected slot changed from `104.0-16.0-30|asset=6|sector=2` to
  `201.0-87.0-30|asset=5|sector=2`, reducing the selected asset count from 6 to
  5.

Observed input drift between the older state used by the 2026-05-31 run and the
newer open-source package:

- `US_Finalprice`: 408 common rows changed, 3271 rows added.
- `US_Balance_sheet`: 616 common rows changed, concentrated in
  `commonStockSharesOutstanding`; 1 row added.
- `US_Earnings`: 2444 common rows changed.
  - `epsActual`: 1172 changed rows.
  - `epsEstimate`: 1654 changed rows.
  - `epsDifference`: 1821 changed rows.
  - `surprisePercent`: 2433 changed rows.
- `SP500Price`: 1 common row changed, 5 rows added.
- `US_Income_statement` and `US_Cash_flow`: 1 row added each, no common-row
  value drift observed in that comparison.

The correct conclusion is not "the model is unstable". The correct conclusion is
that the previous run package was not fully retained and the data publication
layer allowed too much historical mutability without a hard gate.

## What Is Fixed Now

Monthly legacy runs now compute from a retained input package:

- `scripts/run_legacy.py` copies every required model input into
  `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/` before loading any
  data.
- `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json` now records:
  - `input_snapshot_dir`
  - original `source_data_dir`
  - `run_config.source_input_files`
  - `run_config.source_input_sha256`
  - `code_context.git_head`
  - `code_context.git_dirty`
  - `code_context.critical_file_sha256`
  - Python and package versions for core runtime dependencies
  - open-source run id agreement fields
  - active-output versus published-snapshot hash agreement fields
- `--open-source-run-id` resolution now prefers
  `data/open_source/history/output/open_source_output_*/` before the mutable
  `data/open_source/output`.
- `scripts/validate_legacy_replay_package.py` now fails when open-source run ids
  disagree or when the source package differs from the ingestion manifest's
  published output snapshot.

This fixes the retainability of future legacy monthly runs. It does not
magically make older runs complete if they lack `input_snapshot/`.

## Current Data Risk

The fundamental drift is too large to accept as routine noise. It has at least
four separate sources of risk.

### 1. Mixed-Source Open-Source Fundamentals

The current `data/open_source/output` package is multi-source:

1. SEC companyfacts
2. SEC filing-level XBRL
3. SimFin
4. yfinance

That is useful for coverage, but it is not the right default for production
backtest integrity unless every selected value is locked, lineage-visible, and
drift-gated.

### 2. Yahoo Earnings Are Mutable And Not SEC-Only

`US_Earnings.parquet` still carries market-facing fields:

- `epsActual`
- `epsEstimate`
- `epsDifference`
- `surprisePercent`

`epsEstimate` and `surprisePercent` are not SEC facts. Large historical changes
in those columns should be expected unless the raw vendor responses are frozen
and versioned. They must not be treated as stable SEC fundamentals.

### 3. Shares Can Be Rewritten By Derived Logic

The SEC-only contract explicitly forbids recalculating `outstanding_shares` from
`net_income / epsActual`. The drift in `commonStockSharesOutstanding` is a red
flag because the model uses market-cap and PE filters downstream. A historical
share-count rewrite can change the eligible universe and the final portfolio.

### 4. Legacy Export Logic Can Rewrite History

There are current code changes in
`src/alpharank/data/open_source/legacy_export.py` that alter legacy-facing
normalization:

- statement dates moved from nearest month-end logic to quarter-end logic
- earnings period-end normalization changed
- earnings legacy rows with all EPS fields null are filtered out

These may be valid improvements, but they are schema/semantic changes. They must
be versioned and measured before their outputs are allowed to replace a
production history.

## Required Gates Before Trusting A New Package

### P0 - Blockers

- Publish gate: before replacing `data/open_source/output`, compare the new
  package against the previous published package and fail if common historical
  rows drift outside an explicit allowed window.
- History gate: after publication, historize the final package under
  `data/open_source/history/output/open_source_output_*/`; this is now covered
  by `tests/test_open_source_publishing.py`.
- Lineage gate: fail publication if `snapshot_manifest.json`, copied lineage
  `manifest.json`, and ingestion run manifest disagree on run id. The publish
  order now prevents stale lineage manifests in newly retained snapshots; a
  stricter ingest-time failure gate is still needed.
- Legacy export gate: any change to date normalization, natural keys, or derived
  field semantics must write a transformation version into the manifest and
  produce a before/after drift report.
- Production monthly gate: no monthly portfolio is clean unless it has
  `input_snapshot/`, manifest hashes, and code hashes.

### P1 - SEC-First Production Package

- Use `data/sec/output` as the production fundamental source for model features
  that are intended to be SEC-stable.
- Keep `epsEstimate`, `epsDifference`, and `surprisePercent` out of the SEC-only
  production feature set unless a frozen non-SEC vendor snapshot is explicitly
  accepted for that experiment.
- For `commonStockSharesOutstanding`, use SEC reported shares only, with lineage
  to `companyfacts` or filing-level XBRL. Do not derive final shares from
  `net_income / epsActual`.
- Store candidate-level lineage next to the legacy-compatible exported files, not
  only in internal target tables.

### P2 - Diagnostics And Monitoring

- Add a monthly drift report comparing:
  - current output vs previous output
  - current output vs latest retained monthly `input_snapshot/`
  - SEC-only output vs mixed open-source output
- Split drift by source and transformation:
  - raw source value changed
  - candidate selection changed
  - date/key normalization changed
  - legacy export filter changed
- Add ticker-level explanations for changed production selections, starting with
  dropped/added portfolio names.

## Immediate Investigation Checklist

When a future monthly run changes a prior month:

1. Compare the two run manifests first. If either run lacks `input_snapshot/`,
   say the run is not fully replayable.
2. Compare file hashes from `run_config.source_input_sha256` and the manifest
   dataset hashes.
3. Compare common rows by natural key for each input file.
4. For fundamentals, join changed rows to candidate lineage and identify the
   selected source before and after.
5. For portfolio changes, identify the exact model slot and upstream eligibility
   change before discussing performance or robustness.

## Status

Future legacy monthly runs are now retainable at the model-input level.

The open-source publication layer still needs the P0 drift, history, lineage,
and legacy-export gates above before historical backtests can be treated as
production-grade again.
