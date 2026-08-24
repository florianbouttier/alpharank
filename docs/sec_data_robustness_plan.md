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
  by `tests/integration/publishing/test_open_source_publishing.py`.
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

### 2026-08-11 revision audit

The reusable command `scripts/audit_open_source_snapshot_revisions.py` now
compares all Legacy inputs by natural key and then measures downstream Legacy
and Boosting replay impact. Between retained snapshots `20260811_003849` and
`20260811_001503`, it found 136 added price rows and these common-row revisions:

- one UA volume value on 2026-08-07;
- FLR operating income for 2025 Q2;
- AKAM free cash flow for 2026 Q1;
- four Yahoo-sourced earnings rows for FFIV, IPG, KBH, and NVDA from 2005-2007.

The last four are direct evidence that current Yahoo earnings responses can
rewrite historical `epsActual`, estimates, and surprise values. Immutable
snapshots make the revision measurable, but a backtest fed with today's Yahoo
history is not thereby point-in-time. For this comparison, Legacy has no
selection/return change above `2.22e-16` and all 81,267 standard Boosting
predictions are identical. This is an impact result, not permission to remove
the SEC-first P1 work or the missing P0 publish-time historical-drift gate.

### 2026-08-13 source-refresh and storage correction

The August investigation found a second root cause: mutable SEC payloads were
served indefinitely from `_cache/`. A snapshot published in August could thus
reuse June `companyfacts` or submissions while prices were current. The folder
timestamp was not evidence of source freshness.

The production contract now:

- refetches full available Yahoo price history for the latest active universe and SPY on every full ingestion, while retaining inactive histories;
- refetches complete SEC companyfacts and applies all history from `start_date`;
- refetches mutable SEC submissions;
- refreshes StockAnalysis full-history and SimFin bulk fallbacks;
- fetches immutable accession-level filing XBRL only when needed and does not
  persist those payloads;
- records `source_refresh_contract` and separate observed dates in
  `data_freshness`;
- fails before publication when price/benchmark, SEC filing, or membership
  freshness is outside the documented limits;
- requires current network coverage, SEC mapping, submissions, and companyfacts
  success for every active constituent;
- derives financials and earnings actuals from one companyfacts payload per
  company, then releases the payload to keep memory bounded;
- limits filing-level XML fallback to active issuers with no recognized
  companyfacts row for the year, and reuses one Yahoo quarterly fetch across
  refreshed years;
- marks partial repair snapshots non-production and makes the Legacy replay
  validator reject them.
- runs one shared historical-revision gate on all full and partial ingestion
  paths before publication; rows older than 730 days block by default, and the
  report is retained with the run.

All download caches were removed on 2026-08-13. This did not remove
`official/raw`, run deltas, published output snapshots, or monthly input
snapshots. Normalized historical reconstruction and all existing replays remain
available; only the original HTTP payload bodies are no longer retained.

Snapshot history remains directly replayable. Exact duplicate files are now
deduplicated with verified APFS copy-on-write clones via
`scripts/open_source/compact_output_history.py`. The first compaction replaced
1,343 exact duplicates (3,560,991,710 logical bytes) and recovered about 3.3
GiB physically without changing paths, bytes, or hashes.

The first fully fresh candidate, run `20260813_071802` / snapshot
`open_source_output_20260813_075926`, exposed why that gate is mandatory. The
full path had not called the guard before publishing. A reconstruction against
the clean `open_source_output_20260811_014746` snapshot found old-row revisions
in all five guarded datasets: income `+4603/-3032/1163 changed`, balance
`+1893/-1757/6558`, cash flow `+15490/-2369/382`, shares
`+2710/-362/6586`, and earnings `+1028/-220/145`. The candidate and normalized
store are quarantined; production was restored byte-identically to run
`20260811_001503`. The full path now uses the same tested guard helper as both
repair paths, before any publication call.

### 2026-08-13 downstream replay impact

The quarantined candidate was replayed only as a data-revision diagnostic
against production snapshot `open_source_output_20260811_014746`. The exact
machine-readable comparison is
`outputs/data_revision_portfolios_20260813/data_revision_audit.json`.

- Legacy uses run `20260812_171646` for the old snapshot and diagnostic run
  `20260813_154243` for the candidate, with the same 30-trial configuration,
  exclusions, and liquidity policy.
- Legacy `Combined_Frequency` changes 2 of 6 names for the June decision and
  1 of 8 names for the July decision, with two additional weight changes.
- The PE/market-cap eligibility layer contains 76,627 old rows and 78,734 new
  rows: 75,810 keys are common, 2,924 are added, and 817 are removed.
- The complete Legacy holdings histories are materially different: 992 keys
  are added, 1,882 removed, and 2,970 common rows changed.
- Boosting uses the same 81,267 prediction keys in both replays, but every raw
  score changes. Mean absolute score drift is `0.01084998`, maximum drift is
  `0.30761658`.
- Boosting's causal Legacy-winner EMA catalogue changes from 39 to 38 pairs;
  only `(65, 163)` and `(95, 72)` remain common. This is the main indirect
  propagation path from revised Legacy fundamentals into the otherwise
  price/EMA-only public Boosting profile.
- Price history is also revised. Through July 2026, 413 comparable monthly
  returns move by more than 10 bps and 60 by more than 100 bps. The largest
  differences expose stale or incomplete split adjustments in the old
  snapshot, so the incident is not fundamental-only.
- Constituents and the monthly price-eligibility result are identical for the
  June and July decisions. They do not explain the four requested portfolio
  differences.

The old Boosting run was reproduced byte-for-byte (`81,267/81,267`
predictions identical), excluding random seeds as the source of instability.
The candidate remains quarantined: corrected price rows do not make its mass
fundamental rewrite acceptable for production without a reviewed migration.

### 2026-08-14 exhaustive T1/T2 price-vintage audit

The exact stock-price comparison is retained under
`outputs/price_revision_audit_20260814_t1_t2/price_revision_audit.html`, with
all changed rows in `price_changes_exhaustive.parquet`, all changed monthly
returns in `monthly_return_changes.parquet`, and a 503-ticker reconciliation in
`price_changes_by_ticker.parquet`. The reusable entrypoint is
`scripts/open_source/audit_price_snapshot_vintages.py`.

The comparison contract is fixed:

- T1 is production run `20260811_001503`, ingested at
  `2026-08-11T00:15:03Z` and published as
  `open_source_output_20260811_014746`. Its last price date is 2026-08-10 and
  it refreshed only the 2026-07-31 to 2026-08-11 window.
- T2 is quarantined run `20260813_071802`, ingested at
  `2026-08-13T07:18:02Z` and published as
  `open_source_output_20260813_075926`. Its last price date is 2026-08-12 and
  it refreshed the complete active-universe history from 2005-01-01.
- The audit cutoff is 2026-08-10, the maximum stock-price date in T1. No T2-only
  future date participates in the comparison.

Across 3,723,299 common stock/date keys, 2,056,209 rows change and every change
is assigned to one of four reconciled categories:

- 2,013,694 rows are cumulative Yahoo `adjusted_close` factor restatements on
  407 tickers, with raw OHLCV unchanged;
- 36,422 rows on eight tickers repair a full-history corporate-action seam:
  CRWD, CVNA, DD, FDX, HON, KLAC, MNST, and SPGI;
- 667 rows on 494 tickers revise raw bars only on 2026-08-07 or 2026-08-10,
  consistent with Yahoo finalizing the most recent prices or volumes after the
  early T1 download;
- 5,426 EA rows are a provider rewrite with no supporting split event. Seven EA
  price rows from 2026-07-20 through 2026-07-28 disappear in T2, while the
  2026-08-10 row appears with zero volume. This remains a data regression and
  must not be normalized as a legitimate corporate-action correction.

Two additional T1 keys, CPRT and CPT on 2026-07-03, are removed; both rows were
entirely null holiday placeholders, not observed prices. One MNST row on
2026-08-10 becomes null around its 2026-08-11 2-for-1 split.

The root cause is the old rolling-window update policy. Yahoo retroactively
restates full raw history after split-like actions and retroactively restates
the cumulative adjusted-close series after dividends. T1 joined a newly
adjusted tail to an older prefix, creating false seam returns. T2's full-history
download corrects those seams but also exposes mutable-provider revisions such
as EA. Exact-row churn therefore must not be interpreted as equivalent return
churn: 97,783 monthly returns differ exactly, 414 differ by more than 10 bps,
55 by more than 1 percentage point, and six by more than 5 points. The six
largest cases are all explained by the eight corporate actions.

Daily-return reconciliation uses the same formula as Legacy,
`adjusted_close_t / adjusted_close_t-1 - 1`, independently inside each
snapshot. The exhaustive output is
`outputs/price_revision_audit_20260814_t1_t2/daily_adjusted_return_changes.parquet`.
Although 2,037,705 daily returns differ exactly because Yahoo adjustment-factor
precision changed, only 14,882 differ by more than 0.01 bp, 699 by more than
1 bp, 495 by more than 10 bps, 108 by more than 1 percentage point, and eight
by more than 5 points. Eleven rows change return availability because one side
is null. The >10 bps set contains 395 adjustment-factor seam rows, 80 EA rows,
14 corporate-action repair rows, and six recent-bar revisions.

The 2026-08-15 lineage-level drill-down separates a legitimate cumulative
adjustment from an ingestion seam. Of the 495 daily-return changes above
10 bps, 407 fall exactly on a T1 boundary between two different price
ingestion runs; this includes 393 ordinary dividend-adjustment seams and the
14 already identified corporate-action repair rows. Another 80 are EA, six
are recent-bar settlements, and two are isolated AMCR provider restatements.
At the 1 bp threshold, 438 of 699 rows fall on a T1 ingestion boundary.

MSFT is the representative case. T1 rows through 2026-03-23 came from run
`20260503_201459`; rows from 2026-03-24 came from `20260530_231337`. Yahoo
records a $0.91 Microsoft dividend with ex-date 2026-05-21. The second run
knew that dividend and adjusted its pre-ex-date tail, while the older prefix
did not. T1 therefore created a false 2026-03-24 return seam: -2.8892% versus
-2.6789% after the coherent full refresh, a 21.03 bp difference. T2 changes
5,338 MSFT adjusted-close levels by an almost constant -0.2161%; only that one
daily return exceeds 1 bp, and March's monthly return changes by 20.37 bps.
The economic event is the May ex-dividend date; March 24 is only the rolling
ingestion boundary. The exhaustive daily artifact now carries previous and
current ingestion run ids plus explicit T1/T2 vintage-seam flags.

### 2026-08-14 frozen EODHD price-seed gap

The intended price architecture is hybrid: the cancelled EODHD subscription's
retained archive is the immutable historical seed for former constituents and
delisted names, while open sources extend downloadable securities. Incremental
ingestion was intended to preserve this base, not to rebuild the entire past
from Yahoo.

That contract is not met by the current official open package. The frozen
EODHD file has 6,254,372 rows / 835 tickers; the active open output has
3,723,301 rows / 732 tickers and no EODHD source in lineage. After normalizing
the `BF-B`/`BF.B` and `BRK-B`/`BRK.B` aliases, 110 historical constituents and
419,656 rows are absent. The 2005+ gap is 108 tickers / 271,385 rows, and the
2010+ gap is 104 tickers / 168,242 rows. Exact evidence is retained under
`outputs/eodhd_price_seed_audit_20260814/`.

This is a P0 historical-universe issue. Active-universe freshness and retained
open-source inactive histories cannot be presented as full Legacy coverage.
The migration must select the frozen EODHD archive as an immutable canonical
source with explicit `eodhd_frozen_history` lineage, preserve ticker aliases,
and use an audited vendor-transition policy before the open package can replace
the Legacy price base. The source parquet need not be duplicated into mutable
`official/raw/`.

### 2026-08-15 hybrid price candidate and fail-closed gates

The canonical price domain is now implemented under
`src/alpharank/data/prices/` and called by full ingestion before publication.
It hashes and normalizes the immutable EODHD seed, selects one full Yahoo
vintage for all current members, reconstructs recent inactive tails from daily
returns computed within immutable run vintages, rejects tails after a gap over
10 calendar days, and writes exhaustive revision artifacts. Routine thresholds
are 1 bp for historical daily-return revisions and 1 bp for adjustment-factor
transition jumps; both historical-return and historical-key-removal overrides
default to false.

The non-published reviewed candidate is
`outputs/hybrid_price_candidate_20260815_final/`, built from Yahoo run
`20260813_071802` and EODHD seed SHA-256
`0ee4b6d9766fef6942f12bb1591426302b29e19347fb32c87cb04e6777b3f8f5`.
It has 3,708,691 valid rows / 840 tickers, complete single-vintage coverage for
503 active tickers, 143 return-ledger extensions, zero adjustment-transition
findings, and no missing inactive EODHD key. Seven long-gap symbol tails are
rejected; two recent tails (BK and SATS) remain unextended because archived
vintages do not provide every daily return.

The migration review records 23,885 old daily-return differences above 1 bp,
45,739 return-availability changes, and 7,257 removed non-null keys across 43
tickers. The removal list is exhaustive in
`audit/price_historical_key_removals.parquet`; the largest blocks are implausible
post-delisting/reused-symbol series. Both migration overrides were required and
are recorded. A second build from identical inputs produced byte-identical
price, lineage, and daily-revision parquets.

The candidate has not replaced `data/open_source/output`: price correctness does
not waive the separate SEC-only fundamental contract. Promotion must occur only
inside a full composed snapshot that passes both price gates and the SEC-only
fundamental guards.

### 2026-08-14 clarified production source boundary

Official fundamentals are SEC/GAAP only. Allowed final values are SEC
companyfacts, filing-level SEC XBRL, or explicitly labelled derivations using
only SEC facts. EODHD may bridge a historical ticker to a CIK but may not supply
a fundamental value; Yahoo, SimFin, and StockAnalysis fundamental fallbacks are
also excluded from official snapshots.

Prices follow a separate hybrid contract. The frozen EODHD archive remains the
historical seed for delisted/former constituents, while open sources extend
downloadable securities. New split or dividend evidence is applied as a
versioned correction overlay to a new derived snapshot. The frozen EODHD file
and all prior snapshots remain byte-immutable. Splits can restate pre-event
OHLC/inverse volume under the canonical adjustment convention; dividends alter
adjusted/total-return values only, never raw OHLCV.

The retained T1 snapshot is non-compliant with this clarified boundary. Its
consolidated fundamental output selected 406,952 SEC values and 19,243 non-SEC
values (14,754 Yahoo, 4,489 SimFin), and Legacy consumed those files directly.
T1/T2 are retained mixed-contract replay evidence, not current production
truth.

### 2026-08-16 full refresh and causal SEC migration

Full network ingestion `20260816_103942` reached 2026-08-14 for both market
prices and SEC filing dates. The first two price attempts failed closed on
MNST's 2026-08-11 2-for-1 split. The split detector now audits the fresh Yahoo
vintage independently of old retained rows, and the reviewed issuer event is
stored in `configs/data_quality/confirmed_corporate_actions.json`.

The SEC investigation found the root cause of the large historical drift:
the raw Companyfacts upsert key omitted `filing_date`, so later restatements
replaced earlier filing versions. The fixed raw package contains 509,254 rows
and 34,111 fact groups with multiple filing versions. Model exports select the
earliest filing version for causal availability while retaining every version
in raw storage. Relative to the May SEC package, common-row changes are
dominated by outstanding shares (24,649 balance-sheet rows and 23,210 share
rows), followed by revenue (1,705) and net income (1,122). This one-time
migration is not silent: all five datasets triggered the 730-day guard and the
approval note is embedded in the package manifest.

The final price roll-forward is
`outputs/production_refresh_20260816/price_package_roll_forward_v2`. It
preserves 1,232,112 rows for 338 inactive/terminal tickers, refreshes 2,477,583
rows for 502 active tickers from the single Yahoo vintage, and carries EA only
because the official registry records its 2026-08-05 removal. Its strict gate
passes with no override: zero historical keys removed, zero historical return
availability changes, zero daily-return revisions above 1 bp, and zero source
transition findings.

The price and SEC packages are composed and hash-validated under
`outputs/production_refresh_20260816/composed_history/alpharank_input_20260816_115416_2a01288bab06`.
Legacy and Boosting must both consume the Legacy run's retained copy of this
exact snapshot before their performance can be compared.

## Historical identity reconstruction gate

`scripts/open_source/ingestion/reconstruct_historical_sec_companyfacts.py`
consumes the versioned historical ticker-to-CIK bridge, fetches SEC Companyfacts
and submissions, and attempts a filing-level fallback. Its output is always a
candidate. It may enter a model snapshot only after all identities pass review,
the source package is rebuilt immutably, revision guards explain every changed
historical value, and both methods are rerun on that exact new snapshot.

The 2026-08-16 audit fetched all 67 targeted identities, reconstructed 17,971
rows for 63, and left four names without usable machine-readable facts. This
closes most of the mapping gap without pretending that present-day SEC coverage
was historically knowable. Eligibility remains based on filing availability at
each decision date, never on whether a ticker can be resolved today.
