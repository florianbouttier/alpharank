PERFORMANCE_REPORT_STYLES = r"""
:root {
  --navy: #111d55;
  --navy-2: #1e2b68;
  --ink: #182033;
  --muted: #617087;
  --line: #dce3eb;
  --line-strong: #c8d2dd;
  --panel: #ffffff;
  --surface: #f8fafc;
  --soft: #eef2f6;
  --positive: #265511;
  --negative: #802331;
  --gold: #9b8816;
  --warning-bg: #fff8df;
  --warning-line: #d2a927;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--surface);
  font: 14px/1.45 "IBM Plex Sans", Inter, system-ui, -apple-system, sans-serif;
}
button, select, input { font: inherit; }
button:focus-visible, select:focus-visible, input:focus-visible, a:focus-visible {
  outline: 3px solid rgba(17, 29, 85, .2);
  outline-offset: 2px;
}
.shell { min-height: 100vh; }
.sidebar {
  position: fixed;
  inset: 0 auto 0 0;
  z-index: 10;
  width: 248px;
  padding: 24px 18px;
  color: #fff;
  background: var(--navy);
}
.brand { display: flex; gap: 11px; align-items: center; margin-bottom: 26px; }
.brand-mark {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  color: var(--navy);
  background: #fff;
  border-radius: 6px;
  font: 700 13px "IBM Plex Mono", ui-monospace, monospace;
}
.brand strong { display: block; font-size: 15px; letter-spacing: .01em; }
.brand small { color: #b8c2df; }
.nav-label {
  margin: 22px 10px 8px;
  color: #91a0ca;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.nav-link {
  display: flex;
  align-items: center;
  min-height: 38px;
  margin: 2px 0;
  padding: 8px 10px;
  color: #dce3f6;
  border-left: 2px solid transparent;
  border-radius: 3px;
  text-decoration: none;
}
.nav-link:hover { color: #fff; background: rgba(255,255,255,.08); }
.nav-link.is-active { color: #fff; border-left-color: #fff; background: rgba(255,255,255,.12); }
.sidebar-meta {
  position: absolute;
  right: 18px;
  bottom: 20px;
  left: 18px;
  padding-top: 14px;
  color: #b8c2df;
  border-top: 1px solid rgba(255,255,255,.18);
  font: 11px/1.6 "IBM Plex Mono", ui-monospace, monospace;
}
main { margin-left: 248px; }
.content { max-width: 1440px; margin: 0 auto; padding: 28px 32px 64px; }
.eyebrow, .section-kicker {
  color: var(--navy);
  font: 700 11px "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.hero { display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: start; }
h1 { margin: 4px 0 8px; color: var(--navy); font-size: clamp(28px, 4vw, 42px); line-height: 1.06; }
.hero p { max-width: 780px; margin: 0; color: var(--muted); font-size: 15px; }
.status-badge {
  max-width: 300px;
  padding: 13px 15px;
  color: #69550a;
  background: var(--warning-bg);
  border: 1px solid var(--warning-line);
  border-radius: 6px;
  font-size: 12px;
}
.toolbar {
  position: sticky;
  top: 0;
  z-index: 6;
  display: grid;
  grid-template-columns: repeat(2, minmax(150px, .75fr)) minmax(230px, 1.2fr) auto;
  gap: 12px;
  margin: 24px 0;
  padding: 13px;
  background: rgba(248,250,252,.96);
  border: 1px solid var(--line);
  box-shadow: 0 4px 16px rgba(24,32,51,.07);
}
label, .curve-control { display: grid; gap: 5px; color: var(--muted); font-size: 11px; font-weight: 650; }
.field-label { min-height: 16px; }
select, input {
  min-height: 38px;
  padding: 8px 10px;
  color: var(--ink);
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 4px;
}
.multi-select { position: relative; color: var(--ink); font-weight: 400; }
.multi-select summary {
  display: flex;
  min-height: 38px;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  cursor: pointer;
  list-style: none;
}
.multi-select summary::-webkit-details-marker { display: none; }
.multi-select summary::after { content: "▾"; color: var(--muted); }
.multi-select[open] summary::after { content: "▴"; }
.multi-select-menu {
  position: absolute;
  top: calc(100% + 6px);
  right: 0;
  z-index: 12;
  width: min(380px, 86vw);
  padding: 10px;
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 5px;
  box-shadow: 0 12px 28px rgba(24,32,51,.16);
}
.multi-select-actions { display: flex; gap: 7px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.multi-select-actions button {
  padding: 5px 8px;
  color: var(--navy);
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 3px;
  cursor: pointer;
}
.curve-options { display: grid; gap: 3px; max-height: 330px; margin-top: 8px; overflow: auto; }
.curve-option {
  display: grid;
  grid-template-columns: auto auto minmax(0,1fr);
  gap: 8px;
  align-items: center;
  min-height: 31px;
  padding: 4px 5px;
  color: var(--ink);
  border-radius: 3px;
  cursor: pointer;
}
.curve-option:hover { background: var(--soft); }
.curve-option input { min-height: 0; margin: 0; accent-color: var(--navy); }
.curve-option i, .strategy-name i, .metric-table th i {
  display: inline-block;
  width: 9px;
  height: 9px;
  flex: 0 0 auto;
  border-radius: 50%;
}
.button {
  align-self: end;
  min-height: 38px;
  padding: 8px 14px;
  color: #fff;
  background: var(--navy);
  border: 1px solid var(--navy);
  border-radius: 4px;
  cursor: pointer;
}
.button.secondary { color: var(--navy); background: #fff; }
.section { scroll-margin-top: 88px; margin-top: 38px; }
.section-head { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }
.section-head h2 { margin: 3px 0 0; color: var(--navy); font-size: 23px; }
.section-head p { max-width: 720px; margin: 0; color: var(--muted); }
.kpi-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; }
.kpi-card, .panel, .method-card, .lineage-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 7px;
  box-shadow: 0 2px 8px rgba(24,32,51,.04);
}
.kpi-card { padding: 13px; }
.kpi-card header { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 9px; }
.kpi-card header span { color: var(--navy); font-size: 12px; font-weight: 750; }
.kpi-card header small { color: var(--muted); font-size: 9px; text-transform: uppercase; }
.kpi-strategy-list { display: grid; gap: 3px; }
.kpi-strategy-row {
  display: grid;
  grid-template-columns: minmax(0,1fr) auto 48px;
  gap: 6px;
  align-items: center;
  min-height: 31px;
  padding: 4px 6px;
  border-left: 3px solid transparent;
  border-radius: 3px;
}
.strategy-name {
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  color: var(--muted);
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.strategy-name i { flex: 0 0 auto; }
.kpi-strategy-row strong, .metric-value strong { color: var(--navy); font: 700 12px "IBM Plex Mono", ui-monospace, monospace; }
.kpi-strategy-row small, .metric-value small { color: var(--muted); font: 700 8px "IBM Plex Mono", ui-monospace, monospace; text-align: right; text-transform: uppercase; }
.comparison-beats { background: rgba(38,85,17,.09); border-left-color: var(--positive); }
.comparison-trails { background: rgba(128,35,49,.07); border-left-color: var(--negative); }
.comparison-equal { background: var(--soft); border-left-color: var(--line-strong); }
.comparison-benchmark { background: rgba(155,136,22,.11); border-left-color: var(--gold); }
.chart-grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }
.panel { padding: 16px; overflow: hidden; }
.panel h3 { margin: 0 0 3px; color: var(--navy); font-size: 15px; }
.panel-subtitle { margin: 0 0 12px; color: var(--muted); font-size: 12px; }
canvas { display: block; width: 100%; height: 330px; }
.legend { display: flex; flex-wrap: wrap; gap: 7px 14px; margin-top: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; }
.legend i { width: 16px; height: 3px; border-radius: 2px; }
.table-wrap { width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { position: sticky; top: 0; z-index: 1; color: #4d5b71; background: #edf2f6; font-size: 10px; letter-spacing: .055em; text-transform: uppercase; }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover { background: #f6f8fb; }
.value-positive { color: var(--positive); }
.value-negative { color: var(--negative); }
.metric-table { min-width: 760px; }
.metric-table th:first-child, .metric-table td:first-child { position: sticky; left: 0; z-index: 2; background: #edf2f6; }
.metric-table th { min-width: 132px; white-space: normal; }
.metric-table th:first-child { min-width: 180px; }
.metric-table th i { margin-right: 5px; vertical-align: -1px; }
.metric-table .benchmark-head { color: #69550a; background: #f6edc8; }
.metric-value { border-left: 3px solid transparent; font-family: "IBM Plex Mono", ui-monospace, monospace; }
.metric-value.comparison-beats { border-left-color: var(--positive); }
.metric-value.comparison-trails { border-left-color: var(--negative); }
.metric-value.comparison-equal { border-left-color: var(--line-strong); }
.metric-value.comparison-benchmark { border-left-color: var(--gold); }
.metric-value strong, .metric-value small { display: block; }
.metric-definition { min-width: 280px; color: var(--muted); text-align: left; white-space: normal; }
.matrix-controls { display: flex; gap: 8px; margin-bottom: 12px; }
.matrix-controls button {
  padding: 7px 12px;
  color: var(--navy);
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  cursor: pointer;
}
.matrix-controls button.is-active { color: #fff; background: var(--navy); border-color: var(--navy); }
.matrix-block + .matrix-block { margin-top: 28px; padding-top: 24px; border-top: 1px solid var(--line); }
.matrix-block h3 { margin-bottom: 4px; }
.heatmap-wrap { overflow: auto; }
.heatmap { display: grid; gap: 3px; min-width: 1000px; }
.heatmap-cell, .heatmap-head, .heatmap-label {
  display: grid;
  min-height: 42px;
  place-items: center;
  border-radius: 3px;
  font: 600 11px "IBM Plex Mono", ui-monospace, monospace;
}
.heatmap-head { min-height: 28px; color: var(--muted); }
.heatmap-label { justify-items: start; padding: 0 9px; color: var(--ink); background: #edf2f6; font-family: "IBM Plex Sans", sans-serif; }
.viridis-legend { display: flex; align-items: center; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 11px; }
.viridis-bar { width: 180px; height: 9px; border-radius: 2px; background: linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725); }
.composer-layout { display: grid; grid-template-columns: minmax(0,1.6fr) minmax(280px,.7fr); gap: 12px; }
.composer-selection-head { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
.composer-actions { display: flex; flex-wrap: wrap; justify-content: end; gap: 7px; }
.composer-actions button {
  min-height: 32px;
  padding: 6px 9px;
  color: var(--navy);
  background: #fff;
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  cursor: pointer;
}
.composer-options { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 5px; margin-top: 12px; }
.composer-option {
  display: grid;
  grid-template-columns: auto auto minmax(0,1fr) auto;
  gap: 8px;
  align-items: center;
  min-height: 38px;
  padding: 6px 8px;
  color: var(--ink);
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
}
.composer-option input { min-height: 0; margin: 0; accent-color: var(--navy); }
.composer-option i { width: 9px; height: 9px; border-radius: 50%; }
.composer-option strong { color: var(--navy); font: 700 10px "IBM Plex Mono", ui-monospace, monospace; }
.composer-contract {
  padding: 16px;
  color: #5c4d0c;
  background: var(--warning-bg);
  border: 1px solid var(--warning-line);
  border-radius: 7px;
}
.composer-contract strong { color: var(--navy); }
.composer-contract p { margin: 8px 0; }
.composer-contract small { color: #756728; }
.composer-kpis { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 9px; margin-top: 12px; }
.composer-kpi {
  display: grid;
  grid-template-columns: minmax(0,1fr) auto;
  gap: 5px 10px;
  align-items: center;
  min-height: 74px;
  padding: 11px 12px;
  background: #fff;
  border: 1px solid var(--line);
  border-left: 3px solid transparent;
  border-radius: 6px;
}
.composer-kpi span { color: var(--muted); font-size: 11px; }
.composer-kpi strong { color: var(--navy); font: 700 17px "IBM Plex Mono", ui-monospace, monospace; }
.composer-kpi small { grid-column: 1 / -1; color: var(--muted); font: 700 9px "IBM Plex Mono", ui-monospace, monospace; text-transform: uppercase; }
.composer-kpi.comparison-beats { border-left-color: var(--positive); }
.composer-kpi.comparison-trails { border-left-color: var(--negative); }
.composer-kpi.comparison-equal { border-left-color: var(--line-strong); }
.composer-correlation { margin-top: 12px; }
.composer-correlation table { min-width: 620px; }
.composer-correlation th { min-width: 128px; max-width: 190px; white-space: normal; }
.composer-correlation th:first-child { min-width: 190px; }
.correlation-cell { font: 700 11px "IBM Plex Mono", ui-monospace, monospace; text-align: center; }
.correlation-diversifying { color: var(--positive); background: rgba(38,85,17,.11); }
.correlation-moderate { color: #69550a; background: rgba(155,136,22,.10); }
.correlation-high { color: var(--negative); background: rgba(128,35,49,.08); }
.correlation-unknown { color: var(--muted); background: var(--soft); }
.composer-charts { display: grid; gap: 12px; margin-top: 12px; }
.portfolio-controls { display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)) auto; gap: 10px; margin-bottom: 12px; }
.portfolio-summary { margin: 0 0 12px; color: var(--muted); }
.pager { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 10px; color: var(--muted); }
.pager button { padding: 6px 10px; color: var(--navy); background: #fff; border: 1px solid var(--line); border-radius: 4px; cursor: pointer; }
.method-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.method-card { padding: 18px; }
.method-card h3 { margin: 0; color: var(--navy); }
.method-status { display: inline-block; margin: 7px 0 9px; padding: 3px 7px; color: #594906; background: var(--warning-bg); border-radius: 3px; font-size: 11px; }
.method-card p { color: var(--muted); }
.method-card ol { margin: 12px 0 0; padding: 13px 13px 13px 34px; background: #f5f7fa; border-left: 3px solid var(--navy); font: 12px/1.65 "IBM Plex Mono", ui-monospace, monospace; }
.lineage-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.lineage-card { padding: 16px; }
.lineage-card h3 { margin: 0 0 12px; color: var(--navy); }
.definition { display: grid; grid-template-columns: 140px 1fr; gap: 7px 12px; margin: 0; }
.definition dt { color: var(--muted); }
.definition dd { min-width: 0; margin: 0; overflow-wrap: anywhere; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; }
.loading { display: grid; min-height: 100vh; place-items: center; color: var(--navy); font: 700 13px "IBM Plex Mono", monospace; }
.loading[hidden], .shell[hidden] { display: none !important; }
@media (max-width: 1120px) {
  .kpi-grid, .composer-kpis { grid-template-columns: repeat(2, 1fr); }
  .composer-layout { grid-template-columns: 1fr; }
}
@media (max-width: 820px) {
  .sidebar { position: static; width: auto; padding: 14px 18px; }
  .brand { margin: 0; }
  .sidebar nav, .sidebar-meta { display: none; }
  main { margin-left: 0; }
  .content { padding: 20px 16px 48px; }
  .hero { grid-template-columns: 1fr; }
  .toolbar, .portfolio-controls { position: static; grid-template-columns: 1fr 1fr; }
  .method-grid, .lineage-grid, .composer-options { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .toolbar, .portfolio-controls, .kpi-grid, .composer-kpis { grid-template-columns: 1fr; }
  .section-head { display: block; }
  .composer-selection-head { display: block; }
  .composer-actions { justify-content: start; margin-top: 8px; }
}
"""
