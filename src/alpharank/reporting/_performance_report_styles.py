PERFORMANCE_REPORT_STYLES = r"""
:root {
  color-scheme: light;
  --navy: #111d55;
  --navy-2: #1e2b68;
  --accent-cyan: #0369a1;
  --ink: #020617;
  --muted: #475569;
  --line: #d7e0ea;
  --line-strong: #bdc9d8;
  --panel: #ffffff;
  --surface: #f8fafc;
  --surface-raised: #ffffff;
  --soft: #eef2f6;
  --soft-strong: #e7edf4;
  --positive: #265511;
  --negative: #802331;
  --gold: #9b8816;
  --warning-bg: #fff8df;
  --warning-line: #d2a927;
  --grid: #e1e7ee;
  --shadow: 0 1px 2px rgba(15,23,42,.05), 0 10px 30px rgba(15,23,42,.035);
  --radius-panel: 12px;
  --radius-control: 8px;
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
  outline: 3px solid color-mix(in srgb, var(--navy) 24%, transparent);
  outline-offset: 2px;
}
.skip-link {
  position: fixed;
  top: 8px;
  left: 8px;
  z-index: 100;
  padding: 9px 12px;
  color: #fff;
  background: var(--navy);
  border-radius: var(--radius-control);
  transform: translateY(-160%);
}
.skip-link:focus { transform: translateY(0); }
.shell {
  width: min(100%, 1440px);
  min-height: 100vh;
  margin: 0 auto;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 28px;
  align-items: center;
  min-height: 66px;
  padding: 10px 24px;
  color: var(--ink);
  background: color-mix(in srgb, var(--surface) 96%, transparent);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(14px);
}
.brand { display: flex; gap: 10px; align-items: center; color: inherit; text-decoration: none; }
.brand-mark {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  color: #fff;
  background: var(--navy);
  border-radius: 9px;
  font: 700 13px "IBM Plex Mono", ui-monospace, monospace;
}
.brand strong { display: block; font-size: 15px; letter-spacing: .01em; }
.brand small { color: var(--muted); }
.topbar nav { display: flex; gap: 4px; align-items: center; }
.nav-link {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 7px 12px;
  color: var(--muted);
  border-bottom: 2px solid transparent;
  text-decoration: none;
}
.nav-link:hover { color: var(--ink); }
.nav-link.is-active { color: var(--navy); border-bottom-color: var(--navy); font-weight: 700; }
.topbar-actions { display: flex; gap: 10px; align-items: center; }
.topbar-meta { display: grid; justify-items: end; color: var(--muted); font: 10px/1.35 "IBM Plex Mono", ui-monospace, monospace; }
.topbar-meta strong { color: var(--ink); font-weight: 600; }
.topbar-meta span { max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
main { min-width: 0; }
.content { max-width: 1440px; margin: 0 auto; padding: 24px 24px 72px; }
.eyebrow, .section-kicker {
  color: var(--navy);
  font: 700 11px "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.hero { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, 520px); gap: 28px; align-items: center; }
h1 { margin: 3px 0 4px; color: var(--ink); font-size: clamp(28px, 3vw, 40px); line-height: 1.04; letter-spacing: -.035em; }
.hero p { max-width: 760px; margin: 0; color: var(--muted); font-size: 14px; }
.hero-actions { display: flex; gap: 8px; align-items: stretch; }
.status-badge {
  display: grid;
  max-width: none;
  padding: 9px 13px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-control);
  font-size: 10px;
}
.status-badge span { letter-spacing: .06em; text-transform: uppercase; }
.status-badge strong { color: var(--ink); font-size: 11px; }
.theme-toggle {
  display: inline-flex;
  min-width: 92px;
  min-height: 38px;
  gap: 7px;
  align-items: center;
  justify-content: center;
  padding: 7px 10px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-control);
  cursor: pointer;
}
.theme-toggle span { color: var(--navy); font-size: 18px; line-height: 1; }
.theme-toggle strong { font-size: 10px; letter-spacing: .02em; }
.coverage-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)) minmax(150px, 1.2fr);
  margin-top: 18px;
  background: transparent;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  overflow: hidden;
}
.coverage-strip article {
  display: grid;
  gap: 1px;
  min-width: 0;
  padding: 10px 15px;
  border-right: 1px solid var(--line);
}
.coverage-strip article:last-child { border-right: 0; }
.coverage-strip span { color: var(--muted); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; }
.coverage-strip strong { overflow: hidden; color: var(--ink); font: 700 17px "IBM Plex Mono", ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
.coverage-strip small { overflow: hidden; color: var(--muted); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.coverage-strip .coverage-source { background: color-mix(in srgb, var(--soft) 45%, transparent); }
.coverage-strip .coverage-source strong { color: var(--navy); font-family: inherit; font-size: 12px; }
.toolbar {
  display: grid;
  grid-template-columns: repeat(2, minmax(150px, .75fr)) minmax(230px, 1.2fr) auto;
  gap: 12px;
  margin: 0;
  padding: 14px 0 0;
  background: transparent;
  border-top: 1px solid var(--line);
}
label, .curve-control { display: grid; gap: 5px; color: var(--muted); font-size: 11px; font-weight: 650; }
.field-label { min-height: 16px; }
select, input {
  min-height: 38px;
  padding: 8px 10px;
  color: var(--ink);
  background: var(--panel);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
}
.multi-select { position: relative; color: var(--ink); font-weight: 400; }
.multi-select summary {
  display: flex;
  min-height: 38px;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  background: var(--panel);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
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
  background: var(--panel);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  box-shadow: 0 12px 28px rgba(24,32,51,.16);
}
.multi-select-actions { display: flex; gap: 7px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.multi-select-actions button {
  padding: 5px 8px;
  color: var(--navy);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
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
  border-radius: 6px;
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
  border-radius: var(--radius-control);
  cursor: pointer;
}
.button.secondary { color: var(--navy); background: var(--panel); }
.section { scroll-margin-top: 24px; margin-top: 34px; }
.section-head { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }
.section-head h2 { margin: 3px 0 0; color: var(--ink); font-size: 23px; letter-spacing: -.015em; }
.section-head p { max-width: 720px; margin: 0; color: var(--muted); }
.kpi-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1px;
  align-content: stretch;
  overflow: hidden;
  background: var(--line);
  border: 1px solid var(--line);
}
.kpi-card, .panel, .method-card, .lineage-card, .analysis-studio {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-panel);
  box-shadow: var(--shadow);
}
.kpi-card { display: grid; grid-template-columns: 104px minmax(0, 1fr); min-width: 0; padding: 8px; border: 0; border-radius: 0; box-shadow: none; }
.kpi-card header { display: grid; align-content: center; gap: 3px; margin: 0; padding-right: 8px; border-right: 1px solid var(--line); }
.kpi-card header span { color: var(--ink); font-size: 12px; font-weight: 750; }
.kpi-card header small { color: var(--muted); font-size: 9px; text-transform: uppercase; }
.kpi-strategy-list { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px; padding-left: 7px; }
.kpi-strategy-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 2px;
  align-content: center;
  min-height: 50px;
  padding: 4px 5px;
  border-left: 3px solid transparent;
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
.kpi-strategy-row small, .metric-value small { color: var(--muted); font: 600 8px "IBM Plex Mono", ui-monospace, monospace; text-align: right; text-transform: uppercase; }
.kpi-strategy-row strong { padding-left: 15px; }
.kpi-strategy-row .delta-chip { padding-left: 15px; text-align: left; }
.comparison-beats { background: rgba(38,85,17,.11); border-left-color: var(--positive); }
.comparison-trails { background: rgba(128,35,49,.09); border-left-color: var(--negative); }
.comparison-equal { background: var(--soft); border-left-color: var(--line-strong); }
.comparison-benchmark { background: rgba(155,136,22,.11); border-left-color: var(--gold); }
.panel { padding: 16px; overflow: hidden; }
.panel h3 { margin: 0 0 3px; color: var(--ink); font-size: 15px; }
.panel-subtitle { margin: 0 0 12px; color: var(--muted); font-size: 12px; }
canvas { display: block; width: 100%; height: 360px; }
.studio-section { margin-top: 14px; }
.analysis-studio { padding: 16px; }
.studio-head, .chart-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}
.studio-head h2 { margin: 4px 0 0; color: var(--ink); font-size: 21px; letter-spacing: -.012em; }
.mode-switch, .chart-switch {
  display: inline-grid;
  grid-auto-flow: column;
  grid-auto-columns: 1fr;
  gap: 2px;
  padding: 3px;
  background: var(--soft);
  border: 1px solid var(--line);
  border-radius: var(--radius-control);
}
.mode-switch button, .chart-switch button {
  min-height: 32px;
  padding: 6px 10px;
  color: var(--muted);
  background: transparent;
  border: 0;
  border-radius: 2px;
  cursor: pointer;
  white-space: nowrap;
}
.mode-switch button.is-active, .chart-switch button.is-active {
  color: #fff;
  background: var(--navy);
}
.studio-context {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 8px 16px;
  margin: 12px 0 8px;
  color: var(--muted);
  font-size: 11px;
}
.studio-context strong {
  color: var(--navy);
  font: 700 11px "IBM Plex Mono", ui-monospace, monospace;
}
.studio-outcome {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 8px 14px;
  align-items: center;
  min-height: 48px;
  margin-bottom: 9px;
  padding: 9px 12px;
  border: 1px solid var(--line);
  border-left: 3px solid var(--line-strong);
  border-radius: var(--radius-panel);
}
.studio-outcome > span {
  color: var(--muted);
  font: 700 9px "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.studio-outcome strong { color: var(--navy); font-size: 13px; }
.studio-outcome small { color: var(--muted); text-align: right; }
.studio-outcome.comparison-beats { border-left-color: var(--positive); }
.studio-outcome.comparison-trails { border-left-color: var(--negative); }
.delta-chip { white-space: nowrap; }
.studio-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1.75fr) minmax(390px, .95fr);
  gap: 10px;
  align-items: stretch;
}
.studio-chart-panel {
  margin-top: 0;
  padding: 14px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-panel);
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--panel) 82%, transparent);
}
.studio-workspace .studio-chart-panel { min-width: 0; }
.studio-workspace .kpi-grid { min-width: 0; }
.chart-stage { position: relative; }
.chart-tooltip {
  position: absolute;
  z-index: 4;
  width: 236px;
  padding: 10px;
  color: var(--ink);
  background: color-mix(in srgb, var(--panel) 97%, transparent);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  box-shadow: 0 8px 24px rgba(24,32,51,.12);
  pointer-events: none;
}
.chart-tooltip header {
  margin-bottom: 6px;
  color: var(--navy);
  font: 700 11px "IBM Plex Mono", ui-monospace, monospace;
}
.chart-tooltip div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 3px 0;
  font-size: 10px;
}
.chart-tooltip div span {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.chart-tooltip i { width: 8px; height: 8px; flex: 0 0 auto; border-radius: 50%; }
.chart-tooltip strong { color: var(--navy); font-family: "IBM Plex Mono", ui-monospace, monospace; }
.studio-drawers {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}
.studio-drawer {
  min-width: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-panel);
}
.studio-drawer[open] { grid-column: 1 / -1; }
.studio-drawer summary {
  display: grid;
  gap: 2px;
  min-height: 52px;
  padding: 9px 12px;
  cursor: pointer;
  list-style-position: inside;
}
.studio-drawer summary span { color: var(--ink); font-weight: 700; }
.studio-drawer summary small { color: var(--muted); }
.drawer-content { padding: 4px 12px 12px; }
.studio-contract {
  margin-top: 8px;
  padding: 9px 12px;
  color: #69550a;
  background: var(--warning-bg);
  border-left: 3px solid var(--gold);
  font-size: 11px;
}
.composer-summary { margin: 0 0 8px; color: var(--muted); font-size: 11px; }
.composer-menu { width: min(560px, 90vw); }
.frontier-studio { margin-top: 10px; }
.frontier-recommendations {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}
.frontier-recommendation {
  padding: 12px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--profile-color);
  border-radius: var(--radius-panel);
}
.frontier-recommendation header { display: flex; justify-content: space-between; gap: 8px; }
.frontier-recommendation header span { color: var(--navy); font-weight: 750; }
.frontier-recommendation header small { color: var(--muted); }
.frontier-recommendation > strong { display: block; margin: 9px 0; color: var(--ink); font-size: 11px; }
.frontier-recommendation dl { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; margin: 0; }
.frontier-recommendation dl div { display: grid; gap: 2px; }
.frontier-recommendation dt { color: var(--muted); font-size: 9px; text-transform: uppercase; }
.frontier-recommendation dd { margin: 0; color: var(--navy); font: 700 11px "IBM Plex Mono", ui-monospace, monospace; }
.frontier-chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
.frontier-panel { padding: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius-panel); }
.frontier-panel h3 { margin: 0; color: var(--ink); font-size: 14px; }
.frontier-panel p { margin: 2px 0 0; color: var(--muted); font-size: 11px; }
.frontier-panel canvas { height: 330px; }
.frontier-anchor-table { margin-top: 10px; }
.frontier-composition-cell { max-width: 480px; overflow: hidden; text-overflow: ellipsis; }
.profile-dot { display: inline-block; width: 8px; height: 8px; margin-right: 6px; border-radius: 50%; }
.frontier-empty { color: var(--muted); }
[hidden] { display: none !important; }
.legend { display: flex; flex-wrap: wrap; gap: 7px 14px; margin-top: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; }
.legend i { width: 16px; height: 3px; border-radius: 2px; }
.table-wrap { width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: var(--radius-control); }
table { width: 100%; border-collapse: collapse; background: var(--panel); }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { position: sticky; top: 0; z-index: 1; color: var(--muted); background: var(--soft); font-size: 10px; letter-spacing: .055em; text-transform: uppercase; }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover { background: var(--soft); }
.value-positive { color: var(--positive); }
.value-negative { color: var(--negative); }
.metric-table { min-width: 760px; }
.metric-table th:first-child, .metric-table td:first-child { position: sticky; left: 0; z-index: 2; background: var(--soft); }
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
  background: var(--panel);
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
.heatmap-label { justify-items: start; padding: 0 9px; color: var(--ink); background: var(--soft); font-family: "IBM Plex Sans", sans-serif; }
.viridis-legend { display: flex; align-items: center; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 11px; }
.viridis-bar { width: 180px; height: 9px; border-radius: 2px; background: linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725); }
.composer-actions { display: flex; flex-wrap: wrap; justify-content: end; gap: 7px; }
.composer-actions button {
  min-height: 32px;
  padding: 6px 9px;
  color: var(--navy);
  background: var(--panel);
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
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
}
.composer-option input { min-height: 0; margin: 0; accent-color: var(--navy); }
.composer-option i { width: 9px; height: 9px; border-radius: 50%; }
.composer-option strong { color: var(--navy); font: 700 10px "IBM Plex Mono", ui-monospace, monospace; }
#composer-correlation-matrix { min-width: 620px; }
#composer-correlation-matrix th { min-width: 128px; max-width: 190px; white-space: normal; }
#composer-correlation-matrix th:first-child { min-width: 190px; }
.correlation-cell { font: 700 11px "IBM Plex Mono", ui-monospace, monospace; text-align: center; }
.correlation-diversifying { color: var(--positive); background: rgba(38,85,17,.11); }
.correlation-moderate { color: #69550a; background: rgba(155,136,22,.10); }
.correlation-high { color: var(--negative); background: rgba(128,35,49,.08); }
.correlation-unknown { color: var(--muted); background: var(--soft); }
.current-portfolio-meta {
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.current-portfolio-meta article {
  display: grid;
  gap: 6px;
  min-height: 78px;
  padding: 13px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--navy);
  border-radius: 6px;
}
.current-portfolio-meta article.is-pending {
  color: #69550a;
  background: var(--warning-bg);
  border-left-color: var(--gold);
}
.current-portfolio-meta span { color: var(--muted); font-size: 10px; text-transform: uppercase; }
.current-portfolio-meta strong { color: var(--navy); font: 700 13px "IBM Plex Mono", ui-monospace, monospace; }
.current-portfolio-controls {
  display: grid;
  grid-template-columns: minmax(240px, 1fr) auto;
  gap: 10px;
  align-items: end;
  margin-bottom: 12px;
}
.portfolio-controls { display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)) auto; gap: 10px; margin-bottom: 12px; }
.portfolio-summary { margin: 0 0 12px; color: var(--muted); }
.pager { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 10px; color: var(--muted); }
.pager button { padding: 6px 10px; color: var(--navy); background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius-control); cursor: pointer; }
.method-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.method-card { padding: 18px; }
.method-card h3 { margin: 0; color: var(--ink); }
.method-status { display: inline-block; margin: 7px 0 9px; padding: 3px 7px; color: #594906; background: var(--warning-bg); border-radius: 3px; font-size: 11px; }
.method-card p { color: var(--muted); }
.method-card ol { margin: 12px 0 0; padding: 13px 13px 13px 34px; background: var(--soft); border-left: 3px solid var(--navy); font: 12px/1.65 "IBM Plex Mono", ui-monospace, monospace; }
.lineage-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.lineage-card { padding: 16px; }
.lineage-card h3 { margin: 0 0 12px; color: var(--ink); }
.definition { display: grid; grid-template-columns: 140px 1fr; gap: 7px 12px; margin: 0; }
.definition dt { color: var(--muted); }
.definition dd { min-width: 0; margin: 0; overflow-wrap: anywhere; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; }
.loading { display: grid; min-height: 100vh; place-items: center; color: var(--navy); font: 700 13px "IBM Plex Mono", monospace; }
.loading[hidden], .shell[hidden] { display: none !important; }

html[data-theme="dark"] {
  color-scheme: dark;
  --navy: #5d73cc;
  --navy-2: #7890e8;
  --accent-cyan: #38a8d8;
  --ink: #e2e8f0;
  --muted: #94a3b8;
  --line: #243149;
  --line-strong: #384862;
  --panel: #0b1220;
  --surface: #020617;
  --surface-raised: #111c2f;
  --soft: #111c2f;
  --soft-strong: #17233a;
  --positive: #8fcf74;
  --negative: #ed8796;
  --gold: #d5bd38;
  --warning-bg: #211d0d;
  --warning-line: #806f1c;
  --grid: #26334a;
  --shadow: 0 1px 2px rgba(0,0,0,.25), 0 12px 34px rgba(0,0,0,.16);
}
html[data-theme="dark"] .mode-switch button.is-active,
html[data-theme="dark"] .chart-switch button.is-active,
html[data-theme="dark"] .matrix-controls button.is-active { background: #3850a3; }
html[data-theme="dark"] .comparison-beats,
html[data-theme="dark"] .correlation-diversifying { background: rgba(143,207,116,.12); }
html[data-theme="dark"] .comparison-trails,
html[data-theme="dark"] .correlation-high { background: rgba(237,135,150,.11); }
html[data-theme="dark"] .comparison-benchmark,
html[data-theme="dark"] .correlation-moderate { background: rgba(213,189,56,.12); }
html[data-theme="dark"] .metric-table .benchmark-head { color: var(--gold); background: #28240e; }
html[data-theme="dark"] .status-badge,
html[data-theme="dark"] .current-portfolio-meta article.is-pending,
html[data-theme="dark"] .studio-contract,
html[data-theme="dark"] .method-status { color: #e3cf64; }

@media (max-width: 1120px) {
  .coverage-strip { grid-template-columns: repeat(5, minmax(145px, 1fr)); overflow-x: auto; }
  .coverage-strip article { min-width: 145px; }
  .studio-workspace { grid-template-columns: minmax(0, 1.45fr) minmax(360px, 1fr); }
  .topbar { gap: 16px; }
  .nav-link { padding-inline: 8px; }
  .current-portfolio-meta { grid-template-columns: repeat(2, 1fr); }
  .frontier-recommendation dl { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 960px) {
  .topbar { grid-template-columns: auto minmax(0, 1fr) auto; padding-inline: 16px; }
  .topbar nav { justify-content: center; overflow-x: auto; }
  .topbar-meta { display: none; }
  .content { padding: 20px 16px 48px; }
  .hero { grid-template-columns: 1fr; }
  .hero-actions { justify-content: space-between; }
  .studio-workspace { grid-template-columns: 1fr; }
  .toolbar, .portfolio-controls, .current-portfolio-controls { grid-template-columns: 1fr 1fr; }
  .method-grid, .lineage-grid, .composer-options, .frontier-chart-grid, .frontier-recommendations { grid-template-columns: 1fr; }
  .studio-head, .chart-head { align-items: stretch; flex-direction: column; }
  .mode-switch, .chart-switch { align-self: stretch; }
  .studio-drawers { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .topbar { grid-template-columns: 1fr auto; }
  .topbar nav { grid-column: 1 / -1; grid-row: 2; justify-content: start; order: 3; margin: 0 -16px -10px; padding: 0 16px; border-top: 1px solid var(--line); scrollbar-width: none; }
  .topbar nav::-webkit-scrollbar { display: none; }
  .topbar-actions { justify-self: end; }
  .theme-toggle { min-width: 44px; }
  .theme-toggle strong { display: none; }
  .hero-actions { display: grid; grid-template-columns: 1fr auto; }
  .coverage-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); overflow: hidden; }
  .coverage-strip article { min-width: 0; border-bottom: 1px solid var(--line); }
  .coverage-strip article:nth-child(2n) { border-right: 0; }
  .coverage-strip .coverage-source { grid-column: 1 / -1; border-bottom: 0; }
  .analysis-studio { padding: 13px; }
  .kpi-card { grid-template-columns: 1fr; }
  .kpi-card header { display: flex; justify-content: space-between; padding: 0 0 5px; border-right: 0; border-bottom: 1px solid var(--line); }
  .kpi-strategy-list { padding: 5px 0 0; }
  .toolbar, .portfolio-controls, .current-portfolio-controls, .current-portfolio-meta { grid-template-columns: 1fr; }
  .section-head { display: block; }
  .mode-switch, .chart-switch { grid-auto-flow: row; }
  .composer-actions { justify-content: start; margin-top: 8px; }
  .studio-outcome { grid-template-columns: 1fr; }
  .studio-outcome small { text-align: left; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
@media print {
  .topbar, .theme-toggle, .skip-link { display: none !important; }
  .shell { display: block; width: auto; }
  .content { max-width: none; padding: 0; }
}
"""
