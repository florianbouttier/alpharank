"""Semantic HTML shell and styles for the offline SEC explorer."""

from __future__ import annotations

from html import escape

from alpharank.reporting._sec_explorer_script import BROWSER_SCRIPT


def render_sec_explorer_html(*, encoded_payload: str, run_id: str, initial_ticker: str) -> str:
    """Return one self-contained HTML document with an embedded compressed payload."""

    return f"""<!doctype html>
<html lang="fr" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRank — explorateur des données SEC</title>
  <style>{STYLES}</style>
</head>
<body data-initial-ticker="{escape(initial_ticker)}">
  <div id="loading" class="loading"><div><strong>AlphaRank SEC</strong>
  <span>Ouverture des faits téléchargés du run {escape(run_id)}…</span></div></div>
  <div class="shell" id="app" aria-busy="true">
    <aside>
      <div class="brand"><span>α</span><div>AlphaRank<small>SEC DATA EXPLORER</small></div></div>
      <nav>
        <a href="#company" class="active">01 · Entreprise</a>
        <a href="#quarterly">02 · Trimestres</a>
        <a href="#coverage">03 · Couverture</a>
        <a href="#raw">04 · Lignes brutes</a>
        <a href="#proof">05 · Provenance</a>
      </nav>
      <div class="aside-note"><span>RUN D’INGESTION</span><strong>{escape(run_id)}</strong>
      <small>Aucune donnée n’est promue par ce rapport.</small></div>
    </aside>
    <main>
      <header class="topbar">
        <div><span class="kicker">PREUVE STATIQUE · SEC</span>
        <strong id="top-company">Chargement…</strong></div>
        <div class="top-actions"><span class="badge" id="run-badge">{escape(run_id)}</span>
        <button class="icon-button" id="theme-toggle" type="button" aria-label="Changer de thème">◐</button></div>
      </header>
      <div class="page">
        <section id="company" class="hero">
          <div><span class="kicker">FAITS TÉLÉCHARGÉS · SANS CONSOLIDATION</span>
          <h1>Explorer une entreprise, trimestre par trimestre</h1>
          <p>Chaque point et chaque ligne vient du dossier RAW du run. Les versions amendées
          restent visibles ; ce rapport ne choisit pas la valeur finale du modèle.</p></div>
          <div class="company-controls panel">
            <label for="company-search">Chercher ticker ou société</label>
            <input id="company-search" type="search" autocomplete="off" placeholder="Ex. NVDA ou NVIDIA">
            <label for="company-select">Entreprise</label>
            <select id="company-select"></select>
          </div>
        </section>
        <section class="company-strip panel">
          <div><span class="company-symbol" id="company-symbol">—</span>
          <div><h2 id="company-name">—</h2><p id="company-meta">—</p></div></div>
          <div class="company-nav"><button id="previous-company" type="button">←</button>
          <button id="next-company" type="button">→</button></div>
        </section>
        <section class="metrics" id="company-metrics"></section>
        <section id="quarterly">
          <div class="section-head"><div><span class="kicker">SÉRIE FISCALE</span>
          <h2>Valeurs par trimestre</h2><p>La première version déposée est la vue causale ;
          toutes les autres versions restent accessibles.</p></div></div>
          <div class="panel chart-panel">
            <div class="controls-grid">
              <label>Métrique<select id="metric-select"></select></label>
              <label>Version<select id="version-select"><option value="first">Première publication</option>
              <option value="latest">Dernière version téléchargée</option>
              <option value="all">Toutes les versions</option></select></label>
              <label>Fenêtre<select id="quarter-window"><option value="12">12 trimestres</option>
              <option value="24" selected>24 trimestres</option><option value="40">40 trimestres</option>
              <option value="all">Tout l’historique</option></select></label>
            </div>
            <div class="chart-title"><strong id="metric-title">—</strong><span id="metric-note">—</span></div>
            <div class="chart" id="quarter-chart"></div>
            <div class="legend"><span><i class="navy"></i>Valeur retenue pour la vue</span>
            <span><i class="gold"></i>Autres versions brutes</span></div>
          </div>
          <div class="two-columns">
            <article class="panel"><div class="chart-title"><strong>Dépôts reçus</strong>
            <span>10-Q, 10-K et amendements par trimestre fiscal.</span></div>
            <div class="chart compact" id="filings-chart"></div></article>
            <article class="panel"><div class="chart-title"><strong>Lecture du graphique</strong></div>
            <div id="metric-explanation" class="explanation"></div></article>
          </div>
        </section>
        <section id="coverage">
          <div class="section-head"><div><span class="kicker">COUVERTURE</span>
          <h2>Présence des métriques</h2><p>Une cellule compte le nombre de versions RAW
          trouvées pour la métrique et le trimestre.</p></div></div>
          <div class="panel"><div id="coverage-heatmap" class="heatmap"></div></div>
        </section>
        <section id="raw">
          <div class="section-head"><div><span class="kicker">PREUVE LIGNE À LIGNE</span>
          <h2>Tout ce qui a été téléchargé</h2><p>Filtrer, paginer ou exporter les lignes de
          l’entreprise sélectionnée sans perdre les colonnes source.</p></div></div>
          <div class="panel raw-panel">
            <div id="dataset-tabs" class="tabs"></div>
            <div class="raw-toolbar"><input id="raw-search" type="search" placeholder="Filtrer les lignes…">
            <label>Lignes<select id="page-size"><option>25</option><option selected>50</option><option>100</option></select></label>
            <button id="export-csv" type="button">Exporter CSV</button></div>
            <div class="table-wrap"><table><thead id="raw-head"></thead><tbody id="raw-body"></tbody></table></div>
            <div class="pagination"><span id="raw-count">—</span><div>
            <button id="previous-page" type="button">← Précédent</button>
            <button id="next-page" type="button">Suivant →</button></div></div>
          </div>
        </section>
        <section id="proof">
          <div class="section-head"><div><span class="kicker">PROVENANCE</span>
          <h2>Run, sources et empreintes</h2><p>Les hashes relient ce rapport aux fichiers
          Parquet exacts ; le statut d’acquisition reste distinct d’une promotion.</p></div></div>
          <div class="two-columns"><article class="panel" id="source-status"></article>
          <article class="panel" id="source-contract"></article></div>
          <div class="panel"><div class="table-wrap provenance"><table>
          <thead><tr><th>Dataset</th><th>Lignes</th><th>Taille</th><th>SHA-256</th><th>Chemin</th></tr></thead>
          <tbody id="source-files"></tbody></table></div></div>
        </section>
        <footer>AlphaRank · rapport SEC autonome · aucune ressource réseau ·
        <span id="generated-at">—</span></footer>
      </div>
    </main>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script id="sec-explorer-payload" type="application/octet-stream">{encoded_payload}</script>
  <script>{BROWSER_SCRIPT}</script>
</body>
</html>
"""


STYLES = r"""
:root{--bg:#F8FAFC;--panel:#FFFFFF;--surface:#F1F5F9;--border:#D7E0EA;--ink:#020617;
--muted:#475569;--navy:#111D55;--gold:#9B8816;--cyan:#0369A1;--green:#265511;
--amber:#D97706;--red:#802331;--shadow:0 1px 2px rgba(2,6,23,.04);color-scheme:light}
html[data-theme="dark"]{--bg:#020617;--panel:#0B1220;--surface:#111C2F;
--border:rgba(148,163,184,.16);--ink:#E2E8F0;--muted:#94A3B8;--navy:#7C91E8;
--gold:#D4BE46;--cyan:#38BDF8;--green:#72B84F;--amber:#F0A230;--red:#E07886;
--shadow:none;color-scheme:dark}*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",
"Helvetica Neue",Arial,sans-serif}button,input,select{font:inherit;color:inherit}.shell{display:grid;
grid-template-columns:248px minmax(0,1fr);min-height:100vh}aside{position:sticky;top:0;height:100vh;
padding:24px 16px;background:#111D55;color:#fff;display:flex;flex-direction:column}.brand{display:flex;
align-items:center;gap:11px;padding:0 10px 22px;border-bottom:1px solid rgba(255,255,255,.18);
font:600 17px "IBM Plex Mono","SFMono-Regular",Consolas,monospace}.brand>span{font-size:25px}
.brand small{display:block;margin-top:3px;font:500 9px "IBM Plex Sans",sans-serif;letter-spacing:.14em;
opacity:.58}.brand+nav{margin-top:20px}nav{display:grid;gap:4px}nav a{padding:10px 12px;color:
rgba(255,255,255,.7);text-decoration:none;border-radius:4px;font-weight:600}nav a:hover,nav a.active{
color:#fff;background:rgba(255,255,255,.12)}.aside-note{margin-top:auto;padding:14px 10px 0;
border-top:1px solid rgba(255,255,255,.18);font-family:"IBM Plex Mono",monospace}.aside-note span{
display:block;font-size:9px;letter-spacing:.12em;opacity:.55}.aside-note strong{display:block;margin:6px 0;
font-size:12px}.aside-note small{display:block;font:11px/1.45 "IBM Plex Sans",sans-serif;opacity:.62}
main{min-width:0}.topbar{height:64px;display:flex;align-items:center;justify-content:space-between;gap:16px;
padding:0 28px;background:var(--panel);border-bottom:1px solid var(--border);position:sticky;top:0;
z-index:10}.topbar>div:first-child{display:grid}.kicker{color:var(--cyan);font:600 10px "IBM Plex Mono",
monospace;letter-spacing:.11em}.top-actions{display:flex;align-items:center;gap:9px}.badge{display:inline-block;
padding:4px 8px;border:1px solid var(--border);border-radius:999px;color:var(--muted);font:10px
"IBM Plex Mono",monospace}.icon-button,.company-nav button{width:36px;height:36px;border:1px solid
var(--border);background:var(--panel);border-radius:8px;cursor:pointer}.page{max-width:1440px;margin:auto;
padding:28px 28px 64px}.hero{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:24px;
align-items:end}.hero h1{font-size:34px;line-height:1.12;letter-spacing:-.025em;margin:7px 0 10px}.hero p,
.section-head p{margin:0;color:var(--muted);max-width:800px}.panel{background:var(--panel);border:1px solid
var(--border);border-radius:12px;box-shadow:var(--shadow)}.company-controls{padding:16px;display:grid;
grid-template-columns:1fr 1fr;gap:8px 10px}.company-controls label,.controls-grid label,.raw-toolbar label{
color:var(--muted);font:600 10px "IBM Plex Mono",monospace;letter-spacing:.05em;text-transform:uppercase}
.company-controls input,.company-controls select,.controls-grid select,.raw-toolbar input,.raw-toolbar select{
width:100%;min-height:38px;padding:8px 10px;border:1px solid var(--border);border-radius:8px;
background:var(--panel)}.company-controls input{grid-column:1/-1}.company-strip{margin-top:20px;padding:18px;
display:flex;align-items:center;justify-content:space-between;gap:16px}.company-strip>div:first-child{display:flex;
align-items:center;gap:15px}.company-symbol{display:grid;place-items:center;width:58px;height:58px;border-radius:10px;
background:var(--navy);color:#fff;font:600 13px "IBM Plex Mono",monospace}.company-strip h2{margin:0;
font-size:21px}.company-strip p{margin:3px 0 0;color:var(--muted)}.company-nav{display:flex;gap:8px}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:14px 0 0}.metric{
min-height:108px;padding:16px;background:var(--panel);border:1px solid var(--border);border-radius:12px}
.metric span{display:block;color:var(--muted);font-size:11px}.metric strong{display:block;margin:10px 0 4px;
font:600 23px "IBM Plex Mono",monospace}.metric small{color:var(--muted);font-size:11px}.section-head{
display:flex;justify-content:space-between;align-items:end;gap:18px;margin:30px 0 12px}.section-head h2{
font-size:22px;margin:4px 0}.chart-panel{padding:18px}.controls-grid{display:grid;grid-template-columns:2fr 1fr 1fr;
gap:12px}.controls-grid label{display:grid;gap:5px}.chart-title{display:flex;align-items:baseline;
justify-content:space-between;gap:12px;margin:18px 0 7px}.chart-title strong{font-size:16px}.chart-title span{
color:var(--muted);font-size:12px}.chart{height:360px;position:relative;overflow:hidden}.chart.compact{
height:260px}.chart svg{width:100%;height:100%;display:block}.chart text{fill:var(--muted);font:10px
"IBM Plex Mono",monospace}.gridline{stroke:var(--border);stroke-dasharray:3 4}.axis{stroke:var(--border)}
.legend{display:flex;gap:18px;color:var(--muted);font-size:11px}.legend i{display:inline-block;width:14px;
height:3px;margin-right:6px;vertical-align:middle}.legend .navy{background:var(--navy)}.legend .gold{
background:var(--gold)}.two-columns{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.two-columns>.panel{padding:18px}.explanation{display:grid;gap:10px;color:var(--muted)}.explanation strong{
color:var(--ink)}.note{padding:12px 14px;border-left:3px solid var(--cyan);background:var(--surface)}
.heatmap{overflow:auto}.heatmap-grid{display:grid;gap:3px;min-width:760px}.heatmap-cell{min-height:25px;
display:grid;place-items:center;background:var(--surface);font:9px "IBM Plex Mono",monospace;color:var(--muted)}
.heatmap-cell.head{background:transparent;font-weight:600}.heatmap-cell.metric-name{display:block;padding:5px 7px;
position:sticky;left:0;background:var(--panel);z-index:2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.heatmap-cell.present{background:color-mix(in srgb,var(--green) 19%,var(--panel));color:var(--green)}
.heatmap-cell.revised{background:color-mix(in srgb,var(--gold) 24%,var(--panel));color:var(--gold)}
.raw-panel{padding:16px}.tabs{display:flex;gap:6px;overflow:auto;padding-bottom:10px}.tabs button,.raw-toolbar button,
.pagination button{border:1px solid var(--border);background:var(--panel);border-radius:8px;padding:8px 11px;
cursor:pointer;white-space:nowrap}.tabs button.active{background:var(--navy);border-color:var(--navy);color:#fff}
.tabs button span{opacity:.68;margin-left:5px;font-family:"IBM Plex Mono",monospace}.raw-toolbar{display:grid;
grid-template-columns:minmax(220px,1fr) 100px auto;align-items:end;gap:10px;margin-bottom:12px}.raw-toolbar label{
display:grid;gap:4px}.raw-toolbar button{background:var(--surface)}.table-wrap{overflow:auto;border:1px solid
var(--border);max-height:620px}table{width:100%;border-collapse:collapse;min-width:920px}th,td{padding:9px 11px;
border-bottom:1px solid var(--border);white-space:nowrap;text-align:left}th{position:sticky;top:0;z-index:2;
background:var(--surface);color:var(--muted);font:600 10px "IBM Plex Mono",monospace;text-transform:uppercase}
td{font-size:12px}td.number{text-align:right;font-family:"IBM Plex Mono",monospace}tbody tr:hover{background:
var(--surface)}.pagination{display:flex;align-items:center;justify-content:space-between;margin-top:12px;color:
var(--muted);font:11px "IBM Plex Mono",monospace}.pagination>div{display:flex;gap:7px}.provenance{max-height:460px}
.hash{font:10px "IBM Plex Mono",monospace}.status-list{display:grid;gap:9px}.status-row{display:flex;
justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--border)}.status-row span{
font-family:"IBM Plex Mono",monospace;font-size:11px}.status-ok{color:var(--green)}.status-warn{color:var(--amber)}
.failure-list{margin:10px 0 0;padding-left:18px;color:var(--muted);font-size:11px;max-height:170px;overflow:auto}
details{margin-top:12px}summary{cursor:pointer;color:var(--cyan)}pre{white-space:pre-wrap;word-break:break-word;
font:10px/1.45 "IBM Plex Mono",monospace;background:var(--surface);padding:12px;border-radius:8px}
footer{margin-top:38px;padding-top:15px;border-top:1px solid var(--border);color:var(--muted);font-size:11px}
.tooltip{position:fixed;display:none;pointer-events:none;z-index:50;max-width:280px;padding:8px 10px;
background:var(--panel);border:1px solid var(--border);border-radius:4px;color:var(--ink);font:11px/1.45
"IBM Plex Mono",monospace}.loading{position:fixed;inset:0;z-index:100;background:var(--bg);display:grid;
place-items:center}.loading>div{display:grid;gap:6px;padding:22px;border:1px solid var(--border);background:var(--panel);
border-radius:12px}.loading strong{font:600 18px "IBM Plex Mono",monospace}.loading span{color:var(--muted)}
.loading.error strong{color:var(--red)}button:disabled{opacity:.4;cursor:not-allowed}
@media(max-width:1050px){.hero{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}}
@media(max-width:760px){.shell{display:block}aside{position:relative;height:auto;padding:12px}.brand{padding:2px 6px 10px}
nav{display:flex;overflow:auto;margin-top:8px}.aside-note{display:none}.topbar{height:54px;padding:0 14px}.page{
padding:18px 12px 48px}.hero h1{font-size:27px}.company-controls,.controls-grid,.two-columns,.metrics{
grid-template-columns:1fr}.company-strip{align-items:flex-start}.company-strip>div:first-child{align-items:flex-start}.raw-toolbar{
grid-template-columns:1fr}.chart{height:300px}.chart-title{display:grid}.pagination{align-items:flex-start;gap:10px;flex-direction:column}}
"""
