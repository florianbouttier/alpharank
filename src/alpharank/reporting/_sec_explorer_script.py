"""Browser behavior for the self-contained SEC explorer."""

from __future__ import annotations

BROWSER_SCRIPT = r"""
'use strict';
let DATA=null;
const state={ticker:null,dataset:'companyfacts',metric:null,version:'first',quarterWindow:'24',
rawQuery:'',page:0,pageSize:50,visibleCompanies:[]};
const $=selector=>document.querySelector(selector);
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',
'"':'&quot;',"'":'&#039;'}[char]));
const integer=value=>new Intl.NumberFormat('fr-FR',{maximumFractionDigits:0}).format(Number(value||0));
const compact=value=>{const number=Number(value);if(!Number.isFinite(number))return '—';const absolute=Math.abs(number);
const units=[[1e12,'T'],[1e9,'Md'],[1e6,'M'],[1e3,'k']];for(const [scale,suffix] of units){if(absolute>=scale)
return new Intl.NumberFormat('fr-FR',{maximumFractionDigits:2}).format(number/scale)+' '+suffix;}
return new Intl.NumberFormat('fr-FR',{maximumFractionDigits:3}).format(number);};
const bytes=value=>{const number=Number(value||0);if(number<1024)return integer(number)+' o';if(number<1048576)
return (number/1024).toFixed(1)+' Ko';return (number/1048576).toFixed(2)+' Mo';};

async function decodePayload(){
  if(typeof DecompressionStream==='undefined')throw new Error('Ce navigateur ne sait pas ouvrir le payload gzip autonome.');
  const encoded=$('#sec-explorer-payload').textContent.trim();
  const binary=atob(encoded);const compressed=new Uint8Array(binary.length);
  for(let index=0;index<binary.length;index+=1)compressed[index]=binary.charCodeAt(index);
  const stream=new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
}

function dataset(key){return DATA.datasets[key];}
function rows(key,ticker=state.ticker){return dataset(key).rows_by_ticker[ticker]||[];}
function objects(key,ticker=state.ticker){const columns=dataset(key).columns;
return rows(key,ticker).map(values=>Object.fromEntries(columns.map((column,index)=>[column,values[index]])));}
function company(){return DATA.companies.find(item=>item.ticker===state.ticker);}
function financialRows(){return [...objects('companyfacts'),...objects('filing_fallback')];}
function quarterKey(row){if(row.fiscal_year&&/^Q[1-4]$/.test(String(row.fiscal_period||'')))
return `${row.fiscal_year}-${row.fiscal_period}`;const raw=row.date||row.period_end;if(!raw)return 'Sans trimestre';
const date=new Date(String(raw).slice(0,10)+'T00:00:00Z');if(Number.isNaN(date.getTime()))return 'Sans trimestre';
return `${date.getUTCFullYear()}-Q${Math.floor(date.getUTCMonth()/3)+1}`;}
function quarterSort(left,right){const parse=value=>{const match=/^(\d{4})-Q([1-4])$/.exec(value);
return match?Number(match[1])*4+Number(match[2]):Number.MAX_SAFE_INTEGER;};return parse(left)-parse(right);}
function formDate(row){return String(row.filing_date||row.reportDate||row.date||row.period_end||'');}
function metricLabel(metric){return String(metric||'').replaceAll('_',' ').replace(/\b\w/g,char=>char.toUpperCase());}

function initialize(){
  state.visibleCompanies=[...DATA.companies];
  const requested=document.body.dataset.initialTicker;
  state.ticker=DATA.companies.some(item=>item.ticker===requested)?requested:DATA.companies[0].ticker;
  bindEvents();renderCompanyOptions();renderProvenance();setCompany(state.ticker);
  $('#generated-at').textContent=new Date(DATA.meta.generated_at_utc).toLocaleString('fr-FR',{timeZone:'UTC'})+' UTC';
  $('#app').setAttribute('aria-busy','false');$('#loading').remove();
}

function bindEvents(){
  $('#company-search').addEventListener('input',event=>filterCompanies(event.target.value));
  $('#company-select').addEventListener('change',event=>setCompany(event.target.value));
  $('#previous-company').addEventListener('click',()=>moveCompany(-1));
  $('#next-company').addEventListener('click',()=>moveCompany(1));
  $('#metric-select').addEventListener('change',event=>{state.metric=event.target.value;renderQuarterly();});
  $('#version-select').addEventListener('change',event=>{state.version=event.target.value;renderQuarterly();});
  $('#quarter-window').addEventListener('change',event=>{state.quarterWindow=event.target.value;renderQuarterly();});
  $('#raw-search').addEventListener('input',event=>{state.rawQuery=event.target.value.toLowerCase();state.page=0;renderRawTable();});
  $('#page-size').addEventListener('change',event=>{state.pageSize=Number(event.target.value);state.page=0;renderRawTable();});
  $('#previous-page').addEventListener('click',()=>{state.page=Math.max(0,state.page-1);renderRawTable();});
  $('#next-page').addEventListener('click',()=>{state.page+=1;renderRawTable();});
  $('#export-csv').addEventListener('click',exportCsv);
  $('#theme-toggle').addEventListener('click',()=>{const root=document.documentElement;
  root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';renderQuarterly();});
  document.querySelectorAll('aside nav a').forEach(link=>link.addEventListener('click',()=>{
  document.querySelectorAll('aside nav a').forEach(item=>item.classList.remove('active'));link.classList.add('active');}));
  window.addEventListener('resize',()=>{window.clearTimeout(window.__secResize);
  window.__secResize=window.setTimeout(renderQuarterly,120);});
}

function filterCompanies(query){const normalized=query.trim().toLowerCase();state.visibleCompanies=DATA.companies.filter(item=>
!normalized||item.ticker.toLowerCase().includes(normalized)||String(item.name||'').toLowerCase().includes(normalized)||
String(item.cik||'').includes(normalized));renderCompanyOptions();if(state.visibleCompanies.length&&
!state.visibleCompanies.some(item=>item.ticker===state.ticker))setCompany(state.visibleCompanies[0].ticker);}
function renderCompanyOptions(){const select=$('#company-select');select.innerHTML=state.visibleCompanies.map(item=>
`<option value="${esc(item.ticker)}">${esc(item.display_ticker)} · ${esc(item.name)}</option>`).join('');
if(state.visibleCompanies.some(item=>item.ticker===state.ticker))select.value=state.ticker;}
function moveCompany(direction){const list=state.visibleCompanies.length?state.visibleCompanies:DATA.companies;
const index=Math.max(0,list.findIndex(item=>item.ticker===state.ticker));const target=Math.min(list.length-1,
Math.max(0,index+direction));setCompany(list[target].ticker);}

function setCompany(ticker){if(!DATA.companies.some(item=>item.ticker===ticker))return;state.ticker=ticker;state.page=0;
state.rawQuery='';$('#raw-search').value='';$('#company-select').value=ticker;renderCompanyHeader();renderMetrics();
renderMetricOptions();renderQuarterly();renderCoverage();renderDatasetTabs();renderRawTable();}
function renderCompanyHeader(){const item=company();$('#top-company').textContent=`${item.display_ticker} · ${item.name}`;
$('#company-symbol').textContent=item.display_ticker;$('#company-name').textContent=item.name;
const details=[item.exchange,item.sector,item.industry,item.cik?`CIK ${item.cik}`:null,item.sic?`SIC ${item.sic}`:null].filter(Boolean);
$('#company-meta').textContent=details.join(' · ')||'Référentiel société indisponible';}

function renderMetrics(){const facts=financialRows(),calendar=objects('filing_calendar'),actuals=objects('earnings_actuals');
const metrics=new Set(facts.map(row=>row.metric).filter(Boolean));const quarters=new Set([...facts,...calendar,...actuals]
.map(quarterKey).filter(value=>value!=='Sans trimestre'));const dates=[...facts.map(formDate),...calendar.map(formDate)].filter(Boolean).sort();
const accessions=new Set(calendar.map(row=>row.accession_number).filter(Boolean));const amendments=calendar.filter(row=>
String(row.form||'').endsWith('/A')).length;const cards=[['Lignes SEC',facts.length+calendar.length+actuals.length,
'faits, calendriers et EPS'],['Métriques',metrics.size,'concepts normalisés'],['Trimestres',quarters.size,
dates.length?`${dates[0].slice(0,10)} → ${dates.at(-1).slice(0,10)}`:'aucune période'],['Dépôts',accessions.size,
'accessions SEC distinctes'],['Amendements',amendments,'formes /A conservées']];$('#company-metrics').innerHTML=cards.map(card=>
`<article class="metric"><span>${esc(card[0])}</span><strong>${integer(card[1])}</strong><small>${esc(card[2])}</small></article>`).join('');}

function renderMetricOptions(){const counts=new Map();financialRows().forEach(row=>{if(row.metric)counts.set(row.metric,(counts.get(row.metric)||0)+1);});
const metrics=[...counts].sort((left,right)=>right[1]-left[1]||left[0].localeCompare(right[0]));
const select=$('#metric-select');select.innerHTML=metrics.map(([metric,count])=>`<option value="${esc(metric)}">${esc(metricLabel(metric))} · ${integer(count)}</option>`).join('');
const preferred=['revenue','net_income','operating_income','total_assets','outstanding_shares'];
if(!state.metric||!counts.has(state.metric))state.metric=preferred.find(metric=>counts.has(metric))||(metrics[0]||[])[0]||null;
select.value=state.metric||'';}

function metricRows(){return financialRows().filter(row=>row.metric===state.metric&&Number.isFinite(Number(row.value))).sort((a,b)=>
quarterSort(quarterKey(a),quarterKey(b))||formDate(a).localeCompare(formDate(b)));}
function groupMetricRows(){const groups=new Map();metricRows().forEach(row=>{const key=quarterKey(row);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);});
return [...groups].sort((left,right)=>quarterSort(left[0],right[0]));}
function visibleGroups(){const groups=groupMetricRows();return state.quarterWindow==='all'?groups:groups.slice(-Number(state.quarterWindow));}
function selectedRows(groups){return groups.flatMap(([,items])=>{const ordered=[...items].sort((a,b)=>formDate(a).localeCompare(formDate(b)));
if(state.version==='all')return ordered;return [state.version==='latest'?ordered.at(-1):ordered[0]];});}

function renderQuarterly(){if(!DATA||!state.ticker)return;const groups=visibleGroups();const selected=selectedRows(groups);
renderLineChart(groups,selected);renderFilingsChart();renderMetricExplanation(groups,selected);}
function renderLineChart(groups,selected){const element=$('#quarter-chart');const metric=state.metric;
$('#metric-title').textContent=metric?metricLabel(metric):'Aucune métrique';const totalVersions=groups.reduce((sum,item)=>sum+item[1].length,0);
$('#metric-note').textContent=`${integer(groups.length)} trimestres · ${integer(totalVersions)} versions brutes`;
if(!groups.length){element.innerHTML='<div class="note">Aucune valeur financière SEC pour cette métrique.</div>';return;}
const width=Math.max(720,element.clientWidth||900),height=350,p={left:76,right:24,top:25,bottom:48};
const values=groups.flatMap(([,items])=>items.map(row=>Number(row.value))).filter(Number.isFinite);
let minimum=Math.min(...values),maximum=Math.max(...values);if(minimum===maximum){minimum-=Math.abs(minimum||1)*.05;maximum+=Math.abs(maximum||1)*.05;}
const padding=(maximum-minimum)*.08;minimum-=padding;maximum+=padding;const x=index=>p.left+(groups.length===1?0:(index*(width-p.left-p.right)/(groups.length-1)));
const y=value=>p.top+(maximum-value)*(height-p.top-p.bottom)/(maximum-minimum);const selectedSet=new Set(selected);
let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(metricLabel(metric))} par trimestre">`;
for(let tick=0;tick<5;tick+=1){const value=minimum+(maximum-minimum)*tick/4,yy=y(value);svg+=`<line class="gridline" x1="${p.left}" y1="${yy}" x2="${width-p.right}" y2="${yy}"/><text x="${p.left-9}" y="${yy+3}" text-anchor="end">${esc(compact(value))}</text>`;}
const indexByQuarter=new Map(groups.map((item,index)=>[item[0],index]));const linePoints=selected.map(row=>`${x(indexByQuarter.get(quarterKey(row)))},${y(Number(row.value))}`).join(' ');
if(state.version!=='all'&&selected.length>1)svg+=`<polyline points="${linePoints}" fill="none" stroke="var(--navy)" stroke-width="2.5"/>`;
groups.forEach(([quarter,items],index)=>{if(index%Math.max(1,Math.ceil(groups.length/9))===0||index===groups.length-1)
svg+=`<text x="${x(index)}" y="${height-18}" text-anchor="middle">${esc(quarter)}</text>`;items.forEach(row=>{
const isSelected=selectedSet.has(row),color=isSelected?'var(--navy)':'var(--gold)',radius=isSelected?4:3;
const tip=`${quarter} · ${metricLabel(metric)}\n${compact(row.value)}\nDéposé ${formDate(row)||'—'} · ${row.form||'—'}\n${row.source_label||row.source||'SEC'}`;
svg+=`<circle cx="${x(index)}" cy="${y(Number(row.value))}" r="${radius}" fill="${isSelected?color:'var(--panel)'}" stroke="${color}" stroke-width="2" data-tip="${esc(tip)}"/>`;});});
svg+=`<line class="axis" x1="${p.left}" y1="${height-p.bottom}" x2="${width-p.right}" y2="${height-p.bottom}"/></svg>`;
element.innerHTML=svg;bindTooltips(element);}

function renderFilingsChart(){const element=$('#filings-chart');const counts=new Map();objects('filing_calendar').forEach(row=>{
const key=quarterKey(row);if(key!=='Sans trimestre')counts.set(key,(counts.get(key)||0)+1);});let items=[...counts].sort((a,b)=>quarterSort(a[0],b[0]));
if(state.quarterWindow!=='all')items=items.slice(-Number(state.quarterWindow));if(!items.length){element.innerHTML='<div class="note">Aucun dépôt SEC indexé.</div>';return;}
const width=Math.max(520,element.clientWidth||620),height=250,p={left:42,right:15,top:18,bottom:42};const maximum=Math.max(...items.map(item=>item[1]),1);
const slot=(width-p.left-p.right)/items.length,barWidth=Math.max(2,slot-3);let svg=`<svg viewBox="0 0 ${width} ${height}">`;
[0,.5,1].forEach(ratio=>{const yy=p.top+(1-ratio)*(height-p.top-p.bottom);svg+=`<line class="gridline" x1="${p.left}" y1="${yy}" x2="${width-p.right}" y2="${yy}"/><text x="${p.left-7}" y="${yy+3}" text-anchor="end">${Math.round(maximum*ratio)}</text>`;});
items.forEach(([quarter,count],index)=>{const x=p.left+index*slot+(slot-barWidth)/2,barHeight=count/maximum*(height-p.top-p.bottom),y=height-p.bottom-barHeight;
svg+=`<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="var(--navy)" data-tip="${esc(`${quarter}\n${count} dépôts`)}"/>`;
if(index%Math.max(1,Math.ceil(items.length/7))===0||index===items.length-1)svg+=`<text x="${x+barWidth/2}" y="${height-16}" text-anchor="middle">${esc(quarter)}</text>`;});
svg+='</svg>';element.innerHTML=svg;bindTooltips(element);}

function renderMetricExplanation(groups,selected){const total=groups.reduce((sum,item)=>sum+item[1].length,0),revised=groups.filter(item=>item[1].length>1).length;
const forms=[...new Set(selected.map(row=>row.form).filter(Boolean))].sort();const labels=[...new Set(selected.map(row=>row.source_label).filter(Boolean))].slice(0,8);
$('#metric-explanation').innerHTML=`<div class="note"><strong>${esc(metricLabel(state.metric))}</strong>
${integer(total)} observations téléchargées couvrent ${integer(groups.length)} trimestres. ${integer(revised)} trimestres ont plusieurs versions.</div>
<div><strong>Formulaires visibles</strong><br>${forms.length?forms.map(form=>`<span class="badge">${esc(form)}</span>`).join(' '):'—'}</div>
<details><summary>Concepts SEC sources</summary><pre>${esc(labels.join('\n')||'Aucun concept')}</pre></details>
<div class="note"><strong>Règle de lecture</strong> Le mode « première publication » affiche ce qui était disponible en premier. Le tableau brut conserve aussi les dépôts ultérieurs.</div>`;}

function renderCoverage(){const facts=financialRows(),counts=new Map();facts.forEach(row=>{if(!row.metric)return;const key=quarterKey(row);if(key==='Sans trimestre')return;
if(!counts.has(row.metric))counts.set(row.metric,new Map());const quarters=counts.get(row.metric);quarters.set(key,(quarters.get(key)||0)+1);});
const metrics=[...counts].sort((a,b)=>[...b[1].values()].reduce((x,y)=>x+y,0)-[...a[1].values()].reduce((x,y)=>x+y,0)).slice(0,12);
const quarterSet=new Set();metrics.forEach(([,quarters])=>quarters.forEach((value,key)=>quarterSet.add(key)));const quarters=[...quarterSet].sort(quarterSort).slice(-16);
const columns=`150px repeat(${quarters.length},minmax(42px,1fr))`;let html=`<div class="heatmap-grid" style="grid-template-columns:${columns}"><div class="heatmap-cell head metric-name">Métrique</div>`;
html+=quarters.map(quarter=>`<div class="heatmap-cell head">${esc(quarter.replace('-',' '))}</div>`).join('');metrics.forEach(([metric,values])=>{
html+=`<div class="heatmap-cell metric-name" title="${esc(metric)}">${esc(metricLabel(metric))}</div>`;quarters.forEach(quarter=>{const count=values.get(quarter)||0;
html+=`<div class="heatmap-cell ${count>1?'revised':count===1?'present':''}" title="${esc(`${metric} · ${quarter} · ${count} version(s)`)}">${count||'·'}</div>`;});});
html+='</div>';$('#coverage-heatmap').innerHTML=metrics.length?html:'<div class="note">Aucune couverture financière.</div>';}

function renderDatasetTabs(){const container=$('#dataset-tabs');container.innerHTML=Object.entries(DATA.datasets).map(([key,value])=>
`<button type="button" data-dataset="${esc(key)}" class="${key===state.dataset?'active':''}">${esc(value.label)}<span>${integer(rows(key).length)}</span></button>`).join('');
container.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{state.dataset=button.dataset.dataset;state.page=0;renderDatasetTabs();renderRawTable();}));}
function filteredRawRows(){const values=rows(state.dataset);if(!state.rawQuery)return values;return values.filter(row=>row.some(value=>String(value??'').toLowerCase().includes(state.rawQuery)));}
function renderRawTable(){const values=filteredRawRows(),columns=dataset(state.dataset).columns,totalPages=Math.max(1,Math.ceil(values.length/state.pageSize));state.page=Math.min(state.page,totalPages-1);
const start=state.page*state.pageSize,end=Math.min(values.length,start+state.pageSize),visible=values.slice(start,end);$('#raw-head').innerHTML='<tr>'+columns.map(column=>`<th>${esc(column)}</th>`).join('')+'</tr>';
$('#raw-body').innerHTML=visible.map(row=>'<tr>'+row.map(value=>`<td class="${typeof value==='number'?'number':''}">${esc(displayRaw(value))}</td>`).join('')+'</tr>').join('')||`<tr><td colspan="${columns.length}">Aucune ligne.</td></tr>`;
$('#raw-count').textContent=values.length?`${integer(start+1)}–${integer(end)} / ${integer(values.length)} lignes`:'0 ligne';
$('#previous-page').disabled=state.page===0;$('#next-page').disabled=state.page>=totalPages-1;}
function displayRaw(value){if(value===null||value===undefined||value==='')return '∅';if(typeof value==='number')return compact(value);return String(value);}
function exportCsv(){const columns=dataset(state.dataset).columns,values=filteredRawRows();const quote=value=>`"${String(value??'').replaceAll('"','""')}"`;
const content=[columns.map(quote).join(','),...values.map(row=>row.map(quote).join(','))].join('\n');const blob=new Blob([content],{type:'text/csv;charset=utf-8'});
const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=`${state.ticker}_${state.dataset}.csv`;link.click();URL.revokeObjectURL(link.href);}

function renderProvenance(){const statuses=DATA.meta.source_statuses;$('#source-status').innerHTML='<div class="chart-title"><strong>Acquisitions SEC</strong><span>statut du run</span></div><div class="status-list">'+statuses.map(item=>{
const warning=Number(item.failure_count||0)>0||!String(item.status).startsWith('downloaded');return `<div class="status-row"><div><strong>${esc(item.source)}</strong><br><small>${integer(item.downloaded_rows)} lignes</small></div>
<span class="${warning?'status-warn':'status-ok'}">${esc(item.status)} · ${integer(item.failure_count)} échec(s)</span></div>${item.failure_examples.length?`<details><summary>${item.failure_examples.length} exemples d’échec</summary><ul class="failure-list">${item.failure_examples.map(failure=>`<li>${esc(failure.ticker||failure.dataset||'—')} · ${esc(failure.error||'')}</li>`).join('')}</ul></details>`:''}`;}).join('')+'</div>';
const contract=DATA.meta.source_contract;$('#source-contract').innerHTML=`<div class="chart-title"><strong>Contrat du téléchargement</strong><span>pas une promotion</span></div>
<div class="note"><strong>Run explicite ${esc(DATA.meta.run_id)}</strong> Le rapport lit seulement ses fichiers RAW et ne déplace aucun pointeur de production.</div>
<details open><summary>Détail SEC du contrat source</summary><pre>${esc(JSON.stringify(contract,null,2))}</pre></details>`;
$('#source-files').innerHTML=DATA.meta.source_files.map(file=>`<tr><td>${esc(file.label)}</td><td class="number">${integer(file.row_count)}</td><td>${bytes(file.size_bytes)}</td>
<td class="hash" title="${esc(file.sha256)}">${esc(file.sha256.slice(0,16))}…</td><td class="hash">${esc(file.path)}</td></tr>`).join('');}

function bindTooltips(container){const tooltip=$('#tooltip');container.querySelectorAll('[data-tip]').forEach(node=>{
node.addEventListener('mouseenter',event=>{tooltip.innerHTML=esc(event.target.dataset.tip).replaceAll('\n','<br>');tooltip.style.display='block';});
node.addEventListener('mousemove',event=>{tooltip.style.left=`${event.clientX+14}px`;tooltip.style.top=`${event.clientY+14}px`;});
node.addEventListener('mouseleave',()=>{tooltip.style.display='none';});});}

decodePayload().then(payload=>{DATA=payload;initialize();}).catch(error=>{const loading=$('#loading');loading.classList.add('error');
loading.innerHTML=`<div><strong>Rapport impossible à ouvrir</strong><span>${esc(error.message)}</span></div>`;});
"""
