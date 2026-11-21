/* Presentation-only Plotly adapter shared by strategy reports. No portfolio/KPI calculations. */
globalThis.AlphaRankStrategyCharts = (() => {
  function colors() {
    const style = getComputedStyle(document.documentElement);
    const read = (key, fallback) => style.getPropertyValue(key).trim() || fallback;
    return {
      ink: read("--ink", "#17212f"),
      muted: read("--muted", "#64748b"),
      grid: read("--grid", "#edf0f4"),
      panel: read("--panel", "#ffffff"),
      line: read("--line", "#e2e7ee"),
    };
  }

  function layout(height) {
    const c = colors();
    return {
      autosize: true,
      height,
      margin: {l: 62, r: 30, t: 26, b: 48},
      font: {family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12, color: c.muted},
      paper_bgcolor: c.panel,
      plot_bgcolor: c.panel,
      separators: ", ",
      showlegend: false,
      hovermode: "x unified",
      hoverlabel: {bgcolor: c.panel, bordercolor: c.line, font: {family: "Inter, system-ui, sans-serif", size: 12, color: c.ink}},
      xaxis: {type: "date", showgrid: false, zeroline: false, showline: true, linecolor: c.line, tickformat: "%m/%Y", hoverformat: "%d/%m/%Y", fixedrange: false},
      yaxis: {showgrid: true, gridcolor: c.grid, zeroline: false, fixedrange: false, nticks: 6, automargin: true},
      dragmode: "zoom",
    };
  }

  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: false,
    doubleClick: "reset+autosize",
    modeBarButtonsToRemove: ["lasso2d", "select2d", "toggleSpikelines"],
    toImageButtonOptions: {format: "png", filename: "alpharank_analyse", scale: 2},
  };

  function bindViewport(element) {
    if (element._analysisViewportBound) return;
    element.on("plotly_relayout", event => {
      const changed = Object.keys(event).some(key => key.startsWith("xaxis.range"));
      if (changed || event["xaxis.autorange"]) {
        element._analysisModel?.onViewportChange?.(changed);
      }
    });
    element._analysisViewportBound = true;
  }

  function renderTimeSeries(element, model) {
    element._analysisModel = model;
    const drawdown = model.view === "drawdown";
    const traces = model.series.map(series => ({
      type: "scatter",
      mode: "lines",
      name: series.name,
      x: series.values.map(point => point.date),
      y: series.values.map(point => point.value),
      customdata: series.values.map(point => point.label || ""),
      line: {color: series.color, width: series.reference ? 1.8 : 2.6, dash: series.reference ? "dot" : "solid", shape: "linear"},
      connectgaps: false,
      hovertemplate: `${drawdown ? "%{y:.2%}" : "%{y:.3f}×"} %{customdata}<extra>%{fullData.name}</extra>`,
    }));
    const figure = layout(model.height || 440);
    figure.uirevision = model.periodKey;
    figure.yaxis.tickformat = drawdown ? ".0%" : ".2f";
    figure.yaxis.ticksuffix = drawdown ? "" : "×";
    figure.shapes = [{type: "line", xref: "paper", x0: 0, x1: 1, y0: drawdown ? 0 : 1, y1: drawdown ? 0 : 1, line: {color: colors().muted, width: 1, dash: "dot"}, layer: "below"}];
    return Plotly.react(element, traces, figure, config).then(() => {
      bindViewport(element);
      element.dataset.chartEngine = "plotly";
    });
  }

  function renderFrontier(element, model) {
    const trace = (points, name, options={}) => ({
      type: "scatter", mode: "markers", name,
      x: points.map(point => point.x), y: points.map(point => point.y),
      text: points.map(point => point.name || point.label || `Composition ${point.index + 1}`),
      customdata: points.map(point => point.description || ""),
      marker: {size: 5, color: "rgba(100,116,139,.24)", ...options},
      hovertemplate: "%{text}<br>%{customdata}<br>Risque %{x:.2%}<br>CAGR %{y:.2%}<extra></extra>",
    });
    const traces = [
      trace(model.points, "Compositions"),
      {type: "scatter", mode: "lines", name: "Frontière", x: model.frontier.map(p => p.x), y: model.frontier.map(p => p.y), line: {color: "#2563eb", width: 2}, hoverinfo: "skip"},
      ...model.references.map(point => trace([point], point.name, {size: 9, color: point.color, symbol: "square", line: {color: colors().panel, width: 1.5}})),
      ...model.recommendations.map(point => trace([point], point.name, {size: 12, color: point.color, symbol: "diamond", line: {color: colors().panel, width: 1.5}})),
    ];
    const figure = layout(330);
    figure.hovermode = "closest";
    figure.xaxis = {type: "linear", tickformat: ".0%", title: {text: model.xLabel, font: {size: 11}}, gridcolor: colors().grid, zeroline: false, automargin: true};
    figure.yaxis.tickformat = ".0%";
    figure.yaxis.title = {text: "CAGR", font: {size: 11}};
    element._analysisModel = model;
    return Plotly.react(element, traces, figure, config).then(() => {
      element.dataset.chartEngine = "plotly";
    });
  }

  function resize(element) {
    if (element?.data && element.offsetWidth) Plotly.Plots.resize(element);
  }

  function reset(element) {
    if (element?.data) Plotly.relayout(element, {"xaxis.autorange": true, "yaxis.autorange": true});
  }

  function refreshTheme(element) {
    if (!element?.offsetWidth || !element._analysisModel) return;
    const model = element._analysisModel;
    return model.series ? renderTimeSeries(element, model) : renderFrontier(element, model);
  }

  return Object.freeze({renderTimeSeries, renderFrontier, resize, reset, refreshTheme});
})();
