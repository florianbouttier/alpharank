export const BENCHMARK = "SPY · Total return";

export function sourcePolicy(source) {
  return source === "recent"
    ? {
        sidebar: "Même poids à l’entrée, puis maintien sans rééquilibrage.",
        composer:
          "Poches équipondérées à l’entrée, puis maintien sans rééquilibrage, brut de frais.",
        frontier:
          "allocation à l’entrée, puis maintien sans rééquilibrage, brut de frais",
      }
    : {
        sidebar:
          "Chaque poche reçoit le même poids, rééquilibré tous les mois.",
        composer: "Poches équipondérées chaque mois. Frais propres inclus.",
        frontier: "rééquilibrage mensuel, frais propres inclus",
      };
}
export const COMPOSED = "Portefeuille composé";
export const DEFAULT_STRATEGIES = [
  "Boosting · Top 5",
  "Boosting tendance · Top 5",
  BENCHMARK,
];

const COLORS = {
  "Legacy · Frequency": "#8053c5",
  "Legacy · Equal": "#ad8ad8",
  "Boosting · Top 5": "#315cf4",
  "Boosting · Top 10": "#5479f1",
  "Boosting · Top 15": "#7a99f5",
  "Boosting · Top 20": "#a0b7f6",
  "Boosting tendance · Top 5": "#0a9589",
  "Boosting tendance · Top 10": "#32ab9b",
  "Boosting tendance · Top 15": "#66bfae",
  "Boosting tendance · Top 20": "#95d2c4",
  "Boosting · Top 5 sectoriel": "#c5742b",
  "Boosting tendance · Top 5 sectoriel": "#c25483",
  [BENCHMARK]: "#8a97ab",
  [COMPOSED]: "#d45436",
};

export const METRICS = {
  total_turnover: [
    "Turnover cumulé",
    "pct",
    "Somme du turnover des rééquilibrages présents dans la fenêtre.",
    "lower",
  ],
  complete_years: [
    "Années civiles complètes",
    "int",
    "Nombre d’années entièrement couvertes utilisées pour les pires années et le taux de surperformance annuel.",
  ],
  total_return: [
    "Rendement total",
    "pct",
    "Rendement composé entre les deux clôtures de référence.",
    "higher",
  ],
  cagr: [
    "CAGR",
    "pct",
    "Croissance annualisée sur la durée calendaire réelle. Une période courte est extrapolée.",
    "higher",
  ],
  annualized_volatility: [
    "Volatilité annualisée",
    "pct",
    "Écart-type échantillon des rendements journaliers, annualisé sur 252 séances.",
    "lower",
  ],
  max_drawdown: [
    "Max drawdown",
    "pct",
    "Pire perte depuis un plus-haut de la fenêtre, capital initial inclus.",
    "higher",
  ],
  worst_year_return: [
    "Pire année complète",
    "pct",
    "Rendement de la pire année civile entièrement couverte.",
    "higher",
  ],
  worst_year_excess: [
    "Pire année vs SPY",
    "pct",
    "Plus faible écart annuel de rendement au SPY, en points, sur les années complètes.",
    "higher",
  ],
  years_beating_benchmark: [
    "Années devant SPY",
    "years",
    "Nombre d'années civiles complètes où le rendement dépasse celui du SPY.",
    "higher",
  ],
  sharpe: [
    "Sharpe",
    "num",
    "(CAGR − taux sans risque 2 %) / volatilité annualisée.",
    "higher",
  ],
  sortino: [
    "Sortino",
    "num",
    "Rendement annualisé excédentaire rapporté au risque baissier.",
    "higher",
  ],
  calmar: [
    "Calmar",
    "num",
    "CAGR divisé par la profondeur du drawdown maximal.",
    "higher",
  ],
  positive_day_rate: [
    "Séances positives",
    "pct",
    "Part des rendements journaliers strictement positifs.",
    "higher",
  ],
  positive_month_rate: [
    "Mois positifs",
    "pct",
    "Part des rendements mensuels strictement positifs.",
    "higher",
  ],
  annualized_excess_return: [
    "Excès annualisé",
    "pct",
    "CAGR stratégie moins CAGR SPY sur la même fenêtre.",
    "higher",
  ],
  tracking_error: [
    "Tracking error",
    "pct",
    "Volatilité annualisée des rendements actifs quotidiens.",
    "lower",
  ],
  information_ratio: [
    "Information ratio",
    "num",
    "Rendement actif annualisé rapporté à son tracking error.",
    "higher",
  ],
  beta: ["Bêta", "num", "Covariance avec SPY divisée par sa variance."],
  alpha: [
    "Alpha annualisé",
    "pct",
    "Alpha CAPM arithmétique annualisé, taux sans risque 2 %.",
    "higher",
  ],
  correlation: [
    "Corrélation au SPY",
    "num",
    "Corrélation de Pearson des rendements quotidiens, pas des niveaux de capital.",
  ],
  benchmark_hit_rate: [
    "Séances devant SPY",
    "pct",
    "Part des séances où le rendement dépasse celui du SPY.",
    "higher",
  ],
  var_95: [
    "VaR 95 % quotidienne",
    "pct",
    "Quantile à 5 % des rendements journaliers.",
    "higher",
  ],
  cvar_95: [
    "CVaR 95 % quotidienne",
    "pct",
    "Rendement moyen des séances sous la VaR à 95 %.",
    "higher",
  ],
  omega: [
    "Omega",
    "num",
    "Somme des gains quotidiens divisée par la somme absolue des pertes.",
    "higher",
  ],
  up_capture: [
    "Capture haussière",
    "pct",
    "Rendement moyen lorsque SPY monte, relatif à SPY.",
    "higher",
  ],
  down_capture: [
    "Capture baissière",
    "pct",
    "Rendement moyen lorsque SPY baisse, relatif à SPY.",
    "lower",
  ],
  skewness: ["Asymétrie", "num", "Moment centré standardisé d'ordre 3."],
  excess_kurtosis: [
    "Kurtosis excédentaire",
    "num",
    "Moment standardisé d'ordre 4, relativement à la loi normale.",
  ],
  average_monthly_turnover: [
    "Turnover mensuel moyen",
    "pct",
    "Turnover des rééquilibrages présents dans la fenêtre.",
    "lower",
  ],
  annualized_turnover: [
    "Turnover annualisé",
    "pct",
    "Turnover observé, ramené à la durée de la fenêtre.",
    "lower",
  ],
  total_transaction_cost: [
    "Coûts cumulés",
    "pct",
    "Frais déjà facturés aux stratégies, sans frais supplémentaires entre poches.",
    "lower",
  ],
  annualized_transaction_cost: [
    "Coût annualisé",
    "pct",
    "Coûts observés, ramenés à la durée de la fenêtre.",
    "lower",
  ],
  average_positions: [
    "Positions moyennes",
    "num",
    "Nombre moyen de lignes détenues.",
  ],
  minimum_positions: [
    "Positions minimum",
    "int",
    "Plus petit nombre de lignes détenues.",
  ],
  maximum_positions: [
    "Positions maximum",
    "int",
    "Plus grand nombre de lignes détenues.",
  ],
  average_maximum_position_weight: [
    "Poids max moyen",
    "pct",
    "Poids maximal moyen d'une ligne.",
    "lower",
  ],
  maximum_single_name_weight: [
    "Poids individuel maximal",
    "pct",
    "Poids individuel le plus élevé.",
    "lower",
  ],
  average_maximum_sector_weight: [
    "Poids secteur max moyen",
    "pct",
    "Concentration sectorielle moyenne, seulement si tous les secteurs sont connus.",
    "lower",
  ],
  maximum_sector_weight: [
    "Poids secteur maximal",
    "pct",
    "Plus forte concentration sectorielle observable.",
    "lower",
  ],
};

export function metricDefinition(field, frequency = "daily") {
  const definition = METRICS[field] || [field, "num"];
  if (frequency !== "monthly") return definition;
  const monthly = {
    annualized_volatility: [
      "Volatilité annualisée",
      "Écart-type échantillon des rendements mensuels, annualisé sur 12 mois.",
    ],
    max_drawdown: [
      "Max drawdown mensuel",
      "Pire perte observée aux fins de mois, capital initial inclus. Les creux intramensuels ne sont pas mesurés.",
    ],
    tracking_error: [
      "Tracking error",
      "Volatilité annualisée des rendements actifs mensuels, sur 12 mois.",
    ],
    correlation: [
      "Corrélation au SPY",
      "Corrélation de Pearson des rendements mensuels, pas des niveaux de capital.",
    ],
    benchmark_hit_rate: [
      "Mois devant SPY",
      "Part des mois où le rendement dépasse celui du SPY.",
    ],
    var_95: ["VaR 95 % mensuelle", "Quantile à 5 % des rendements mensuels."],
    cvar_95: [
      "CVaR 95 % mensuelle",
      "Rendement moyen des mois sous la VaR à 95 %.",
    ],
    omega: [
      "Omega",
      "Somme des gains mensuels divisée par la somme absolue des pertes.",
    ],
  }[field];
  return monthly
    ? [monthly[0], definition[1], monthly[1], definition[3]]
    : definition;
}

export const PRIMARY_METRICS = [
  "total_return",
  "cagr",
  "annualized_volatility",
  "max_drawdown",
  "worst_year_return",
  "worst_year_excess",
  "years_beating_benchmark",
];
export const colorFor = (name) => COLORS[name] || "#526d82";
export const shortName = (name) =>
  name === BENCHMARK
    ? "SPY"
    : name === COMPOSED
      ? "Mon portefeuille"
      : name
          .replace("Legacy · Frequency", "Legacy fréquence")
          .replace("Legacy · Equal", "Legacy équipondéré")
          .replace("Boosting tendance · ", "Tendance ")
          .replace("Boosting · ", "Boosting ");
export function groupFor(name) {
  if (name.startsWith("Legacy")) return "Legacy";
  if (name.includes("sectoriel")) return "Sectorielles";
  return name.includes("tendance") ? "Boosting tendance" : "Boosting";
}
export const dateLabel = (value, options = {}) =>
  value
    ? new Date(`${value.slice(0, 10)}T12:00:00Z`).toLocaleDateString("fr-FR", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
        ...options,
      })
    : "—";
export function format(value, type = "num") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (type === "pct")
    return new Intl.NumberFormat("fr-FR", {
      style: "percent",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  if (type === "int")
    return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(
      value,
    );
  if (type === "score")
    return new Intl.NumberFormat("fr-FR", {
      minimumFractionDigits: 6,
      maximumFractionDigits: 6,
    }).format(value);
  if (type === "wealth")
    return new Intl.NumberFormat("fr-FR", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    }).format(value);
  return new Intl.NumberFormat("fr-FR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}
export const timeKey = (time) =>
  typeof time === "string"
    ? time
    : time &&
      `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`;

export function sampledPoints(points, resolution, field) {
  if (resolution === "D1" || points.length < 3) return points;
  const result = [points[0]];
  const period = (point) => {
    if (resolution === "M1") return point.date.slice(0, 7);
    const day = new Date(`${point.date}T12:00:00Z`);
    day.setUTCDate(day.getUTCDate() - ((day.getUTCDay() + 6) % 7));
    return day.toISOString().slice(0, 10);
  };
  points.slice(1).forEach((point, index, rest) => {
    if (
      typeof point[field] !== "number" ||
      !Number.isFinite(point[field]) ||
      index === rest.length - 1 ||
      period(point) !== period(rest[index + 1])
    )
      result.push(point);
  });
  return result;
}

export function contiguousSegments(points, field) {
  const segments = [];
  let current = [];
  points.forEach((point) => {
    if (typeof point[field] === "number" && Number.isFinite(point[field]))
      current.push({ time: point.date, value: point[field] });
    else if (current.length) {
      segments.push(current);
      current = [];
    }
  });
  if (current.length) segments.push(current);
  return segments;
}

export function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const fields = Object.keys(rows[0]);
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [
    fields.map(quote).join(","),
    ...rows.map((row) => fields.map((field) => quote(row[field])).join(",")),
  ].join("\n");
  const url = URL.createObjectURL(
    new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function viridis(position) {
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const bounded = Math.max(0, Math.min(1, position)) * 4;
  const index = Math.min(3, Math.floor(bounded)),
    blend = bounded - index;
  return `rgb(${stops[index].map((value, channel) => Math.round(value + blend * (stops[index + 1][channel] - value))).join(",")})`;
}
