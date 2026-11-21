# Composant frontal d'analyse de stratégies

Responsabilité : interface React d'analyse des stratégies, graphiques financiers
Lightweight Charts, tableaux et navigation du rapport canonique.

Entrées : payload source complet, points et KPI calculés dans Python. Le module
expose `AlphaRankAnalysis.mount(element, payload)` ; les dates libres appellent
l'API locale explicite décrite dans `../strategy_analysis/`.

Sorties : bundle autonome, styles, survol/zoom et filtres. Aucun Plotly, CDN ou
police distante. `client.jsx`, `charts.jsx`, `tables.jsx`, `audit.jsx` et
`format.js` séparent état, graphiques, mesures, audit et formatage. Le bundle
construit est versionné pour que le rendu Python et la wheel fonctionnent sans
Node ni réseau ; le code maintenu reste dans ces sources.

Construction reproductible : `npm ci`, `npm test`, `npm run format:check`,
`npm run build`. Versions exactes dans `package.json` et `package-lock.json` ;
notices complètes dans `THIRD_PARTY_LICENSES.txt`, attribution TradingView visible.

Dossiers enfants : aucun.

Interdit ici : acquisition, sélection de titres, calcul de rendement, KPI,
interpolation d'un trou ou changement de données. Jour/semaine/mois ne changent
que la résolution d'affichage, jamais les statistiques reçues de Python.
