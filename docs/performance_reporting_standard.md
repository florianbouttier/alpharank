# Standard de reporting des backtests AlphaRank

**Rôle : contrat canonique.** Ce document définit le rapport de performance
interactif commun à Legacy, Boosting et SPY. Les formules restent définies dans
[`common_portfolio_backtest_engine.md`](common_portfolio_backtest_engine.md).

## Objectif

Un rapport de backtest doit permettre de répondre dans une seule page à cinq
questions : quelle méthode est mesurée, sur quelle période, avec quelles
performances, avec quels portefeuilles et à partir de quelles données. Il ne
doit jamais transformer une variante R&D en recommandation ou en méthode
promue par la seule qualité de son affichage.

## Entrées obligatoires

Le générateur reçoit trois chemins explicites :

1. un replay commun contenant `comparison_common_monthly.parquet`,
   `comparison_common_holdings.parquet` et son `manifest.json` ;
2. le run Legacy correspondant, afin d'exposer aussi l'agrégation
   `Combined_Equal` ;
3. le manifeste du snapshot immuable consommé par ce run.

Aucun de ces chemins n'est résolu par récence, par nom de dossier ou par un
pointeur `latest`. Les fichiers et leurs hashes sont recopiés dans le manifeste
du rapport.

## KPI et filtres temporels

Le filtre accepte chaque année du calendrier commun : le début est janvier ou
le premier mois OOS disponible, et la fin décembre ou le dernier mois réalisé.
Pour toutes les combinaisons annuelles inclusives `début -> fin`, les KPI sont
calculés avant rendu par
`alpharank.portfolio.performance.portfolio_period_statistics()` puis assemblés
par `subperiod_portfolio_metric_grid()`.

Ce choix garde le rapport publiable dans une page autonome tout en couvrant
toutes les lectures « depuis 2011 », « depuis 2012 », ou entre deux années. Les
portefeuilles restent filtrables mois par mois. Le navigateur sélectionne une
ligne pré-calculée ; il ne possède pas de seconde formule de CAGR, volatilité,
Sharpe, drawdown, risque relatif, turnover, coûts ou concentration. Les courbes
de richesse et de drawdown sont des projections graphiques des rendements
mensuels déjà produits par le moteur ; les valeurs affichées dans les cartes et
tableaux proviennent du cube canonique.

Le multiselect accepte toute combinaison des onze séries et pilote toutes les
vues comparatives : cartes synthétiques, courbes de richesse et de drawdown,
colonnes du tableau complet et lignes des model cards. Une stratégie non cochée
n'apparaît dans aucune de ces vues. Les filtres de holdings restent indépendants
car ils servent à auditer un panier historique précis.

SPY reste la référence de comparaison, même lorsqu'il n'est pas coché. Une
couleur indique une surperformance ou sous-performance uniquement lorsqu'un
sens économique est défini : rendement et ratios plus élevés, ou risque, coûts
et turnover plus faibles. Les métriques descriptives sans ordre économique
restent neutres. Cette comparaison porte toujours sur la même fenêtre
pré-calculée.

La croissance composée occupe une ligne complète. Le drawdown utilise le même
format de graphique pleine largeur sur la ligne suivante, avec les mêmes
couleurs, la même fenêtre et exactement les mêmes stratégies.

## Laboratoire de portefeuille composé

Le rapport permet de cocher plusieurs stratégies puis de comparer leur
portefeuille composé au SPY. Chaque mois, 100 % du capital est réparti à parts
égales entre les poches cochées ; le poids de chaque poche vaut donc `1 / N` et
le rééquilibrage est mensuel. Les rendements d'entrée sont les `net_return` de
chaque stratégie, déjà diminués des frais facturés dans son propre replay.
Aucun coût supplémentaire de transfert entre poches n'est modélisé dans cette
première version ; cette limite doit rester visible à côté du sélecteur. Les
poches ne sont pas fusionnées au niveau des titres : si deux stratégies
détiennent la même action, cette exposition existe dans chacune de leurs
poches et contribue deux fois selon leurs poids respectifs.

Pour les dix stratégies hors SPY, les 1 023 combinaisons non vides, leurs
rendements mensuels et leurs KPI par fenêtre annuelle sont calculés avant le
rendu par `equal_weight_strategy_combination_grid()`. Le navigateur transforme
la sélection en masque pour choisir la ligne pré-calculée ; il ne calcule aucun
rendement total, CAGR, volatilité, Sharpe, Sortino ou max drawdown. Il projette
seulement les rendements mensuels pré-calculés en courbes de richesse et de
drawdown. Ce laboratoire reste un diagnostic post-hoc non promu : il mesure la
diversification entre méthodes, il ne constitue ni une optimisation de poids
ni une nouvelle stratégie de production.

Deux lectures complémentaires sont obligatoires dans ce laboratoire :

- la corrélation de Pearson porte sur les rendements mensuels, jamais sur les
  niveaux de richesse cumulée. Une matrice bornée par la fenêtre active contient
  uniquement les poches cochées ; le KPI du portefeuille composé mesure de la
  même façon sa corrélation avec les rendements mensuels du SPY ;
- la richesse relative n'est pas une corrélation. Pour chaque mois `t`, elle
  vaut `produit(1 + rendement portefeuille) / produit(1 + rendement SPY)` depuis
  le début de la fenêtre. Elle commence à 1 ; au-dessus de 1 le portefeuille a
  davantage composé que SPY, et sa pente indique la direction récente de la
  performance relative. Les pourcentages de performance ne sont jamais divisés
  directement entre eux.

Une corrélation faible ou négative signale une diversification historique plus
forte mais ne prouve ni la robustesse future ni l'intérêt économique de la
poche. Rendement, drawdown, coûts et concentration restent nécessaires pour
l'interprétation.

Le rapport expose au minimum :

- rendement total, CAGR, volatilité, Sharpe, drawdown et mois positifs ;
- Sortino, Calmar, alpha, bêta, corrélation, tracking error et information ratio ;
- VaR, CVaR, Omega, captures haussière et baissière, asymétrie et kurtosis ;
- turnover, coûts facturés, nombre de positions et concentrations maximales.

Une concentration sectorielle n'est affichée que si chaque holding de la
stratégie porte un secteur observable. Une colonne secteur absente ou nulle
produit `indisponible`, jamais un faux secteur unique à 100 %.

## Model cards cumulées et annuelles

Deux familles de matrices sont obligatoires pour le CAGR, la volatilité
annualisée et le max drawdown. L'axe X contient uniquement les années comprises
entre le début et la fin sélectionnés ; l'axe Y contient uniquement les
stratégies cochées dans le multiselect. La première famille cumule chaque année
de départ jusqu'à la fin sélectionnée. La seconde isole chaque année civile :
sa première et sa dernière colonne peuvent être partielles lorsque les bornes
le sont.

Dans la matrice annuelle, l'onglet CAGR affiche le rendement composé de l'année
isolée. Il est égal au CAGR sur une année civile complète et évite d'annualiser
trompeusement une année de bord partielle. Les onglets volatilité et drawdown
conservent leurs KPI canoniques sur cette seule année. Toutes les cellules sont
sélectionnées dans `metric_windows` ; aucun indicateur n'est recalculé en
JavaScript.

La palette Viridis encode la valeur brute pour le CAGR ou rendement annuel et
la volatilité ; pour le drawdown elle encode la profondeur absolue de la perte.
La valeur numérique reste écrite dans chaque cellule.

Le premier portefeuille Boosting hors échantillon commence en août 2011. La
colonne 2011 est donc marquée comme couverture partielle et aucun rendement de
janvier à juillet n'est inventé.

## Portefeuilles historiques

Chaque panier affiche la stratégie, le mois de décision, le mois de détention,
le ticker, le poids cible, le rang et le score lorsqu'ils existent, le secteur,
le nombre de votes Legacy et le rendement réalisé. Le rendement réalisé est une
information d'audit affichée après la sélection ; il ne sert jamais à choisir
les positions.

Les filtres stratégie, mois et ticker doivent permettre de retrouver toute
ligne du Parquet de holdings source. L'export CSV est construit depuis ces
lignes sans enrichissement externe.

## Méthodologies et statuts

La page décrit au minimum Legacy Frequency, Legacy Equal, Boosting natif,
Boosting filtré par tendance et SPY. Elle rappelle le pseudo-code, le statut de
promotion, la décision en `t`, la détention en `t+1`, le coût de 10 points de
base multiplié par le turnover et la règle d'arrêt sur rendement sélectionné
manquant.

La source détaillée des signaux reste
[`legacy_boosting_methodology.md`](legacy_boosting_methodology.md). Le rapport
est une projection de lecture, pas une nouvelle définition de la méthode.

## Publication dans le site Portfolio

AlphaRank génère le HTML autonome et son manifeste sous
`outputs/performance_reports/<run_id>/`. Le dépôt Portfolio copie ces deux
artefacts dans son espace public et ajoute seulement la navigation. Aucun KPI,
holding ou statut méthodologique n'est recalculé dans Portfolio.

Commande canonique :

```bash
python scripts/research/build_backtest_performance_report.py \
  --common-replay-dir <replay-commun> \
  --legacy-run-dir <run-legacy> \
  --snapshot-manifest <snapshot-manifest.json> \
  --output-dir outputs/performance_reports/<run-id>
```

## Validation et preuve

Un nouveau standard ou une nouvelle version du payload exige : tests du moteur
de KPI, test du payload, vérification de la syntaxe JavaScript, génération sur
un replay réel, manifeste hashé, vérification des combinaisons et changements
de fenêtre dans le navigateur, validation documentaire et build du site
consommateur. L'artefact généré reste ignoré par Git dans AlphaRank ; son run,
son snapshot, ses hashes et son statut sont conservés dans un rapport daté sous
`docs/research/`.
