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

**Livrable utilisateur obligatoire : le HTML canonique complet.** Une demande
de performances, y compris un simple suivi prix, régénère ce rapport sans
perdre ses méthodes, filtres, KPI historiques, portefeuilles, composeur ou
frontières. Le tableau de chat, Markdown, CSV et JSON sont des compléments
techniques ; ils ne remplacent jamais le HTML. Réutiliser le générateur du
dépôt, pas une page réduite ou un second moteur de reporting.

## Architecture du dashboard

Le générateur commun utilise **React et Lightweight Charts**, sans Plotly.
Le même panneau réunit sélection des séries, dates libres, capital ou richesse
relative au SPY, drawdown quotidien en dessous, résumé KPI, tableaux complets,
model cards, composition et frontières. Les onglets Portefeuilles, Méthodes et
Sources gardent toutes les informations d'audit, sans obliger à redescendre une
longue page. Les HTML déjà archivés restent inchangés.

La sélection par familles et recherche est explicite ; seules les stratégies
cochées et SPY apparaissent dans les mesures et graphiques. SPY est fixe.
Les deux Boosting Top 5 sont la sélection initiale lorsqu'ils existent.
Début et fin sont des champs calendrier au jour près, toujours visibles :
aucun preset annuel arbitraire ni menu limité aux années. Jour/semaine/mois
change la résolution visuelle, pas la fenêtre ou les KPI quotidiens.

Le survol expose simultanément les séries et le jour, capital et drawdown sont
synchronisés, et le zoom dispose d'une action explicite « Analyser cette plage »
pour appliquer ses dates aux KPI. Tant que cette action n'est pas effectuée,
le zoom n'est qu'une loupe et le filtre reste affiché. Les erreurs de requête
ne laissent jamais d'anciens chiffres présentés comme ceux des nouvelles dates.

### Frontend partagé et moteur local (`REPORT-024`)

`reporting/analysis_ui/` possède les sources React, CSS, bibliothèque financière,
notices et build verrouillé. Le bundle est embarqué, sans CDN, police distante
ou dépendance Node à l'ouverture. Le logo/lien TradingView et sa notice sont
conservés. Les anciens adaptateurs/scripts/styles Python du frontend ont été
retirés après inventaire des lecteurs ; ils sont récupérables dans Git,
notamment au commit `b658363`, et dans les anciens HTML autonomes.

`reporting/strategy_analysis/` prépare et sert les vues, sans implémentation de
KPI. Les requêtes de dates libres appellent le **moteur Python canonique**
`portfolio/daily/` via un service HTTP local en lecture seule, lié aux seuls
datasets hashés de ce rapport. Il n'y a ni appel fournisseur, ni entraînement,
ni ordre, ni parcours libre du disque, ni publication tierce. L'écoute est
limitée à `127.0.0.1`, avec vérification de l'origine Host.

Le HTML reste complet et conserve son état initial, les méthodes et les données
embarquées. **Les nouvelles fenêtres de KPI nécessitent le service local** :
ouvrir le fichier hors serveur ne prétend pas fournir un calcul Python hors
ligne. Une indisponibilité est indiquée ; aucun KPI JavaScript ou repli silencieux
sur une autre période n'est autorisé. L'API calcule aussi compositions,
corrélations et frontières sur les mêmes dates. Les archives mensuelles de
référence restent consultables séparément, sans être prises pour du journalier.

Sur mobile, la sélection se replie et les tables défilent localement. La page
reste complète, sans débordement global. Labels, focus clavier, états de
sélection, contrastes et thèmes clair/sombre font partie de la validation.

### Versions économiques et catalogue (`REPORT-030`)

Une version correspond à un ensemble cohérent de runs, snapshot, code,
configuration, stratégies, positions, frais et résultats scellés. Sa clé
économique ne dépend ni de la date de rendu du HTML, ni de son style, ni d'une
valorisation ultérieure d'un panier figé. Un téléchargement de prix destiné
seulement au suivi n'est donc pas un nouveau backtest.

Le générateur reçoit la version par défaut avec `--source-report` et les
archives explicitement retenues avec `--archive-report`, répétable. Il produit
un nouveau dossier immuable contenant les rapports complets, leurs ledgers,
leurs manifestes et `versions_catalog.json`. Il ne modifie pas les anciens
rapports. Aucune découverte par nom de dossier, récence ou pointeur de données
mutable ne remplace cette liste. Deux rendus de la même version sont refusés
comme doublon économique.

Le catalogue lie les chemins relatifs et les hashes des HTML et manifestes.
Le service vérifie ces liens, les identités économiques et la cohérence des
métadonnées embarquées avant d'ouvrir son port local. `?version=<identifiant>`
choisit uniquement une version déjà capturée ; chaque requête API porte le
même identifiant. Le catalogue complet doit être servi depuis son dossier
racine, pas depuis une de ses archives enfants isolée.

`/api/version-comparison` appelle le moteur commun sur les **mêmes dates
demandées et effectives**, avec la même fréquence quotidienne ou mensuelle.
Une borne hors couverture ou un calendrier effectif différent est refusé,
jamais recadré silencieusement. Le suivi hypothétique n'entre pas dans cette
comparaison. Les seules stratégies communes sont comparées, les autres sont
nommées ; les titres/poids cibles différents sont comptés par panier/mois.
Ces rapprochements d'archives ne sont ni un réentraînement, ni l'audit causal
inter-vintages exigé lors d'un refresh.

Les différences de snapshot, code, conventions et frais sont visibles : une
différence n'est jamais attribuée automatiquement aux seules données. Les
archives sans politique de frontière ne reçoivent pas une politique inventée.
Date du replay central, date du snapshot, dernière séance prix et fin de
performance restent quatre repères distincts ; un rendu du 22 novembre ne
transforme pas un backtest arrêté au 30 septembre en backtest de novembre.

## Entrées obligatoires

Le générateur reçoit quatre chemins explicites :

1. un replay commun contenant `comparison_common_monthly.parquet`,
   `comparison_common_holdings.parquet` et son `manifest.json` ;
2. le run Legacy correspondant, afin d'exposer aussi l'agrégation
   `Combined_Equal` ;
3. le manifeste du snapshot immuable consommé par ce run ;
4. une preuve de prolongation prix explicitement nommée dont la dernière date
   de marché borne le portefeuille présenté comme « en vigueur ».

Aucun de ces chemins n'est résolu par récence, par nom de dossier ou par un
pointeur `latest`. Les fichiers et leurs hashes sont recopiés dans le manifeste
du rapport.

Le profil unifié `REPORT-019` ajoute deux chemins explicites,
`--sector-backtest-dir` et `--sector-history-dir`. Il montre les onze séries
standard plus les deux Top 5 sectoriels de `REPLAY-014` : Legacy Frequency et
Equal, Boosting natif et tendance en Top 5/10/15/20, Boosting natif et tendance
Top 5 sectoriel, et SPY total return. Les variantes Legacy PE expérimentales
du replay commun ne font pas partie du comparatif standard.

Les hashes du replay commun, des sorties sectorielles et des allocations
sources sont vérifiés. Les quatre références de `REPLAY-014` doivent être
identiques au replay commun avant de n'ajouter que les deux séries sectorielles.
La commande prépare `legacy_equal_common_monthly.parquet` en valorisant les
positions Equal figées au même calendrier, au même benchmark SPY et au même
coût par turnover que le replay commun. Cette étape précède le rendu et ne
reconstruit aucun signal. Le profil standard sans les deux options sectorielles
conserve son ancienne source Equal, à 0 bps dans ce run historique.

Les allocations non valorisées viennent des Parquet scellés du même run,
séparément des positions historiques. Un secteur absent du référentiel de
positions ne vaut jamais une concentration de 0 %.

## KPI et filtres temporels

Les bornes sont personnalisables au jour près **dans la couverture annoncée
de la source choisie**. Le début est inclusif pour les rendements : le repère
initial est la clôture observée précédente, capital et richesse relative à 1,
drawdown à 0. Le premier rendement n'est jamais supprimé pour fabriquer ce
départ. Dates demandées, dates des séances effectives et date de base sont
distinguées ; fêtes et week-ends ne deviennent pas de fausses observations.

Les valorisations proviennent des prix ajustés et des positions scellées du même
snapshot, avec frais et poids dérivés explicités dans le
[contrat quotidien](common_portfolio_backtest_engine.md#daily-research-valuation-and-free-date-windows-report-023).
Aucun rendement mensuel n'est interpolé. Les trous restent visibles ; un chemin
incomplet rend ses risques quotidiens indisponibles, pas égaux à zéro.

Les KPI quotidiens utilisent la même fenêtre que le graphique : CAGR sur durée
calendaire réelle, volatilité et statistiques de rendement sur 252 séances/an,
drawdown avec capital initial. Ils peuvent différer des anciens risques
mensuels, même lorsque le rendement total est identique. Les jours positifs et
VaR/CVaR sont bien journaliers ; ils ne sont pas étiquetés « mois ».

Le mode Backtest conserve l'historique réalisé net de frais. Le mode Suivi
récent conserve séparément les prix des paniers figés, bruts de frais ; un
maintien au-delà du mois prévu est hypothétique. Aucun suivi récent n'est
ajouté aux rendements mensuels historiques ou présenté comme un nouveau signal.

Le résumé répond au rendement, CAGR, volatilité, max drawdown, pire année,
pire année vs SPY et nombre d'années devant SPY. Les années partielles restent
visibles mais ne déterminent pas les pires années complètes ni leur décompte.
Les indicateurs détaillés et leurs définitions restent accessibles dans le
même panneau. Un écart au SPY compare deux KPI de même fréquence et période ;
les mesures descriptives sans ordre économique restent neutres.

### Référence mensuelle conservée

Le cube historique `metric_windows` et ses 33 champs ne sont ni supprimés ni
recalculés côté navigateur. Il contient les fenêtres annuelles pré-calculées,
avec premier/dernier mois partiels. La vue « Archive mensuelle » est nommée
comme référence scellée distincte, jamais substituée à la sélection quotidienne.
Les API historiques peuvent demander le profil explicite `monthly_archive` ;
les CLI de reporting produisent désormais le journalier par défaut.

### Diagnostic des trous et repli explicite (`REPORT-028/029`)

Une mesure quotidienne indisponible ne signifie pas que toute l'histoire manque.
Le diagnostic nomme, pour les seules poches utilisées, le titre, le mois détenu,
les séances exactes absentes et le poids cible. Dans Frontières, les trois
poches du simplex définissent cette sélection ; dans Composer, les poches
cochées la définissent. Les références complètes, notamment SPY, restent
affichées même si aucun mélange n'est quotidiennement optimisable.

La source distincte **Référence mensuelle · repli** mesure les rendements
mensuels déjà scellés selon le
[contrat commun](common_portfolio_backtest_engine.md#sealed-monthly-reference-report-028).
Son activation est volontaire, jamais automatique à la première donnée manquante.
Elle conserve les portefeuilles et le snapshot mais ne crée aucun prix quotidien
et ne corrige pas les anciennes hypothèses de valorisation terminale.

Le bouton affiche ses bornes avant le clic : soit les mêmes bornes calendaires
complètes, soit uniquement les mois complets inclus dans une sélection partielle.
Le service refuse un intervalle mensuel non conforme, sans l'arrondir. Aucun
changement de source/période n'est appliqué par simple survol ou découverte du
trou. Les dates de formulaire modifiées mais non appliquées sont signalées,
distinctement de la fenêtre des résultats déjà calculés.

En repli, graphiques, KPI, model cards, composition et corrélations utilisent
la même grille mensuelle à 12 observations/an. Les labels VaR/CVaR, jours/mois
positifs et Pearson suivent cette fréquence ; aucun faux journalier ni bouton
« Analyser cette plage » au jour près n'est proposé pour un zoom mensuel.
Le journalier strict conserve ses dates libres et son ancienne absence de risque
sur un chemin incomplet. L'audit des trous reste consultable dans les deux modes.

## Laboratoire de portefeuille composé

Le composeur du panneau Analyse permet de cocher plusieurs
stratégies puis de comparer leur portefeuille au SPY. Chaque mois, 100 % du capital est réparti à parts
égales entre les poches cochées ; le poids de chaque poche vaut donc `1 / N` et
le rééquilibrage est mensuel. Les rendements d'entrée sont les `net_return` de
chaque stratégie, déjà diminués des frais facturés dans son propre replay.
Aucun coût supplémentaire de transfert entre poches n'est modélisé dans cette
première version ; cette limite doit rester visible à côté du sélecteur. Les
poches ne sont pas fusionnées au niveau des titres : si deux stratégies
détiennent la même action, cette exposition existe dans chacune de leurs
poches et contribue deux fois selon leurs poids respectifs.

Le profil standard compose dix poches (1 023 combinaisons). Le profil unifié
compose les deux Legacy et les quatre Top 5 (63 combinaisons), limitant le poids
du HTML autonome ; les Top 10/15/20 restent sélectionnables dans « Comparer ».
Les archives annuelles de `equal_weight_strategy_combination_grid()` restent
accessibles. La vue quotidienne appelle `combine_daily_sleeves()` et
`daily_period_analysis()` dans le moteur Python, sur les dates exactes choisies.
Les poids dérivent entre rééquilibrages ; ils ne sont pas remis à `1 / N` chaque
jour. En suivi récent, l'équipondération se fait à l'entrée du panier figé,
sans inventer de rééquilibrage mensuel. Le navigateur ne calcule aucun KPI.
Les textes de la sidebar, du bandeau de composition et des frontières suivent
cette même source via un helper de présentation partagé : le suivi récent ne
revendique jamais le rééquilibrage mensuel ni les frais nets de l'historique.
Ce laboratoire reste un diagnostic post-hoc non promu : il mesure la
diversification entre méthodes, il ne constitue ni une optimisation de poids
ni une nouvelle stratégie de production.

Deux lectures complémentaires sont obligatoires dans ce laboratoire :

- la corrélation de Pearson porte sur les rendements quotidiens, jamais sur les
  niveaux de richesse cumulée. Une matrice bornée par la fenêtre active contient
  uniquement les poches cochées ; le KPI du portefeuille composé mesure de la
  même façon sa corrélation avec les rendements du SPY. Une seconde matrice porte
  sur les rendements actifs `rendement poche - rendement SPY`. L'archive conserve
  séparément ses corrélations mensuelles ;
- la richesse relative n'est pas une corrélation. Pour chaque jour `t`, elle
  vaut `produit(1 + rendement portefeuille) / produit(1 + rendement SPY)` depuis
  le début de la fenêtre. Elle commence à 1 ; au-dessus de 1 le portefeuille a
  davantage composé que SPY, et sa pente indique la direction récente de la
  performance relative. Les pourcentages de performance ne sont jamais divisés
  directement entre eux.

Une corrélation faible ou négative signale une diversification historique plus
forte mais ne prouve ni la robustesse future ni l'intérêt économique de la
poche. Rendement, drawdown, coûts et concentration restent nécessaires pour
l'interprétation.

## Frontières efficientes entre stratégies

Le troisième mode du studio compare des pondérations entre trois poches
méthodologiquement distinctes : `Legacy · Frequency`, `Boosting · Top 5` et
`Boosting tendance · Top 5`. Il ne mélange pas les variantes Top 10/15/20, qui
sont des déclinaisons de concentration des mêmes signaux, ni les variantes PE
diagnostiques. Les contraintes sont fixes : poids longs, somme à 100 %, aucun
levier, pas de 5 %, plafond de 70 % par poche et rééquilibrage mensuel. Les
rendements d'entrée sont déjà nets des frais propres à chaque stratégie ; aucun
coût inter-poche supplémentaire n'est ajouté.

`daily_strategy_frontier()` évalue les 168 pondérations admissibles sur la
fenêtre quotidienne exacte. Le moteur commun calcule les KPI et les profils ;
le navigateur ne fait que les afficher. L'archive conserve aussi la grille
mensuelle `constrained_strategy_frontier_grid()` et ses fenêtres annuelles.
Les deux plans montrent CAGR face à la volatilité annualisée,
puis CAGR face à la valeur absolue du max drawdown. Un point appartient à la
frontière si aucune autre composition n'offre simultanément un risque inférieur
ou égal et un CAGR supérieur ou égal, avec au moins une inégalité stricte.

Trois profils descriptifs sont présentés sur chaque fenêtre : volatilité
minimale, Sharpe maximal et Calmar maximal, lorsque ces métriques sont définies
(aucun faux Calmar si le drawdown est nul). Les égalités sont départagées de
façon déterministe par le second critère de risque puis par l'ordre des poids.
Ces profils sont sélectionnés après observation de la fenêtre : ils constituent
un diagnostic post-hoc, pas une règle causale, un conseil d'investissement ou
une promotion. Une allocation promouvable exigerait une sélection walk-forward
sur la seule histoire disponible à chaque décision, des frais de transfert
inter-poches et une validation temporelle scellée.

Le tableau multi-horizon expose 2010, 2015, 2020, 2023 et 2025 jusqu'au dernier
mois commun lorsque ces fenêtres existent. L'historique comparable commence en
août 2011 : la ligne demandée « 2010 à aujourd'hui » affiche donc explicitement
août 2011 comme début effectif et n'invente aucun mois antérieur.

Les mêmes cartes, graphiques, tiroirs KPI et model cards sont alimentés par la
combinaison active et SPY. La matrice de corrélation entre poches reste dans un
tiroir propre au mode composé. Le rapport expose au minimum :

- rendement total, CAGR, volatilité, Sharpe, drawdown et jours positifs ;
- Sortino, Calmar, alpha, bêta, corrélation, tracking error et information ratio ;
- VaR, CVaR, Omega, captures haussière et baissière, asymétrie et kurtosis ;
- turnover, coûts facturés, nombre de positions et concentrations maximales.

Une concentration sectorielle n'est affichée que si chaque holding de la
stratégie porte un secteur observable. Une colonne secteur absente ou nulle
produit `indisponible`, jamais un faux secteur unique à 100 %.

## Model cards cumulées et annuelles

Deux matrices restent disponibles pour CAGR, volatilité et max drawdown :
cumul depuis chaque année jusqu'à la fin choisie, puis chaque année isolée.
Elles utilisent uniquement les stratégies cochées, SPY et les années de la
fenêtre active. La première et la dernière année respectent les bornes exactes
personnalisées ; aucun janvier ou décembre hors fenêtre n'est réintroduit.

Les valeurs sont calculées par le moteur Python. En annuel isolé, la lecture
rendement/CAGR affiche le rendement effectivement composé de l'année, sans
extrapoler une année de bord partielle. Les mesures de risque restent
quotidiennes et deviennent null si le chemin nécessaire est incomplet.

La palette Viridis reste un encodage visuel, avec valeur numérique dans chaque
cellule. Les anciennes matrices mensuelles restent dans l'archive de référence.
L'histoire comparable commence en août 2011 : aucun mois OOS antérieur n'est
inventé pour répondre à une demande « depuis 2010 ».

## Portefeuille en vigueur

Le dernier mois de performance entièrement réalisé et le portefeuille encore
détenu sont deux objets distincts. Le rapport conserve les courbes, KPI et
model cards jusqu'au dernier mois complet, puis affiche séparément le panier en
vigueur lorsque son `holding_month` est postérieur.

Cette vue expose obligatoirement la dernière séance observée, le mois de
décision, le mois de détention et le statut `rendement mensuel non réalisé`.
Elle lit les holdings live Boosting et les agrégations courantes Legacy du run
explicitement fourni. La date « en vigueur au » provient du maximum de la
preuve prix explicite et doit appartenir au mois de détention ; elle n'est
jamais déduite de l'heure du navigateur ou du nom d'un dossier. Aucun rendement
partiel du mois n'entre dans les KPI et aucun nouveau signal de milieu de mois
n'est inventé.

Dans le profil sectoriel, « Dernier panier du replay » désigne un panier figé
et non un portefeuille en vigueur au moment de la consultation. L'état
`archived_replay_return_pending` signifie seulement que son rendement n'a pas
été évalué par ce replay ; des prix plus récents n'étendent pas ses KPI.

Un suivi prix seul utilise `scripts/research/value_frozen_portfolios.py`, un
rapport source et un reçu RAW nommés explicitement. Il produit par défaut un
nouveau `backtest_performance_report.html` complet et son manifeste, ainsi que
les compléments Markdown/CSV/Parquet, sans réécrire le HTML source historique.
Le [contrat de valorisation](common_portfolio_backtest_engine.md#frozen-basket-price-valuation)
impose les poids dérivés depuis l'entrée, les dates communes à SPY et le libellé
« maintien hypothétique » après le mois prévu. Ce suivi ne remplace ni un
refresh complet des entrées modèles, ni le contrôle inter-vintages obligatoire
lors d'un tel refresh.

Dans le même studio, le mode « Suivi récent » choisit une fenêtre datée et les
mêmes courbes que le comparateur. Performance, drawdown quotidien et richesse
relative au SPY sont pré-calculés par le moteur portefeuille ; le survol donne
le jour exact. Les KPI de ce suivi utilisent la fréquence quotidienne et
signalent la brièveté de l'échantillon ; un CAGR annualisé n'est pas une
prévision. Cette source reste brute de frais, hypothétique après le mois prévu,
et séparée du backtest historique net de frais et de ses 33 KPI mensuels archivés.
Une carte Méthodes expose aussi le pseudo-code du suivi prix et ses limites.

`outputs/performance_reports/index.html` est l'entrée locale stable : elle
désigne le dernier HTML complet généré par le suivi, sans écraser les rapports
datés. Seul cet index identifié comme appartenant au générateur est remplacé,
atomiquement ; une page préexistante étrangère fait échouer sa mise à jour.
Le lien stable et l'état initial du HTML fonctionnent hors ligne. Les nouvelles
fenêtres quotidiennes nécessitent le service Python local, pas une API externe.

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
Le profil sectoriel ajoute une carte qui reprend la règle des cinq titres de
cinq secteurs et affiche les onze libellés lus dans la taxonomie versionnée.
Le [contrat portefeuille](common_portfolio_backtest_engine.md#canonical-holdings-contract)
définit cette règle et liste les secteurs autorisés.

## Publication dans le site Portfolio

AlphaRank génère le HTML complet, ses manifestes et ses ledgers hashés sous
`outputs/performance_reports/<run_id>/`. Le service local charge uniquement ces
artefacts immuables. Portfolio peut consommer l'archive statique et sa navigation,
mais une copie du seul HTML ne fournit pas le moteur Python des dates libres.
Aucun KPI, holding ou statut méthodologique n'est recalculé dans Portfolio ;
aucune publication distante n'est implicite dans une refonte locale.

Commande canonique :

```bash
python scripts/research/build_backtest_performance_report.py \
  --common-replay-dir <replay-commun> \
  --legacy-run-dir <run-legacy> \
  --snapshot-manifest <snapshot-manifest.json> \
  --portfolio-as-of-evidence <price-return-extension-audit.parquet> \
  --output-dir outputs/performance_reports/<run-id>
```

Pour reconstruire l'analyse quotidienne depuis un rapport complet scellé,
sans téléchargement ni relance des modèles :

```bash
python scripts/research/build_daily_performance_report.py \
  --source-report outputs/performance_reports/<source>/backtest_performance_report.html \
  --archive-report outputs/performance_reports/<ancienne-version>/backtest_performance_report.html \
  --output-dir outputs/performance_reports/<nouveau-run>
python scripts/research/serve_performance_report.py \
  --report-dir outputs/performance_reports/<nouveau-run> --port 8765
```

Ouvrir `http://127.0.0.1:8765/`. Les CLI de génération et de suivi prix utilisent
`--analysis-frequency daily` par défaut ; `monthly_archive` est un profil
historique nommé, pas un repli silencieux sur une fenêtre annuelle.

## Validation et preuve

Un nouveau standard ou une nouvelle version du payload exige : tests du moteur
de KPI, test du payload, vérification de la syntaxe JavaScript, génération sur
un replay réel, manifeste hashé, vérification des combinaisons et changements
de fenêtre dans le vrai service et le navigateur, build React verrouillé,
validation documentaire et contrôle des licences embarquées. La QA vérifie
dates de milieu de mois, base 1, journalier, composition, frontières, source
récente distincte, erreurs sans chiffres périmés, archives complètes et mobile.
Un build consommateur est requis si ce consommateur change ; une tâche locale
ne modifie pas Portfolio. L'artefact généré reste ignoré par Git dans AlphaRank ; son run,
son snapshot, ses hashes et son statut sont conservés dans un rapport daté sous
`docs/research/`.
