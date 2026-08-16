# SEC Fundamentals Contract

Pour le statut courant du package, le package recommande a date et les derniers resultats de couverture, lire aussi `docs/sec_open_source_status.md`.
Pour le plan de robustesse data/replay suite a l'incident de drift de juin 2026, lire aussi `docs/sec_data_robustness_plan.md`.

Ce document fixe le contrat officiel du package `data/sec/output`.

Le but est d'avoir un package fondamental simple a comprendre:

- une seule source de verite pour les fondamentaux: la SEC
- un lineage officiel publie a cote du package
- une documentation qui explique les choix de consolidation, pas seulement les fichiers produits

## Perimetre

`data/sec/output` est un package **fondamentaux SEC-only**.

Il contient:

- `US_Income_statement.parquet`
- `US_share.parquet`
- `US_Earnings.parquet`
- `US_General.parquet`
- `lineage/manifest.json`
- les parquets de lineage SEC associes

Il ne contient pas:

- de prix
- de fallback Yahoo
- de fallback SimFin
- de fallback EODHD

Cette exclusion est semantique, pas seulement technique: les fondamentaux
EODHD historiques peuvent utiliser des definitions non-GAAP qui ne sont pas
comparables aux facts GAAP SEC. Une valeur EODHD ne doit donc jamais completer
un trou SEC dans un snapshot officiel, meme si elle existe localement.

Nuance importante:

- un bridge d'identifiant historique peut etre utilise pour retrouver un `CIK` SEC ancien ou delisted
- ce bridge peut venir d'un referentiel local comme `data/eodhd/output/US_General.parquet`
- un bridge manuel versionne peut aussi etre maintenu dans `src/alpharank/data/open_source/reference/sec_historical_ticker_bridge.csv`
- ce bridge ne fournit jamais une valeur fondamentale finale
- la valeur fondamentale finale doit toujours venir de la SEC

Les prix doivent venir d'un package separe:

- socle historique: archive EODHD gelee pour conserver les delisted
- extension/correction: sources de prix ouvertes et overlays corporate actions
- cible: un package prix historise, joint au package SEC-only seulement lors de
  la composition du snapshot modele

Une correction de split ou dividende peut modifier le package prix derive mais
ne change jamais le contrat fondamental SEC-only. L'archive EODHD brute reste
immutable; les corrections sont versionnees et tracees dans le lineage prix.

## Composition Du Snapshot Modele

Un snapshot officiel Legacy doit reunir dans un package immutable:

- `US_Finalprice.parquet` et `SP500Price.parquet` issus du package prix hybride
  approuve;
- income, balance, cash flow, shares, earnings et general issus exclusivement
  d'un snapshot historise de `data/sec/output`;
- le calendrier historique des constituants;
- les deux manifests de lineage, leurs hashes et l'identifiant de composition.

Copier directement tous les fichiers de `data/open_source/output` ne respecte
pas ce contrat: ce dossier contient encore des fondamentaux Yahoo/SimFin. Un
snapshot compose doit echouer si `selected_source` contient autre chose que
`sec_companyfacts`, `sec_filing` ou une derivation explicitement SEC-only.

## Politique Source

Pour ce package, la regle est volontairement stricte:

- source canonique = SEC
- acces SEC niveau 1 = `companyfacts`
- acces SEC niveau 2 = `filing-level XBRL`

Donc `companyfacts` et `filing-level` ne sont pas deux vendors differents.
Ce sont deux facons d'extraire la meme source officielle.

Pour les tickers anciens, renommes ou delisted, l'acces a la SEC peut necessiter un bridge `ticker -> CIK`.
Ce bridge fait partie de la plomberie d'acces, pas de la source fondamentale.

Quand un ticker a ete recycle par une autre societe plus recente:

- le bridge historique doit porter une fenetre de validite (`start_date`, `end_date`)
- le package SEC final doit exclure les rows SEC hors de cette fenetre pour ce ticker
- cela evite de melanger deux societes differentes sous un meme symbole historique

Si la SEC ne permet pas de reconstruire proprement une valeur:

- on laisse `null`
- on ne complete pas avec un vendor externe

## KPI Couverts

Les KPI cibles aujourd'hui sont:

- `revenue`
- `net_income`
- `outstanding_shares`
- `epsActual`

Autres points:

- `free_cash_flow` n'est pas un KPI SEC natif. Si on l'ajoute un jour, il devra etre marque `derived_from_sec`.
- `epsActual` est prioritairement le `EPS` SEC publie. Si ce tag SEC manque mais que `net_income` et une base d'actions SEC existent pour le meme quarter fiscal, un fallback `sec_derived_eps` peut etre publie. La base d'actions est prise en priorite sur `outstanding_shares`, puis sur `weighted_average_diluted_shares` si la serie diluee existe mais pas les actions en circulation. Ce fallback doit rester explicitement trace dans le lineage.
- pour l'EPS SEC publie, on privilegie les facts trimestriels directs `Q1..Q4`. On n'utilise pas de `Q4` synthetique derive d'un `FY` annuel dans la couche `earnings_sec_actuals`, car cela cree trop de mismatches historiques.
- `epsEstimate` et `surprisePercent` ne font pas partie du package SEC-only. La SEC n'est pas la bonne source pour ces champs.

## Fichiers Officiels

Package user-facing:

- `data/sec/output/US_Income_statement.parquet`
- `data/sec/output/US_share.parquet`
- `data/sec/output/US_Earnings.parquet`
- `data/sec/output/US_General.parquet`

Lineage user-facing:

- `data/sec/output/lineage/financials_sec_consolidated.parquet`
- `data/sec/output/lineage/financials_sec_lineage.parquet`
- `data/sec/output/lineage/earnings_sec_consolidated.parquet`
- `data/sec/output/lineage/earnings_sec_lineage.parquet`
- `data/sec/output/lineage/general_reference.parquet`
- `data/sec/output/lineage/general_reference_lineage.parquet`
- `data/sec/output/lineage/manifest.json`

Snapshots historises:

- `data/sec/history/output/sec_output_<timestamp>/`

## Regles de Consolidation

### Financials

Ordre de selection:

1. `sec_companyfacts`
2. `sec_filing`

Exception importante:

- pour `outstanding_shares`, on privilegie `sec_filing` avant `sec_companyfacts`
- raison: la cover page DEI et les facts filing-level sont plus auditables et evitent une partie des ambiguities de `companyfacts`

Cle logique d'un fait financier:

- `ticker`
- `statement`
- `metric`
- `fiscal_year`
- `fiscal_period`

Cle de conservation dans le raw SEC Companyfacts:

- `ticker`
- `statement`
- `metric`
- `date`
- `filing_date`
- `source`

`filing_date` est obligatoire dans cette cle: deux depots qui republient le
meme fait sont deux versions distinctes. Le raw conserve toutes les versions.
Pour le snapshot causal utilise par les modeles, on selectionne la premiere
version deposee pour chaque `(ticker, statement, metric, date, source)` avant la
normalisation trimestrielle. Une version ulterieure ne doit jamais remplacer la
valeur qui etait disponible a la date initiale dans un replay historique.

Quand plusieurs candidats SEC existent pour le meme quarter logique:

- on privilegie `companyfacts`
- sinon `filing-level`
- le fichier de lineage doit garder les candidats vus et la source retenue

Quand plusieurs lignes `outstanding_shares` SEC existent pour le meme quarter fiscal:

- on ne garde qu'une seule ligne finale par `(ticker, fiscal_year, fiscal_period)`
- on choisit la ligne la plus canonique pour le quarter, avec une preference pour la date de fin de quarter plutot qu'une date de filing intermediaire
- les doublons de type cover page / reprise du meme quarter dans un autre filing ne doivent pas apparaitre comme plusieurs rows finales

Derivations acceptees dans ce package:

- `free_cash_flow = operating_cash_flow - capital_expenditures`
- `total_liabilities = total_assets - stockholders_equity` uniquement quand le filing SEC ne fournit pas directement la ligne et que les deux composantes existent

Derivation non acceptee dans ce package:

- recalculer les `outstanding_shares` a partir de `net_income / epsActual`

Filtres qualite acceptes dans ce package:

- supprimer les `outstanding_shares <= 0`
- supprimer les `outstanding_shares` manifestement absurdes a plusieurs ordres de grandeur du voisinage
- ce filtrage est un filtre qualite SEC-only, pas un fallback vendor

### Earnings

Pour `US_Earnings.parquet`:

- calendrier canonique: SEC
- `period_end`: SEC
- `reportDate`: SEC
- `epsActual`: SEC publie si disponible, sinon `sec_derived_eps = net_income / share_base` en fallback explicite
- `epsEstimate`: `null`
- `surprisePercent`: `null`

Cle logique:

- `ticker`
- `fiscal_year`
- `fiscal_period`

Si plusieurs filings SEC concernent le meme quarter:

- la premiere version exploitable par `reportDate` est retenue pour le replay
  causal; les versions ulterieures restent dans le raw
- le lineage doit permettre de remonter au filing retenu
- au moment de publier le package SEC-only, les `sec_actuals` sont realignes une premiere fois sur le calendrier SEC par `reportDate`, puis une seconde fois par `ticker + fiscal_year + fiscal_period` si la date de publication seule ne suffit pas

### General

`US_General.parquet` dans ce package reste un support de mapping SEC:

- `ticker`
- `name`
- `exchange`
- `cik`
- `sic`
- `sic_description`

Si des champs sectoriels existent:

- ils doivent etre explicitement etiquetes comme venant de la logique SEC de mapping
- ils ne doivent pas etre presentes comme des metadonnees vendor generiques

## Normalisation des Quarters

Le package SEC-only raisonne en quarter fiscal canonique, pas uniquement en date brute.

Pourquoi:

- des boites ont des exercices decales
- la meme information peut apparaitre avec des dates legerement differentes
- un audit sur date brute surestime artificiellement les trous

La normalisation officielle actuelle repose sur:

- `fiscal_year`
- `fiscal_period`
- la date de quarter retenue apres normalisation canonique

Regles complementaires importantes:

- quand `fiscal_period` source est deja valide (`Q1..Q4`), on le preserve en priorite
- le mois de cloture modal du `Q4` sert a recanoniser l'annee fiscale
- pour les clotures janvier/fevrier, la convention d'annee fiscale est detectee par ticker, car certaines societes nomment l'exercice sur l'annee de cloture et d'autres sur l'annee precedente
- si une sous-sequence source recente est localement coherente (`Q1 -> Q2 -> Q3 -> Q1` apres un quarter manquant, par exemple), on preserve ces labels source meme si l'historique ancien du ticker est bruité
- l'annee fiscale source n'est conservee que si elle reste compatible avec le mois reel de cloture observe; sinon, l'annee est recalculee a partir du calendrier fiscal detecte
- apres canonicalisation, on ne garde qu'une seule ligne finale par `ticker x metric x fiscal_year x fiscal_period`

Pour `outstanding_shares`, la normalisation inclut aussi:

- une selection d'une seule ligne canonique par quarter fiscal
- une preference pour la date la plus representative du quarter plutot qu'une date de filing plus tardive

Si cette logique change:

- il faut mettre a jour ce document
- il faut mettre a jour le rapport d'audit
- il faut l'indiquer dans `manifest.json` ou le changelog de run associe

## Ce Que Le Lineage Doit Toujours Expliquer

Pour chaque ligne finale, on doit pouvoir repondre a ces questions:

1. Quelle source a gagne?
2. Quels autres candidats existaient?
3. Quel quarter logique est vise?
4. Quel filing ou quel endpoint SEC a servi?
5. Est-ce une valeur reportee ou derivee?

Champs minimaux attendus dans les exports de lineage:

- `selected_source`
- `candidate_sources`
- `selected_fiscal_year`
- `selected_fiscal_period`
- `selected_form`
- `selected_accession_number` quand disponible
- `is_derived` pour les KPI derives, si on en ajoute plus tard

## Regle De Regeneration Du Raw SEC

Changer le parseur SEC ou la liste de tags SEC ne suffit pas a ameliorer le package publie.

Si on modifie:

- la logique de parsing `companyfacts`
- la logique de derivation trimestrielle
- la liste de tags SEC supportes

alors il faut aussi regenerer les raw SEC correspondants avant de rebuild `data/sec/output`, au minimum:

- `data/open_source/official/raw/financials_sec_companyfacts.parquet`
- `data/open_source/official/raw/earnings_sec_actuals.parquet`

Sinon:

- le code source parait corrige
- mais le package publie et les dashboards continuent a lire un raw stale
- et les gains de couverture n'apparaissent pas

## Limitations Acceptees

Ce package privilegie la coherence semantique sur la couverture maximale.

Limitations acceptees:

- certains vieux filings pre-XBRL restent difficiles a parser
- certains concepts SEC ne sont pas universels selon les secteurs
- certaines banques/assureurs ne mappent pas naturellement sur un concept simple de `revenue`
- des valeurs peuvent rester `null` si la SEC n'est pas exploitable proprement

Limitation non acceptee:

- melanger silencieusement un autre vendor dans ce package

## Regle de Documentation

A chaque fois qu'on modifie un de ces points:

- source priority
- natural key
- quarter normalization
- fallback policy
- schema des fichiers user-facing
- schema des fichiers de lineage

il faut mettre a jour dans le meme travail:

1. ce document
2. le `README.md` si le package user-facing change
3. le document d'architecture global si le rangement ou le contrat de stockage change

Le repo doit rester lisible pour quelqu'un qui revient des semaines plus tard sans relire tout l'historique des discussions.
