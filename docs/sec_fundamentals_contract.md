# SEC Fundamentals Contract

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

Les prix doivent venir d'un package separe:

- aujourd'hui: un package prix dedie ou un mirror existant
- cible: `data/stitched_prices/output`

## Politique Source

Pour ce package, la regle est volontairement stricte:

- source canonique = SEC
- acces SEC niveau 1 = `companyfacts`
- acces SEC niveau 2 = `filing-level XBRL`

Donc `companyfacts` et `filing-level` ne sont pas deux vendors differents.
Ce sont deux facons d'extraire la meme source officielle.

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

Cle logique d'un fait financier:

- `ticker`
- `statement`
- `metric`
- `fiscal_year`
- `fiscal_period`

Quand plusieurs candidats SEC existent pour le meme quarter logique:

- on privilegie `companyfacts`
- sinon `filing-level`
- le fichier de lineage doit garder les candidats vus et la source retenue

### Earnings

Pour `US_Earnings.parquet`:

- calendrier canonique: SEC
- `period_end`: SEC
- `reportDate`: SEC
- `epsActual`: SEC
- `epsEstimate`: `null`
- `surprisePercent`: `null`

Cle logique:

- `ticker`
- `fiscal_year`
- `fiscal_period`

Si plusieurs filings SEC concernent le meme quarter:

- la version la plus recente et la plus exploitable est retenue
- le lineage doit permettre de remonter au filing retenu

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
- `accession_number` quand disponible
- `is_derived` pour les KPI derives, si on en ajoute plus tard

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
