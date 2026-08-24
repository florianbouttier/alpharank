# SEC Open-Source Status

Derniere mise a jour: `2026-08-16`

Ce document est le point d'entree a lire en premier pour comprendre l'etat reel des fondamentaux open source dans AlphaRank.

## Statut courant - refresh du 2026-08-16

Le nouveau package SEC-only valide est
`outputs/production_refresh_20260816/sec_package_pit_validated`, run source
`20260816_103942`. Les depots SEC atteignent le 2026-08-14. Il est integre au
snapshot modele immutable
`outputs/production_refresh_20260816/composed_history/alpharank_input_20260816_115416_2a01288bab06`.

Correction structurelle: le raw Companyfacts conserve maintenant
`filing_date` dans sa cle de version. Il contient 509 254 lignes et 34 111
groupes de facts ayant plusieurs dates de depot. Le package modele choisit la
premiere version deposee; il ne rejoue plus une valeur restatee ulterieurement
comme si elle etait la seule version historique.

Cette migration modifie massivement l'ancien package de mai, surtout les
actions en circulation. Elle a donc declenche la garde 730 jours sur les cinq
datasets. L'autorisation est une migration ponctuelle documentee dans
`lineage/manifest.json`, avec sa note de revue; ce n'est pas la tolerance par
defaut des refreshs suivants.

## Alerte actuelle sur le package mixte

Clarification de source: les fondamentaux officiels sont SEC/GAAP only. EODHD
est conserve pour les prix historiques et les delisted, pas pour fournir une
valeur fondamentale. Un bridge ticker/CIK EODHD reste autorise pour retrouver
la societe SEC correspondante.

Le dernier package mixte publie est le run `20260811_001503`, snapshot retenu
`open_source_output_20260811_014746`. Il est propre du point de vue publication,
hashes, couverture prix et replay, mais il n'est pas SEC-only. L'audit contre le
snapshot propre precedent trouve quatre valeurs earnings Yahoo 2005-2007
reecrites par la reponse fournisseur du 11 aout. Le backtest commun courant ne
change pas, mais ce constat confirme que `US_Earnings.parquet` mixte ne doit pas
etre presente comme une source point-in-time stable. Voir
`docs/sec_data_robustness_plan.md` et le `data_revision_audit.json` du replay
commun `20260811_001503_035522_standard`.

Ce snapshot a selectionne 19 243 valeurs fondamentales non-SEC dans sa couche
consolidee: 14 754 Yahoo et 4 489 SimFin. Les runs Legacy T1/T2 qui l'utilisent
sont donc des replays reproductibles de l'ancien contrat mixte, pas des runs de
production conformes au contrat SEC-only clarifie.

Ce package est maintenant un repli gele, pas un snapshot conforme au nouveau
contrat: son manifeste precede `source_refresh_contract`, `data_freshness` et la
garde historique prepublication. Il peut rejouer les decisions deja produites,
mais aucune nouvelle production mensuelle ne doit pretendre que les donnees du
13 aout sont validees a partir de ce package.

Le refresh complet `20260813_071802` est conserve mais quarantaine. Son audit
reconstruit contre le snapshot propre du 11 aout detecte des revisions
historiques massives dans income, balance, cash flow, shares et earnings. Il ne
remplace donc pas le package production `20260811_001503`. La garde est
desormais appelee avant publication par le chemin d'ingestion complete; une
garde absente ou un marqueur de quarantaine invalide le package.

## Objectif

Pour les fondamentaux, le chantier ne cherche pas a refaire EODHD: il l'exclut.
Pour les prix, l'archive EODHD gelee reste au contraire necessaire pour les
delisted et doit etre prolongee/corrigee par une couche prix separee.
La cible active est plus etroite et plus mesurable:

- package fondamental `SEC-only`
- KPI coeur: `epsActual`, `revenue`, `net_income`
- objectif qualite: **moins de 1 % de trous sur la pire annee**

## Packages historiques de qualite KPI

La production modele courante n'utilise plus directement les packages de ce
tableau. Elle utilise `sec_package_pit_validated` dans le snapshot compose
immutable pointe par `data/model_inputs/manifests/latest.json`. Les packages
ci-dessous restent des benchmarks historiques du chantier de couverture KPI de
mai; ils ne doivent pas etre confondus avec la production aout 2026.

Pour relire ces experiences, il faut raisonner sur quatre packages differents:

| Package | Role historique | Statut au `2026-05-25` |
| --- | --- | --- |
| `data/sec/output/` | ancienne baseline SEC-only de qualite | **baseline historique, non canonique pour les nouveaux runs** |
| `outputs/sec_kpi_hybrid_output_latest/` | fusion selective KPI entre le meilleur chemin `EPS` et le meilleur chemin `revenue` / `net_income` | **meilleur candidat global observe a date** |
| `outputs/sec_q4_fix2_candidate_combo_output_latest/` | workflow autonome parser Q4 + refresh cible + overlay historique | **meilleur candidat package-entier avant fusion selective** |
| `outputs/sec_overlay_fix2_output/` | meilleur overlay historique teste seul | **meilleur overlay pur a date** |
| `outputs/sec_overlay_multi_history_output/` | overlay multi-snapshots large | **experience utile, non promue** |

Ce qu'il faut retenir pour ces benchmarks historiques:

- `data/open_source/output/` n'est **pas** le bon package pour piloter l'objectif `<1 %` sur `EPS`, `revenue` et `net_income`
- le travail KPI coeur se faisait sur la branche `SEC-only`
- `outputs/sec_kpi_hybrid_output_latest/` est le meilleur resultat global concret observe jusqu'ici
- l'overlay multi-history a enrichi la couverture brute, mais il a aussi rouvert des annees anciennes plus trouees et a donc degrade le pire cas

## Score actuel

Fenetre de lecture standardisee pour ce document:

- `2010` -> `2025`
- meme fenetre pour la baseline SEC et pour les overlays compares
- le fichier le plus court a publier apres chaque run est `worst_year_brief.md`

### Baseline historique: `data/sec/output/`

Rapports de reference:

- `outputs/sec_quality_dashboard_latest/`
- `outputs/sec_core_kpi_yearly_report_latest/`
- `outputs/sec_core_kpi_yearly_report_latest/worst_year_brief.md`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 70 | `2.73 %` |
| `net_income` | `2023` | 88 | `3.42 %` |
| `revenue` | `2023` | 90 | `3.50 %` |

### Meilleur overlay a date: `outputs/sec_overlay_fix2_output/`

Rapports associes:

- `outputs/sec_overlay_fix2_quality/`
- `outputs/sec_overlay_fix2_yearly/`
- `outputs/sec_overlay_fix2_yearly/worst_year_brief.md`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 87 | `3.40 %` |
| `net_income` | `2023` | 56 | `2.17 %` |
| `revenue` | `2023` | 62 | `2.40 %` |

Interpretation:

- `fix2` reste la meilleure brique historique pure pour `revenue` et `net_income`
- il n'est **pas** le meilleur sur `EPS`
- il sert maintenant de composant du package hybride, pas de candidat final recommande
- seul, il ne permet pas d'atteindre la cible `<1 %`

### Probe parser Q4: `outputs/sec_q4_probe_output/`

Cette experience correspond a:

- un correctif de derive `Q4` dans le parseur `companyfacts`
- un refresh cible de `20` tickers tres presents dans les trous `2023`
- un rebuild SEC-only de staging sur la fenetre `2010 -> 2025`

Rapports associes:

- `outputs/sec_q4_probe_quality/`
- `outputs/sec_q4_probe_yearly/`
- `outputs/sec_q4_probe_yearly/worst_year_brief.md`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 64 | `2.50 %` |
| `net_income` | `2023` | 74 | `2.87 %` |
| `revenue` | `2023` | 76 | `2.95 %` |

Interpretation:

- le correctif parser apporte un **gain reel** par rapport a la baseline SEC actuelle sur les trois KPI coeur
- il a servi de base au chemin `q4_fix2_candidate`, puis au package hybride final
- seul, il reste moins bon que `fix2` sur `revenue` et `net_income`

### Experience non promue: `outputs/sec_overlay_multi_history_output/`

Rapports associes:

- `outputs/sec_overlay_multi_history_quality/`
- `outputs/sec_overlay_multi_history_yearly/`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2009` | 96 | `5.12 %` |
| `net_income` | `2008` | 65 | `6.35 %` |
| `revenue` | `2008` | 65 | `6.48 %` |

Pourquoi ce package n'est pas promu:

- il assemble plusieurs snapshots SEC historiques
- cela recupere bien des quarters absents du snapshot courant
- mais cela agrandit aussi le perimetre historique audite sur des annees plus anciennes et beaucoup plus trouees
- le pire cas se degrade donc, ce qui va a l'encontre de l'objectif metier

Conclusion:

- garder cette experience comme outillage de diagnostic
- ne pas la presenter comme le nouveau package de reference

### Meilleur candidat global actuel: `outputs/sec_kpi_hybrid_output_latest`

Cette experience combine vraiment les deux methodes qui avaient des avantages differents:

- base `q4_fix2_candidate` pour garder le gain de refresh cible et le correctif `Q4`
- `revenue` et `net_income` repris depuis `fix2`, puis backfilles avec la base si besoin
- `epsActual` garde la base `q4_fix2_candidate`, puis recupere les quarts ou `fix2` publie un `EPS` non nul alors que la base ne l'a pas

Rapports associes:

- `outputs/sec_kpi_hybrid_quality_latest/`
- `outputs/sec_kpi_hybrid_yearly_latest/`
- `outputs/sec_kpi_hybrid_yearly_latest/worst_year_brief.md`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 22 | `0.86 %` |
| `net_income` | `2023` | 49 | `1.90 %` |
| `revenue` | `2023` | 49 | `1.90 %` |

Interpretation:

- oui, les deux methodes peuvent etre combinees proprement
- la fusion selective domine maintenant tous les autres scenarios testes
- `EPS` passe sous `1 %`
- `revenue` et `net_income` n'y sont pas encore, mais on descend de `3.50 %` sur la baseline a `1.90 %`

### Meilleur candidat package-entier avant fusion selective: `outputs/sec_q4_fix2_candidate_combo_output_latest`

Cette experience combine:

- le parseur `Q4` corrige cote `companyfacts`
- un refresh cible automatique de la cohorte KPI la plus trouee
- l'overlay `fix2` comme couche de comblement des trous residuels

Rapports associes:

- `outputs/sec_q4_fix2_candidate_combo_quality_latest/`
- `outputs/sec_q4_fix2_candidate_combo_yearly_latest/`
- `outputs/sec_q4_fix2_candidate_combo_yearly_latest/worst_year_brief.md`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 53 | `2.07 %` |
| `net_income` | `2023` | 67 | `2.60 %` |
| `revenue` | `2023` | 68 | `2.64 %` |

Interpretation:

- c'etait le meilleur **package entier** avant de passer a une fusion selective par KPI
- il garde le gain `EPS` du probe parser Q4
- il recupere une partie du gain `revenue` / `net_income` de `fix2`
- il reste utile comme base de travail pour le workflow autonome

## Ce qui est deja solide

- le package `SEC-only` est bien separe du package multi-source
- les fondamentaux publies restent 100 % SEC dans le package cible
- les snapshots historiques existent sous `data/sec/history/output/`
- un script d'overlay reproductible existe maintenant: `scripts/open_source/build_sec_overlay_package.py`
- ce script sait merger plusieurs snapshots et conserver la provenance via `overlay_origin`
- un script de fusion selective KPI existe maintenant: `scripts/open_source/build_sec_metric_hybrid_package.py`

## Ce qui bloque encore la cible `<1 %`

Les vrais trous restants ne se reglent plus seulement par de l'empilage de snapshots.
Les chantiers prioritaires sont:

1. **Canonicalisation fiscale**
   - quarters mal alignes pour les societes a exercice decale
   - conventions d'annee fiscale heterogenes autour des clotures janvier/fevrier

2. **EPS**
   - trous persistants sur `epsActual`
   - besoin de mieux exploiter le filing-level per-share et les cas de fallback derives deja admis par le contrat SEC

3. **Historique ancien / tickers delistes**
   - certains snapshots recuperent plus de profondeur, mais avec une qualite tres inegale sur 2008-2010
   - il faut distinguer les vrais gains de couverture des regressions de perimetre

4. **Cohorte auditee**
   - les comparaisons peuvent bouger quand le nombre de tickers attendus change
   - il faut toujours lire ensemble `expected_quarters`, `present_quarters` et la plage d'annees analysee

## Workflow recommande

### 1. Baseline officielle

Le package a utiliser par defaut reste:

- `data/sec/output/`

### 2. Overlay candidat

Pour rejouer l'overlay candidat le plus utile:

```bash
./.venv/bin/python scripts/open_source/build_sec_overlay_package.py \
  --primary-sec-dir data/sec/output \
  --secondary-sec-dir outputs/sec_candidate_fix2_output \
  --output-dir outputs/sec_overlay_fix2_output
```

### 3. Recalcul des rapports

```bash
./.venv/bin/python scripts/open_source/build_sec_quality_dashboard.py \
  --sec-output-dir outputs/sec_overlay_fix2_output \
  --output-dir outputs/sec_overlay_fix2_quality
```

```bash
./.venv/bin/python scripts/open_source/build_sec_core_kpi_yearly_report.py \
  --sec-output-dir outputs/sec_overlay_fix2_output \
  --quality-dir outputs/sec_overlay_fix2_quality \
  --start-year 2010 \
  --end-year 2025 \
  --output-dir outputs/sec_overlay_fix2_yearly
```

### 4. Overlay multi-history

Le script accepte aussi plusieurs `--secondary-sec-dir`, mais cette voie doit etre traitee comme une experience de recherche tant qu'elle degrade le pire cas annuel.

### 5. Workflow autonome recommande

Pour rejouer sans bricolage manuel le pipeline qui:

- selectionne automatiquement les tickers les plus troues sur les KPI coeur
- refresh les `companyfacts` SEC de cette cohorte
- rebuild un package probe
- applique `fix2` par-dessus
- construit ensuite le package hybride KPI
- republie les rapports annuels et le brief KPI

utiliser:

```bash
./.venv/bin/python scripts/open_source/run_sec_q4_fix2_candidate.py
```

Le script ecrit un manifest horodate et republie aussi:

- `outputs/sec_kpi_hybrid_output_latest/`
- `outputs/sec_kpi_hybrid_quality_latest/`
- `outputs/sec_kpi_hybrid_yearly_latest/`

### 6. Comparaison lisible des scenarios

Pour publier une vue non ambigue de type:

- scenario par scenario
- par KPI
- pire annee
- nombre de trous
- `%` de missing
- classement global

utiliser:

```bash
./.venv/bin/python scripts/open_source/build_sec_kpi_scenario_comparison.py
```

Sortie principale:

- `outputs/sec_kpi_scenario_comparison_latest/summary.md`

## Documents a lire ensuite

Une fois ce statut lu, les docs utiles sont:

- `docs/sec_fundamentals_contract.md`
- `docs/open_source_ingestion_architecture.md`
- `docs/archive/reports/open_source_cadrage_status_2025.md` pour le contexte historique du package multi-source
- `docs/archive/reports/audit_donnees_financieres_2025.md` pour l'audit detaille du chantier

## Regle de mise a jour

Ce document doit etre mis a jour a chaque fois qu'un de ces points change:

- package de reference recommande
- meilleur overlay candidat
- score de pire annee sur `epsActual`, `revenue` ou `net_income`
- interpretation officielle des experimentations historiques

## Politique de fraicheur production - 2026-08-13

Le cache HTTP n'est plus une source de verite. Un snapshot production doit
porter `source_refresh_contract.snapshot_scope=full_ingestion` et exposer
`data_freshness`.

- `companyfacts` SEC est retelcharge en entier a chaque ingestion complete, y
  compris son historique potentiellement revise;
- les index `submissions` SEC mutables sont retelcharges;
- les XML XBRL par accession, immuables, sont charges a la demande mais ne sont
  plus stockes durablement;
- les historiques prix Yahoo/StockAnalysis et les fichiers bulk SimFin sont
  rafraichis selon le contrat de production;
- `official/raw` reste la base normalisee durable; `_cache/` est jetable.

La fraicheur doit toujours distinguer la fin de periode fiscale de la date de
filing SEC. La premiere peut etre en juin pour un snapshot d'aout; la seconde
doit passer le seuil de fraicheur et prouver que SEC a reellement ete interroge.

## Reconstruction de la couverture SEC historique - 2026-08-16

Le snapshot modele immutable courant reste inchange. Une reconstruction
candidate separee a audite les 67 identites de tickers historiques qui
manquaient auparavant dans le chemin de mapping SEC:

- bridge: `src/alpharank/data/open_source/reference/sec_historical_ticker_bridge.csv`;
- run: `outputs/production_refresh_20260816/sec_historical_reconstruction_67_v3`;
- 67/67 identites ticker-vers-CIK telechargees et concordantes;
- 63/67 identites avec des lignes financieres SEC lisibles par machine;
- 17 971 lignes candidates Companyfacts/filing-level reconstruites;
- FRC, MIL, RX et SBNY restent sans ligne exploitable par ce chemin
  d'extraction.

Cette candidate n'est volontairement pas fusionnee dans le package officiel.
Sa promotion exige la revue des identites, le garde de revision du package
complet, un nouveau snapshot immutable et de nouveaux runs Legacy/Boosting.
La comparaison courante reste donc reproductible et le passe n'est pas reecrit
silencieusement.
