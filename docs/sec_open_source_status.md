# SEC Open-Source Status

Derniere mise a jour: `2026-05-24`

Ce document est le point d'entree a lire en premier pour comprendre l'etat reel des fondamentaux open source dans AlphaRank.

## Objectif

Le chantier actuel ne cherche pas a refaire tout EODHD.
La cible active est plus etroite et plus mesurable:

- package fondamental `SEC-only`
- KPI coeur: `epsActual`, `revenue`, `net_income`
- objectif qualite: **moins de 1 % de trous sur la pire annee**

## Decision actuelle

Pour les fondamentaux, il faut raisonner sur trois packages differents:

| Package | Role | Statut au `2026-05-24` |
| --- | --- | --- |
| `data/sec/output/` | package SEC-only canonique | **base officielle actuelle** |
| `outputs/sec_q4_fix2_combo_output/` | combinaison parser Q4 + overlay historique | **meilleur candidat global observe a date** |
| `outputs/sec_overlay_fix2_output/` | meilleur overlay historique teste seul | **meilleur overlay pur a date** |
| `outputs/sec_overlay_multi_history_output/` | overlay multi-snapshots large | **experience utile, non promue** |

Ce qu'il faut retenir:

- `data/open_source/output/` n'est **pas** le bon package pour piloter l'objectif `<1 %` sur `EPS`, `revenue` et `net_income`
- le travail KPI coeur se fait aujourd'hui sur la branche `SEC-only`
- `outputs/sec_q4_fix2_combo_output/` est le meilleur resultat global concret observe jusqu'ici
- l'overlay multi-history a enrichi la couverture brute, mais il a aussi rouvert des annees anciennes plus trouees et a donc degrade le pire cas

## Score actuel

Fenetre de lecture standardisee pour ce document:

- `2010` -> `2025`
- meme fenetre pour la baseline SEC et pour les overlays compares
- le fichier le plus court a publier apres chaque run est `worst_year_brief.md`

### Baseline officielle: `data/sec/output/`

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

- l'overlay `fix2` est le **meilleur compromis** observe
- il n'est **pas** le meilleur sur `EPS`: la baseline SEC actuelle fait mieux sur ce KPI
- il ameliore nettement `revenue` et `net_income`
- il reste le meilleur compromis global sur les trois KPI coeur, car son pire cas global (`3.40 %`) est legerement meilleur que celui de la baseline (`3.50 %`)
- il ne permet pas encore d'atteindre la cible `<1 %`

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
- c'est le meilleur resultat observe a date sur `EPS`
- il reste moins bon que `fix2` sur `revenue` et `net_income`
- la prochaine experience logique est donc la combinaison `q4_probe + fix2`

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

### Meilleur candidat global actuel: `outputs/sec_q4_fix2_combo_output/`

Cette experience combine:

- le parseur `Q4` corrige cote `companyfacts`
- le refresh cible de la cohorte 2023 la plus trouee
- l'overlay `fix2` comme couche de comblement des trous residuels

Rapports associes:

- `outputs/sec_q4_fix2_combo_quality/`
- `outputs/sec_q4_fix2_combo_yearly/`
- `outputs/sec_q4_fix2_combo_yearly/worst_year_brief.md`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 64 | `2.50 %` |
| `net_income` | `2023` | 70 | `2.72 %` |
| `revenue` | `2023` | 72 | `2.80 %` |

Interpretation:

- c'est le meilleur **pire cas global** observe jusqu'ici sur les trois KPI coeur
- il garde le gain `EPS` du probe parser Q4
- il recupere une partie du gain `revenue` / `net_income` de `fix2`
- la cible `<1 %` reste loin, mais le frontiere actuelle descend de `3.50 %` pour la baseline a `2.80 %` sur ce combo

## Ce qui est deja solide

- le package `SEC-only` est bien separe du package multi-source
- les fondamentaux publies restent 100 % SEC dans le package cible
- les snapshots historiques existent sous `data/sec/history/output/`
- un script d'overlay reproductible existe maintenant: `scripts/open_source/build_sec_overlay_package.py`
- ce script sait merger plusieurs snapshots et conserver la provenance via `overlay_origin`

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

## Documents a lire ensuite

Une fois ce statut lu, les docs utiles sont:

- `docs/sec_fundamentals_contract.md`
- `docs/open_source_ingestion_architecture.md`
- `docs/open_source_cadrage_status_2025.md` pour le contexte historique du package multi-source
- `docs/audit_donnees_financieres_2025.md` pour l'audit detaille du chantier

## Regle de mise a jour

Ce document doit etre mis a jour a chaque fois qu'un de ces points change:

- package de reference recommande
- meilleur overlay candidat
- score de pire annee sur `epsActual`, `revenue` ou `net_income`
- interpretation officielle des experimentations historiques
