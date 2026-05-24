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
| `outputs/sec_overlay_fix2_output/` | meilleur overlay historique teste | **meilleur candidat d'amelioration a date** |
| `outputs/sec_overlay_multi_history_output/` | overlay multi-snapshots large | **experience utile, non promue** |

Ce qu'il faut retenir:

- `data/open_source/output/` n'est **pas** le bon package pour piloter l'objectif `<1 %` sur `EPS`, `revenue` et `net_income`
- le travail KPI coeur se fait aujourd'hui sur la branche `SEC-only`
- `outputs/sec_overlay_fix2_output/` est le meilleur resultat concret observe jusqu'ici
- l'overlay multi-history a enrichi la couverture brute, mais il a aussi rouvert des annees anciennes plus trouees et a donc degrade le pire cas

## Score actuel

### Baseline officielle: `data/sec/output/`

Rapports de reference:

- `outputs/sec_quality_dashboard_latest/`
- `outputs/sec_core_kpi_yearly_report_latest/`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 117 | `4.57 %` |
| `net_income` | `2023` | 57 | `2.21 %` |
| `revenue` | `2023` | 63 | `2.44 %` |

Resume global:

| KPI | Trous totaux | Taux global |
| --- | ---: | ---: |
| `epsActual` | 779 | `1.83 %` |
| `net_income` | 914 | `2.14 %` |
| `revenue` | 943 | `2.21 %` |

### Meilleur overlay a date: `outputs/sec_overlay_fix2_output/`

Rapports associes:

- `outputs/sec_overlay_fix2_quality/`
- `outputs/sec_overlay_fix2_yearly/`

Pire annee par KPI:

| KPI | Pire annee | Trous | Taux de manque |
| --- | --- | ---: | ---: |
| `epsActual` | `2022` | 87 | `3.40 %` |
| `net_income` | `2023` | 56 | `2.17 %` |
| `revenue` | `2023` | 62 | `2.40 %` |

Resume global:

| KPI | Trous totaux | Taux global |
| --- | ---: | ---: |
| `epsActual` | 815 | `1.91 %` |
| `net_income` | 714 | `1.65 %` |
| `revenue` | 849 | `1.96 %` |

Interpretation:

- l'overlay `fix2` est le **meilleur compromis** observe
- il ameliore nettement `EPS` sur la pire annee
- il ameliore aussi `net_income` et `revenue`, mais plus modestement
- il ne permet pas encore d'atteindre la cible `<1 %`

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
