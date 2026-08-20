# Audit des Données Financières — AlphaRank

> **Archive datée.** Ce rapport est conservé comme preuve historique et ne
> décrit pas automatiquement l'état courant des données.

**Date de l'audit :** 10 mai 2026  
**Auditeur :** OpenCode (agent IA)  
**Périmètre :** Données fondamentales (financials) dans les packages `data/sec/output/` et `data/open_source/output/`, avec comparaison legacy EODHD  
**Objectif :** Challenger la structure et les résultats produits par Codex, identifier les blocages pour le backtest S&P 500, et recommander des actions concrètes.

---

## 1. Résumé Exécutif

| Package | Utilisable pour backtest historique ? | Verdict |
|---------|--------------------------------------|---------|
| **EODHD legacy** (`data/eodhd/output/`) | Oui, mais source payante coupée | Référence historique, non maintenu |
| **SEC-only** (`data/sec/output/`) | **Oui, avec réserves** | **Meilleur candidat actuel** |
| **Open Source** (`data/open_source/output/`) | **Non** — historique quasi nul | Inutilisable en l'état |

**Conclusion immédiate :** Si tu dois lancer un backtest aujourd'hui, utilise le package **SEC-only** pour les fondamentaux. L'open-source output n'a quasiment pas d'historique (seulement 2025-2026) et dépend à 46 % de sources non-SEC (Yahoo, SimFin). Le SEC package a une couverture historique solide (2006-2026) et est 100 % GAAP/SEC.

---

## 2. Méthodologie

L'audit s'appuie sur :
1. **Lecture du code source** : `config.py`, `sec.py`, `sec_filing.py`, `consolidation.py`, `legacy_export.py`, `sec_only.py`, `earnings.py`
2. **Analyse des données réelles** : parquets dans `data/sec/output/`, `data/open_source/output/`, `data/open_source/official/raw/`, `data/eodhd/output/`
3. **Vérification des contrats** : `docs/sec_fundamentals_contract.md`, `docs/open_source_ingestion_architecture.md`
4. **Exécution des tests** : `tests/test_open_source_*.py`
5. **Contrôle de couverture S&P 500** : comparaison avec `SP500_Constituents.csv`

---

## 3. Résultats par Package

### 3.1 Package SEC-only (`data/sec/output/`)

#### 3.1.1 Qualité de la source
✅ **100 % SEC** — Le lineage montre 324 404 lignes sur 326 631 provenant de `sec_companyfacts`, et 2 227 de `sec_filing`. Aucune donnée Yahoo, SimFin ou EODHD ne s'est infiltrée. Cela respecte strictement le contrat SEC-only.

#### 3.1.2 Couverture

| Fichier | Lignes | Tickers uniques | Date min | Date max |
|---------|--------|-----------------|----------|----------|
| `US_Income_statement.parquet` | 39 239 | 625 | 2006-12-31 | 2026-03-31 |
| `US_Balance_sheet.parquet` | 40 258 | 626 | 2006-09-30 | 2026-03-31 |
| `US_Cash_flow.parquet` | 10 881 | 622 | 2007-12-31 | 2026-03-31 |
| `US_share.parquet` | 36 586 | 626 | 2006-12-31 | 2026-04-30 |
| `US_Earnings.parquet` | 45 908 | 628 | 2004-12-31 | 2026-03-31 |
| `US_General.parquet` | 635 | — | — | — |

**Analyse :**
- **Income statement** : couverture très correcte (~63 rows/ticker en moyenne = ~15 ans de trimestres).
- **Balance sheet** : similaire, légèrement supérieure.
- **Cash flow** : **point faible majeur** — seulement 10 881 rows, soit ~4x moins que l'income statement. Cela signifie que beaucoup de tickers n'ont pas de cash flow trimestriel dans les tags SEC actuellement recherchés.
- **Shares** : bonne couverture.
- **Earnings** : bonne couverture, mais `epsEstimate` et `surprisePercent` sont `null` par design (conforme au contrat).

#### 3.1.3 Densité des colonnes (format legacy wide)

Le package SEC exporte des fichiers au **format legacy EODHD wide** (colonnes comme `totalRevenue`, `netIncome`, `researchDevelopment`, etc.). Problème : la plupart des colonnes sont **100 % NULL** car elles ne sont pas mappées dans `METRIC_SPECS`.

Exemple pour `US_Income_statement.parquet` :

| Colonne | Taux de null |
|---------|-------------|
| `totalRevenue` | 0.9 % |
| `netIncome` | 0.2 % |
| `grossProfit` | **56.6 %** |
| `operatingIncome` | **22.7 %** |
| `researchDevelopment` | **100 %** |
| `ebit` | **100 %** |
| `ebitda` | **100 %** |
| `incomeBeforeTax` | **100 %** |
| `sellingGeneralAdministrative` | **100 %** |

**Pourquoi c'est un problème :**
- Le fichier prétend être compatible EODHD mais n'a de la donnée que pour 4-5 métriques.
- Un utilisateur qui ouvre le parquet voit 35 colonnes et pense que les données sont incomplètes, alors que c'est juste que ces concepts ne sont pas encore mappés depuis la SEC.
- **Risque de confusion** : on ne sait pas si une valeur est NULL parce que la SEC ne l'a pas reportée, ou parce que le mapping n'existe pas.

#### 3.1.4 Problèmes structuraux dans le package SEC

**A. Modification des shares outstanding par logique earnings-implied**

Dans `legacy_export.py`, la fonction `_align_balance_shares_with_earnings_semantics` recalcule :
```
implied_shares = abs(netIncome / epsActual)
```
Si le ratio `implied_shares / reported_shares` est entre 0.8 et 1.2, la valeur exportée dans `commonStockSharesOutstanding` est **remplacée** par la valeur impliquée.

**Verdict :** ❌ **Dangereux pour un package SEC-only.** Cela signifie que la donnée exportée n'est pas strictement la donnée SEC reportée. Le contrat dit "source canonique = SEC", mais ici on fait une approximation mathématique qui peut masquer des restatements ou des différences de share classes.

**B. Absence d'`accession_number` dans le lineage**

Le contrat SEC (`docs/sec_fundamentals_contract.md`) exige :
> "`accession_number` quand disponible"

Or le lineage `data/sec/output/lineage/financials_sec_lineage.parquet` ne contient **pas** cette colonne. C'est une violation du contrat documenté.

**C. `total_liabilities` dérivé dans `sec_filing.py`**

Le client filing-level dérive :
```python
total_liabilities = total_assets - stockholders_equity
```
quand le tag SEC manque. C'est correct comptablement, mais ce n'est **pas documenté** dans le contrat SEC. Le contrat dit "on laisse null" si la SEC ne permet pas de reconstruire proprement.

**D. `free_cash_flow` dérivé**

`free_cash_flow = operating_cash_flow - capital_expenditures` est bien marqué `derived_from_sec` dans le code, ce qui est conforme au contrat. Mais dans le fichier wide, il apparaît comme une colonne normale sans metadata.

### 3.2 Package Open Source (`data/open_source/output/`)

#### 3.2.1 Couverture historique — CRITIQUE

| Fichier | Lignes | Tickers | Date min | Date max |
|---------|--------|---------|----------|----------|
| `US_Income_statement.parquet` | **2 617** | 644 | 2024-12-31 | 2026-03-31 |
| `US_Balance_sheet.parquet` | **2 623** | 644 | 2024-12-31 | 2026-03-31 |
| `US_Cash_flow.parquet` | **2 604** | 644 | 2024-12-31 | 2026-03-31 |
| `US_share.parquet` | **3 231** | 636 | 2024-12-31 | 2026-04-30 |

**Verdict :** ❌❌ **Inutilisable pour backtest.** En moyenne 4 rows/ticker = 1 an d'historique. Même pour du walk-forward sur 2 ans, c'est insuffisant.

#### 3.2.2 Pourquoi l'open source est si pauvre ?

Les **raw data** dans `data/open_source/official/raw/` contiennent pourtant :
- `financials_sec_companyfacts.parquet` : **324 404 rows**, 626 tickers, 2006-2026
- `financials_sec_filing.parquet` : 24 101 rows, 624 tickers, 2011-2026

Mais le **target** (fichiers consolidés dans `official/target/`) et le **output** ne contiennent que **38 444 lignes** de lineage, datées de 2025 uniquement.

**Cause racine identifiée :**
Dans `run_open_source_ingestion()`, le paramètre `financial_lookback_years=2` en mode `daily` fait que le pipeline ne récupère que les 2 dernières années de financials. Mais le problème est plus profond : les raw data contiennent bien l'historique complet, **mais le target est reconstruit à partir des données fraîchement fetchées** et écrase l'ancien consolidated. Si le dernier run était un daily récent, le target ne contient que les années rafraîchies.

**En d'autres termes :** le raw est bien, le target est mal reconstruit. Il faudrait que le target soit toujours rebuildé depuis la **totalité** du raw, pas seulement depuis les deltas du dernier run.

#### 3.2.3 Mix de sources — Fallback massif

Le lineage open-source montre :

| Source | Lignes | Part |
|--------|--------|------|
| `sec_companyfacts` | 20 820 | 54 % |
| `yfinance` | 13 733 | **36 %** |
| `sec_filing` | 1 991 | 5 % |
| `simfin` | 1 900 | 5 % |

**46 % des données sont des fallback non-SEC.** Pour certaines métriques, c'est pire :
- `free_cash_flow` : 76 % yfinance
- `operating_cash_flow` : 73 % yfinance
- `gross_profit` : 55 % yfinance
- `capital_expenditures` : 74 % yfinance

**Pourquoi ?** Parce que les raw yfinance et simfin ne couvrent que 2025+, donc pour les dates récentes, quand SEC manque une métrique (souvent parce que le tag n'est pas dans `METRIC_SPECS`), le fallback Yahoo prend le relais.

**Problème :** Yahoo ne garantit pas le GAAP. Pour un backtest où on veut de la rigueur, 46 % de fallback non-SEC est inacceptable.

### 3.3 Legacy EODHD (`data/eodhd/output/`)

| Fichier | Lignes | Tickers |
|---------|--------|---------|
| `US_Income_statement.parquet` | 91 109 | 792 |
| `US_Balance_sheet.parquet` | 91 000 | 791 |
| `US_Cash_flow.parquet` | 85 052 | 792 |

EODHD reste supérieur en volume (2.3x plus de rows que SEC), mais c'est une source payante que tu as coupée. Les données existantes peuvent servir de référence pour auditer le SEC package, mais ne doivent pas être utilisées pour du nouveau backtest.

---

## 4. Problèmes Structurels Identifiés (Challenge du travail de Codex)

### 4.1 Le format legacy wide est une mauvaise abstraction pour le SEC package

Codex a réutilisé `export_legacy_compatible_fundamental_outputs()` pour construire le package SEC. Cela produit des fichiers avec 35+ colonnes dont 90 % sont NULL. C'est une **fuite d'abstraction** : le package SEC devrait soit :
- Exporter un format "slim" avec uniquement les métriques effectivement peuplées + colonnes de metadata (source, tag SEC, form, accession_number), ou
- Documenter très clairement que les colonnes NULL sont "non mappées par design".

**Recommandation :** Créer un second format d'export pour le SEC package : `US_Income_statement_slim.parquet` avec les colonnes `date`, `filing_date`, `ticker`, `totalRevenue`, `grossProfit`, `operatingIncome`, `netIncome`, `totalRevenue_source_tag`, `totalRevenue_form`, `totalRevenue_accession_number`, etc.

### 4.2 La consolidation `consolidation.py` a une logique spéciale pour les shares qui contamine le SEC package

```python
if default.get("statement") != "shares":
    return default
# ... override avec yfinance si ratio >= 1.5
```

Cette logique existe dans le code de consolidation générique. Heureusement, dans le SEC package, le lineage montre que shares vient à 99 % de SEC. Mais le fait que cette logique existe dans un module partagé entre open-source et SEC est **risqué**.

**Recommandation :** Isoler la consolidation SEC dans un module dédié qui n'a aucune logique de fallback vendor, même pour les shares.

### 4.3 Le contrat SEC n'est pas respecté sur 3 points

1. **`accession_number` manquant** dans le lineage (exigé par le contrat)
2. **`total_liabilities` dérivé** dans `sec_filing.py` sans mention dans le contrat
3. **Shares modifiés** par logique earnings-implied dans `legacy_export.py` — le contrat dit "ne pas compléter avec un vendor externe", or ici on complète avec une approximation mathématique externe aux données SEC brutes

### 4.4 `METRIC_SPECS` est trop restrictif

Codex n'a mappé que **12 métriques** : `revenue`, `gross_profit`, `operating_income`, `net_income`, `total_assets`, `total_liabilities`, `stockholders_equity`, `cash_and_equivalents`, `operating_cash_flow`, `capital_expenditures`, `free_cash_flow`, `outstanding_shares`.

Pour un backtest complet sur le S&P 500, il manque des métriques critiques :
- `cost_of_revenue` / `costOfRevenue` (pour calculer les marges)
- `research_and_development`
- `selling_general_and_administrative`
- `ebit`, `ebitda`
- `total_debt`, `long_term_debt`
- `current_assets`, `current_liabilities`
- `inventory`, `accounts_receivable`
- `goodwill`, `intangible_assets`

Cela explique pourquoi les colonnes correspondantes dans le fichier wide sont 100 % NULL : ce n'est pas que la SEC ne les a pas, c'est qu'on ne les cherche pas.

### 4.5 Le Q4 derived est bien fait mais pas assez testé

```python
# sec.py
derived_q4 = annual - q1 - q2 - q3
```

C'est la méthode standard. Cependant, il n'y a **aucun test** vérifiant que pour un ticker connu (ex: AAPL), le Q4 dérivé correspond bien au Q4 réel quand il existe. Risque : si un quarter a été restaté, le Q4 dérivé peut être faux.

### 4.6 La normalisation des dates dans `legacy_export.py` est agressive

```python
_normalize_statement_date_for_legacy
# arrondit au month-end le plus proche
```

Cela peut créer des décalages de quelques jours entre la date réelle du quarter et la date exportée. Pour un backtest où les features sont mergeées avec les prix par date, cela peut entraîner du **lookahead bias** si la `filing_date` n'est pas correctement utilisée comme date d'availability.

---

## 5. Recommandations Prioritaires

### 🔴 P0 — Bloquant pour le backtest

1. **Utiliser `data/sec/output/` pour les fondamentaux, pas `data/open_source/output/`**
   - L'open-source output n'a pas d'historique. Le SEC output est le seul utilisable.
   - Si le backtest charge actuellement `data/open_source/output/US_Income_statement.parquet`, il faut switcher immédiatement.

2. **Reconstruire le target open-source à partir de la totalité du raw**
   - Le raw SEC companyfacts a 324k rows. Le target n'en a que 38k.
   - Modifier `run_open_source_ingestion()` ou `refresh_open_source_reference_layers()` pour que le consolidated financials soit toujours rebuildé depuis le raw complet, pas seulement depuis les deltas du run courant.
   - Alternative rapide : lancer un script one-off qui lit `raw/financials_sec_companyfacts.parquet` + `raw/financials_sec_filing.parquet` et écrit un nouveau target consolidé avec tout l'historique.

3. **Désactiver la logique earnings-implied pour les shares dans le package SEC**
   - Dans `legacy_export.py`, conditionner `_align_balance_shares_with_earnings_semantics` pour qu'elle ne s'applique pas quand on build le package SEC.
   - Les shares SEC doivent être exportés tels quels.

### 🟡 P1 — Qualité et confiance

4. **Ajouter `accession_number` au lineage SEC**
   - Le contrat l'exige. C'est indispensable pour l'auditabilité.
   - `sec_filing.py` a déjà l'`accession_number` dans `FilingMetadata`. Il faut le propager jusqu'au lineage.

5. **Documenter les colonnes dérivées dans le contrat SEC**
   - `total_liabilities` dérivé dans `sec_filing.py`
   - `free_cash_flow` dérivé dans `sec.py`
   - Mettre à jour `docs/sec_fundamentals_contract.md` en conséquence.

6. **Améliorer la couverture cash flow SEC**
   - 10 881 rows est très faible. Investiguer si des tags XBRL manquent dans `METRIC_SPECS` pour le cash flow.
   - Exemples de tags SEC additionnels potentiels : `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations`, `NetCashProvidedByUsedInInvestingActivities`, etc.

7. **Ajouter des métriques manquantes à `METRIC_SPECS`**
   - Au minimum : `cost_of_revenue`, `rd`, `sga`, `ebit`, `total_debt`, `current_assets`, `current_liabilities`.
   - Cela permettra de peupler les colonnes legacy actuellement 100 % NULL.

8. **Créer un rapport de densité automatique**
   - Un script qui, pour chaque fichier wide SEC, calcule le taux de null par colonne et l'exporte en JSON/CSV.
   - Cela permettra de suivre l'amélioration de la couverture au fil du temps.

### 🟢 P2 — Robustesse et tests

9. **Ajouter des tests pour les cas limites**
   - Banques / REITs / assurances : la revenue est souvent un concept différent (net interest income vs total revenue). Vérifier que `_select_revenue_facts` gère bien ces cas.
   - Fiscal year décalé : un ticker avec FY ending en janvier (ex: NVDA ? Non, mais certains retailers).
   - Q4 derived : comparer le Q4 dérivé avec le Q4 réel quand disponible.
   - Restatements : si une entreprise restate un quarter, est-ce que le pipeline prend la dernière version ? (L'upsert sur natural key + `ingested_at` devrait le faire, mais ce n'est pas testé.)

10. **Séparer complètement la logique de consolidation SEC vs open-source**
    - Créer `consolidation_sec_only.py` qui n'a aucune logique de fallback ou d'override vendor.
    - Le partage de code entre open-source et SEC est pratique mais dangereux pour la rigueur du contrat.

11. **Vérifier le mapping S&P 500**
    - Le SEC package a 625 tickers. Le S&P 500 a ~503 tickers actuels. Mais certains tickers historiques ou récents pourraient manquer.
    - Faire une table de correspondance `SP500_Constituents.csv` ↔ `US_General.parquet` pour identifier les trous.

---

## 6. Points d'Attention pour le Backtest

Si tu switches sur le SEC package pour les fondamentaux, voici les points à surveiller :

| Risque | Mitigation |
|--------|------------|
| `grossProfit` 56 % null | Vérifier si tes features utilisent `grossProfit`. Si oui, tu vas avoir des NaN fréquents. Alternative : calculer `grossProfit = revenue - costOfRevenue` si on ajoute `costOfRevenue`. |
| `operatingIncome` 22 % null | Même chose. Certaines entreprises ne reportent pas explicitement l'operating income sous ce tag. |
| Cash flow faible | Si tes stratégies utilisent `freeCashFlow` ou `operatingCashFlow`, la couverture est faible. |
| Dates normalisées au month-end | Quand tu merges fondamentaux avec prix, utiliser la `filing_date` comme date d'availability, pas la `date` du quarter. Sinon tu risques du lookahead. |
| Shares outstanding | Après fix P0 #3, utiliser les shares SEC directement. Avant fix, vérifier que `commonStockSharesOutstanding` n'a pas été override par la logique earnings. |

---

## 7. Plan d'Action Suggéré

**Semaine 1 (Immédiat)**
1. Switcher le backtest sur `data/sec/output/` pour les fondamentaux.
2. Désactiver la logique earnings-implied dans `legacy_export.py` pour le build SEC.
3. Vérifier que tes features principales (revenue, netIncome, totalAssets, etc.) ont une couverture > 90 % sur l'univers S&P 500.

**Semaine 2**
4. Rebuild le target open-source à partir du raw complet (324k rows).
5. Ajouter `accession_number` au lineage SEC.
6. Mettre à jour le contrat SEC avec les dérivations documentées.

**Semaine 3-4**
7. Étendre `METRIC_SPECS` avec les métriques manquantes (costOfRevenue, R&D, SG&A, EBIT, debt, current assets/liabilities).
8. Investiguer les tags SEC manquants pour améliorer la couverture cash flow.
9. Ajouter des tests de non-régression sur les cas limites (banques, REITs, FY décalé).

---

## 8. Annexe — Commandes de Vérification Rapide

```python
import polars as pl

# Vérifier la couverture d'un ticker
sec = pl.read_parquet("data/sec/output/US_Income_statement.parquet")
aapl = sec.filter(pl.col("ticker") == "AAPL.US")
print(aapl.shape[0], "rows")
print(aapl.select(["date", "totalRevenue", "netIncome", "grossProfit", "operatingIncome"]))

# Vérifier le lineage SEC
lineage = pl.read_parquet("data/sec/output/lineage/financials_sec_lineage.parquet")
print(lineage["selected_source"].value_counts())

# Vérifier le taux de null par colonne
for col in sec.columns:
    nulls = sec[col].is_null().sum()
    if nulls > 0:
        print(f"{col}: {100*nulls/sec.shape[0]:.1f}% null")
```

---

*Fin du rapport.*
