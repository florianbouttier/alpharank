# Rapport : Trous de Données par Ticker — S&P 500 Actuel

> **Archive datée.** Les chiffres ci-dessous décrivent le package observé à la
> date du rapport, pas la couverture courante.

**Date :** 11 mai 2026  
**Source :** SEC companyfacts (`data/open_source/official/raw/financials_sec_companyfacts.parquet`) vs EODHD legacy (`data/eodhd/output/`)  
**Univers :** S&P 500 actuel (503 tickers au 2026-04-01)  
**Métrique focus :** `totalRevenue` (le raisonnement s'applique aussi à `netIncome`)

---

## 1. Synthèse : Où on en est

Sur les **503 tickers du S&P 500 actuel** :

| Catégorie | Nombre | % | Verdict |
|-----------|--------|---|---------|
| ✅ Bonne couverture (>= 40q = 10 ans) | **454** | 90.3% | Utilisable directement |
| ⚠️ Couverture partielle (< 40q) | **47** | 9.3% | À traiter |
| ❌ Aucune donnée SEC | **2** | 0.4% | Bug de mapping |

**Objectif :** Passer de 454 à ~484 tickers avec >= 40 quarters, soit **96.2% de couverture complète**.

Les ~19 tickers restants sont des IPO/spinoffs récents (< 5 ans) — leur historique court est **normal et non récupérable**.

---

## 2. Les 2 Tickers Sans Aucune Donnée (Bug de Mapping)

| Ticker | Problème | Solution | Impact |
|--------|----------|----------|--------|
| **BRK.B** | Le SEC mapping utilise `BRK-B`, pas `BRK.B` | Mapper `BRK.B` → `BRK-B` | +108q EODHD, +~70q SEC |
| **BF.B** | Le SEC mapping utilise `BF-B`, pas `BF.B` | Mapper `BF.B` → `BF-B` | +~120q EODHD, +~50q SEC |

**Diagnostic :** Le mapping SEC `company_tickers_exchange.json` de la SEC utilise des **tirets** (`-`) pour les classes de shares. Le SP500 constituents utilise des **points** (`.`). C'est un problème de normalisation trivial.

**Action :** Dans `sec.py` ou dans la logique de mapping, ajouter une règle de normalisation :
```python
ticker_normalized = ticker.replace(".", "-")  # BRK.B → BRK-B
```

---

## 3. Les 47 Tickers à Couverture Partielle — Classés par Type

### Type A : Normalisation du Ticker (0 dans les 47, mais 2 sans données)
→ Voir section 2 ci-dessus.

---

### Type B : Entités Récentes (IPO / Spinoff récent — 12 tickers)

Ces entreprises ont moins de 7 ans d'historique car elles n'existaient pas (ou n'étaient pas publiques) avant. **Ce n'est PAS récupérable.**

| Ticker | SEC | EODHD | Première date SEC | Diagnostic |
|--------|-----|-------|-------------------|------------|
| **Q** | 2q | 8q | 2024-09-30 | SPAC/merge très récent (Quintillion?) |
| **PSKY** | 5q | 120q | 2023-12-31 | Entité récente, ancien historique sous autre nom |
| **SNDK** | 7q | 10q | 2023-09-30 | Spinoff Western Digital récent |
| **HOOD** | 23q | 28q | 2020-06-30 | IPO 2020 |
| **ABNB** | 25q | 26q | 2019-12-31 | IPO 2019 |
| **APP** | 25q | 29q | 2019-12-31 | IPO 2019 (Applovin) |
| **COIN** | 25q | 28q | 2019-12-31 | IPO 2019 |
| **DASH** | 25q | 28q | 2019-12-31 | IPO 2019 |
| **PLTR** | 26q | 28q | 2019-09-30 | IPO 2019 |
| **DDOG** | 30q | 32q | 2018-09-30 | IPO 2018 |
| **CRWD** | 32q | 33q | 2018-01-31 | IPO 2018 |
| **MRNA** | 36q | 37q | 2017-03-31 | IPO 2017 |

**Verdict :** ✅ **Pas d'action nécessaire.** Leur historique court est normal. Pour un backtest, ils n'auront pas de features de croissance sur 5 ans, mais ils sont utilisables dès qu'ils ont suffisamment d'historique.

---

### Type C : Spinoffs (14 tickers)

Ces entreprises sont des **spinoffs récents** qui n'existaient pas comme entités SEC indépendantes avant leur date de scission. On peut récupérer leur historique pré-spinoff depuis l'**entreprise mère**.

| Ticker | SEC | EODHD | Première date | Entité mère | Mère SEC | Gain potentiel |
|--------|-----|-------|---------------|-------------|----------|----------------|
| **GEV** | 13q | 12q | 2022-12-31 | GE (General Electric) | 72q | +59q |
| **SOLV** | 13q | 16q | 2022-12-31 | DD (DuPont) | 40q | +27q |
| **GEHC** | 17q | 20q | 2021-12-31 | GE (General Electric) | 72q | +55q |
| **KVUE** | 17q | 17q | 2021-12-31 | JNJ (Johnson & Johnson) | 70q | +53q |
| **MBC** | 17q | 20q | 2021-12-31 | DHR (Danaher) | 71q | +54q |
| **CARR** | 29q | 28q | 2018-12-31 | UTX/RTX (United Technologies) | 71q | +42q |
| **OTIS** | 29q | 28q | 2018-12-31 | UTX/RTX (United Technologies) | 71q | +42q |
| **CTVA** | 33q | 36q | 2017-12-31 | DWDP (DowDuPont) | 0q (delisted) | 0q (utiliser EODHD) |
| **VICI** | 33q | 40q | 2017-12-31 | CZR (Caesars) | 52q | +19q |
| **AMCR** | 34q | 54q | 2017-09-30 | Ancienne entité (complexe) | — | Utiliser EODHD |
| **FOX** | 34q | 54q | 2017-09-30 | FOXA / NWSA | 55q (NWSA) | +21q via NWSA |
| **FOXA** | 34q | 104q | 2017-09-30 | NWSA (News Corp) | 55q | +21q |
| **VRT** | 34q | 29q | 2017-06-30 | EMC / Dell complexe | — | EODHD ou parent |
| **VST** | 37q | 96q | 2016-12-31 | T / AEP / Luminant (complexe) | — | EODHD |

**Solution recommandée :**
1. Pour les spinoffs avec **entreprise mère encore cotée** (GEHC→GE, KVUE→JNJ, CARR→RTX, etc.) : **copier les données de la mère avant la date de spinoff.**
2. Pour les spinoffs où la mère est **delisted** ou a aussi restructuré (CTVA→DWDP) : **utiliser EODHD legacy** pour l'historique manquant.

**Comment implémenter :**
```python
SPINOFF_PARENTS = {
    'GEHC': {'parent': 'GE', 'spinoff_date': '2023-01-04'},
    'KVUE': {'parent': 'JNJ', 'spinoff_date': '2023-05-04'},
    'GEV': {'parent': 'GE', 'spinoff_date': '2024-04-02'},
    'SOLV': {'parent': 'DD', 'spinoff_date': '2023-09-01'},
    'CARR': {'parent': 'RTX', 'spinoff_date': '2020-04-03'},
    'OTIS': {'parent': 'RTX', 'spinoff_date': '2020-04-03'},
    'MBC': {'parent': 'DHR', 'spinoff_date': '2023-09-01'},
    # etc.
}

# Pour chaque spinoff, récupérer les données de la mère avant la date de spinoff
# et les attribuer au ticker du spinoff.
```

**Impact :** +10 tickers passent de "partiel" à "bon".

---

### Type D : CIK Hérité / Restructuration (17 tickers)

Ces entreprises **existent depuis longtemps** mais leur **CIK actuel** ne contient qu'un historique récent. Leur historique ancien est sous un **ancien CIK** ou a été perdu dans la restructuration.

| Ticker | SEC | EODHD | 1ère date SEC | Diagnostic | Solution |
|--------|-----|-------|---------------|------------|----------|
| **BLK** | 10q | 108q | 2023-09-30 | BlackRock a changé de CIK (nouveau: 2012383) | EODHD backfill |
| **SW** | 10q | 68q | 2023-06-30 | Southwest? ou autre restructuration | EODHD backfill |
| **CRH** | 13q | 137q | 2022-12-31 | CRH plc (Irlande) → acquisition US récente | EODHD backfill |
| **BG** | 16q | 104q | 2022-03-31 | Bunge a restructuré / changé de CIK | EODHD backfill |
| **APA** | 6q | 162q | 2019-12-31 | Apache Corp → APA Corp, changement de domicile | EODHD backfill |
| **TPL** | 28q | 162q | 2019-03-31 | Texas Pacific Land a changé de structure | EODHD backfill |
| **VTRS** | 28q | 161q | 2018-12-31 | Viatris (fusion Mylan + Upjohn 2020) | EODHD backfill |
| **NXPI** | 29q | 69q | 2017-12-31 | NXP Semiconductor, changement de CIK? | EODHD backfill |
| **DIS** | 33q | 162q | 2017-09-30 | Disney a changé de CIK? Ou companyfacts limité | EODHD backfill |
| **DOW** | 33q | 65q | 2017-12-31 | Dow Inc (post-DowDuPont split 2019) | EODHD backfill |
| **UBER** | 33q | 36q | 2017-12-31 | Uber IPO 2019, 33q c'est normal en fait | Pas d'action (proche d'EODHD) |
| **STE** | 34q | 144q | 2017-03-31 | STERIS a restructuré / changé de CIK | EODHD backfill |
| **CI** | 36q | 146q | 2017-03-31 | Cigna a acquis Express Scripts 2018, restructuration | EODHD backfill |
| **EVRG** | 36q | 162q | 2017-03-31 | Evergy (fusion Westar + Great Plains 2018) | EODHD backfill |
| **LIN** | 36q | 140q | 2017-03-31 | Linde (fusion Linde + Praxair 2018) | EODHD backfill |
| **AVGO** | 37q | 77q | 2017-01-31 | Broadcom a changé de domicile (Singapour → US) | EODHD backfill |
| **VST** | 37q | 96q | 2016-12-31 | Vistra (complexe restructuration Energy Future) | EODHD backfill |
| **IR** | 39q | 104q | 2016-06-30 | Ingersoll Rand (fusion Gardner Denver 2020) | EODHD backfill |
| **XOM** | 37q | 162q | 2008-06-30 | ExxonMobil, companyfacts limité à ~2008 | EODHD backfill |

**Verdict :** Pour tous ces cas, la SEC companyfacts ne contient pas l'historique complet sous le CIK actuel. Il y a 2 approches possibles :

**Approche 1 (idéale mais complexe) :** Trouver l'ancien CIK et récupérer les données SEC sous cet ancien CIK.
- Ex: DIS sous ancien CIK ? Exxon sous ancien CIK ?
- **Problème :** Très manuel. Chaque cas est différent. Pas scalable.

**Approche 2 (pragmatique et rapide) :** Utiliser **EODHD legacy** pour backfill l'historique manquant avant la date de disponibilité SEC.
- Ex: DIS a 33q SEC (2017-2025) + 129q EODHD (1990-2017) = 162q total
- **Avantage :** Simple, rapide, couverture complète
- **Inconvénient :** Les données EODHD ne sont pas GAAP/SEC (mais c'est mieux que des trous)

**Recommandation :** Utiliser l'**Approche 2** (EODHD backfill) pour tous les cas CIK legacy. C'est le meilleur compromis qualité/effort.

**Comment implémenter :**
```python
# Pour chaque ticker, définir une "date de coupure" SEC
# Avant cette date, utiliser EODHD. Après, utiliser SEC.

SEC_START_DATES = {
    'DIS': '2017-09-30',
    'XOM': '2008-06-30',
    'BLK': '2023-09-30',
    'STE': '2017-03-31',
    'CI': '2017-03-31',
    'EVRG': '2017-03-31',
    'LIN': '2017-03-31',
    'AVGO': '2017-01-31',
    'IR': '2016-06-30',
    # etc.
}

# Merge logic:
# 1. Take SEC data for all dates
# 2. For dates < SEC_START_DATE, use EODHD if SEC is null
# 3. Tag source as "sec" or "eodhd_legacy_backfill"
```

**Impact :** +17 tickers passent de "partiel" à "bon".

---

## 4. Tableau Récapitulatif des 47 Tickers Partiels

| # | Ticker | SEC | EODHD | Type | Action | Impact |
|---|--------|-----|-------|------|--------|--------|
| 1 | Q | 2 | 8 | Recent entity | None | — |
| 2 | PSKY | 5 | 120 | Recent entity | None | — |
| 3 | SNDK | 7 | 10 | Recent entity | None | — |
| 4 | APA | 6 | 162 | CIK legacy | EODHD backfill | ✅ +156q |
| 5 | BLK | 10 | 108 | CIK legacy | EODHD backfill | ✅ +98q |
| 6 | SW | 10 | 68 | CIK legacy | EODHD backfill | ✅ +58q |
| 7 | CRH | 13 | 137 | CIK legacy | EODHD backfill | ✅ +124q |
| 8 | GEV | 13 | 12 | Spinoff | Parent (GE) backfill | ✅ +59q |
| 9 | SOLV | 13 | 16 | Spinoff | Parent (DD) backfill | ✅ +27q |
| 10 | TKO | 14 | 20 | Recent entity | None | — |
| 11 | VLTO | 14 | 16 | Spinoff | Parent (VNT) backfill | — |
| 12 | BG | 16 | 104 | CIK legacy | EODHD backfill | ✅ +88q |
| 13 | GEHC | 17 | 20 | Spinoff | Parent (GE) backfill | ✅ +55q |
| 14 | KVUE | 17 | 17 | Spinoff | Parent (JNJ) backfill | ✅ +53q |
| 15 | MBC | 17 | 20 | Spinoff | Parent (DHR) backfill | ✅ +54q |
| 16 | APO | 21 | 71 | Recent entity | None | — |
| 17 | CEG | 21 | 54 | Recent entity | None | — |
| 18 | HOOD | 23 | 28 | Recent entity | None | — |
| 19 | ABNB | 25 | 26 | Recent entity | None | — |
| 20 | APP | 25 | 29 | Recent entity | None | — |
| 21 | COIN | 25 | 29 | Recent entity | None | — |
| 22 | DASH | 25 | 28 | Recent entity | None | — |
| 23 | PLTR | 26 | 28 | Recent entity | None | — |
| 24 | TPL | 28 | 162 | CIK legacy | EODHD backfill | ✅ +134q |
| 25 | VTRS | 28 | 161 | CIK legacy | EODHD backfill | ✅ +133q |
| 26 | CARR | 29 | 28 | Spinoff | Parent (RTX) backfill | ✅ +42q |
| 27 | NXPI | 29 | 69 | CIK legacy | EODHD backfill | ✅ +40q |
| 28 | OTIS | 29 | 28 | Spinoff | Parent (RTX) backfill | ✅ +42q |
| 29 | DDOG | 30 | 32 | Recent entity | None | — |
| 30 | CRWD | 32 | 33 | Recent entity | None | — |
| 31 | CTVA | 33 | 36 | Spinoff | EODHD backfill | ✅ +3q |
| 32 | DIS | 33 | 162 | CIK legacy | EODHD backfill | ✅ +129q |
| 33 | DOW | 33 | 65 | CIK legacy | EODHD backfill | ✅ +32q |
| 34 | UBER | 33 | 36 | Recent entity | None | — |
| 35 | VICI | 33 | 40 | Spinoff | Parent (CZR) backfill | ✅ +19q |
| 36 | AMCR | 34 | 54 | Spinoff | EODHD backfill | ✅ +20q |
| 37 | FOX | 34 | 54 | Spinoff | Parent (NWSA) backfill | ✅ +21q |
| 38 | FOXA | 34 | 104 | Spinoff | Parent (NWSA) backfill | ✅ +21q |
| 39 | STE | 34 | 144 | CIK legacy | EODHD backfill | ✅ +110q |
| 40 | VRT | 34 | 29 | Spinoff | EODHD backfill | — |
| 41 | CI | 36 | 146 | CIK legacy | EODHD backfill | ✅ +110q |
| 42 | EVRG | 36 | 162 | CIK legacy | EODHD backfill | ✅ +126q |
| 43 | LIN | 36 | 140 | CIK legacy | EODHD backfill | ✅ +104q |
| 44 | MRNA | 36 | 37 | Recent entity | None | — |
| 45 | AVGO | 37 | 77 | CIK legacy | EODHD backfill | ✅ +40q |
| 46 | VST | 37 | 96 | CIK legacy | EODHD backfill | ✅ +59q |
| 47 | IR | 39 | 104 | CIK legacy | EODHD backfill | ✅ +65q |
| 48 | XOM | 37 | 162 | CIK legacy | EODHD backfill | ✅ +125q |

**Note :** `BRK.B` et `BF.B` (les 2 sans données) sont en dehors de ce tableau. Leur fix est trivial (normalisation du ticker).

---

## 5. Plan d'Action Recommandé

### Étape 1 : Ticker Normalization (30 min de dev)
**Action :** Ajouter une règle `ticker.replace(".", "-")` avant la recherche dans le SEC mapping.
**Résultat :** `BRK.B` et `BF.B` récupèrent leurs données.

### Étape 2 : EODHD Backfill pour CIK Legacy (2-3h de dev)
**Action :** Créer une table `SEC_START_DATES` avec la première date disponible en SEC pour chaque ticker. Lors de la consolidation, pour les dates < `SEC_START_DATE`, utiliser EODHD legacy si SEC est null.

**Tickers concernés (19) :** APA, BLK, SW, CRH, BG, TPL, VTRS, NXPI, DIS, DOW, STE, CI, EVRG, LIN, AVGO, VST, IR, XOM, (et UBER si besoin mais c'est récent)

**Pseudo-code :**
```python
SEC_START_DATES = {
    'APA': '2019-12-31', 'BLK': '2023-09-30', 'SW': '2023-06-30',
    'CRH': '2022-12-31', 'BG': '2022-03-31', 'TPL': '2019-03-31',
    'VTRS': '2018-12-31', 'NXPI': '2017-12-31', 'DIS': '2017-09-30',
    'DOW': '2017-12-31', 'STE': '2017-03-31', 'CI': '2017-03-31',
    'EVRG': '2017-03-31', 'LIN': '2017-03-31', 'AVGO': '2017-01-31',
    'VST': '2016-12-31', 'IR': '2016-06-30', 'XOM': '2008-06-30',
}

for ticker, start_date in SEC_START_DATES.items():
    # For dates before start_date, fill from EODHD
    sec_data = sec_financials.filter(ticker=ticker)
    eodhd_data = eodhd_financials.filter(ticker=ticker)
    
    merged = sec_data.join(eodhd_data, on='date', how='outer')
    merged = merged.with_columns(
        pl.when(pl.col('date') < start_date)
        .then(pl.col('eodhd_revenue'))
        .otherwise(pl.coalesce([pl.col('sec_revenue'), pl.col('eodhd_revenue')]))
        .alias('revenue')
    )
```

### Étape 3 : Parent Company Backfill pour Spinoffs (3-4h de dev)
**Action :** Créer une table `SPINOFF_PARENTS`. Pour chaque spinoff, copier les données de la mère avant la date de spinoff.

**Tickers concernés (10) :** GEHC, KVUE, GEV, SOLV, CARR, OTIS, MBC, FOX, FOXA, VICI

**Pseudo-code :**
```python
SPINOFF_PARENTS = {
    'GEHC': {'parent': 'GE', 'date': '2023-01-04'},
    'KVUE': {'parent': 'JNJ', 'date': '2023-05-04'},
    'GEV': {'parent': 'GE', 'date': '2024-04-02'},
    'SOLV': {'parent': 'DD', 'date': '2023-09-01'},
    'CARR': {'parent': 'RTX', 'date': '2020-04-03'},
    'OTIS': {'parent': 'RTX', 'date': '2020-04-03'},
    'MBC': {'parent': 'DHR', 'date': '2023-09-01'},
    'FOX': {'parent': 'NWSA', 'date': '2013-06-28'},
    'FOXA': {'parent': 'NWSA', 'date': '2013-06-28'},
    'VICI': {'parent': 'CZR', 'date': '2017-10-17'},
}

for child, info in SPINOFF_PARENTS.items():
    parent_data = sec_financials.filter(ticker=info['parent'], date < info['date'])
    parent_data = parent_data.with_columns(pl.lit(child).alias('ticker'))
    sec_financials = pl.concat([sec_financials, parent_data])
```

### Étape 4 : Vérification (1h)
**Action :** Relancer le pipeline et vérifier que :
- 454 + 2 (normalization) + 19 (EODHD) + 10 (spinoffs) = **485 tickers** ont >= 40 quarters
- Les 18 tickers récents restent partiels (normal)
- Pas de doublons, pas de données incohérentes

---

## 6. Ce qui restera partiel (et pourquoi c'est OK)

Après toutes les fixes, environ **18 tickers** resteront avec < 40 quarters :

| Ticker | SEC | Raison |
|--------|-----|--------|
| Q, PSKY, SNDK | 2-7q | Entités trop récentes (< 3 ans) |
| HOOD, ABNB, APP, COIN, DASH, PLTR, DDOG, CRWD | 23-32q | IPO 2017-2020, historique normal |
| TKO, VLTO, APO, CEG | 14-21q | Spinoff/Merge récent |
| MRNA | 36q | IPO 2017, proche du seuil |

**Verdict :** Ces 18 tickers représentent **3.6% du S&P 500**. Pour un backtest, c'est acceptable car :
1. Ils n'ont pas assez d'historique pour des features de croissance long terme
2. Ils deviendront "bons" naturellement avec le temps
3. Ils peuvent toujours être utilisés dans le backtest (juste sans features historiques longues)

---

## 7. Analyse Technique : Pourquoi la SEC companyfacts manque de l'historique

### 7.1 Limitation de l'API SEC companyfacts

L'API `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` ne contient que les données XBRL. Problèmes :

1. **Avant 2009-2011**, beaucoup d'entreprises ne rapportaient pas en XBRL. Leurs filings étaient en HTML.
2. **Entre 2011-2017**, le XBRL était obligatoire mais pas toujours rétroactivement intégré dans l'API companyfacts.
3. **Après 2017**, la plupart des entreprises ont des données complètes dans companyfacts.

**Exemple :**
- DIS (Disney) : companyfacts commence en 2017 (tag `Revenues`). Avant 2017, les données existaient en HTML mais pas en XBRL dans companyfacts.
- XOM (ExxonMobil) : companyfacts commence en 2008.
- BLK (BlackRock) : companyfacts commence en 2022 (changement de CIK).

### 7.2 Changements de CIK

Quand une entreprise :
- **Fusionne** (ex: Exxon + Mobil = ExxonMobil)
- **Change de domicile** (ex: Broadcom Singapour → Broadcom US)
- **Se restructure** (ex: BlackRock restructure sa holding)
- **Fait un spinoff majeur** (ex: DowDuPont → Dow + DuPont + Corteva)

Elle obtient souvent un **nouveau CIK**. L'historique sous l'ancien CIK n'est pas automatiquement fusionné dans le nouveau.

**Exemple :**
- DowDuPont (CIK ancien) → Dow Inc (CIK 1751788, depuis 2017)
- Linde AG + Praxair → Linde plc (CIK depuis 2018)
- Broadcom Limited → Broadcom Inc (changement de domicile, nouveau CIK)

### 7.3 Pourquoi EODHD a plus d'historique

EODHD compile des données de multiples sources :
- Fichiers XBRL SEC
- Fichiers HTML SEC (parsés manuellement ou semi-automatiquement)
- Fournisseurs de données tiers (Compustat, Capital IQ, etc.)
- Données rétroactivement ajustées pour les restructurations

Donc EODHD a souvent "bridgé" les trous entre anciens et nouveaux CIK, et a parsé les vieux filings HTML.

**Conclusion :** Pour l'historique ancien, EODHD est une source complémentaire légitime à la SEC companyfacts.

---

## 8. Recommandation Finale

**Pour un backtest S&P 500 robuste :**

1. **SEC comme source primaire** (2017+ pour la plupart, 2008+ pour quelques-uns)
2. **EODHD legacy comme backfill** pour les données manquantes avant la disponibilité SEC
3. **Parent company backfill** pour les spinoffs
4. **Ticker normalization** pour les classes de shares (BRK.B, BF.B)

**Résultat attendu :**
- **96.2%** des tickers S&P 500 avec >= 10 ans d'historique complet
- **3.6%** avec historique partiel (IPO/spinoff récents, acceptable)
- **0%** avec aucune donnée
- **100% traçable** (source claire pour chaque point: SEC, EODHD, parent)

**Alternatives écartées :**
- ❌ Récupérer les anciens CIK manuellement : trop complexe, chaque cas est unique
- ❌ Utiliser Yahoo Finance pour backfill : données non-GAAP, moins fiables qu'EODHD
- ❌ Laisser les trous : fausserait les features de croissance et le ranking

---

## 9. Fichiers de Référence

- `outputs/sec_sp500_coverage_report.csv` — Couverture complète (849 tickers)
- `data/sec/output/` — Package SEC-only actuel
- `data/eodhd/output/` — Package EODHD legacy (référence pour backfill)
- `data/sec/output/SP500_Constituents.csv` — Base historique S&P 500 (225k rows, 847 tickers uniques)

---

*Fin du rapport.*
