# Rapport de Couverture : totalRevenue & netIncome — S&P 500 Historique

> **Archive datée.** Les chiffres ci-dessous décrivent le package observé à la
> date du rapport, pas la couverture courante.

**Date :** 10 mai 2026
**Source :** Package SEC-only (`data/sec/output/`) comparé à EODHD legacy (`data/eodhd/output/`)
**Métriques focus :** `totalRevenue`, `netIncome` (demandé à Codex)

---

## 1. Résumé Global

- **Univers analysé :** 849 tickers (union SP500 actuel + SEC + EODHD)
- **SP500 actuel (2026-04-01) :** 846 tickers
- **Tickers avec revenue SEC :** 624 / 849
- **Tickers avec >= 40 quarters (10 ans) revenue SEC :** 558
- **Tickers SANS revenue SEC :** 225

| Métrique | SEC rows | EODHD rows | Delta |
|----------|----------|------------|-------|
| totalRevenue | 38,874 | 90,585 | 51,711 |
| netIncome | 39,171 | 90,290 | 51,119 |

**Verdict :** Le package SEC a ~57 % du volume de rows revenue par rapport à EODHD. 225 tickers n'ont aucun revenue SEC, dont beaucoup sont des anciens constituents delisted.

---

## 2. Focus S&P 500 Actuel (n = 846)

- **Avec revenue SEC :** 621 / 846 (73.4%)
- **Avec >= 10 ans d'historique (40q) :** 555 / 846 (65.6%)
- **SANS revenue SEC du tout :** 225 / 846 (26.6%)

### 2.1 SP500 Actuel — SANS totalRevenue SEC (225 tickers)

Ces tickers n'ont **aucune** valeur `totalRevenue` dans le package SEC. La raison principale est qu'ils sont **delisted** ou ont changé de CIK/ticker, et ne sont donc pas dans le mapping SEC actuel.

| Ticker | EODHD revenue rows | Note |
|--------|-------------------|------|
| IPG | 161 | 161q disponibles en EODHD — potentiellement récupérable |
| K | 161 | 161q disponibles en EODHD — potentiellement récupérable |
| FL | 160 | 160q disponibles en EODHD — potentiellement récupérable |
| WBA | 160 | 160q disponibles en EODHD — potentiellement récupérable |
| HES | 159 | 159q disponibles en EODHD — potentiellement récupérable |
| JWN | 159 | 159q disponibles en EODHD — potentiellement récupérable |
| SWN | 157 | 157q disponibles en EODHD — potentiellement récupérable |
| GPS | 156 | 156q disponibles en EODHD — potentiellement récupérable |
| ODP | 155 | 155q disponibles en EODHD — potentiellement récupérable |
| BIG | 154 | 154q disponibles en EODHD — potentiellement récupérable |
| RAD | 153 | 153q disponibles en EODHD — potentiellement récupérable |
| ABMD | 146 | 146q disponibles en EODHD — potentiellement récupérable |
| RRD | 146 | 146q disponibles en EODHD — potentiellement récupérable |
| CERN | 145 | 145q disponibles en EODHD — potentiellement récupérable |
| KSU | 145 | 145q disponibles en EODHD — potentiellement récupérable |
| MDP | 145 | 145q disponibles en EODHD — potentiellement récupérable |
| CMA | 144 | 144q disponibles en EODHD — potentiellement récupérable |
| X | 141 | 141q disponibles en EODHD — potentiellement récupérable |
| MXIM | 140 | 140q disponibles en EODHD — potentiellement récupérable |
| MRO | 139 | 139q disponibles en EODHD — potentiellement récupérable |
| PDCO | 139 | 139q disponibles en EODHD — potentiellement récupérable |
| BMS | 135 | 135q disponibles en EODHD — potentiellement récupérable |
| AET | 133 | 133q disponibles en EODHD — potentiellement récupérable |
| DRE | 133 | 133q disponibles en EODHD — potentiellement récupérable |
| CHK | 128 | 128q disponibles en EODHD — potentiellement récupérable |
| HFC | 128 | 128q disponibles en EODHD — potentiellement récupérable |
| PBCT | 128 | 128q disponibles en EODHD — potentiellement récupérable |
| ATVI | 125 | 125q disponibles en EODHD — potentiellement récupérable |
| SIVB | 125 | 125q disponibles en EODHD — potentiellement récupérable |
| FRC | 124 | 124q disponibles en EODHD — potentiellement récupérable |
| SRCL | 119 | 119q disponibles en EODHD — potentiellement récupérable |
| ANSS | 117 | 117q disponibles en EODHD — potentiellement récupérable |
| DISH | 117 | 117q disponibles en EODHD — potentiellement récupérable |
| NCR | 115 | 115q disponibles en EODHD — potentiellement récupérable |
| DO | 113 | 113q disponibles en EODHD — potentiellement récupérable |
| LSI | 110 | 110q disponibles en EODHD — potentiellement récupérable |
| PXD | 110 | 110q disponibles en EODHD — potentiellement récupérable |
| TWX | 110 | 110q disponibles en EODHD — potentiellement récupérable |
| VAR | 108 | 108q disponibles en EODHD — potentiellement récupérable |
| ESRX | 107 | 107q disponibles en EODHD — potentiellement récupérable |
| CTXS | 106 | 106q disponibles en EODHD — potentiellement récupérable |
| FTR | 105 | 105q disponibles en EODHD — potentiellement récupérable |
| JNPR | 105 | 105q disponibles en EODHD — potentiellement récupérable |
| ALXN | 103 | 103q disponibles en EODHD — potentiellement récupérable |
| XLNX | 103 | 103q disponibles en EODHD — potentiellement récupérable |
| DNB | 102 | 102q disponibles en EODHD — potentiellement récupérable |
| FLIR | 100 | 100q disponibles en EODHD — potentiellement récupérable |
| GRA | 94 | 94q disponibles en EODHD — potentiellement récupérable |
| ENDP | 91 | 91q disponibles en EODHD — potentiellement récupérable |
| ADS | 89 | 89q disponibles en EODHD — potentiellement récupérable |
| HBI | 89 | 89q disponibles en EODHD — potentiellement récupérable |
| STR | 88 | 88q disponibles en EODHD — potentiellement récupérable |
| TIF | 84 | 84q disponibles en EODHD — potentiellement récupérable |
| DFS | 82 | 82q disponibles en EODHD — potentiellement récupérable |
| DNR | 82 | 82q disponibles en EODHD — potentiellement récupérable |
| LM | 82 | 82q disponibles en EODHD — potentiellement récupérable |
| NBL | 82 | 82q disponibles en EODHD — potentiellement récupérable |
| AGN | 81 | 81q disponibles en EODHD — potentiellement récupérable |
| ETFC | 81 | 81q disponibles en EODHD — potentiellement récupérable |
| FII | 81 | 81q disponibles en EODHD — potentiellement récupérable |
| JCP | 81 | 81q disponibles en EODHD — potentiellement récupérable |
| LVLT | 81 | 81q disponibles en EODHD — potentiellement récupérable |
| NLSN | 81 | 81q disponibles en EODHD — potentiellement récupérable |
| AKS | 80 | 80q disponibles en EODHD — potentiellement récupérable |
| DF | 80 | 80q disponibles en EODHD — potentiellement récupérable |
| QEP | 80 | 80q disponibles en EODHD — potentiellement récupérable |
| RTN | 80 | 80q disponibles en EODHD — potentiellement récupérable |
| SBNY | 80 | 80q disponibles en EODHD — potentiellement récupérable |
| XEC | 80 | 80q disponibles en EODHD — potentiellement récupérable |
| AVP | 79 | 79q disponibles en EODHD — potentiellement récupérable |
| CELG | 79 | 79q disponibles en EODHD — potentiellement récupérable |
| RHT | 79 | 79q disponibles en EODHD — potentiellement récupérable |
| APC | 78 | 78q disponibles en EODHD — potentiellement récupérable |
| ESV | 78 | 78q disponibles en EODHD — potentiellement récupérable |
| FDC | 78 | 78q disponibles en EODHD — potentiellement récupérable |
| TSS | 78 | 78q disponibles en EODHD — potentiellement récupérable |
| DWDP | 77 | 77q disponibles en EODHD — potentiellement récupérable |
| RDC | 76 | 76q disponibles en EODHD — potentiellement récupérable |
| SCG | 76 | 76q disponibles en EODHD — potentiellement récupérable |
| COL | 75 | 75q disponibles en EODHD — potentiellement récupérable |
| NFX | 75 | 75q disponibles en EODHD — potentiellement récupérable |
| SII | 75 | 75q disponibles en EODHD — potentiellement récupérable |
| SVU | 75 | 75q disponibles en EODHD — potentiellement récupérable |
| XL | 75 | 75q disponibles en EODHD — potentiellement récupérable |
| ANDV | 74 | 74q disponibles en EODHD — potentiellement récupérable |
| CVG | 74 | 74q disponibles en EODHD — potentiellement récupérable |
| DISCA | 74 | 74q disponibles en EODHD — potentiellement récupérable |
| EVHC | 74 | 74q disponibles en EODHD — potentiellement récupérable |
| GGP | 74 | 74q disponibles en EODHD — potentiellement récupérable |
| DISCK | 73 | 73q disponibles en EODHD — potentiellement récupérable |
| BCR | 71 | 71q disponibles en EODHD — potentiellement récupérable |
| WCG | 71 | 71q disponibles en EODHD — potentiellement récupérable |
| WFM | 71 | 71q disponibles en EODHD — potentiellement récupérable |
| BHI | 70 | 70q disponibles en EODHD — potentiellement récupérable |
| RAI | 70 | 70q disponibles en EODHD — potentiellement récupérable |
| DYN | 69 | 69q disponibles en EODHD — potentiellement récupérable |
| JNS | 69 | 69q disponibles en EODHD — potentiellement récupérable |
| JOY | 69 | 69q disponibles en EODHD — potentiellement récupérable |
| POM | 69 | 69q disponibles en EODHD — potentiellement récupérable |
| APOL | 68 | 68q disponibles en EODHD — potentiellement récupérable |
| HAR | 68 | 68q disponibles en EODHD — potentiellement récupérable |
| LLTC | 68 | 68q disponibles en EODHD — potentiellement récupérable |
| ACAS | 67 | 67q disponibles en EODHD — potentiellement récupérable |
| LXK | 67 | 67q disponibles en EODHD — potentiellement récupérable |
| MWW | 67 | 67q disponibles en EODHD — potentiellement récupérable |
| STJ | 67 | 67q disponibles en EODHD — potentiellement récupérable |
| HOT | 66 | 66q disponibles en EODHD — potentiellement récupérable |
| IGT | 66 | 66q disponibles en EODHD — potentiellement récupérable |
| ARG | 65 | 65q disponibles en EODHD — potentiellement récupérable |
| PCP | 64 | 64q disponibles en EODHD — potentiellement récupérable |
| BRCM | 63 | 63q disponibles en EODHD — potentiellement récupérable |
| SIAL | 63 | 63q disponibles en EODHD — potentiellement récupérable |
| VIAB | 63 | 63q disponibles en EODHD — potentiellement récupérable |
| FDO | 62 | 62q disponibles en EODHD — potentiellement récupérable |
| PETM | 61 | 61q disponibles en EODHD — potentiellement récupérable |
| CXO | 60 | 60q disponibles en EODHD — potentiellement récupérable |
| SWY | 59 | 59q disponibles en EODHD — potentiellement récupérable |
| LO | 58 | 58q disponibles en EODHD — potentiellement récupérable |
| MON | 57 | 57q disponibles en EODHD — potentiellement récupérable |
| WIN | 57 | 57q disponibles en EODHD — potentiellement récupérable |
| JNY | 56 | 56q disponibles en EODHD — potentiellement récupérable |
| ARNC | 55 | 55q disponibles en EODHD — potentiellement récupérable |
| OMX | 55 | 55q disponibles en EODHD — potentiellement récupérable |
| CVH | 53 | 53q disponibles en EODHD — potentiellement récupérable |
| WB | 53 | 53q disponibles en EODHD — potentiellement récupérable |
| FBHS | 52 | 52q disponibles en EODHD — potentiellement récupérable |
| HSP | 51 | 51q disponibles en EODHD — potentiellement récupérable |
| TIE | 51 | 51q disponibles en EODHD — potentiellement récupérable |
| CTLT | 50 | 50q disponibles en EODHD — potentiellement récupérable |
| DPS | 47 | 47q disponibles en EODHD — potentiellement récupérable |
| TWTR | 46 | 46q disponibles en EODHD — potentiellement récupérable |
| MNK | 45 | 45q disponibles en EODHD — potentiellement récupérable |
| SNI | 45 | 45q disponibles en EODHD — potentiellement récupérable |
| AYE | 44 | 44q disponibles en EODHD — potentiellement récupérable |
| MFE | 44 | 44q disponibles en EODHD — potentiellement récupérable |
| WPX | 44 | 44q disponibles en EODHD — potentiellement récupérable |
| MHS | 42 | 42q disponibles en EODHD — potentiellement récupérable |
| SE | 41 | 41q disponibles en EODHD — potentiellement récupérable |
| TWC | 41 | 41q disponibles en EODHD — potentiellement récupérable |
| XTO | 41 | 41q disponibles en EODHD — potentiellement récupérable |
| TSG | 40 | 40q disponibles en EODHD — potentiellement récupérable |
| DAY | 39 | 39q disponibles en EODHD — potentiellement récupérable |
| MJN | 39 | 39q disponibles en EODHD — potentiellement récupérable |
| ALTR | 37 | 37q disponibles en EODHD — potentiellement récupérable |
| COV | 36 | 36q disponibles en EODHD — potentiellement récupérable |
| CFN | 34 | 34q disponibles en EODHD — potentiellement récupérable |
| NSM | 32 | 32q disponibles en EODHD — potentiellement récupérable |
| PLL | 29 | 29q disponibles en EODHD — potentiellement récupérable |
| DTV | 25 | 25q disponibles en EODHD — potentiellement récupérable |
| CSC | 20 | 20q disponibles en EODHD — potentiellement récupérable |
| KRFT | 19 | 19q disponibles en EODHD — potentiellement récupérable |
| CSRA | 16 | 16q disponibles en EODHD — potentiellement récupérable |
| HPH | 16 | 16q disponibles en EODHD — potentiellement récupérable |
| MI | 16 | 16q disponibles en EODHD — potentiellement récupérable |
| TEG | 16 | 16q disponibles en EODHD — potentiellement récupérable |
| LLL | 15 | 15q disponibles en EODHD — potentiellement récupérable |
| GAS | 14 | 14q disponibles en EODHD — potentiellement récupérable |
| BXLT | 10 | 10q disponibles en EODHD — potentiellement récupérable |
| CCE | 10 | 10q disponibles en EODHD — potentiellement récupérable |
| CPGX | 10 | 10q disponibles en EODHD — potentiellement récupérable |
| WYN | 8 | Très peu d'historique EODHD |
| SAI | 6 | Très peu d'historique EODHD |
| YHOO | 5 | Très peu d'historique EODHD |
| BJS | 3 | Très peu d'historique EODHD |
| ABS | 2 | Très peu d'historique EODHD |
| LIFE | 2 | Très peu d'historique EODHD |
| ABK | 0 | Aucune donnée financière historique connue |
| ACE | 0 | Aucune donnée financière historique connue |
| ANR | 0 | Aucune donnée financière historique connue |
| AV | 0 | Aucune donnée financière historique connue |
| BF.B | 0 | Aucune donnée financière historique connue |
| BMC | 0 | Aucune donnée financière historique connue |
| BRK.B | 0 | Aucune donnée financière historique connue |
| BS | 0 | Aucune donnée financière historique connue |
| CA | 0 | Aucune donnée financière historique connue |
| CAM | 0 | Aucune donnée financière historique connue |
| CBE | 0 | Aucune donnée financière historique connue |
| CEPH | 0 | Aucune donnée financière historique connue |
| CFC | 0 | Aucune donnée financière historique connue |
| CMCSK | 0 | Aucune donnée financière historique connue |
| CTX | 0 | Aucune donnée financière historique connue |
| CVC | 0 | Aucune donnée financière historique connue |
| DJ | 0 | Aucune donnée financière historique connue |
| EK | 0 | Aucune donnée financière historique connue |
| EMC | 0 | Aucune donnée financière historique connue |
| FNM | 0 | Aucune donnée financière historique connue |
| FRE | 0 | Aucune donnée financière historique connue |
| FRX | 0 | Aucune donnée financière historique connue |
| GENZ | 0 | Aucune donnée financière historique connue |
| GLK | 0 | Aucune donnée financière historique connue |
| GMCR | 0 | Aucune donnée financière historique connue |
| GR | 0 | Aucune donnée financière historique connue |
| GRN | 0 | Aucune donnée financière historique connue |
| HCBK | 0 | Aucune donnée financière historique connue |
| HNZ | 0 | Aucune donnée financière historique connue |
| INFO | 0 | Aucune donnée financière historique connue |
| JDSU | 0 | Aucune donnée financière historique connue |
| KFT | 0 | Aucune donnée financière historique connue |
| KSE | 0 | Aucune donnée financière historique connue |
| LDW | 0 | Aucune donnée financière historique connue |
| LEH | 0 | Aucune donnée financière historique connue |
| MEE | 0 | Aucune donnée financière historique connue |
| MIL | 0 | Aucune donnée financière historique connue |
| MOLX | 0 | Aucune donnée financière historique connue |
| NOVL | 0 | Aucune donnée financière historique connue |
| NVLS | 0 | Aucune donnée financière historique connue |
| NYX | 0 | Aucune donnée financière historique connue |
| PCL | 0 | Aucune donnée financière historique connue |
| PCS | 0 | Aucune donnée financière historique connue |
| PGN | 0 | Aucune donnée financière historique connue |
| PTV | 0 | Aucune donnée financière historique connue |
| QTRN | 0 | Aucune donnée financière historique connue |
| RSH | 0 | Aucune donnée financière historique connue |
| RX | 0 | Aucune donnée financière historique connue |
| SBL | 0 | Aucune donnée financière historique connue |
| SGP | 0 | Aucune donnée financière historique connue |
| SHLD | 0 | Aucune donnée financière historique connue |
| SLR | 0 | Aucune donnée financière historique connue |
| SMS | 0 | Aucune donnée financière historique connue |
| SPLS | 0 | Aucune donnée financière historique connue |
| TLAB | 0 | Aucune donnée financière historique connue |
| TRB | 0 | Aucune donnée financière historique connue |
| TYC | 0 | Aucune donnée financière historique connue |
| USL | 0 | Aucune donnée financière historique connue |
| WFR | 0 | Aucune donnée financière historique connue |

### 2.2 SP500 Actuel — Couverture Faible revenue SEC (< 20 quarters = 5 ans)

Ces tickers ont du revenue SEC mais pas assez pour un backtest historique robuste. Il s'agit souvent de spinoffs, mergers récents, ou tickers dont le CIK a changé.

| Ticker | SEC revenue | EODHD revenue | SEC range | Note |
|--------|-------------|---------------|-----------|------|
| Q | 2q | 8q | 2024-09-30 → 2025-09-30 | Spinoff / IPO très récent |
| SOLS | 2q | 6q | 2024-09-30 → 2025-09-30 | Spinoff / IPO très récent |
| KG | 4q | 53q | 2024-06-30 → 2025-09-30 | Spinoff / IPO très récent |
| STI | 4q | 61q | 2021-09-30 → 2025-09-30 | Spinoff / IPO très récent |
| PSKY | 5q | 120q | 2023-12-31 → 2025-06-30 | Nouvel entrant / fusion récente |
| TMC | 5q | 21q | 2019-12-31 → 2025-09-30 | Nouvel entrant / fusion récente |
| APA | 6q | 162q | 2019-12-31 → 2025-12-31 | Nouvel entrant / fusion récente |
| TE | 6q | 67q | 2023-03-31 → 2025-09-30 | Nouvel entrant / fusion récente |
| SNDK | 7q | 10q | 2023-09-30 → 2025-12-31 | Nouvel entrant / fusion récente |
| AMTM | 10q | 13q | 2023-09-30 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| BLK | 10q | 108q | 2023-09-30 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| SW | 10q | 68q | 2023-06-30 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| ADCT | 12q | 32q | 2023-03-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| CRH | 13q | 137q | 2022-12-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| GEV | 13q | 12q | 2022-12-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| SOLV | 13q | 16q | 2022-12-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| TKO | 14q | 20q | 2022-09-30 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| VLTO | 14q | 16q | 2022-09-30 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| BG | 16q | 104q | 2022-03-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| GEHC | 17q | 20q | 2021-12-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| KVUE | 17q | 17q | 2021-12-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| MBC | 17q | 20q | 2021-12-31 → 2025-12-31 | Couverture partielle — probablement CIK hérité |
| NE | 18q | 162q | 2021-09-30 → 2025-12-31 | Couverture partielle — probablement CIK hérité |

### 2.3 SP500 Actuel — Couverture Faible netIncome SEC (< 20 quarters)

| Ticker | SEC netIncome | EODHD netIncome | Note |
|--------|---------------|-----------------|------|
| Q | 2q | 8q | |
| SOLS | 2q | 6q | |
| KG | 4q | 42q | |
| PSKY | 5q | 120q | |
| SNDK | 8q | 10q | |
| AMTM | 10q | 13q | |
| BLK | 10q | 108q | |
| SW | 11q | 68q | |
| TE | 11q | 82q | |
| ADCT | 12q | 32q | |
| CRH | 13q | 136q | |
| GEV | 13q | 12q | |
| SOLV | 13q | 16q | |
| TKO | 14q | 20q | |
| VLTO | 14q | 16q | |
| BG | 16q | 104q | |
| STI | 16q | 85q | |
| GEHC | 17q | 20q | |
| KVUE | 17q | 17q | |
| MBC | 17q | 20q | |
| NE | 18q | 147q | |

---

## 3. Tickers avec Revenue EODHD mais PAS SEC (166 tickers)

Ce sont des cas où EODHD a des données mais le pipeline SEC n'en a pas. Pour beaucoup, c'est parce que le ticker a changé de CIK ou est delisted. Pour d'autres, c'est un problème de mapping ou de tags XBRL.

**Top 30 par volume EODHD manquant :**

| Ticker | EODHD revenue | SP500 actuel ? | Diagnostic probable |
|--------|---------------|----------------|---------------------|
| IPG | 161q | Oui | Delisted / ticker changé — historique EODHD existe |
| K | 161q | Oui | Delisted / ticker changé — historique EODHD existe |
| FL | 160q | Oui | Delisted / ticker changé — historique EODHD existe |
| WBA | 160q | Oui | Delisted / ticker changé — historique EODHD existe |
| HES | 159q | Oui | Delisted / ticker changé — historique EODHD existe |
| JWN | 159q | Oui | Delisted / ticker changé — historique EODHD existe |
| SWN | 157q | Oui | Delisted / ticker changé — historique EODHD existe |
| GPS | 156q | Oui | Delisted / ticker changé — historique EODHD existe |
| ODP | 155q | Oui | Delisted / ticker changé — historique EODHD existe |
| BIG | 154q | Oui | Delisted / ticker changé — historique EODHD existe |
| RAD | 153q | Oui | Delisted / ticker changé — historique EODHD existe |
| ABMD | 146q | Oui | Delisted / ticker changé — historique EODHD existe |
| RRD | 146q | Oui | Delisted / ticker changé — historique EODHD existe |
| CERN | 145q | Oui | Delisted / ticker changé — historique EODHD existe |
| KSU | 145q | Oui | Delisted / ticker changé — historique EODHD existe |
| MDP | 145q | Oui | Delisted / ticker changé — historique EODHD existe |
| CMA | 144q | Oui | Delisted / ticker changé — historique EODHD existe |
| X | 141q | Oui | Delisted / ticker changé — historique EODHD existe |
| MXIM | 140q | Oui | Delisted / ticker changé — historique EODHD existe |
| MRO | 139q | Oui | Delisted / ticker changé — historique EODHD existe |
| PDCO | 139q | Oui | Delisted / ticker changé — historique EODHD existe |
| BMS | 135q | Oui | Delisted / ticker changé — historique EODHD existe |
| AET | 133q | Oui | Delisted / ticker changé — historique EODHD existe |
| DRE | 133q | Oui | Delisted / ticker changé — historique EODHD existe |
| CHK | 128q | Oui | Delisted / ticker changé — historique EODHD existe |
| HFC | 128q | Oui | Delisted / ticker changé — historique EODHD existe |
| PBCT | 128q | Oui | Delisted / ticker changé — historique EODHD existe |
| ATVI | 125q | Oui | Delisted / ticker changé — historique EODHD existe |
| SIVB | 125q | Oui | Delisted / ticker changé — historique EODHD existe |
| FRC | 124q | Oui | Delisted / ticker changé — historique EODHD existe |

---

## 4. Top Couverture SEC (Référence)

Tickers avec la meilleure couverture revenue SEC (>= 73 quarters = ~18 ans) :

| Ticker | Revenue quarters | Range |
|--------|------------------|-------|
| BBY | 74q | 2008-08-31 → 2025-10-31 |
| HRB | 74q | 2008-04-30 → 2025-12-31 |
| MSFT | 74q | 2007-09-30 → 2025-12-31 |
| ADSK | 73q | 2008-01-31 → 2026-01-31 |
| CSX | 73q | 2007-12-31 → 2025-12-31 |
| CVX | 73q | 2007-12-31 → 2025-12-31 |
| FE | 73q | 2007-12-31 → 2025-12-31 |
| FLR | 73q | 2007-12-31 → 2025-12-31 |
| GPC | 73q | 2006-12-31 → 2025-12-31 |
| J | 73q | 2007-12-31 → 2025-12-31 |
| LUV | 73q | 2007-12-31 → 2025-12-31 |
| NEM | 73q | 2007-12-31 → 2025-12-31 |
| WU | 73q | 2007-12-31 → 2025-12-31 |
| ABT | 72q | 2008-03-31 → 2025-12-31 |
| GE | 72q | 2007-12-31 → 2025-12-31 |

---

## 5. Analyse et Recommandations

### Pourquoi 225 tickers du SP500 actuel n'ont pas de revenue SEC ?

1. **Delisted / Acquis :** Beaucoup de ces tickers ne sont plus cotés (ex: ATVI acquis par Microsoft, CELG acquis par BMS, etc.). Le SEC mapping actuel ne les inclut pas car ils n'ont plus de ticker actif.
2. **Changement de ticker/CIK :** Des fusions ou restructurations créent de nouveaux CIK. Le mapping `company_tickers_exchange.json` de la SEC ne contient que les tickers actuellement cotés.
3. **Spinoffs récents :** Des tickers comme GEHC (spinoff GE), KVUE (spinoff JNJ), VLTO (spinoff VNT) ont des historiques courts car ils n'existaient pas comme entités SEC indépendantes avant.
4. **Données manquantes dans les tags :** Certains tickers (rare) pourraient rapporter sous des tags XBRL que `METRIC_SPECS` ne capture pas encore.

### Qu'est-ce qui est récupérable ?

- **141 tickers** du SP500 actuel ont >= 40 quarters de revenue EODHD mais 0 en SEC.
- Ces tickers sont principalement des anciens constituents delisted mais avec un historique riche en EODHD.
- Pour un backtest S&P 500, ce n'est pas critique : quand un ticker sort du SP500, il n'est plus dans l'univers de trading.
- Cependant, si tu veux calculer des features de croissance sur des fenêtres glissantes avant l'entrée dans le SP500, ces données manquantes posent problème.

### Tickers critiques à surveiller

Ces tickers sont dans le SP500 actuel, ont une couverture EODHD > 100q, mais 0 en SEC. Si tu fais du walk-forward depuis 2010, ces trous vont fausser tes features de croissance historique :

- **IPG** : 161q EODHD, 0 SEC
- **K** : 161q EODHD, 0 SEC
- **FL** : 160q EODHD, 0 SEC
- **WBA** : 160q EODHD, 0 SEC
- **HES** : 159q EODHD, 0 SEC
- **JWN** : 159q EODHD, 0 SEC
- **SWN** : 157q EODHD, 0 SEC
- **GPS** : 156q EODHD, 0 SEC
- **ODP** : 155q EODHD, 0 SEC
- **BIG** : 154q EODHD, 0 SEC
- **RAD** : 153q EODHD, 0 SEC
- **ABMD** : 146q EODHD, 0 SEC
- **RRD** : 146q EODHD, 0 SEC
- **CERN** : 145q EODHD, 0 SEC
- **KSU** : 145q EODHD, 0 SEC
- **MDP** : 145q EODHD, 0 SEC
- **CMA** : 144q EODHD, 0 SEC
- **X** : 141q EODHD, 0 SEC
- **MXIM** : 140q EODHD, 0 SEC
- **MRO** : 139q EODHD, 0 SEC
- **PDCO** : 139q EODHD, 0 SEC
- **BMS** : 135q EODHD, 0 SEC
- **AET** : 133q EODHD, 0 SEC
- **DRE** : 133q EODHD, 0 SEC
- **CHK** : 128q EODHD, 0 SEC
- **HFC** : 128q EODHD, 0 SEC
- **PBCT** : 128q EODHD, 0 SEC
- **ATVI** : 125q EODHD, 0 SEC
- **SIVB** : 125q EODHD, 0 SEC
- **FRC** : 124q EODHD, 0 SEC
- **SRCL** : 119q EODHD, 0 SEC
- **ANSS** : 117q EODHD, 0 SEC
- **DISH** : 117q EODHD, 0 SEC
- **NCR** : 115q EODHD, 0 SEC
- **DO** : 113q EODHD, 0 SEC
- **LSI** : 110q EODHD, 0 SEC
- **PXD** : 110q EODHD, 0 SEC
- **TWX** : 110q EODHD, 0 SEC
- **VAR** : 108q EODHD, 0 SEC
- **ESRX** : 107q EODHD, 0 SEC
- **CTXS** : 106q EODHD, 0 SEC
- **FTR** : 105q EODHD, 0 SEC
- **JNPR** : 105q EODHD, 0 SEC
- **ALXN** : 103q EODHD, 0 SEC
- **XLNX** : 103q EODHD, 0 SEC
- **DNB** : 102q EODHD, 0 SEC
- **FLIR** : 100q EODHD, 0 SEC
- **GRA** : 94q EODHD, 0 SEC
- **ENDP** : 91q EODHD, 0 SEC
- **ADS** : 89q EODHD, 0 SEC
- **HBI** : 89q EODHD, 0 SEC
- **STR** : 88q EODHD, 0 SEC
- **TIF** : 84q EODHD, 0 SEC
- **DFS** : 82q EODHD, 0 SEC
- **DNR** : 82q EODHD, 0 SEC
- **LM** : 82q EODHD, 0 SEC
- **NBL** : 82q EODHD, 0 SEC
- **AGN** : 81q EODHD, 0 SEC
- **ETFC** : 81q EODHD, 0 SEC
- **FII** : 81q EODHD, 0 SEC
- **JCP** : 81q EODHD, 0 SEC
- **LVLT** : 81q EODHD, 0 SEC
- **NLSN** : 81q EODHD, 0 SEC

---

## 6. Fichier de Données Brutes

Le fichier complet de couverture (849 tickers) est disponible ici :
- `outputs/sec_sp500_coverage_report.csv`
- `outputs/sec_sp500_coverage_report.parquet`

Colonnes disponibles :
- `ticker` : symbole
- `in_current_sp500` : présent dans le SP500 actuel (2026-04-01)
- `sec_revenue_rows` / `sec_netincome_rows` : nombre de quarters avec données SEC
- `eodhd_revenue_rows` / `eodhd_netincome_rows` : nombre de quarters avec données EODHD legacy
- `sec_first_date` / `sec_last_date` : range de dates SEC
- `eodhd_first_date` / `eodhd_last_date` : range de dates EODHD

---

## 7. Plan d'Action Noté (P0 #1 et P0 #2 reportés)

Comme convenu, les plans 1 et 2 identifiés dans l'audit général sont notés pour plus tard :

### P0 #1 — Rebuild target open-source depuis le raw complet
Le raw `financials_sec_companyfacts.parquet` contient 324 404 rows. Le target n'en a que 38 444.
**Action :** Modifier le runner pour rebuild le consolidated depuis la totalité du raw à chaque run.

### P0 #2 — Désactiver earnings-implied shares pour le package SEC
Dans `legacy_export.py`, la logique `netIncome / epsActual` modifie les shares SEC.
**Action :** Conditionner cette logique pour qu'elle ne s'applique pas au build SEC-only.

### P1 — Étendre METRIC_SPECS
Les 225 tickers sans revenue SEC ne sont pas tous récupérables, mais certains le sont via des tags XBRL additionnels (ex: `RevenuesNetOfInterestExpense` pour les banques).
**Action :** Ajouter des tags SEC et investiguer les cas spécifiques (banques, REITs, etc.).

---

*Fin du rapport.*
