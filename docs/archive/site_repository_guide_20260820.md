# Archive du guide publié AlphaRank au 20 août 2026

> Cette page contient une ancienne vue du dépôt et des résultats. Elle est
> conservée pour traçabilité mais ne doit plus alimenter le site. Utiliser
> [`../../README.md`](../../README.md) et [`../README.md`](../README.md) pour
> l'état courant.

# AlphaRank : guide du dépôt

## Ce que fait le projet

AlphaRank construit et audite des portefeuilles actions mensuels. Deux méthodes
coexistent volontairement :

- **Legacy** est la baseline de production mensuelle ;
- **Boosting** est le challenger XGBoost évalué hors échantillon.

Les deux méthodes génèrent leurs signaux séparément, puis utilisent le même
moteur de portefeuille pour les poids, frais, rendements, benchmark et KPI.

## Le parcours d'une décision

1. Une ingestion horodatée télécharge les prix actifs, SPY, les déclarations SEC
   et la composition historique du S&P 500.
2. Les prix ouverts sont combinés avec l'archive EODHD figée afin de conserver
   les sociétés delistées.
3. Les fondamentaux officiels sont reconstruits uniquement depuis la SEC.
4. Un snapshot modèle immutable est publié avec ses hashes et manifestes.
5. Legacy et Boosting consomment exactement le même snapshot.
6. Une information du mois `t` produit uniquement une détention en `t+1`.
7. Le replay commun compare les stratégies sur les mêmes mois, coûts et prix SPY.

## Arborescence AlphaRank

```text
alpharank/
├── README.md              orientation et commandes
├── AGENTS.md              règles impératives pour les agents
├── AGENT.md               guide technique détaillé
├── configs/               décisions versionnées de qualité et de recherche
├── data/                  sources et snapshots locaux volumineux
├── docs/                  contrats canoniques et rapports historiques
├── scripts/               entrées CLI et orchestration
├── src/alpharank/         logique Python réutilisable
├── tests/                 protections unitaires et de replay
├── outputs/               artefacts de runs, ignorés par Git
└── logs/                  journaux d'exécution, ignorés par Git
```

Chaque dossier actif possède un README local qui décrit ses enfants et renvoie
vers le contrat canonique approprié.

## Méthode Legacy

Pour chaque mois de décision, Legacy :

1. utilise uniquement la composition S&P 500 connue à cette date ;
2. applique les contrôles de prix et de liquidité ;
3. calcule les signaux EMA relatifs à SPY ;
4. applique capitalisation, secteurs et `0 < PE < 100` avec des données SEC déjà
   publiées à la date de décision ;
5. exécute quatre recherches annuelles walk-forward ;
6. agrège leurs votes en `Combined_Equal` et `Combined_Frequency`.

Le portefeuille mensuel canonique est `Combined_Frequency` et se lance avec
`scripts/run_legacy.py`.

## Méthode Boosting

Le challenger public utilise XGBoost avec des folds chronologiques expansifs.
Le modèle courant apprend principalement depuis des signaux de prix et EMA ; il
charge et hash le snapshot fondamental commun mais n'injecte pas les valeurs SEC
dans son score final.

Les Top 5, 10, 15 et 20 utilisent les mêmes scores hors échantillon. Seul le
nombre de positions change. Une vue supplémentaire applique avant ranking le
même univers SEC/PE que Legacy afin de mesurer l'asymétrie de couverture.

## Contrôles anti-fuite

- décision `t`, détention `t+1` ;
- fondamentaux disponibles à partir de la date de filing, pas de la fin fiscale ;
- composition historique de l'indice ;
- prix et liquidité évalués mois par mois ;
- mois courant incomplet exclu des performances ;
- labels futurs immatures exclus des métriques modèle ;
- folds train/validation/test chronologiques et purgés ;
- hashes identiques obligatoires pour une comparaison Legacy/Boosting ;
- snapshots publiés immuables et révisions de prix fail-closed.

## Données et sociétés delistées

Les prix suivent un contrat hybride :

- EODHD reste le seed historique figé pour les titres anciens et delistés ;
- Yahoo rafraîchit les trajectoires encore disponibles ;
- une indisponibilité actuelle ne supprime jamais un historique déjà publié ;
- splits et dividendes confirmés créent un nouveau package versionné.

Les fondamentaux officiels sont SEC-only. Yahoo, SimFin, StockAnalysis et EODHD
peuvent servir à l'audit ou au mapping d'identité, jamais à remplir une valeur
fondamentale officielle manquante.

## Référence comparable actuelle

- ingestion : `20260816_103942` ;
- snapshot : `alpharank_input_20260816_120458_2a01288bab06` ;
- calendrier réalisé : août 2011 à juillet 2026, 180 mois ;
- coûts : 10 bps multipliés par le turnover pour les stratégies ;
- benchmark : SPY total return depuis `adjusted_close` ;
- replay : `common_replay_v4_sec_universe`.

Sur l'univers natif, les CAGR sont 28,16 % pour Top 5, 26,57 % pour Top 10,
24,31 % pour Top 15, 20,61 % pour Top 20, 19,00 % pour Legacy et 14,40 % pour
SPY. Top 5 atteint toutefois un drawdown de -51,97 %.

Sur l'univers PE identique à Legacy, Top 10 conserve 26,07 % de CAGR, avec
26,15 % de volatilité, 0,92 de Sharpe et -21,84 % de drawdown maximal. Le filtre
SEC n'explique donc pas son avantage historique, mais le bootstrap ne prouve pas
encore une supériorité statistique sur Legacy.

## Quelle documentation lire

- **Run mensuel** : `docs/monthly_portfolio_runbook.md`
- **Legacy et Boosting** : `docs/legacy_boosting_methodology.md`
- **KPI et replay commun** : `docs/common_portfolio_backtest_engine.md`
- **Ingestion et stockage** : `docs/open_source_ingestion_architecture.md`
- **Contrat SEC** : `docs/sec_fundamentals_contract.md`
- **État SEC** : `docs/sec_open_source_status.md`
- **Catalogue R&D** : `docs/boosting_signal_copy_model_catalog.md`

Le sommaire complet se trouve dans `docs/README.md`. Les rapports datés et
`docs/research/` sont des preuves historiques, pas la source de vérité courante.

## Commandes essentielles

```bash
# Installer le package
python -m pip install -e .

# Lancer Legacy sur un snapshot immutable
./.venv/bin/python scripts/run_legacy.py --data-dir <snapshot>

# Valider un package Legacy
./.venv/bin/python scripts/validate_legacy_replay_package.py <manifest>

# Contrôler la documentation
./.venv/bin/python scripts/validate_documentation.py

# Tests ciblés
./.venv/bin/pytest -q tests/test_documentation_structure.py
```

## Règle de changement

Une modification de données, méthode, portefeuille ou publication est terminée
seulement lorsque le code, les tests, le manifeste du run et le document
canonique correspondant racontent la même chose.
