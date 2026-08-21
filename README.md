# AlphaRank

AlphaRank est un projet de recherche quantitative actions avec deux méthodes :

- **Legacy** : méthode mensuelle de référence ;
- **Boosting** : challenger XGBoost évalué dans le temps.

Les deux méthodes produisent leurs signaux séparément. Toute comparaison
publiée doit ensuite utiliser le même snapshot de données, le même moteur de
portefeuille, les mêmes frais et le même benchmark.

## Commencer ici

1. [`ROADMAP.md`](ROADMAP.md) : état du rangement et prochaines tâches.
2. [`CONTRIBUTING.md`](CONTRIBUTING.md) : normes de développement actives.
3. [`docs/README.md`](docs/README.md) : carte de toute la documentation.
4. [`docs/architecture/repository_map.md`](docs/architecture/repository_map.md) :
   où se trouvent le code, les données et les résultats.
5. [`docs/monthly_portfolio_runbook.md`](docs/monthly_portfolio_runbook.md) :
   procédure mensuelle Legacy.

Le registre détaillé et append-only de l'audit méthodologique est conservé dans
[`METHODOLOGY_AUDIT_ROADMAP.md`](METHODOLOGY_AUDIT_ROADMAP.md). Il n'ordonne pas
les priorités : seule [`ROADMAP.md`](ROADMAP.md) le fait.

## Carte du dépôt

```text
configs/          décisions versionnées de qualité et de recherche
data/             données locales, transformations et snapshots
docs/             contrats courants, recherche et archives
scripts/          commandes et orchestration
src/alpharank/    logique Python réutilisable
tests/            tests unitaires, intégration et replay
outputs/          résultats de runs, ignorés par Git
logs/             journaux d'exécution, ignorés par Git
```

Les responsabilités détaillées sont dans les README locaux :
[`src/alpharank/README.md`](src/alpharank/README.md),
[`scripts/README.md`](scripts/README.md), [`data/README.md`](data/README.md) et
[`tests/README.md`](tests/README.md).

## Donnée de production courante

Ne pas choisir un fichier en parcourant `data/` ou `outputs/`. Le seul pointeur
actuel vers le snapshot composé de production est :

```text
data/model_inputs/manifests/latest.json
```

Sa cible est immuable. `data/open_source/output/`, `data/sec/output/`,
`data/eodhd/output/` et les Parquet directement sous `data/` ont encore des
rôles historiques ou de replay, mais ne sont pas des substituts libres à ce
pointeur.

La cible de production doit être sous `data/warehouse/mart/`. Sans
`--data-dir`, `scripts/run_legacy.py` résout ce pointeur, vérifie le manifeste
MART, les neuf hashes de fichiers et la parité DEF/source avant de créer son
`input_snapshot/`. Un `--data-dir` explicite reste réservé aux replays décrits
par le runbook.

Une publication snapshot ne recopie pas ce MART : son manifeste immuable
inventorie et hashe l'arbre complet, puis `latest.json` référence ce manifeste
et le même dossier MART. Le pointeur est mutable ; le contenu publié ne l'est
pas.

L'explication de `raw -> stg -> def -> mart -> snapshot` est dans
[`docs/architecture/data_lifecycle.md`](docs/architecture/data_lifecycle.md).

## Commandes principales

| Besoin | Commande |
| --- | --- |
| portefeuille Legacy mensuel | `scripts/run_legacy.py` |
| recherche/backtest Boosting | `scripts/run_backtest.py` |
| comparaison commune | `scripts/build_common_legacy_boosting_replay.py` |
| validation stricte Legacy | `scripts/validate_legacy_replay_package.py` |
| ingestion open source | `scripts/open_source/ingestion/run_ingestion.py` |
| composition du snapshot | `scripts/open_source/publication/build_composed_model_snapshot.py` |
| validation documentaire | `scripts/validate_documentation.py` |

Avant une production mensuelle, lire le
[`runbook`](docs/monthly_portfolio_runbook.md). Avant de modifier les méthodes,
lire [`docs/legacy_boosting_methodology.md`](docs/legacy_boosting_methodology.md)
et [`docs/common_portfolio_backtest_engine.md`](docs/common_portfolio_backtest_engine.md).

## Installation

Le projet utilise une structure Python `src/` et des imports `alpharank.*`.

```bash
python -m pip install -e '.[dev]'
```

L'environnement Conda existant peut aussi être créé avec :

```bash
bash scripts/setup_conda_env.sh alpharank
conda activate alpharank
```

Les dépendances sont déclarées uniquement dans `pyproject.toml`.
`requirements.txt` est sa vue runtime générée et `environment.yml` délègue
l'installation Python à l'extra `dev` du même fichier.

## Vérifications courantes

```bash
./.venv/bin/python -m pytest
./.venv/bin/python scripts/validate_documentation.py
```

Les validations supplémentaires de données, causalité et replay dépendent du
périmètre. Elles sont listées dans le runbook et les contrats spécialisés ; une
suite unitaire verte ne suffit pas à déclarer un snapshot ou un backtest prêt
pour publication.

## Règles essentielles

- Pas de logique métier nouvelle dans `scripts/` lorsqu'elle peut vivre dans
  `src/alpharank/`.
- Pas de comparaison économique entre snapshots différents.
- Pas de réécriture silencieuse d'une donnée brute ou d'un résultat historique.
- Pas de chiffre de performance sans run, snapshot, période, convention
  d'exécution, frais et benchmark.
- Une tâche de roadmap par commit, avec l'identifiant de tâche dans le message.
- Les rapports datés et anciens journaux sont sous `docs/archive/`, pas dans le
  parcours d'onboarding.

## Statut de la remise en ordre

Le dépôt fonctionne mais son organisation a accumulé plusieurs générations de
code, données et sorties. La remise en ordre est suivie dans
[`ROADMAP.md`](ROADMAP.md). Aucun déplacement de code ou de données ne doit être
fait avant inventaire des lecteurs et preuve de parité.
