# Documentation AlphaRank

**Rôle : sommaire canonique.** Ce fichier indique quel document fait autorité et
sépare volontairement les règles courantes des preuves historiques.

## Parcours humain recommandé

1. [`../README.md`](../README.md) : comprendre le projet en cinq minutes.
2. [`../ROADMAP.md`](../ROADMAP.md) : voir ce qui doit être rangé et dans quel
   ordre.
3. [`architecture/repository_map.md`](architecture/repository_map.md) : savoir
   où chercher code, données et résultats.
4. [`monthly_portfolio_runbook.md`](monthly_portfolio_runbook.md) : exécuter la
   production mensuelle Legacy.
5. [`legacy_boosting_methodology.md`](legacy_boosting_methodology.md) :
   comprendre les deux méthodes.
6. [`common_portfolio_backtest_engine.md`](common_portfolio_backtest_engine.md) :
   comprendre la simulation et les KPI communs.

## À la racine du dépôt

| Document | Rôle |
| --- | --- |
| [`README.md`](../README.md) | onboarding court |
| [`ROADMAP.md`](../ROADMAP.md) | remise en ordre du dépôt |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | normes de développement actives |
| [`METHODOLOGY_AUDIT_ROADMAP.md`](../METHODOLOGY_AUDIT_ROADMAP.md) | registre détaillé et historique des remédiations méthodologiques |
| [`AGENTS.md`](../AGENTS.md) | contraintes normatives pour les agents |

## Architecture

Le dossier [`architecture/`](architecture/) explique l'organisation sans
dupliquer les contrats métier :

- [`architecture/repository_map.md`](architecture/repository_map.md) : réalité
  actuelle, cible et responsabilités ;
- [`architecture/data_lifecycle.md`](architecture/data_lifecycle.md) : sens de
  `raw`, `stg`, `def`, `mart`, `snapshot` et `run`.
- [`architecture/data_location_inventory_v1.json`](architecture/data_location_inventory_v1.json) :
  carte machine-lisible des emplacements actuels et de leurs lecteurs.
- [`architecture/data_reader_migration_v1.json`](architecture/data_reader_migration_v1.json) :
  comparaison ancien/MART et décision de migration de chaque lecteur Legacy.
- [`architecture/legacy_data_archive_policy_v1.json`](architecture/legacy_data_archive_policy_v1.json) :
  gel, archivage par référence et procédure de retour arrière des anciennes racines.
- [`architecture/run_root_inventory_v1.json`](architecture/run_root_inventory_v1.json) :
  registre consultable des anciennes racines de résultats.
- [`architecture/run_retention_report_v1.json`](architecture/run_retention_report_v1.json) :
  espace dupliqué exact et proposition de rétention réversible.

## Standards de développement

[`standards/`](standards/) contient les règles détaillées applicables au nouveau
code et aux nouvelles données :

- [`standards/python.md`](standards/python.md) : style et qualité Python ;
- [`standards/data.md`](standards/data.md) : modélisation et ingénierie data ;
- [`standards/repository.md`](standards/repository.md) : organisation et
  dépendances entre dossiers ;
- [`standards/git.md`](standards/git.md) : une tâche de roadmap, un commit et ses
  preuves.

## Contrats et procédures courants

| Sujet | Source de vérité |
| --- | --- |
| Production mensuelle | [`monthly_portfolio_runbook.md`](monthly_portfolio_runbook.md) |
| Méthodes Legacy et Boosting | [`legacy_boosting_methodology.md`](legacy_boosting_methodology.md) |
| Simulation, KPI et comparaison | [`common_portfolio_backtest_engine.md`](common_portfolio_backtest_engine.md) |
| Gouvernance des résultats | [`research_governance.md`](research_governance.md) |
| Organisation des chemins de runs | [`run_organization.md`](run_organization.md) |
| Référence des features | [`backtest_feature_reference.md`](backtest_feature_reference.md) |
| Ingestion open source | [`open_source_ingestion_architecture.md`](open_source_ingestion_architecture.md) |
| Fondamentaux SEC | [`sec_fundamentals_contract.md`](sec_fundamentals_contract.md) |
| Robustesse et migrations SEC | [`sec_data_robustness_plan.md`](sec_data_robustness_plan.md) |
| État courant de la couverture SEC | [`sec_open_source_status.md`](sec_open_source_status.md) |

Les tests découvrent ces contrats par leur contenu normatif et ne verrouillent
plus leur emplacement. Ils restent directement sous `docs/` pour conserver les
liens humains et publics actuels ; un classement ultérieur devra inventorier
ces lecteurs dans une tâche distincte.

## Recherche

[`research/`](research/) contient les expériences reproductibles. La
[`synthèse Boosting`](research/boosting_signal_copy_model_catalog.md) expose les
conclusions durables et renvoie vers quatre journaux chronologiques datés. Les
entrées historiques restent intactes sans alourdir le parcours courant.

## Archives

[`archive/`](archive/) contient :

- les audits et rapports datés sous [`archive/reports/`](archive/reports/) ;
- l'ancien journal [`archive/CODEX_HANDOFF.md`](archive/CODEX_HANDOFF.md) ;
- les anciennes pages d'onboarding et instructions agents.

Un document archivé peut expliquer une décision passée, mais ne doit jamais être
utilisé seul pour annoncer l'état ou les performances courantes.

## Règles d'entretien

- Une seule source canonique par sujet.
- Toute modification significative respecte le standard Python, data ou dépôt
  correspondant ; la dette historique est traitée séparément.
- Une modification de workflow met à jour le contrat courant dans le même
  changement.
- Une expérience datée reste dans `research/` ou `archive/`, jamais mélangée à
  l'onboarding.
- Un chiffre de performance indique run, snapshot, période, calendrier, frais et
  benchmark.
- Chaque dossier maintenu garde un README local court ; les dossiers générés
  sont expliqués par leur parent.
- Après toute modification de structure ou de liens, exécuter
  `../scripts/validate_documentation.py`.
