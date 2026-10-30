# Refonte du dashboard de performance du 30 octobre 2026

## Statut

Cette preuve clôt `REPORT-015`. Elle remplace le renderer du rapport autonome
par un dashboard AlphaRank plus dense et responsive sans modifier le payload,
les formules, les rendements, les portefeuilles sources ou le statut des
stratégies. Le replay reste un candidat de recherche non promu.

## Entrées explicites

- replay commun candidat :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/common_end_2026_09_trend_clean` ;
- run Legacy candidat :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/legacy/2026-10-24/runs/20261024_172912` ;
- manifeste de snapshot :
  `outputs/data_refresh_replay_20261024/replay_head5545e2e/candidate/legacy/2026-10-24/runs/20261024_172912/input_snapshot/snapshot_manifest.json` ;
- preuve de marché :
  `data/open_source/official/runs/20261024_140104/price_return_extension_audit.parquet` ;
- composition immuable :
  `c340a9b455095722b539352d364094cf4a9e238e40cf7733a31b8fca7a528b7e`.

Le calendrier réalisé contient 182 mois, d'août 2011 à septembre 2026. Le
portefeuille courant reste séparé : 126 lignes, décision septembre 2026,
détention octobre 2026, dernière séance observée le 23 octobre et rendement du
mois absent des KPI.

## Contrat de complétude

Le dashboard conserve les 11 séries, les 33 KPI, les trois modes stratégies,
portefeuille composé et frontières efficientes, ainsi que les trois vues
performance, drawdown et richesse relative au SPY. Il conserve aussi les deux
model cards, les corrélations, les 168 compositions de frontière, les trois
profils descriptifs, les 24 650 lignes de positions historiques, le panier en
vigueur, les exports, les pseudo-codes et tous les hashes de lignée.

La refonte ajoute un résumé de couverture alimenté par le payload, une
navigation analyse/audit, des tokens visuels cohérents avec AlphaRank, un thème
clair/sombre et une réorganisation mobile. Elle n'ajoute aucun appel réseau et
ne déplace aucun calcul financier dans le navigateur.

## Parité économique

Le payload compressé du rapport `strategy_frontiers_20261025` a été décodé et
comparé au nouveau payload après retrait du seul champ de génération
`generated_at_utc`. Les deux objets JSON sont exactement égaux. Les contrôles
confirment 11 stratégies, 33 métriques, 182 mois, 24 650 holdings historiques,
126 holdings courants et le même `composition_id`.

## Artefacts et QA

- rapport autonome :
  `outputs/performance_reports/dashboard_refactor_20261030/backtest_performance_report.html` ;
- SHA-256 du rapport :
  `e2f07d398ca7dbe9e2e1d5c9985d781776ba08793ea18c957ec78d320ca72aba` ;
- manifeste :
  `outputs/performance_reports/dashboard_refactor_20261030/backtest_performance_report_manifest.json` ;
- SHA-256 du manifeste :
  `3bd86a28fc76af62bd9ef220b22b032dc38c335ab4d542ab62ecfad8c5acd3fa`.

Le test reporting ciblé passe avec 3 tests et le package reporting avec 19
tests. Ruff, le contrôle de format et la
syntaxe des 15 909 100 caractères JavaScript extraits du rapport réel passent.
La QA navigateur a servi l'artefact par HTTP local et validé : chargement sans
erreur console, compteurs de couverture, absence d'identifiant dupliqué,
bascule stratégies/composé/frontières, richesse relative, six cartes KPI,
matrice de corrélation, trois recommandations, 15 lignes multi-horizon et
thème sombre.

À 390 × 844, aucune section n'est masquée, les cinq blocs de couverture sont
présents, les trois recommandations de frontière restent accessibles et le
document ne produit aucun débordement horizontal global. À 1 139 × 863, le
rail, le studio, les filtres, les cartes et le graphique restent simultanément
lisibles. Le rapport généré reste ignoré par Git ; cette preuve et le code du
générateur assurent sa reproductibilité.

Le gate de taille ne relève aucune régression de `REPORT-015`. Il reste rouge
sur deux régressions hors diff : `_sec_explorer_html.py` et
`build_sec_output_package.py`. Elles ne sont ni modifiées ni masquées ici.
