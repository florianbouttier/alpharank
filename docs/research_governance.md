# Gouvernance des résultats de recherche

Dernière mise à jour : 2026-08-17.

Ce document définit les règles approuvées pour conserver, comparer et promouvoir
les résultats Legacy et Boosting. Il complète les contrats méthodologiques et de
simulation sans transformer une baseline reproductible en preuve causale.

## Baseline `v1-audited-biased`

Les artefacts audités du 16 août 2026 sont figés comme témoin historique :

- Legacy : run `20260816_142810` ;
- Boosting : `outputs/production_refresh_20260816/boosting_latest_common_v3` ;
- comparaison : `outputs/production_refresh_20260816/common_replay_v3` ;
- dashboard : `outputs/research_dashboard/alpharank_common_20260816_pit_validated`.

Cette baseline conserve les entrées, configurations, prédictions, positions,
rendements et rapports qui ont produit les métriques auditées. Son statut est
`audited_biased_not_causal`. Elle ne doit jamais être présentée comme validation
du Boosting ni être modifiée en place.

Le contrat est implémenté dans `src/alpharank/governance.py` et le package est
créé avec `scripts/seal_methodology_baseline.py`. Chaque fichier
du payload possède une taille et un SHA-256 dans `baseline_manifest.json`. Le
SHA-256 du manifeste est conservé séparément dans
`baseline_manifest.sha256`. Les fichiers et répertoires perdent tous leurs bits
d'écriture après renommage atomique du package temporaire.

## Décisions approuvées

- Les structures, mois, tickers, rangs et décisions doivent être strictement
  identiques pour une migration déclarée neutre.
- La tolérance maximale sur les calculs numériques est `1e-12`; les fichiers
  seulement copiés doivent être identiques par SHA-256.
- Toute différence économique crée une nouvelle version et un rapprochement
  mois par mois. Elle ne réécrit jamais `v1`.
- Un run R&D sale reste autorisé si son patch est capturé. Une promotion finale
  exige un commit propre.
- Commit, état Git, diff, code critique, commande, configuration, seeds,
  interpréteur, dépendances, données et modèles appartiennent à la provenance
  obligatoire. Les secrets en sont exclus.
- Toute dérogation doit porter une approbation humaine explicite dans le
  manifeste.

## Validation

`validate_baseline_package` recalcule l'inventaire et échoue si un fichier a été
ajouté, supprimé, modifié ou rendu inscriptible. La baseline n'est valide que si
le manifeste, son sceau détaché et l'intégralité du payload concordent.

## Garde du préfixe économique

`scripts/validate_economic_prefix.py` compare une référence publiée et un
candidat de migration. Le dernier mois de la référence définit le préfixe ; les
nouveaux mois du candidat restent hors comparaison. Les clés
stratégie/décision/détention/ticker, les rangs et les champs de décision sont
exacts. Les poids, rendements, turnover et coûts utilisent la tolérance absolue
approuvée de `1e-12`, obligatoirement accompagnée de sa justification.

Le rapport contient les SHA-256 canoniques, les clés manquantes ou inattendues et
l'écart maximal de chaque colonne. Une différence sur un mois publié interdit de
qualifier la migration de neutre ; elle doit être traitée comme correction
économique et produire une nouvelle version.
