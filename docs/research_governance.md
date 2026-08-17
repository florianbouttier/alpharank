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

## Provenance runtime

Chaque nouveau run Legacy ou Boosting écrit un bloc `runtime_provenance` dans
son `data_input_manifest.json` et un artefact `runtime_git_patch.json` dans son
répertoire immuable. Le bloc enregistre la commande exacte, la configuration
résolue, les seeds, l'interpréteur et la plateforme, l'inventaire complet des
dépendances installées, les hashes du code critique et les identifiants de
données. Les valeurs dont le nom évoque un secret, mot de passe, token, clé API
ou credential sont remplacées par `<redacted>`.

La section Git contient le commit, la branche, l'état dirty réel, le hash de
l'état porcelain, le hash et la taille du patch suivi, ainsi que le nombre et le
hash d'inventaire des fichiers non suivis. Le bundle conserve le patch Git
binaire complet des fichiers suivis et les empreintes SHA-256 des fichiers non
suivis. Un run R&D sale est donc rejouable et attribuable sans embarquer le
contenu potentiellement sensible ou volumineux des fichiers non suivis.

`validate_runtime_provenance` échoue si un champ obligatoire ou l'artefact de
patch manque, si un hash diverge, ou si le manifeste déclare `git_dirty=false`
alors que le dépôt est sale. La validation contre le worktree courant est une
preuve de capture immédiate ; un replay ultérieur vérifie le manifeste et son
bundle sans exiger que le dépôt soit resté dans le même état.

## Mutations du futur

`tests/test_future_mutation_invariance.py` constitue le garde transversal
anti-look-ahead. Il modifie séparément une cible future, un prix futur, un
événement de membership futur, un reclassement sectoriel futur et un filing
futur, puis exige l'identité des décisions, features ou attributs antérieurs au
cutoff. Les attributs datés utilisent `join_point_in_time_attributes`, qui
retient dans la sortie le timestamp effectif sélectionné et refuse les versions
dupliquées. Ce garde est une condition nécessaire de promotion ; les tâches
UNI/FND restent responsables de brancher toutes les sources de production sur
ces contrats point-in-time.

## Appartenance à l'univers

Les événements S&P sont appliqués à leur instant effectif, par défaut minuit
`America/New_York` de `effective_date` lorsqu'aucune heure plus précise n'est
fournie. `membership_at_decision_time` exige une décision timezone-aware et
rejoue les opérations jusqu'à cet instant inclus. Les snapshots mensuels datés
du premier jour représentent l'univers utilisable à la décision de fin de ce
mois : un événement effectif au milieu du mois n'est donc plus décalé au mois
suivant. Le snapshot de base est lui aussi rapproché avec les événements de son
mois et chaque opération reste dans l'audit.

La clé canonique d'un snapshot est `(Date, Ticker)`. Les doublons sont résolus
par le nom normalisé le plus fréquent ; une égalité est tranchée
lexicographiquement. Chaque groupe dupliqué conserve le nombre de lignes, tous
les noms candidats et leurs fréquences, le nom retenu et l'identifiant de la
règle. Sur le fichier actif contrôlé le 2026-08-17, 1 067 groupes à date non
nulle sont ainsi audités et la sortie de 225 620 lignes ne contient plus aucune
clé dupliquée.

Chaque événement d'univers doit désormais fournir `event_id`, `source_url`,
`observed_at`, `effective_at`, `effective_date` et `confidence`. Le registre est
refusé avant toute décision si l'un de ces champs manque, si l'identifiant est
dupliqué, si les timestamps ne sont pas zonés ou si l'observation est postérieure
à l'effet. Lorsqu'une source officielle publie seulement une date sans heure, la
connaissance est placée conservativement à 23:59:59 `America/New_York`. Le journal
de reconstruction propage ces champs jusqu'à chaque opération appliquée.
