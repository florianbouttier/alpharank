# Rapport d'analyse à dates libres

Responsabilité : préparer et servir en lecture seule le rapport React complet,
sans définir un deuxième moteur financier.

Entrées : rapport scellé explicite, manifeste du même snapshot et prix hashés,
valorisations quotidiennes du moteur `portfolio/daily/`.

Sorties : HTML autonome pour son état initial, datasets et manifeste immuables,
API locale de fenêtres libres déléguée au moteur Python commun.

`versions.py` décrit l'identité économique et rapproche les archives à dates
identiques ; `version_catalog.py` capture les versions explicitement choisies
et valide leurs chemins, hashes et métadonnées. Une refonte UI ou un suivi prix
ne crée pas un nouveau run économique. Le catalogue est servi depuis sa racine.

Le diagnostic joint les séances manquantes capturées aux positions scellées.
La source `monthly_reference` constitue un repli explicitement demandé, limité
aux mois calendaires complets. Elle ne remplace pas le journalier et ne résout
pas les anciennes hypothèses de valorisation terminale.

La CLI peut déclarer `--valuation-policy last_observed_mark_v1` pour la seule
version par défaut. Cette convention entre dans son identité économique ; les
archives gardent leur politique. Les cours conservés et rapprochements sont
capturés et hashés ; aucune source mutable ni formule financière de reporting
ne remplace le moteur commun. Le scénario n'est ni une vente au dernier cours,
ni une correction des contreparties terminales, ni un nouveau signal.

Dossiers enfants : aucun. Le frontend appartient à `../analysis_ui/`.

Interdit ici : téléchargement, entraînement, sélection de titres, mutation des
entrées, publication tierce, exécution d'ordres et formules KPI locales.
