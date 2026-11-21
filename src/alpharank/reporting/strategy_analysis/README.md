# Rapport d'analyse à dates libres

Responsabilité : préparer et servir en lecture seule le rapport React complet,
sans définir un deuxième moteur financier.

Entrées : rapport scellé explicite, manifeste du même snapshot et prix hashés,
valorisations quotidiennes du moteur `portfolio/daily/`.

Sorties : HTML autonome pour son état initial, datasets et manifeste immuables,
API locale de fenêtres libres déléguée au moteur Python commun.

Le diagnostic joint les séances manquantes capturées aux positions scellées.
La source `monthly_reference` constitue un repli explicitement demandé, limité
aux mois calendaires complets. Elle ne remplace pas le journalier et ne résout
pas les anciennes hypothèses de valorisation terminale.

Dossiers enfants : aucun. Le frontend appartient à `../analysis_ui/`.

Interdit ici : téléchargement, entraînement, sélection de titres, mutation des
entrées, publication tierce, exécution d'ordres et formules KPI locales.
