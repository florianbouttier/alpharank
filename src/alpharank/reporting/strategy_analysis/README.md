# Rapport d'analyse à dates libres

Responsabilité : préparer et servir en lecture seule le rapport React complet,
sans définir un deuxième moteur financier.

Entrées : rapport scellé explicite, manifeste du même snapshot et prix hashés,
valorisations quotidiennes du moteur `portfolio/daily/`.

Sorties : HTML autonome pour son état initial, datasets et manifeste immuables,
API locale de fenêtres libres déléguée au moteur Python commun.

Dossiers enfants : aucun. Le frontend appartient à `../analysis_ui/`.

Interdit ici : téléchargement, entraînement, sélection de titres, mutation des
entrées, publication tierce, exécution d'ordres et formules KPI locales.
