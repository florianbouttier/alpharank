# Composant frontal d'analyse de stratégies

Responsabilité : rendu interactif partagé des séries temporelles et frontières,
indépendant de la méthode qui produit les données.

Entrées : courbes et points pré-calculés, libellés, couleurs et thème.

Sorties : graphiques Plotly autonomes, zoom, survol et export visuel. Le runtime
Plotly fourni par la dépendance Python existante est embarqué, jamais chargé par CDN.
Sa version est inscrite dans le HTML et sa notice MIT est embarquée avec le runtime.

Dossiers enfants : aucun.

Interdit ici : acquisition, sélection de titres, calcul de rendement, KPI ou
changement de données. Les appels propres au rapport restent dans ses adaptateurs.
