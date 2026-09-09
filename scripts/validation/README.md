# Validation

Responsabilité : contrôles sans mutation des sources observées.

Entrées : manifestes, snapshots et inventaires.

Sorties : diagnostics et code retour explicite.

`audit_refresh_replay.py` compare la trajectoire mûre au cutoff historique et,
avec `--latest-decision-month`, le dernier portefeuille déjà formé même si son
rendement n'est pas encore complet.

`build_refresh_replay_report.py` rend en HTML autonome l'audit baseline,
prix-seuls, SEC-seuls et candidat complet. Lorsque les quatre replays communs
sont disponibles, il exige une attribution additive des écarts de portefeuille
avant de qualifier le drift d'expliqué ; il ne publie aucune donnée.

`build_ticker_transition_replay_report.py` compare un replay baseline et un
replay candidat après continuité de ticker. Il lit les prix, prédictions,
holdings et résultats du moteur commun, puis écrit JSON, HTML autonome et
manifeste de hashes sans déplacer le pointeur de production.

Dossiers enfants : aucun.

Interdit ici : promotion, correction destructive et téléchargement.
