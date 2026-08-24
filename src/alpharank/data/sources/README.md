# Adaptateurs fournisseurs

Responsabilité : une frontière explicite par fournisseur ou référentiel externe.

Entrées : requêtes typées, caches nommés et réponses fournisseur.

Sorties : observations sourcées avant arbitrage entre fournisseurs.

`sec_historical.py` reconstruit uniquement un candidat diagnostique à partir du
bridge SEC versionné. Il essaie Companyfacts, trace le fallback filing-level et
bloque explicitement toute promotion automatique.

Dossiers enfants : aucun.

Interdit ici : publication, priorité DEF et simulation de portefeuille.
