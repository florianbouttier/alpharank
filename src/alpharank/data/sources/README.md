# Adaptateurs fournisseurs

Responsabilité : une frontière explicite par fournisseur ou référentiel externe.

Entrées : requêtes typées, caches nommés et réponses fournisseur.

Sorties : observations sourcées avant arbitrage entre fournisseurs.

`sec_historical.py` reconstruit uniquement un candidat diagnostique à partir du
bridge SEC versionné. Il essaie Companyfacts, trace le fallback filing-level et
bloque explicitement toute promotion automatique.

`sec_filing_headers.py` acquiert les index de soumissions SEC et les en-têtes
SGML immuables des dépôts annuels. Il parse le CIK, le SIC et l'heure officielle
d'acceptation, conserve chaque réponse ou échec dans le RAW adressé par contenu
et ne décide jamais du secteur final. `sector_identity.py` borne séparément la
relation ticker/CIK dans le temps ; une identité non résolue reste inconnue.
Les ponts manuels nécessaires aux nouveaux symboles vivent dans
`../open_source/reference/sector_identity_overrides_v1.csv` avec date de début
et preuves SEC et S&P ; ils ne rendent jamais l'identité disponible avant
l'événement sourcé.

Dossiers enfants : aucun.

Interdit ici : publication, priorité DEF et simulation de portefeuille.
