# Ingestion open source

Implémentation réutilisable de l'ingestion : Yahoo, SEC, SimFin et
StockAnalysis, consolidation, qualité, fraîcheur, transactions atomiques et
publication.

Le pipeline actif est séparé par responsabilité :

- `ingestion.py` orchestre les transactions et conserve la façade historique ;
- `ingestion_frames.py` possède les schémas et normalisations communes ;
- `ingestion_prices.py` possède acquisition, conservation et validation prix ;
- `ingestion_reference.py` possède référentiel, résultats et fondamentaux.

Les modules d'étape ne dépendent jamais de l'orchestrateur.

## Dossier enfant

- `reference/` : petits registres versionnés nécessaires au mapping et à
  l'identité historique.

Les valeurs fondamentales officielles sont SEC-only. Les autres fournisseurs
servent aux audits ou aux packages de recherche explicitement étiquetés.
