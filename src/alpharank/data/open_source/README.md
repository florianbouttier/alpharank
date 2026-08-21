# Compatibilité open source

Ce dossier conserve la façade publique `alpharank.data.open_source`, les trois
modules SEC dont les lecteurs historiques utilisent encore le chemin, deux
contrats de compatibilité Legacy et les petits registres de référence.

L'implémentation active est désormais séparée par responsabilité dans le parent :

- `../sources/` : adaptateurs Yahoo, SEC-only, SimFin et StockAnalysis ;
- `../ingestion/` : orchestration, transactions et stockage ;
- `../quality/` : diagnostics fournisseurs ;
- `../publishing/` : consolidation et publication.

Les modules d'étape ne dépendent jamais de l'orchestrateur. Aucun nouveau
provider, audit ou publisher n'est placé dans cette façade.

## Dossier enfant

- `reference/` : petits registres versionnés nécessaires au mapping et à
  l'identité historique.

Les valeurs fondamentales officielles sont SEC-only. Les autres fournisseurs
servent aux audits ou aux packages de recherche explicitement étiquetés.
