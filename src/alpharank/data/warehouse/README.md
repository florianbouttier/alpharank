# Entrepôt data

Responsabilité : contrats de chemins et transformations canoniques RAW, STG,
DEF et MART.

Entrées : reçus fournisseurs, observations normalisées et règles de priorité.

Sorties : tables gouvernées et entrée modèle MART validée. La release DEF
`sector_history_<hash>/` conserve les classifications sectorielles au premier
instant réellement observé, leurs rejets et leurs hashes RAW.

Dossiers enfants : aucun.

Interdit ici : connecteur fournisseur, logique de modèle et rendu.
