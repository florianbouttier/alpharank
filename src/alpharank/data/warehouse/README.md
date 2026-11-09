# Entrepôt data

Responsabilité : contrats de chemins et transformations canoniques RAW, STG,
DEF et MART.

Entrées : reçus fournisseurs, observations normalisées et règles de priorité.

Sorties : tables gouvernées et entrée modèle MART validée. La release DEF
`sector_history_<hash>/` conserve les classifications sectorielles au premier
instant réellement observé, leurs rejets et leurs hashes RAW.

`sector_sic_history.py` transforme les SIC observés dans les en-têtes SEC avec
la taxonomie versionnée, puis `sector_history.py` matérialise la release DEF
point-in-time. Une release sectorielle v2 embarque les observations SEC, la
carte d'identité, les identités non résolues et les échecs comme preuves
Parquet hashées ; aucun de ces éléments ne promeut un snapshot modèle.

Dossiers enfants : aucun.

Interdit ici : connecteur fournisseur, logique de modèle et rendu.
