# Orchestration d'ingestion

Responsabilité : transactions, schémas intermédiaires et étapes d'acquisition
multi-source.

Entrées : adaptateurs de `sources/`, politiques de refresh et chemins résolus.

Sorties : observations reçues, normalisées et conservées avec leur lignée.

`raw_archive.py` porte l'API d'archivage et les reçus ;
`raw_archive_frames.py` reconstruit et compare les événements RAW v1 en colonnes,
sans dictionnaire Python de toutes les lignes. Les hashes logiques et les
anciennes archives restent compatibles ; les fichiers historiques ne sont pas
migrés ni supprimés.

Dossiers enfants : aucun.

Interdit ici : décision DEF, calcul modèle et rendu de rapport.
