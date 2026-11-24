# Refresh et portefeuilles du 23 novembre 2026

**Rôle : preuve datée du refresh, des apprentissages et des contrôles ; aucun
résultat non exécuté n'est présenté comme acquis.**

## Demande et bornes

Le propriétaire demande un téléchargement frais, les apprentissages classiques
et tous les paniers pour un achat le mardi 24 novembre. Les prix de référence
sont ceux du vendredi 20 novembre. En méthode mensuelle classique, le signal
vient du dernier mois clôturé, octobre, pour les paniers de novembre. Un éventuel
scénario traitant le 20 novembre comme une fin de mois doit rester distinct.
Le propriétaire confirme le 24 novembre les paniers mensuels classiques :
aucun scénario de fausse fin de mois n'est demandé.

Le dernier HTML `REPORT-032` n'était pas un nouvel apprentissage : ses 126
positions provenaient de la décision de septembre pour octobre ; leur suivi
prix atteignait le 20 novembre. Le pointeur de production référence encore la
composition `9a2058c9…425ad` du 20 août, alors que ce rapport utilise le candidat
`e0e6045a…ef95a` du 31 octobre. Ces deux références doivent rester distinguées
dans l'audit, sans promotion implicite de l'une ou de l'autre.

## Univers officiel de novembre (`DATA-049`)

L'[annonce primaire S&P du 30 octobre](https://press.spglobal.com/2026-10-30-Okta-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600)
annonce OKTA en remplacement de KVUE avant l'ouverture du 4 novembre. Le registre
reçoit uniquement cet événement additionnel ; les événements déjà enregistrés
restent identiques. La date de connaissance conservatrice est le 30 octobre
à 23:59:59 America/New_York (UTC−4), l'effet le 4 novembre à 00:00 (UTC−5).
Cette date d'effet n'est ni un prix ni une contrepartie actionnaire KVUE.

La régression exige que KVUE reste membre aux décisions du 30 octobre et du
3 novembre, puis qu'OKTA le remplace au 20 novembre ; la reconstruction mensuelle
conserve octobre et ajoute novembre. L'heure d'effet du registre est
conservatrice à l'intérieur de la date officielle ; elle ne prétend pas dater
une transaction de bourse.

Validation : huit tests d'ingestion/membership verts (la nouvelle régression
échouait avant l'ajout), 27 configurations validées. Les 228 559 lignes jusqu'à
octobre sont exactement égales avant/après, les anciens événements sont
inchangés et novembre conserve 503 membres. Registre SHA-256
`e5db75f0bde24085f4d09aea722d5f01a9390a0c00e7ddbb13cf58df59d5c095`.
Audit préalable en lecture seule sous
`outputs/data_refresh_replay_20261123/constituents/` ; le fichier de membership
actif n'a pas été modifié pendant cette validation.

## Préflight d'acquisition (`DATA-050`)

- Worktree initial propre, branche `main`, commit `d566c949…7fae0`.
- Espace libre observé : environ 2,1 Gio ; aucun fichier utilisateur supprimé.
- Le verrou `20261122_011505` référence le PID 62657, absent ; il ne représente
  pas une acquisition actuellement en cours.
- La transaction interrompue conserve exactement les mêmes 140 enfants de
  `history/output` qu'à son ouverture : aucun ajout tiers n'est exposé au
  rollback à cet instant. À revérifier avant la reprise automatique.
- Onze des treize fichiers raw sont identiques aux sauvegardes ; seules deux
  tables de référence générale, neuf lignes, ont été modifiées. Aucun prix ou
  fondamental frais du 22 novembre n'est revendiqué ; cause d'arrêt inconnue.
- Le lot comparable du 31 octobre a acquis les sources en environ 51 minutes.
  Ses nouveaux fichiers, hors clones COW, représentent environ 518 Mio. Une
  enveloppe de 1–1,5 Gio paraît plausible, sans garantie du pic de temporaires
  ou du système : la marge doit être surveillée avant chaque phase.

## Première tentative d'acquisition, incomplète

Le registre a été committé et poussé avant l'acquisition (`DATA-049`). La
reconstruction mensuelle réelle `20261123_011622` conserve les 228 559 lignes
antérieures et ajoute les 503 membres de novembre. L'ancien CSV a été figé
par clone APFS dans `outputs/data_refresh_replay_20261123/frozen_reference/`.

La commande normale d'ingestion a démarré le 23 novembre, run
`20261123_001623`, avec `--start-date 2005-01-01 --end-date 2026-11-21` : la
borne fournisseur exclusive conserve comme dernière séance attendue le
20 novembre. La résolution des références et leurs hashes a été capturée avant
le lancement dans `baseline_resolution.json`, sous le même dossier de refresh.
Le journal est `logs/data_refresh/20261123_acquisition.log`.

Après le lancement, la marge disque système est descendue vers 370 Mio et la
mémoire virtuelle utilisait environ 2,8 Gio. Un échantillonnage natif du
processus d'ingestion a mesuré une empreinte de 9,3 Gio pendant un tri Python ;
le journal est resté arrêté à 01:18:01. Le 24 novembre, le PID n'existe plus et
aucun résultat final n'est présent : la cause exacte de sa terminaison n'est
pas établie. Seules les petites tables de référence du run sont persistées ;
aucun téléchargement complet ni nouvel apprentissage n'est revendiqué.

## Correction mémoire préalable (`DATA-051`)

L'archive RAW v1 reconstruisait toutes les valeurs de toutes les versions en
dictionnaires Python avant chaque acquisition. Le correctif garde l'état et
les empreintes en colonnes, avec un index SHA-256 binaire sans perte. Les
recherches de références construisent leur table sur les clés demandées, pas
sur toutes les valeurs historiques. Seuls les payloads nécessaires à l'état
demandé sont retenus ; chaque Parquet visité est néanmoins lu séparément.

Une première validation grandeur nature a aussi exposé la rétention des blocs
par les vues de chaînes Arrow. Une seule empreinte de 64 octets retenait un
bloc de 64 000 octets sur la petite régression. Une matérialisation colonnaire
compacte détache maintenant l'état et les valeurs sélectionnées des anciens
blocs, sans modifier leur contenu. La régression échoue lorsque cette
compaction est désactivée. Aucun ancien fichier RAW n'est réécrit ni supprimé.

Le contrat v1 reste inchangé : hashes JSON logiques identiques, événements et
restaurations non dupliqués, validation des fichiers, snapshots parents,
cycles et références. Deux tests supplémentaires conservent la distinction
entre un événement `missing` et un hash absent sur un autre événement.
Les premiers tests de non-régression échouaient bien avant les corrections.
Le croisement des deux writers et des deux lecteurs sur cinq vintages donne
les mêmes données, événements et manifestes logiques ; les fixtures conservent
aussi l'ordre historique des colonnes d'un résultat vide.

Validation finale : 22 tests RAW/transaction et 60 tests ciblés distincts
d'ingestion, d'archive et de qualité verts ; Ruff ciblé et différentiel sans
nouveau diagnostic ; documentation et liens verts. Le contrôle de tailles
global conserve huit régressions antérieures hors périmètre, aucune sur le
correctif ; sa baseline n'est pas réinitialisée.

Le probe final en lecture seule valide les **73 vintages** et reconstruit le
dernier état de **2 781 482 lignes / 506 tickers** du vrai run RAW
`20261109_011506`. Chaque hash de fichier
et de snapshot parent est vérifié ; le pointeur reste inchangé et le digest
logique demeure `5eeed64e7cc0d719a81790a5e8a2abf61adde8b21397aed4e637ea6903dc9d93`.
Exécution terminée avec code 0 en **385,027 secondes**, pic RSS macOS de
**4 154 802 176 octets**. Cette mesure RSS n'est pas directement interchangeable
avec l'empreinte native de 9,3 Gio échantillonnée sur la première tentative ;
ce n'est ni une mesure avant/après stricte, ni le pic garanti du pipeline entier.

Preuves ignorées par Git, sous la racine de travail datée :

- `raw_archive_memory_probe.py`, exécuté avec `POLARS_MAX_THREADS=2` ;
- `logs/data_refresh/20261124_raw_archive_probe_final.log`, SHA-256
  `035506ebce17a2592f1e0372b1f36e8f31ff814090fa216f31b6dad218846ee6` ;
- `DATA-051_junit.xml`, SHA-256
  `dc54ae65a3f7e4f694977d297b1bcffed8c2548e2d6d09dcd59d4c344a203cca` ;
- `DATA-051_size_check.json` et `DATA-051_ruff_report.json`.

Le disque, à nouveau à 3,8 Gio au début de la reprise du 24 novembre, est
retombé vers **350 Mio** avec la réservation de swap. Aucun fichier utilisateur
n'est supprimé. Cette marge ne suffit pas aux nouveaux fichiers d'acquisition
et aux apprentissages ; une libération d'au moins 3 Gio est demandée. Le
correctif est validé, mais la reprise réseau reste à effectuer.

Les références figées sont encore intactes : pointeur de production SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`
et manifeste du candidat d'octobre SHA-256
`e2d9381fc08a0f5510df508e0a9a24b4e6af24b27980cfef04da3f4f671b1d58`.
La transaction interrompue conserve ses mêmes 140 enfants historiques,
zéro ajout et zéro retrait ; aucune récupération manuelle n'a été forcée.

## Exécution et verdict

Acquisition complète à reprendre ; composition, apprentissages, contrôles
historiques et nouveaux portefeuilles restent à exécuter. Aucun ancien panier n'est publié comme un
nouveau résultat et aucun ordre ne doit être envoyé par ce workflow.
