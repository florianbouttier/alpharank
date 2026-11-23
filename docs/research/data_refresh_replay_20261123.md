# Refresh et portefeuilles du 23 novembre 2026

**Rôle : preuve datée du refresh, des apprentissages et des contrôles ; aucun
résultat non exécuté n'est présenté comme acquis.**

## Demande et bornes

Le propriétaire demande un téléchargement frais, les apprentissages classiques
et tous les paniers pour un achat le mardi 24 novembre. Les prix de référence
sont ceux du vendredi 20 novembre. En méthode mensuelle classique, le signal
vient du dernier mois clôturé, octobre, pour les paniers de novembre. Un éventuel
scénario traitant le 20 novembre comme une fin de mois doit rester distinct.

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

## Préflight d'acquisition (`DATA-050`, non exécuté à ce stade)

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

## Exécution et verdict

Acquisition, composition, apprentissages, contrôles historiques et nouveaux
portefeuilles restent à exécuter. Aucun ancien panier n'est publié comme un
nouveau résultat et aucun ordre ne doit être envoyé par ce workflow.
