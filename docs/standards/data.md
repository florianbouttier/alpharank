# Standard data AlphaRank

**Rôle : norme de modélisation, stockage, transformation, lignée et publication.**

Ce document précise le contrat présenté dans
[`../architecture/data_lifecycle.md`](../architecture/data_lifecycle.md). Il
s'applique aux prix, fondamentaux, univers, corporate actions, features,
prédictions, portefeuilles et KPI.

## 1. Principes obligatoires

1. Une table possède un grain déclaré et une clé contrôlée.
2. Une valeur conserve sa source, son instant de connaissance et sa règle de
   sélection.
3. Une donnée brute est append-only ; une correction crée une nouvelle version.
4. Une couche ne prend pas une décision réservée à la couche suivante.
5. `null`, zéro, non applicable, confidentiel, non publié et non encore mature
   sont des états différents.
6. Toute jointure temporelle est causale et explicitement orientée.
7. Toute publication est immuable, hashée et reproductible.
8. Un pointeur `latest` ne contient pas de données ; il désigne atomiquement une
   release immuable.
9. Un dataset n'est pas « bon » parce qu'il a beaucoup de lignes : clés,
   couverture, révisions et usages doivent être contrôlés.
10. Une absence de donnée ne déclenche jamais un repli silencieux.

## 2. Couches et responsabilités

### `raw` — observation fournisseur

Autorisé : payload original, ligne source, reçu de téléchargement, métadonnées
HTTP/API, hash et version fournisseur.

Interdit : renommage métier irréversible, arbitrage entre sources, forward-fill,
calcul de KPI ou correction en place.

### `stg` — forme commune

Autorisé : typage, renommage, conversion d'unité, normalisation d'identifiant,
détection de doublons et contrôles de format.

Interdit : choisir Yahoo plutôt qu'EODHD, sélectionner une révision SEC,
remplacer une absence par zéro ou publier un dataset modèle.

### `def` — décision gouvernée

Autorisé : choisir une observation selon une règle versionnée, appliquer une
correction sourcée, résoudre un conflit, exprimer un statut de disponibilité.

Interdit : feature spécifique à un modèle, allocation, score ou KPI de
performance.

### `mart` — vue consommateur

Autorisé : jointures causales de valeurs définitives, features, agrégats et
tables adaptées à Legacy, Boosting, simulation ou reporting.

Interdit : téléchargement, préférence fournisseur non documentée ou dépendance à
un dossier de run mutable.

### `snapshot` — release

Un snapshot fige les marts et leurs preuves. Il n'est pas une couche de calcul.

### `run` — consommation

Un run résout un snapshot une fois au démarrage et produit des artefacts sous
`outputs/`. Il ne modifie pas son entrée.

## 3. Nommage des datasets et fichiers

- Dossiers, datasets, tables et colonnes en `lower_snake_case`.
- Nom basé sur le contenu métier, pas sur l'étape manuelle :
  `daily_security_prices`, pas `prices_final_fixed`.
- Le fournisseur apparaît dans `raw` : `raw/yahoo/daily_prices`.
- À partir de `stg`, le nom décrit le concept commun, et la source reste une
  colonne de lignée.
- Les mots `new`, `old`, `final`, `copy`, `test2`, `latest_data` sont interdits.
- Une version contractuelle utilise un identifiant explicite :
  `schema_version: 2` ou `price_selection_policy_v2`.
- Une date dans un identifiant utilise `YYYYMMDD` ou `YYYYMMDD_HHMMSS` en UTC
  selon le besoin.
- `latest.json` est autorisé uniquement comme petit pointeur atomique vers un
  identifiant immuable.
- Une extension correspond au format réel ; pas de CSV renommé en `.txt`.

## 4. Grain, clés et unicité

Tout contrat de dataset déclare :

- le grain en une phrase ;
- la clé métier ;
- la clé technique éventuelle ;
- les dimensions temporelles ;
- les colonnes obligatoires et optionnelles ;
- l'unité et la devise ;
- les règles de nullité ;
- la politique de révision ;
- le propriétaire et les consommateurs.

Exemples de grains acceptables :

- « une observation fournisseur par titre, séance et vintage de téléchargement » ;
- « une valeur fondamentale par société, concept, période fiscale et filing » ;
- « une valeur définitive par titre, concept et date de connaissance » ;
- « une position par stratégie, décision mensuelle et titre ».

`drop_duplicates` n'est jamais une règle d'unicité. En cas de doublon, la table
explique s'il s'agit d'une révision, d'un conflit, d'un doublon technique ou
d'une violation.

## 5. Identifiants de titres et sociétés

- `ticker` est un symbole observé sur une période, pas une identité durable.
- `security_id` identifie un instrument économique à travers les changements de
  symbole lorsqu'une preuve existe.
- `issuer_id` identifie la société émettrice ; il peut différer de
  `security_id`.
- Les identifiants fournisseurs restent dans des colonnes nommées : `cik`,
  `eodhd_code`, `yahoo_symbol`, etc.
- Une table de correspondance est historisée avec `valid_from`, `valid_to`,
  source et confiance.
- Une réutilisation de ticker n'est jamais fusionnée automatiquement avec
  l'ancien titre.
- Une corporate action terminale ou une succession conserve l'ancien et le
  nouveau titre, le type d'événement et la contrepartie sourcée.

## 6. Colonnes temporelles

Le mot `date` seul est insuffisant dans un nouveau schéma. Utiliser :

| Colonne | Sens |
| --- | --- |
| `event_date` | date économique de l'événement |
| `period_start` / `period_end` | période couverte par la valeur |
| `filing_date` | date du dépôt réglementaire |
| `available_at` | premier instant où AlphaRank pouvait connaître la valeur |
| `retrieved_at` | instant où le fournisseur a été interrogé |
| `ingested_at` | instant d'entrée dans la plateforme |
| `valid_from` / `valid_to` | période de validité de l'identité ou règle |
| `decision_month` | mois où le signal est décidé |
| `holding_month` | mois de rendement réalisé |
| `published_at` | instant de publication du snapshot ou rapport |

Règles :

- timestamps techniques stockés en UTC ;
- dates de marché accompagnées du calendrier/venue lorsque nécessaire ;
- `available_at` ne peut pas précéder la preuve de disponibilité publique ;
- `retrieved_at` ne remplace jamais `filing_date` ou `available_at` ;
- toute période utilise des bornes clairement inclusives ou exclusives ;
- une table mensuelle conserve la date de décision réelle, pas seulement un
  label `YYYY-MM` ambigu.

## 7. Temps de connaissance et point-in-time

Pour une décision à `t`, seules les observations dont `available_at <= t` sont
éligibles. La dernière valeur connue est choisie après ce filtre, jamais avant.

Sont interdits :

- utiliser la dernière révision actuelle pour reconstruire une décision passée
  sans conserver la version historique ;
- filtrer des candidats selon la disponibilité de leur rendement réalisé ;
- utiliser la composition actuelle de l'indice sur toute l'histoire ;
- connaître un delisting avant son annonce publique ;
- compléter une feature passée avec une donnée publiée plus tard.

Un test de causalité modifie les données futures et prouve que les décisions
antérieures ne changent pas.

## 8. Valeurs manquantes et statuts

Une valeur numérique nullable possède, lorsque le sens métier l'exige, un statut
séparé. Vocabulaire recommandé :

| Statut | Sens |
| --- | --- |
| `observed` | valeur réellement observée |
| `derived` | valeur calculée depuis des observations identifiées |
| `carried_forward` | ancienne valeur reprise selon une règle autorisée |
| `not_reported` | aucune valeur publiée trouvée |
| `not_applicable` | concept sans sens pour cette ligne |
| `confidential` | valeur masquée ou non publiable |
| `source_unavailable` | fournisseur indisponible |
| `horizon_pending` | horizon futur pas encore arrivé |
| `terminal_event` | absence expliquée par fin de cotation/événement |
| `quarantined` | valeur conservée mais interdite à la production |

Règles :

- zéro est une valeur observée, jamais le remplacement générique d'un null ;
- `NaN` et infini sont refusés aux frontières publiées sauf contrat explicite ;
- un forward-fill indique source, valeur originale, âge et règle ;
- aucun report entre deux titres ou deux dates métier différentes sans règle
  sourcée ;
- les suppressions de lignes sont comptées par motif ;
- un modèle peut imputer selon une politique versionnée, mais la donnée source
  reste manquante dans les couches précédentes.

## 9. Contrat RAW

Chaque acquisition crée un reçu, même si le contenu est identique à un payload
déjà stocké. Champs minimaux :

- `source_name` et `dataset_name` ;
- `request_id` ou requête normalisée ;
- `retrieved_at` ;
- statut de réponse et erreur éventuelle ;
- `payload_sha256` ;
- taille et format ;
- plage ou univers demandé ;
- référence vers le payload ;
- version de l'ingesteur.

Si deux acquisitions ont le même hash, les deux reçus peuvent référencer le
même objet physique. Si une seule ligne change, le nouveau payload ou delta est
conservé avec son propre hash. Aucun ancien payload n'est réécrit.

EODHD est une preuve brute durable. Une correction corporate action se place
dans un overlay versionné ; elle ne modifie pas l'archive figée.

## 10. Contrat STG

Chaque table staging conserve :

- clé de la ligne raw ou `payload_sha256` ;
- `source_name` ;
- identifiants bruts et normalisés ;
- timestamps source et ingestion ;
- version du schéma staging ;
- erreurs de parsing séparées des lignes valides.

Une conversion d'unité indique valeur brute, unité brute, valeur normalisée et
unité normalisée lorsqu'une perte d'information est possible.

## 11. Contrat DEF

Chaque ligne définitive expose :

- clé métier et grain ;
- valeur retenue et statut ;
- `source_name`, clé raw et hash du payload ;
- `available_at` ;
- `selection_policy_id` et version ;
- `selected_at` ;
- motif de sélection ou de rejet des alternatives ;
- identifiant de correction éventuelle ;
- niveau de confiance ou état de revue lorsqu'il existe.

Les observations candidates rejetées restent accessibles. `def` ne supprime pas
la preuve du conflit.

## 12. Contrat MART

Un mart possède :

- `mart_id` et `schema_version` ;
- consommateur et objectif ;
- grain et clé ;
- période demandée et effective ;
- sources `def` et hashes ;
- configuration de feature/jointure ;
- statistiques de qualité ;
- timestamp de build ;
- code commit et environnement de calcul ;
- statut `candidate`, `validated`, `published` ou `quarantined`.

Legacy et Boosting peuvent avoir des marts différents. Une comparaison publique
doit prouver qu'ils proviennent de la même composition économique ou utiliser un
mart commun explicitement déclaré.

## 13. Jointures, agrégations et remplissages

- Déclarer cardinalité attendue : `1:1`, `1:n`, `n:1` ou `n:n` justifié.
- Une jointure `n:n` nouvelle est interdite sans table de pont et test du nombre
  de lignes.
- Avant/après chaque jointure : lignes, clés, nulls introduits et lignes non
  appariées par côté.
- Une agrégation liste explicitement clés et fonction de chaque colonne.
- Aucune colonne non agrégée ne survit par ordre accidentel.
- Les fenêtres sont précédées d'un tri stable et partitionnées par identité
  durable.
- Un `asof` indique direction et tolérance ; la direction `forward` est interdite
  pour une feature causale sauf événement explicitement futur.
- Un forward-fill possède `max_age`, partition, colonnes autorisées et indicateur
  de provenance.
- Une ligne perdue est comptée et expliquée ; `inner join` n'est pas utilisé
  uniquement pour faire disparaître les cas difficiles.

## 14. Prix et rendements

- Conserver OHLCV brut tel que reçu et `adjusted_close` séparément.
- Déclarer fournisseur, venue, devise, calendrier et type d'ajustement.
- Un split corrige l'historique via un overlay sourcé ; un dividende ne réécrit
  pas les OHLC bruts.
- `close`, `adjusted_close`, prochaine ouverture et VWAP ne sont jamais
  interchangeables.
- La date d'achat, la date de vente et la durée de détention sont explicites.
- Les rendements sont recalculés depuis les prix de la même vintage lorsque le
  contrat l'exige.
- Un prix absent au terme d'une position suit une politique terminale sourcée ;
  il ne fait pas remplacer le titre par un candidat moins bien classé.

## 15. Formats et partitionnement

- Parquet est le format tabulaire de référence pour les datasets volumineux et
  typés.
- JSON est réservé aux manifestes, configurations et petits registres.
- CSV est autorisé pour échange humain ou source fournisseur ; il n'est pas le
  format canonique d'un mart volumineux.
- UTF-8, séparateur et représentation du null explicites pour tout CSV.
- Compression Parquet standardisée par dataset ; un changement est versionné.
- Partitionner par colonnes peu cardinales et fréquemment filtrées, typiquement
  source, dataset, année ou vintage.
- Partitionner par ticker est interdit par défaut : cela crée trop de petits
  fichiers.
- Cible indicative d'un fichier Parquet : 64 à 512 Mo. Un dataset plus petit
  reste dans un seul fichier si cela améliore la lisibilité.
- L'ordre de lignes ne fait pas partie du contrat sauf déclaration explicite ;
  trier avant hash logique ou comparaison.

## 16. Schémas et versions

- Chaque dataset publiable possède un contrat versionné et lisible par machine.
- Changement compatible : ajout nullable ou métadonnée sans changement de sens.
- Changement incompatible : clé, grain, type, unité, sémantique temporelle ou
  suppression de colonne ; incrément majeur obligatoire.
- Une migration indique source, cible, transformateur, validation et possibilité
  de retour arrière.
- Aucun consommateur n'interprète une version inconnue comme la dernière connue.
- Un renommage conserve une table de correspondance pendant la migration.

La cible d'organisation est un contrat déclaratif sous
`configs/data_contracts/` et ses validateurs sous
`src/alpharank/data/contracts/`. Leur création est une tâche de code/config
distincte ; ce document ne prétend pas qu'ils existent déjà.

## 17. Hashes et lignée

- SHA-256 pour les fichiers, payloads et compositions publiées.
- Distinguer hash physique du fichier et hash logique d'une table normalisée.
- Un manifeste liste chaque entrée, chemin relatif, taille, hash, schéma et rôle.
- Les chemins absolus utilisateur sont exclus du contenu reproductible.
- Le code commit seul ne suffit pas : conserver aussi les hashes des fichiers
  critiques lorsque le worktree n'est pas propre.
- Une copie, clone copy-on-write ou déduplication conserve la même sémantique
  immuable ; un symlink mutable ne remplace pas un snapshot.
- La chaîne permet de remonter d'une position mensuelle jusqu'aux lignes `def`,
  `stg` et `raw` qui l'expliquent.

## 18. Contrôles de qualité minimaux

À chaque frontière :

- conformité du schéma ;
- unicité de la clé ;
- nombre de lignes et clés ;
- min/max temporels ;
- nulls par colonne obligatoire ;
- doublons et conflits ;
- valeurs infinies et domaines invalides ;
- couverture par période, source et univers ;
- révisions par rapport au snapshot précédent ;
- lignes ajoutées, modifiées et supprimées ;
- échantillon traçable jusqu'au raw.

Une anomalie possède un niveau : `info`, `warning`, `blocking`, et une politique
versionnée. Un seuil dépassé ne supprime pas les données reçues : il bloque la
promotion et conserve le candidat pour audit.

Une anomalie d'une source ne coupe pas les acquisitions indépendantes du même
run : chaque source déclarée est tentée et reçoit un statut avant la décision
de publication. La quarantaine peut bloquer une valeur ou un package, jamais
l'enregistrement du téléchargement ni l'audit des autres fournisseurs.

Pour les prix, un refresh complet reste obligatoire mais ne remplace jamais une
clé `ticker,date` déjà publiée avec une nouvelle valeur fournisseur. Le RAW
différentiel conserve l'observation et ses changements ; le canonique conserve
exactement l'historique validé et ajoute uniquement les dates postérieures,
reconstruites depuis les rendements journaliers du fournisseur et ancrées sur le
dernier cours validé. Une correction historique exige toujours un overlay
sourcé et approuvé. Le run conserve séparément le diagnostic des révisions, le
motif de non-remplacement et les lignes nouvelles effectivement sélectionnées.
Le détail Parquet des révisions de rendement ne répète que les écarts supérieurs
au seuil matériel versionné, actuellement 1 point de base, ou un changement de
disponibilité ; les écarts plus fins restent reconstructibles dans le RAW.

## 19. Publication et promotion

Ordre obligatoire :

1. construire dans un emplacement candidat unique ;
2. écrire les fichiers et manifestes ;
3. calculer les hashes ;
4. exécuter les contrôles de qualité et de révision ;
5. exécuter les replays requis ;
6. marquer le candidat `validated` ;
7. promouvoir le pointeur atomiquement ;
8. conserver l'ancienne cible et le rapport de comparaison.

Une publication partielle n'est jamais présentée comme réussie. Un run échoué
reste identifiable avec son statut et ses journaux.

## 20. Correction et suppression

- Une correction ne réécrit ni raw ni snapshot publié.
- Utiliser un overlay ou une nouvelle release avec événement, preuve, récupéré à,
  auteur/reviewer, clés affectées et hashes avant/après.
- Une suppression logique conserve motif, portée et date d'effet.
- Une suppression physique exige un inventaire, une preuve de duplication ou de
  rétention, un retour arrière testé et une décision distincte.
- Aucun script de maintenance ne vise un glob ou un pointeur non résolu pour une
  suppression.

## 21. Checklist de revue data

- [ ] grain, clé et identité durable déclarés ;
- [ ] dates économiques et dates de connaissance séparées ;
- [ ] null, zéro et statuts métier distincts ;
- [ ] couche correcte (`raw`, `stg`, `def`, `mart`) ;
- [ ] jointures, cardinalités et pertes de lignes mesurées ;
- [ ] aucune information future dans une décision passée ;
- [ ] source, règle, version et hashes présents ;
- [ ] révisions comparées au snapshot précédent ;
- [ ] publication atomique et retour arrière possible ;
- [ ] aucun ancien fichier réécrit ou supprimé silencieusement.
