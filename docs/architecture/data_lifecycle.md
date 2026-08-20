# Cycle de vie des données AlphaRank

**Rôle : contrat canonique de rangement et de provenance des données.**

## 1. Le trajet cible

```text
RAW  ->  STG  ->  DEF  ->  MART  ->  SNAPSHOT  ->  RUN
reçu     propre    choisi   prêt      figé          calculé
```

Chaque étape répond à une question différente. Sauter une étape rend difficile
de savoir si une valeur a été reçue, transformée, arbitrée ou calculée.

## 2. RAW — ce qui a été reçu

`raw` conserve la preuve fournisseur : payload, observations, instant de
téléchargement, requête, statut et hash.

Règles :

- une valeur reçue n'est jamais corrigée en place ;
- un nouveau téléchargement strictement identique réutilise le même payload par
  hash, mais son reçu de téléchargement reste enregistré ;
- une valeur changée crée une nouvelle version ; l'ancienne reste consultable ;
- EODHD reste conservé comme source brute historique, y compris les titres
  inactifs et delistés ;
- un cache réseau jetable n'est pas une preuve `raw` ;
- aucune préférence entre fournisseurs n'est appliquée ici.

Le registre strict
[`../../configs/data_contracts/raw_provider_contracts_v1.json`](../../configs/data_contracts/raw_provider_contracts_v1.json)
attribue EODHD, yfinance, SEC Companyfacts, SEC Submissions, les documents de
filing, SimFin, StockAnalysis et les preuves de composition S&P 500 à une racine
unique `data/warehouse/raw/<provider_id>`. Il définit aussi les champs communs
des reçus et manifestes ; déclarer une cible ne déplace ni ne publie encore un
payload.

Ainsi, « ne pas stocker deux fois les mêmes octets » et « conserver tout
l'historique des téléchargements » sont compatibles : on garde chaque reçu et
on référence un payload déjà présent lorsque son hash est identique.

Le writer canonique `record_raw_download` applique ce contrat : le reçu est
créé même lorsque la réponse échoue sans payload, un identifiant de reçu ne peut
pas être réutilisé, et le manifeste `manifests/latest.json` n'est régénéré
qu'après vérification de tous les objets référencés. Les anciennes racines ne
sont ni copiées ni basculées par cette étape ; cette migration est suivie par
`DATA-008` et `DATA-009`.

## 3. STG — la même information dans une forme commune

`stg` normalise sans choisir la vérité économique :

- noms et types de colonnes ;
- fuseaux et calendriers ;
- identifiants de titres et alias explicites ;
- unités, devises et conventions de signe ;
- clés techniques et contrôles de format.

Deux sources en désaccord restent deux observations distinctes dans `stg`.

Le contrat exécutable `alpharank_staging_observations_v1` refuse explicitement
les colonnes de sélection (`selected_source`, `source_priority`,
`fallback_used`, `selection_reason`). Son API ne reçoit aucune priorité : elle
normalise les types, conserve le nombre de lignes et impose une lignée
`source_name`, `dataset_name`, `receipt_id`, `payload_sha256`, `retrieved_at`.
Le choix entre candidats appartient exclusivement à `def`.

## 4. DEF — la valeur retenue et la raison

`def` signifie **définitif pour une règle versionnée et une date de
connaissance**, pas « vérité éternelle ».

Cette couche reste nécessaire même si `raw` évite les doublons exacts, car il
faut encore :

- choisir entre deux fournisseurs ;
- tenir compte d'une date de publication ou d'un restatement ;
- appliquer une correction corporate-action sourcée ;
- distinguer valeur absente, confidentielle, non applicable et zéro observé ;
- expliquer un report autorisé de valeur sans inventer une observation à une
  autre date.

Chaque ligne retenue porte au minimum la clé métier, la valeur, la source brute,
la version de règle, l'instant de connaissance et le motif de sélection.

Le contrat `alpharank_definitive_observations_v1` reçoit la liste de priorité,
sa version et un cutoff de connaissance. Pour chaque clé, il ne considère que
les reçus connus à ce cutoff, retient la dernière observation de chaque source,
choisit la première valeur non nulle, et écrit le reçu, le hash, la date reçue
et le motif. Un zéro observé reste une valeur ; un `null` peut déclencher un
fallback explicite. Si aucune valeur n'est disponible, une décision irrésolue
est tout de même produite. Les consolidateurs historiques restent lisibles mais
ne constituent pas la nouvelle interface canonique DEF.

## 5. MART — prêt pour un consommateur précis

`mart` assemble les données définitives pour un usage identifié, par exemple :

- entrées mensuelles Legacy ;
- entrées Boosting ;
- comparaison Legacy/Boosting sur snapshot commun ;
- exploration publique du site.

Un mart peut calculer des features ou joindre des tables, mais ne doit pas
réinventer la préférence fournisseur définie dans `def`.

Le résolveur `resolve_mart_model_input` impose que la cible consommée par
Legacy soit sous `data/warehouse/mart/`, que le manifeste atteste les parités
DEF-vers-MART et snapshot-source-vers-MART, puis recalcule tous les hashes. Le
chemin par défaut de `scripts/run_legacy.py` passe par ce contrôle avant de
figer son propre `input_snapshot/`.

## 6. SNAPSHOT — une publication figée, pas une nouvelle couche

Un snapshot est une release immuable d'un ou plusieurs marts. Il contient ou
référence de manière reproductible :

- les fichiers réellement consommés ;
- leurs hashes ;
- la composition et les règles utilisées ;
- les versions de schéma ;
- la date maximale connue par type de donnée ;
- les sources et corrections ;
- le commit ou les hashes des fichiers de code critiques.

Le pointeur de production actuel est
`data/model_inputs/manifests/latest.json`. Le mot `latest` désigne seulement le
pointeur ; sa cible est immuable.

## 7. RUN — le calcul et ses résultats

Un run consomme une cible de snapshot résolue au démarrage. Il écrit sous
`outputs/` : configuration, manifeste, résultats, rapports et preuves de
validation. Ses journaux vont sous `logs/` et portent le même `run_id`.

Un run ne doit pas choisir silencieusement « le dernier fichier présent » si la
source déclarée manque.

## 8. Correspondance avec les dossiers actuels

Le relevé machine-lisible daté, avec volumes et lecteurs de code actifs, est
[`data_location_inventory_v1.json`](data_location_inventory_v1.json). Le tableau
ci-dessous en donne la lecture contractuelle stable.

| Emplacement actuel | Lecture correcte | État cible |
| --- | --- | --- |
| `data/US_*.parquet` | ancienne interface Legacy | référencée puis retirée des lecteurs actifs |
| `data/eodhd/output/` | archive EODHD figée | cataloguée sous la source `raw/eodhd` sans réécriture |
| `data/open_source/official/raw/` | matières premières de l'ingestion open source | convergence vers `warehouse/raw` |
| `data/open_source/output/` | ancien package mixte de recherche/replay | non canonique pour une nouvelle production |
| `data/sec/` | anciennes générations SEC-only et staging | convergence vers raw/stg/def puis mart |
| `data/warehouse/raw/` | début de la cible brute historisée | cible |
| `data/warehouse/stg/` | début de la cible normalisée | cible |
| `data/warehouse/def/` | début de la cible arbitrée | cible |
| `data/warehouse/mart/` | début des packages consommateurs | cible |
| `data/model_inputs/history/` | snapshots composés déjà publiés | releases immuables conservées |
| `data/_snapshots/` | anciennes captures locales | inventaire puis archive explicite |

## 9. Migration sans mauvaise surprise

La migration suivra ces règles :

1. ne déplacer aucune donnée tant que ses lecteurs ne sont pas connus ;
2. ne retélécharger aucune histoire déjà prouvée localement ;
3. enregistrer les contenus par hash pour détecter les doublons exacts ;
4. comparer nombre de clés, valeurs, dates et hashes entre ancien et nouveau
   trajet ;
5. conserver les anciennes racines en lecture seule tant que tous les replays
   nécessaires ne passent pas ;
6. ne dédupliquer physiquement qu'après une décision séparée et une procédure de
   restauration testée.

Les tâches correspondantes sont `DATA-001` à `DATA-010` dans
[`../../ROADMAP.md`](../../ROADMAP.md).
Les conventions de grain, clés, temps, nulls, formats et publication sont
normatives dans [`../standards/data.md`](../standards/data.md).
