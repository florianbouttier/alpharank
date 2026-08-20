# Standard Python AlphaRank

**Rôle : norme de code Python, de calcul et de test.**

**Périmètre : `src/`, `scripts/`, `tests/` et outils Python maintenus.**

## 1. Socle et formatage

| Règle | Niveau |
| --- | --- |
| Compatibilité minimale Python 3.10 tant qu'une migration dédiée n'est pas validée | OBLIGATOIRE |
| `from __future__ import annotations` dans tout nouveau module maintenu | OBLIGATOIRE |
| Format cible `ruff format`, longueur de ligne 100 | OBLIGATOIRE pour le nouveau code |
| Fichier UTF-8, fin de ligne Unix, une ligne vide finale | OBLIGATOIRE |
| Formatage manuel d'un fichier entier dans un correctif fonctionnel | INTERDIT |
| Plusieurs instructions sur une ligne | INTERDIT |
| Code commenté laissé « au cas où » | INTERDIT ; Git conserve l'historique |

Ruff sera activé progressivement. Tant que l'automatisation n'est pas terminée,
la règle reste applicable en revue sur le périmètre modifié.

## 2. Imports et dépendances

Ordre obligatoire : bibliothèque standard, dépendances tierces, package
`alpharank`, imports locaux. Un groupe est séparé du suivant par une ligne vide.

- Imports absolus `alpharank.*` obligatoires entre packages.
- Imports relatifs autorisés uniquement à l'intérieur d'un même sous-package et
  sur un niveau maximum.
- `from module import *` interdit.
- Modification de `sys.path`, lecture de `PYTHONPATH` ou dépendance au dossier
  courant interdites.
- Import conditionnel autorisé seulement pour une dépendance réellement
  optionnelle, avec erreur explicite lors de l'utilisation.
- Aucune requête réseau, lecture de fichier, création de dossier, entraînement
  ou journalisation lors de l'import d'un module.
- Une dépendance externe nouvelle doit être justifiée, versionnée dans la source
  de dépendances canonique et entourée d'un adaptateur si elle touche le cœur
  métier.

## 3. Nommage

| Élément | Convention | Exemple |
| --- | --- | --- |
| module | `snake_case`, nom métier | `price_eligibility.py` |
| fonction/variable | `snake_case` | `compute_monthly_return` |
| classe | `PascalCase` | `SnapshotManifest` |
| constante | `UPPER_SNAKE_CASE` | `MAX_GAP_DAYS` |
| protocole/interface | nom métier, suffixe éventuel `Protocol` | `PriceProviderProtocol` |
| exception | suffixe `Error` | `SnapshotLineageError` |
| booléen | préfixe qui forme une question | `is_evaluable`, `has_price` |
| unité | suffixe explicite lorsque nécessaire | `cost_bps`, `price_usd`, `age_days` |
| horizon | suffixe explicite | `return_1m`, `target_h6` |

Sont interdits dans un nouveau nom : `final`, `new`, `latest2`, `temp`, `misc`,
`helper`, `stuff`, ou un suffixe `_v2` sans contrat public versionné. Une version
vit dans un schéma, une configuration ou un identifiant de protocole, pas dans
une succession indéfinie de fichiers.

Un ticker n'est jamais nommé `id` : utiliser `ticker` lorsqu'il s'agit bien du
symbole observé et `security_id` pour l'identité durable.

## 4. Taille et responsabilité

| Élément | Cible | Limite de nouveau code |
| --- | ---: | ---: |
| fonction | 50 lignes | revue bloquante au-delà de 80 |
| module de bibliothèque | 500 lignes | plan de découpage au-delà de 800 |
| script | 150 lignes | maximum 250 |
| paramètres de fonction | 5 | config nommée au-delà |
| complexité cyclomatique | 8 | maximum 10 |
| niveaux d'imbrication | 3 | maximum 4 |

Une fonction a un verbe et une responsabilité. Une classe représente un objet,
un service ou une politique stable ; elle ne sert pas à cacher un fichier de
fonctions dans un état mutable.

Préférer :

- une fonction pure pour un calcul ;
- une `dataclass(frozen=True, slots=True)` pour une configuration ou une valeur ;
- un petit service injecté pour une frontière réseau ou disque ;
- la composition à l'héritage.

Éviter les méthodes statiques regroupées uniquement pour créer un « namespace ».

## 5. Signatures de fonctions

- Toute fonction publique possède des types d'entrée et de sortie.
- Les options qui ne sont pas évidentes sont keyword-only après `*`.
- Les valeurs par défaut mutables sont interdites.
- Un booléen qui change profondément l'algorithme est remplacé par un `Enum` ou
  une politique nommée.
- `*args` et `**kwargs` sont interdits dans une API métier, sauf adaptateur dont
  le contrat est explicitement limité.
- `None` n'a qu'un sens documenté. Il ne signifie pas simultanément « absent »,
  « automatique » et « désactivé ».
- Une fonction ne retourne pas des tuples positionnels de plus de deux valeurs ;
  utiliser une structure nommée.
- Une collection reçue en lecture seule utilise `Sequence`, `Mapping` ou
  `Collection` plutôt qu'un type concret inutilement restrictif.

Exemple attendu :

```python
def simulate_month(
    holdings: Sequence[Holding],
    returns: Mapping[str, float],
    *,
    missing_return_policy: MissingReturnPolicy,
    transaction_cost_bps: float,
) -> MonthlySimulationResult:
    ...
```

## 6. Typage

- `Any` est interdit dans le cœur métier. Il est toléré à une frontière externe
  non typée, puis converti immédiatement vers un type interne.
- `cast()` ne remplace pas une validation runtime d'une donnée externe.
- Les dictionnaires stables utilisent `TypedDict`, une dataclass ou un modèle de
  schéma versionné.
- Les valeurs à états fermés utilisent `Enum` ou `Literal`.
- Les alias de types portent un nom métier, pas `Data` ou `Result` seuls.
- Une annotation `DataFrame` ne constitue pas un schéma : les colonnes, types,
  clés et unités sont validés à la frontière.
- Les suppressions de contrôle (`type: ignore`, `noqa`) indiquent la règle, la
  raison et la portée minimale.

Le typage est introduit package par package. Une API déjà déclarée stricte ne
peut pas redevenir permissive pour faciliter un appel historique.

## 7. Structures de données et état

- Entrées non mutées par défaut. Si une mutation est nécessaire pour la
  performance, elle est nommée et documentée.
- Pas d'état global mutable.
- Pas de singleton implicite pour les chemins, caches, seeds ou configurations.
- Les structures métier importantes sont immuables après création.
- Les objets de configuration sont séparés des résultats calculés.
- Les valeurs dérivées ne sont pas recopiées dans plusieurs objets si elles
  peuvent être recalculées de façon déterministe et peu coûteuse.
- Une API ne mélange pas lecture réseau, transformation, sélection et
  publication dans une seule méthode.

## 8. Polars, pandas et NumPy

### Choix du moteur

- Polars est le choix par défaut pour les transformations tabulaires de
  production et les volumes importants.
- pandas est autorisé aux frontières des bibliothèques qui l'exigent, dans le
  code Legacy non migré et pour une analyse locale justifiée.
- Une conversion pandas/Polars se fait dans un adaptateur nommé, une seule fois
  par trajet, avec contrôle du schéma avant et après.
- NumPy est utilisé pour les calculs vectorisés et les interfaces de modèles,
  pas comme stockage sans noms de colonnes à travers plusieurs couches.

### Transformations

- Sélectionner explicitement les colonnes utiles.
- Trier explicitement avant `lag`, fenêtre, `asof join`, déduplication ou choix
  « premier/dernier ».
- Déclarer et contrôler la cardinalité attendue de chaque jointure.
- Vérifier que la jointure n'a pas multiplié les clés de manière inattendue.
- Interdire `drop_nulls()`, `fill_null(0)` ou `dropna()` sans liste de colonnes
  et justification métier.
- Interdire `unique(keep="first")` ou équivalent sans ordre déterministe et règle
  de préférence documentée.
- Éviter `iterrows`, boucles par ligne, `apply` Python et UDF lorsque
  l'expression vectorisée est lisible.
- Ne pas chaîner une expression gigantesque : nommer les étapes selon leur sens
  métier et valider les frontières.
- Ne pas mélanger renommage technique et décision économique dans la même étape.

Toute transformation de production expose au minimum : nombre de lignes avant
et après, nombre de clés distinctes, doublons de clé, nulls des colonnes
obligatoires et période couverte.

## 9. Dates, temps et causalité

- `date` pour un jour sans heure ; `datetime` avec fuseau pour un instant.
- UTC pour le stockage des instants techniques ; conversion locale seulement à
  l'affichage ou à une frontière de marché explicitement nommée.
- Aucun `datetime.now()` caché dans un calcul reproductible. Injecter l'instant
  du run.
- Ne pas stocker une date métier sous forme de chaîne à l'intérieur du cœur.
- Les noms précisent le sens : `event_date`, `filing_date`, `available_at`,
  `retrieved_at`, `decision_month`, `holding_month`.
- Une jointure temporelle indique direction, égalité autorisée et tolérance.
- Un forward-fill précise les clés, la durée maximale et la source de la valeur.
- Aucune feature ou sélection à la date `t` ne consulte une disponibilité,
  révision ou performance réalisée après `t`.

## 10. Nombres financiers

- Rendements, poids et calculs statistiques utilisent `float64` sauf raison
  documentée ; aucune conversion silencieuse vers `float32`.
- Les montants comptables destinés à une exécution réelle utilisent `Decimal` ou
  des unités mineures entières à la frontière d'ordre ; les backtests peuvent
  conserver `float64` avec convention explicite.
- Aucune comparaison exacte de flottants calculés ; utiliser une tolérance
  nommée. Les rapprochements économiques critiques conservent leur tolérance
  contractuelle, actuellement souvent `1e-12`.
- Aucun arrondi intermédiaire pour rendre un rapprochement vert. Arrondir
  uniquement à l'affichage ou selon une règle d'exécution documentée.
- `NaN`, `null`, zéro, infini et non applicable sont des états différents.
- Les pourcentages sont stockés en fraction (`0.05`) et affichés en pourcentage
  (`5 %`). Un nom en `_pct` n'est utilisé que pour une valeur réellement stockée
  sur l'échelle 0–100.

## 11. Erreurs et validations

- Lever une exception métier précise avec le contexte nécessaire à la
  correction.
- `except:` est interdit.
- `except Exception` est autorisé uniquement à la frontière d'un processus ; il
  journalise puis échoue ou produit un statut d'échec explicite.
- Une validation n'utilise pas `assert` dans le code de production : `assert`
  peut être désactivé.
- Ne pas convertir une erreur en liste vide, zéro ou ancien fichier sans
  politique nommée.
- Conserver la cause avec `raise ... from error` lors d'une traduction
  d'exception.
- Les messages contiennent la clé, le dataset, la période et le contrat violé,
  sans exposer de secret.

## 12. Journalisation

- Aucun `print()` dans `src/alpharank/`.
- Un script peut afficher un résumé final destiné à l'humain ; les événements
  d'exécution utilisent le logger.
- Niveaux : `DEBUG` détail diagnostic, `INFO` jalon normal, `WARNING` anomalie
  tolérée et tracée, `ERROR` étape échouée.
- Toute exécution durable porte `run_id`, `snapshot_id`, composant et étape.
- Ne pas journaliser des DataFrames complets, tokens, secrets ou payloads bruts.
- Les compteurs de qualité sont des champs structurés, pas une phrase libre
  impossible à agréger.

## 13. Configuration et secrets

- Aucun secret, token ou chemin utilisateur dans le code, les configs
  versionnées ou les rapports.
- Variables d'environnement lues dans la couche d'application, puis converties
  en configuration typée.
- Aucun appel profond à `os.getenv()` dans le domaine.
- Une configuration utilisée par un run est copiée ou hashée dans son manifeste.
- Un défaut qui affecte l'économie du résultat est explicite dans la CLI et le
  manifeste ; il n'est pas choisi dans une fonction profonde.

## 14. Déterminisme et performance

- Seed explicite pour tout aléa ; seed et versions de bibliothèques dans le run.
- Ordre explicite avant toute opération où l'ordre influence le résultat.
- Pas de parallélisme non déterministe dans une confirmation scellée.
- Une optimisation conserve d'abord un test de parité fonctionnelle.
- Mesurer avant d'optimiser ; ne pas sacrifier les contrôles de lignée pour une
  accélération.
- Les caches sont jetables et invalidés par contenu/configuration, jamais
  considérés comme source de replay.

## 15. Scripts et CLI

- Un script expose `main() -> int` et termine par
  `raise SystemExit(main())` lorsqu'un code retour est utile.
- L'import d'un script n'exécute rien.
- Le script se limite à : parser, résoudre la configuration, appeler la
  bibliothèque, écrire le manifeste et afficher le résumé.
- La logique testable vit sous `src/alpharank/`.
- Toute commande destructive ou de promotion possède une vérification préalable
  et affiche la cible résolue.
- Les codes retour distinguent succès, échec de validation et erreur
  d'exécution lorsque cela aide l'automatisation.

## 16. Tests

- Nommer le comportement, la condition et le résultat.
- Structure Arrange / Act / Assert visible sans commentaires artificiels.
- Un test vérifie une raison principale d'échec.
- Une correction commence par un test qui échoue sur l'ancien comportement.
- Aucun réseau dans un test unitaire.
- Aucun pointeur `latest` de production dans un test unitaire.
- Temps, hasard, fichiers et providers injectés ou remplacés par des fixtures
  minimales.
- Préférer de petites données explicites à une copie d'un snapshot réel.
- Les tests de replay utilisent des artefacts immuables et vérifient hashes,
  calendrier, lignée et sorties économiques.
- Les tests de causalité modifient le futur et prouvent l'invariance du passé.
- Les comparaisons flottantes utilisent la tolérance du contrat, pas une valeur
  agrandie pour faire passer le test.
- Un test paramétré est utilisé lorsque le même contrat doit couvrir plusieurs
  politiques ; il ne doit pas cacher des scénarios sans noms lisibles.

## 17. Documentation du code

- Docstring obligatoire pour toute API publique non triviale.
- La docstring donne contrat, grain, unités, calendrier, retours et exceptions.
- Un commentaire explique une contrainte historique, causale ou fournisseur.
- Un `TODO` comporte un identifiant de roadmap ; sinon il n'est pas traçable.
- Le README du package explique sa responsabilité, ses entrées/sorties,
  dépendances et interdits ; il ne duplique pas tout le pseudocode métier.

## 18. Anti-patterns bloquants

Tout nouveau code contenant l'un des éléments suivants est refusé :

- `sys.path.append` ou `sys.path.insert` ;
- chemin absolu utilisateur ;
- source de données choisie parce qu'un fichier « semble le plus récent » ;
- `fill_null(0)` global ;
- classement des candidats après consultation du rendement futur ;
- moteur de CAGR, Sharpe, drawdown ou turnover local hors package partagé ;
- `except Exception: pass` ;
- logique métier dans un générateur HTML ;
- téléchargement, calcul et publication dans une seule fonction ;
- fichier nommé `final_v2_fixed.py` ;
- test dépendant du réseau ou d'un état local non déclaré.

## 19. Checklist de revue Python

- [ ] responsabilité et emplacement corrects ;
- [ ] types publics et schémas de DataFrame contrôlés ;
- [ ] aucune dépendance au dossier courant, au réseau ou à l'heure cachée ;
- [ ] dates, unités et conventions financières explicites ;
- [ ] jointures et nulls contrôlés ;
- [ ] erreurs précises et journaux structurés ;
- [ ] test du comportement et, si nécessaire, test causal/replay ;
- [ ] documentation et roadmap mises à jour ;
- [ ] aucune remise en forme ou migration sans rapport dans le commit.
