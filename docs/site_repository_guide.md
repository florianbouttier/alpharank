# AlphaRank : méthodes, décisions et portefeuille

**Rôle : projection lisible dans le site Portfolio — mise à jour le 2026-08-29.**

Cette page explique les méthodes sans les redéfinir. Les contrats canoniques
restent `docs/legacy_boosting_methodology.md` pour les signaux et
`docs/common_portfolio_backtest_engine.md` pour la simulation. Le site affiche
la présente projection, synchronisée depuis le dépôt AlphaRank.

## La réponse courte

AlphaRank possède une méthode de production, **Legacy**, et un challenger de
recherche, **Boosting**. Legacy privilégie explicitement la tendance relative à
SPY ; Boosting apprend un classement non linéaire à partir de variantes causales
de ces signaux EMA, sans utiliser actuellement les fondamentaux SEC dans son
score. Les deux méthodes décident à la fin du mois `t`, avant de connaître le
rendement de la détention en `t+1`, puis passent dans le même moteur de
portefeuille.

L'objectif d'une alternative n'est pas nécessairement de battre Legacy seule.
Une poche qui ajoute des titres différents, réduit la concentration du
portefeuille total ou améliore son comportement dans certains régimes peut être
utile même avec un rendement autonome inférieur.

## Statut exact des méthodes

| Méthode | Rôle | Données de décision | Statut au 2026-08-29 |
| --- | --- | --- | --- |
| Legacy `Combined_Frequency` sans SEC | portefeuille mensuel canonique | prix, SPY, composition historique, qualité/liquidité et secteur | production promue |
| Legacy avec filtre PE | reproduction de l'ancienne règle `0 < PE < 100` | mêmes données plus fondamentaux SEC point-in-time | compatibilité historique uniquement |
| Boosting natif | challenger XGBoost Top 5/10/15/20 | signaux EMA relatifs causaux appris fold par fold | R&D comparable, non promu |
| Boosting filtré par tendance | diagnostic Top 5/10 | même score, mais candidats limités à une majorité d'EMA positives | R&D post-hoc, non promu |
| Poche Boosting complémentaire | alternative conçue pour diversifier Legacy | score Boosting inchangé, titres Legacy du même mois exclus | planifiée, pas encore implémentée |

« Comparable » signifie même snapshot, calendrier, benchmark, coûts et moteur.
Cela ne signifie ni « validé pour le futur », ni « recommandé à l'achat ».

## De la source à une décision

```text
source téléchargée
    -> raw : payload reçu, versionné et jamais écrasé silencieusement
    -> stg : types, identifiants et calendriers normalisés
    -> def : valeur retenue et raison du choix
    -> mart : table préparée pour AlphaRank
    -> snapshot : mart immuable, sources et hashes figés
    -> signal Legacy ou Boosting
    -> panier finalisé
    -> moteur de portefeuille commun
    -> rapport et manifeste du run
```

Pseudo-code de la donnée :

```text
download_cutoff = date demandée
payload = télécharger toutes les sources prévues jusqu'au cutoff
raw_version = conserver payload + heure + source + hash

stg = normaliser(raw_version)
def = arbitrer_par_clé(stg, politique_versionnée)
mart = construire_les_entrées_modèle(def)

si contrôles de schéma, temps de connaissance, identité ou lignée échouent :
    conserver ce qui a été reçu
    interdire la promotion
sinon :
    publier un nouveau snapshot immuable

ne jamais modifier un ancien snapshot publié
```

Les prix des titres inactifs ou retirés de la cote conservent leur preuve
historique EODHD. Une source ouverte peut prolonger une trajectoire active, mais
elle ne doit jamais effacer un ancien préfixe parce que le symbole n'est plus
téléchargeable aujourd'hui. Les fondamentaux officiels sont SEC/GAAP ; ils sont
conservés et auditables même si la politique Legacy courante ne les utilise pas.

## Garde commune avant tout modèle

Pour un titre et un mois de décision, les deux méthodes exigent :

| Contrôle | Seuil courant |
| --- | ---: |
| observations de clôture non nulles | au moins 10 jours |
| médiane quotidienne de `close × volume` | au moins 1 million USD |
| jours OHLC incohérents | au plus 5 % |

La composition du S&P 500 est celle connue à la date de décision. La quarantaine
historique d'identifiants cassés s'applique à toute leur trajectoire et doit être
identique pour Legacy et Boosting. Un titre exclu un mois peut redevenir éligible
un autre mois si le défaut était seulement mensuel.

Pseudo-code :

```text
pour chaque mois t et chaque titre :
    membre = appartient_au_SP500_connu_à(t)
    assez_de_prix = nombre_de_clôtures(t) >= 10
    liquide = médiane(clôture * volume, t) >= 1_000_000_USD
    ohlc_valide = taux_anomalies_ohlc(t) <= 5%
    non_quarantainé = titre absent de la quarantaine versionnée

    éligible[titre, t] = membre et assez_de_prix et liquide \
                         et ohlc_valide et non_quarantainé
```

## Méthode 1 — Legacy

Legacy est un trend-following relatif : il cherche les titres dont le prix
ajusté progresse le mieux relativement à SPY selon un rapport de deux moyennes
mobiles exponentielles. Quatre recherches indépendantes choisissent chaque
année leurs paramètres sur le seul passé. Le portefeuille canonique additionne
ensuite leurs votes.

La politique courante `no_sec_fundamentals_v1` n'utilise ni PE, ni
capitalisation, ni autre valeur SEC pour choisir les titres. Les fichiers SEC
restent présents et hashés dans le snapshot, mais ne sont pas lus par cette
sélection. L'ancienne politique PE demeure appelable uniquement pour reproduire
un historique explicitement étiqueté.

Pseudo-code complet simplifié :

```text
INPUT = snapshot immuable
relative_close[titre, jour] = adjusted_close[titre, jour] / close_SP500[jour]

pour chacune des 4 pistes Legacy :
    pour chaque frontière annuelle S :
        train = mois strictement antérieurs à S

        Optuna évalue 30 combinaisons sur train :
            long_ema entre 50 et 400 jours
            short_ema entre 1 et 100 jours
            nombre_de_titres entre 5 et 30
            maximum_par_secteur entre 1 et 2

            signal = EMA(relative_close, short) / EMA(relative_close, long)
            candidats = éligibles au mois t
            préselection = 30 meilleurs signaux
            sélection = plafond_sectoriel(préselection)
            portefeuille = meilleurs N, poids égaux
            rendement = détention au mois t+1

            score_optuna = moyenne décroissante dans le temps
                           de la performance relative à SP500

        paramètres_annuels = meilleur candidat, tie-break déterministe
        produire les paniers hors échantillon jusqu'à la frontière suivante

Combined_Frequency[t] :
    votes[titre] = nombre de pistes ayant choisi le titre
    poids[titre] = votes[titre] / somme(votes)

envoyer Combined_Frequency au moteur commun
```

Les secteurs plafonnent la concentration. Les fondamentaux ne prédisent pas le
rendement dans Legacy : dans l'ancienne politique, ils étaient seulement un
filtre d'éligibilité appliqué avec leur date réelle de disponibilité.

## Méthode 2 — Boosting natif

Boosting entraîne un classifieur XGBoost pour estimer quels titres appartiendront
aux 10 % les plus élevés en rendement relatif à SPY sur les six mois suivants.
Cette cible sert uniquement à apprendre dans le passé. Pour construire un
portefeuille mensuel, le modèle classe ses scores à `t`, retient les `N`
premiers et les détient en `t+1`.

Le profil public courant ne contient aucun fondamentaux, aucun PE et aucune
capitalisation. Il conserve, pour chaque paire EMA causale gagnée par Legacy,
cinq formes : ratio brut, percentile mensuel, z-score mensuel, indicateur du
quartile supérieur et indicateur du quartile inférieur.

Pseudo-code complet simplifié :

```text
INPUT = le même snapshot et le même run Legacy

pour chaque titre et mois t :
    construire les EMA avec les prix connus jusqu'à la fin de t
    future_excess_6m = rendement_titre(t -> t+6) relatif à SPY
    cible = 1 si future_excess_6m est dans les 10% supérieurs, sinon 0

construire des folds chronologiques expansifs :
    train >= 62 mois
    validation = 6 mois
    purge = 5 mois pour séparer les labels à horizon 6 mois
    test = 12 mois, avec dernière fenêtre partielle conservée

pour chaque fold :
    cutoff = dernier mois du train
    paires_ema = gagnants Legacy observables au cutoff
    features = 5 formes de chaque paire_ema
    supprimer toutes les autres features, dont les fondamentaux

    apprendre filtre de colonnes et imputation sur train seulement
    entraîner XGBoost sur train
    arrêter les itérations selon validation
    calibrer sur validation seulement si les deux classes existent
    scorer le test resté intact
    calculer SHAP sur toutes les prédictions test, sans échantillonnage

pour chaque mois test t et chaque N dans {5, 10, 15, 20} :
    classer score XGBoost décroissant, ticker comme tie-break
    sélectionner avant de regarder le rendement réalisé
    prendre Top N à poids égaux
    détenir au mois t+1
```

Les SHAP expliquent pourquoi le modèle a produit un score ; ils ne servent pas
à filtrer les titres et ne prouvent pas à eux seuls que le raisonnement reste
stable dans le temps. Leur étude de stabilité doit suivre, par mois et par
feature économique, le signe, le rang, la dispersion et les changements de
régime sur les lignes OOS exhaustives.

## Variante — Boosting filtré par tendance

Cette variante ne réentraîne pas XGBoost et ne modifie aucun score. Elle exige
qu'une majorité stricte des ratios EMA relatifs bruts du fold indique une
tendance positive, puis classe par le score Boosting habituel.

```text
pour chaque prédiction OOS (titre, mois, fold) :
    paires = paires EMA causales gelées au cutoff du fold
    signes = []

    pour chaque paire (short, long) :
        ratio = EMA_relative_short / EMA_relative_long au mois t
        si la paire est stockée dans l'ordre inverse : inverser le signe
        signes.ajouter(ratio > 1)

    éligible = toutes les paires sont observées \
               et somme(signes) > nombre_de_paires / 2

pour chaque mois t :
    classer par score Boosting uniquement parmi les éligibles
    prendre Top N avant de joindre les rendements t+1
```

Le replay Top 5/10 améliore les estimations ponctuelles de rendement et de
risque, mais les intervalles bootstrap du gain de rendement et de Sharpe
traversent zéro. De plus, l'idée a été formulée après lecture des SHAP récents :
c'est donc un diagnostic post-hoc, pas une confirmation indépendante.

Surtout, ce filtre ne diversifie presque pas Legacy aujourd'hui : son Top 10
d'août 2026 partage 9 titres sur 10 avec Legacy. Il crée un panier davantage
trend-following, mais pas une vraie seconde source de positions.

## Proposition — poche Boosting complémentaire

Pour répondre à l'objectif « plus de titres et plus de diversification », la
meilleure prochaine variante ne doit pas forcer Boosting à ressembler davantage
à Legacy. Elle doit conserver le modèle natif et empêcher seulement la poche
complémentaire d'acheter les titres déjà présents dans Legacy au même instant.

```text
pour chaque mois de décision t :
    legacy_t = panier Legacy calculé avec les informations disponibles à t
    scores_t = scores Boosting OOS inchangés à t

    candidats = titres éligibles à t
    candidats = candidats moins les titres de legacy_t
    candidats = appliquer un plafond sectoriel préenregistré, si retenu

    boost_complémentaire_t = Top K candidats par score Boosting
    poids_boost = poids égaux dans la poche

    portefeuille_total_t = (1 - poids_poche) * legacy_t \
                           + poids_poche * boost_complémentaire_t
```

Le premier replay doit figer avant calcul les poids de poche `10 %`, `20 %` et
`30 %`, plutôt que choisir a posteriori celui qui a le meilleur rendement. On
jugera cette alternative sur :

- le nombre total et le nombre effectif de positions ;
- le recouvrement mensuel et le Jaccard avec Legacy ;
- la concentration par titre et par secteur ;
- la corrélation des rendements et la contribution marginale au risque ;
- le turnover, les coûts, le drawdown et la volatilité du portefeuille total ;
- le rendement du portefeuille total, sans exiger que la poche batte Legacy
  lorsqu'elle est regardée seule.

Sur la décision de juillet pour détention en août 2026, Legacy contient 15
titres. Le Top 10 Boosting natif en partage un seul, DELL, et apporte donc neuf
titres distincts ; leur combinaison simple contient actuellement 24 titres
distincts. La future règle complémentaire garantirait jusqu'à 25 titres
distincts avec une poche Top 10, sous réserve d'une couverture suffisante.

## Derniers paniers observables

Les tableaux suivants décrivent le snapshot acquis jusqu'au 2026-08-26. Ce ne
sont pas des recommandations d'achat et le rendement d'août n'entre pas dans le
score de juillet.

### Legacy, détention cible août 2026

| Titre | Poids | Titre | Poids | Titre | Poids |
| --- | ---: | --- | ---: | --- | ---: |
| DELL | 12,50 % | AMD | 9,38 % | HPE | 9,38 % |
| MU | 9,38 % | SNDK | 9,38 % | STX | 9,38 % |
| WDC | 9,38 % | INTC | 6,25 % | MRVL | 6,25 % |
| AMAT | 3,13 % | DDOG | 3,13 % | FLEX | 3,13 % |
| FTNT | 3,13 % | HUM | 3,13 % | PANW | 3,13 % |

### Boosting natif Top 10, détention cible août 2026

| Rang | Titre | Score | Déjà dans Legacy ? |
| ---: | --- | ---: | --- |
| 1 | BSX | 0,452313 | non |
| 2 | CHTR | 0,444015 | non |
| 3 | ZTS | 0,420190 | non |
| 4 | LULU | 0,397275 | non |
| 5 | INTU | 0,385244 | non |
| 6 | IT | 0,377782 | non |
| 7 | FISV | 0,365073 | non |
| 8 | DELL | 0,349872 | oui |
| 9 | ACN | 0,346119 | non |
| 10 | TSCO | 0,340164 | non |

### Boosting tendance Top 10, détention cible août 2026

| Rang | Titre | Score | Déjà dans Legacy ? |
| ---: | --- | ---: | --- |
| 1 | DELL | 0,349872 | oui |
| 2 | SNDK | 0,337033 | oui |
| 3 | MU | 0,318474 | oui |
| 4 | WDC | 0,314044 | oui |
| 5 | STX | 0,311204 | oui |
| 6 | INTC | 0,299029 | oui |
| 7 | MRVL | 0,295930 | oui |
| 8 | AMD | 0,287544 | oui |
| 9 | LRCX | 0,281418 | non |
| 10 | FLEX | 0,259821 | oui |

## Cas SATS — ce qui se serait réellement passé

Le premier replay de la variante tendance a tenté les Top 5, 10, 15 et 20. Les
Top 15 et 20 auraient sélectionné `SATS.US` avec la décision d'avril 2026 pour
une détention en mai. Le snapshot possède le signal d'avril, mais pas la
trajectoire de prix nécessaire pour valoriser mai ; le moteur a donc arrêté le
run au lieu de transformer artificiellement cette absence en rendement nul.

SATS n'était pourtant ni disparu ni un événement terminal en mai. EchoStar
déposait encore auprès de la SEC sous ce symbole le 22 mai. La société a annoncé
seulement le 22 juin que `SATS` deviendrait `ECHO` le 24 juin ; elle précise que
le CUSIP, la structure du capital et les droits des actionnaires ne changent pas.
Utiliser cette annonce de juin pour retirer SATS de la décision d'avril serait
une fuite d'information future.

Avec une donnée complète, le comportement correct aurait été :

```text
décision fin avril : SATS est sélectionnable avec les informations d'avril
détention en mai   : appliquer son vrai rendement total de marché, pas zéro
24 juin            : continuer la même sécurité sous le symbole ECHO
détentions futures : résoudre l'identité SATS -> ECHO, sans vente fictive
```

La correction proposée n'est pas de remplir une valeur à la main. Il faut :

1. télécharger la trajectoire ajustée SATS jusqu'au 23 juin et ECHO à partir du
   24 juin auprès d'une source couvrant les deux symboles ;
2. confirmer l'identité continue par le CUSIP et vérifier les facteurs
   d'ajustement, splits et dividendes ;
3. publier un nouvel overlay prix versionné et immuable, sans modifier le
   snapshot du 28 août ;
4. reconstruire le snapshot composé puis rejouer Legacy, Boosting et le moteur
   commun, car le prix réparé peut aussi modifier des EMA et des cibles ;
5. conserver les Top 15/20 comme non valides tant que ce replay n'existe pas.

Sources officielles : [annonce EchoStar du changement SATS vers ECHO](https://ir.echostar.com/news-releases/news-release-details/echostar-changing-stocker-ticker-sats-echo-marking-companys-next),
[dépôt SEC du 22 mai 2026](https://www.sec.gov/Archives/edgar/data/1415404/000141540426000013/0001415404-26-000013-index.htm)
et [historique de prix EchoStar fourni par LSEG](https://ir.echostar.com/stock-information/historical-price-lookup).

## Moteur de portefeuille commun

Legacy et Boosting s'arrêtent de différer une fois leurs paniers finalisés. Le
moteur commun impose alors le même contrat :

```text
pour chaque panier décidé à la fin du mois t :
    vérifier holding_month = t + 1 mois
    vérifier poids longs, finis et somme = 1
    figer les titres avant de joindre les rendements réalisés

    si un rendement sélectionné manque :
        utiliser seulement un événement terminal sourcé dans le mois de détention
        sinon arrêter le replay ; ne jamais promouvoir le titre suivant

    rendement_brut = somme(poids * rendement_titre)
    turnover = 0.5 * somme(abs(poids_t - poids_t-1))
    coût = turnover * points_de_base / 10_000
    rendement_net = rendement_brut - coût

    comparer à SPY total return construit depuis adjusted_close
```

Le replay commun exige les mêmes hashes de snapshot, exclusions historiques,
contrôle mensuel de prix, convention d'exécution et calendrier. Les coûts de la
comparaison courante sont de 10 points de base multipliés par le turnover pour
les stratégies. Legacy conserve séparément sa convention historique de
production à zéro coût.

## Règles anti-fuite indispensables

- information et composition d'indice disponibles à `t`, détention à `t+1` ;
- sélection des titres avant toute inspection du rendement réalisé ;
- prétraitement appris sur le train puis réutilisé sans recalcul sur test ;
- folds chronologiques avec purge adaptée à la cible de six mois ;
- cible immature conservée comme non évaluable, jamais changée en zéro ;
- mois courant partiel exclu des performances ;
- événement terminal utilisable seulement s'il est sourcé et effectif pendant
  la détention concernée ;
- comparaison économique seulement sur le même snapshot et le même calendrier ;
- SHAP utilisé pour expliquer une prédiction, jamais comme information de
  sélection future cachée.

## Référence chiffrée comparable

Le replay courant utilise le snapshot
`alpharank_input_20260827_122648_5bfbc1d3cb04`, 180 mois réalisés d'août 2011 à
juillet 2026, SPY total return et 10 points de base multipliés par le turnover.

| Stratégie | CAGR | Volatilité | Sharpe | Drawdown maximal |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 5 natif | 19,94 % | 35,92 % | 0,50 | -37,28 % |
| Boosting Top 10 natif | 19,93 % | 31,49 % | 0,57 | -31,36 % |
| Boosting Top 15 natif | 18,90 % | 29,70 % | 0,57 | -32,71 % |
| Boosting Top 20 natif | 17,81 % | 28,68 % | 0,55 | -31,84 % |
| Legacy sur calendrier commun | 19,64 % | 26,57 % | 0,66 | -26,49 % |
| SPY total return | 14,40 % | 14,30 % | 0,87 | -23,93 % |

Ces chiffres montrent que Boosting peut constituer une source de décisions
différente. Ils ne prouvent pas que son avantage futur est certain. La prochaine
question utile est donc celle du portefeuille combiné : combien de diversification
réelle obtient-on, à quel coût et avec quel impact sur le risque total ?

## Où approfondir

- méthodes et pseudo-codes exhaustifs : `docs/legacy_boosting_methodology.md` ;
- simulation, coûts, KPI et causalité :
  `docs/common_portfolio_backtest_engine.md` ;
- cycle des données : `docs/architecture/data_lifecycle.md` ;
- production mensuelle Legacy : `docs/monthly_portfolio_runbook.md` ;
- priorités et statut des variantes : `ROADMAP.md`.
