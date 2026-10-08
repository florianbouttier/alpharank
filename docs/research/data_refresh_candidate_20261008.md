# Candidat data complet du 8 octobre 2026

## Verdict data

Le run réseau `20261008_072222` a tenté toutes les sources déclarées et a
conservé leurs réponses. Après correction des deux faux blocages de portée et
revue sourcée des quatre séances BE, le candidat composé
`eac32074b914c8842aabc5f3345bc38ae6631f091599748d56955d720f0e1fe6`
passe les gates data. Il reste non promu jusqu'au replay `REPLAY-010`.

Le pointeur de production `data/model_inputs/manifests/latest.json` reste
byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`,
sur la composition
`9a2058c98ecda33bda77170f67c5c73c0d69efb51d5d26948ca44f70d91425ad`.

## Acquisition et durée

L'acquisition s'est déroulée de 09:22:22 à 10:14:23, heure de Paris, soit
environ 52 minutes. Les prix et références étaient déjà largement parallélisés ;
la plus grande partie du temps restant vient des appels SEC séquentiels et
bornés. Le manifeste d'acquisition a le SHA-256
`bc4b6328fdf4fc053245dcf29511d8796d7cd93eb6b7ca2ba0956dc3ea7ee92d`.

| Source | Statut | Lignes | Échecs | Lecture |
| --- | --- | ---: | ---: | --- |
| Yahoo prix | `downloaded_quarantined` | 2 487 879 | 0 | 503/503 actifs, RAW conservé avant revue |
| Yahoo metadata | `downloaded` | 8 | 0 | diagnostic |
| yfinance earnings | `downloaded` | 42 134 | 0 | fallback diagnostic |
| SEC submissions | `downloaded` | 3 951 | 0 | univers actif complet |
| SEC Companyfacts | `downloaded_with_failures` | 505 865 | 7 | uniquement anciens symboles 404 hors univers actif |
| documents SEC | `downloaded` | 17 | 0 | fallback filing ciblé |
| SimFin | `downloaded_with_failures` | 14 090 | 2 | fallback diagnostic, jamais valeur officielle finale |
| yfinance fondamentaux | `downloaded` | 26 575 | 0 | fallback diagnostic |

Les sept réponses SEC 404 sont `ABS`, `CFC`, `GLK`, `PLL`, `RX`, `SBNY` et
`TMC`. Aucun membre actif ne manque dans Companyfacts ou submissions.

## Prix : observer tout, ne réécrire aucun passé validé

Le téléchargement complet contient 44 révisions de rendement fournisseur. Le
RAW les conserve ; le canonique conserve toutes les anciennes valeurs publiées
et ajoute 16 841 lignes dérivées des nouveaux rendements. Le package final
contient 3 745 807 lignes pour 847 tickers et valide :

- prix des 503 membres et SPY jusqu'à la séance close du 7 octobre ;
- zéro ancienne ligne canonique modifiée et zéro clé historique supprimée ;
- zéro transition d'ajustement dans le candidat réconcilié ;
- 1 200 483 clés EODHD historiques attendues et zéro clé manquante ;
- cinq mouvements extrêmes approuvés dans leurs bornes exactes, dont quatre BE ;
- préfixe SPY publié inchangé et 34 séances ajoutées jusqu'au 7 octobre.

Le manifeste prix, SHA-256
`1ca798fd4043df3f577a89b9816b5411a2690e0933f1a11bde88083ebcc6e7a8`,
est dans
`outputs/data_refresh_replay_20261008/price_candidate/lineage/manifest.json`.

## SEC cumulatif et point-in-time

Le dossier RAW du run est un delta et n'est pas utilisé seul. Il est fusionné
avec le dernier RAW point-in-time retenu en conservant la clé de version
`ticker, statement, metric, date, filing_date, source`. Le résultat contient
513 123 faits Companyfacts, 145 650 faits filing, 58 064 calendriers, 39 568
actuals et 1 656 références. Son manifeste a le SHA-256
`51cc2b6fb872ba56c95a61b22b7fee273a21ade6374f586e419fda7c9baf0555`.

Le package SEC-only contient 469 496 lignées financières, 56 659 lignées
earnings et 847 références. Les révisions officielles sont conservées dans le
candidat diagnostic et explicitement soumises au replay avant toute promotion.
Son manifeste a le SHA-256
`9851d2df833729211efd8a62000896611435e628d2dc65489d6d4008dbb6fb3b`.

Une invocation du script historique
`build_sec_output_package_with_backfill.py --help` a exécuté ses valeurs par
défaut au lieu d'afficher une aide. Elle a sauvegardé l'ancien `data/sec/output`
dans `data/sec/history/output/sec_output_20261008_083112`. Ni ce package legacy
ni cette archive ne sont utilisés par le candidat ; les chemins explicites
ci-dessus font foi. Aucun fichier n'a été supprimé.

## Snapshot candidat commun

La composition a été créée avec un pointeur local :

```bash
./.venv/bin/python scripts/open_source/build_composed_model_snapshot.py \
  --price-package-dir outputs/data_refresh_replay_20261008/price_candidate \
  --sec-package-dir outputs/data_refresh_replay_20261008/sec_candidate \
  --history-root outputs/data_refresh_replay_20261008/composed_history \
  --latest-manifest outputs/data_refresh_replay_20261008/candidate_latest.json \
  --expected-through 2026-10-07
```

Le snapshot se trouve dans
`outputs/data_refresh_replay_20261008/composed_history/alpharank_input_20261008_083342_eac32074b914`.
Son manifeste, SHA-256
`9b83c81bb6477a956fb023fef948d76f4d7b50b304a7a09581392f70c0babb67`,
valide neuf fichiers, l'unicité des 3 745 807 clés prix, les identités, le
registre des historiques persistants et l'usage du même snapshot par Legacy et
Boosting.

## Gate suivante obligatoire

`REPLAY-010` doit maintenant recalculer Legacy et Boosting sur la production et
sur ce candidat avec le même code et les mêmes cutoffs. Il doit comparer le
dernier portefeuille déjà formé entre les deux vintages, puis expliquer toute
différence d'entrée, univers, score, position, poids ou rendement. La présence
de nouvelles séances d'août, septembre et octobre ne dispense jamais ce test.
