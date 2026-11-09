# Historique sectoriel causal du 9 novembre 2026

## Verdict

`DATA-047` publie la release DEF candidate
`55fbc0c519ea8b50d77fb87b7c0d62ef7ab02d6ee2694c0c2e4faf0f8b5eda84`.
Elle contient 11 486 classifications pour 754 tickers, connues entre le
23 janvier 2009 et le 9 novembre 2026. Chaque secteur vient du SIC d'un en-tête
de dépôt annuel SEC immuable et n'est disponible qu'à l'heure officielle
d'acceptation du dépôt.

La release rend les 183 mois Boosting de juillet 2011 à septembre 2026
calculables avec au moins cinq secteurs distincts. Elle ne constitue ni un
nouveau backtest de performance, ni une promotion de stratégie, ni une
modification du snapshot modèle ou des scores.

## Contrat causal

Les seuls formulaires admis sont `10-K`, `10-KT`, `20-F` et `40-F`, hors
amendements. `ACCEPTANCE-DATETIME` est interprété à New York puis converti en
UTC ; le secteur n'existe donc jamais avant cette heure. Le mapping
`sec_sic_to_alpharank_sector_v1` transforme le SIC en onze secteurs de
recherche AlphaRank : il ne prétend pas reproduire GICS et n'a pas été calibré
sur les rendements.

Les identités ticker/CIK sont séparées des secteurs. Les classes d'actions SEC
avec tiret sont normalisées vers la convention modèle avec point. Les nouveaux
symboles 2026 utilisent sept ponts revus, bornés à la date effective de leur
événement S&P ou changement de ticker. Les symboles réutilisés conservent leurs
intervalles revus ; zéro ticker de l'univers demandé reste sans identité.

## Acquisition et lignée

| Élément | Valeur |
| --- | --- |
| CIK acquis | 743 |
| en-têtes SEC parsés | 11 249 |
| classifications après rattachement aux identités | 11 486 |
| classifications rejetées | 0 |
| échecs conservés | 54 : 48 en-têtes indisponibles, 6 sans SIC correspondant |
| manifeste DEF SHA-256 | `fae156d924d57d7830a94d752a2542a74a8d8ab17c16e697cc04514100ed5652` |
| Parquet secteur SHA-256 | `4a67a6ac6b1d137bac0ec18cc6e22d6d041345a74182a79e5534baada48884e6` |

Le RAW contient 2 905 reçus de soumissions pour 1 890 objets et 11 362 reçus
d'en-têtes pour 11 246 objets. Les reçus supplémentaires viennent des reprises
du téléchargement, tandis que les octets identiques ne sont stockés qu'une
fois par hash. Les deux manifestes passent leurs validations de contrat et
d'intégrité des objets.

## Couverture du replay existant

La validation réutilise exactement les prédictions Boosting du candidat du
31 octobre et le calendrier SPY du snapshot `e0e6045a…ef95a`. Elle applique le
registre terminal avant classement et le moteur
`boosting_sector_champions_top5_v1` sans modifier les scores.

| Variante | Mois sélectionnés | Candidats classés | Couverture | Secteurs minimum par mois |
| --- | ---: | ---: | ---: | ---: |
| Boosting natif | 183 / 183 | 87 974 / 89 944 | 97,81 % | 11 |
| Boosting tendance | 183 / 183 | 40 067 / 41 005 | 97,71 % | 9 |

Le premier mois contient 433 candidats classés sur 453 en natif et 189 sur 200
en tendance, avec onze secteurs. En septembre 2026, le natif couvre 499/500
candidats — seul `HONA` n'a encore aucun dépôt annuel admissible — et la
variante tendance couvre 161/161 candidats. Les candidats sans classification
causale sont exclus et listés dans l'audit mensuel ; ils ne reçoivent jamais le
secteur actuel par défaut.

Cette couverture n'est pas 100 %. Les lacunes longues viennent surtout de
réorganisations où le ticker a gardé son nom mais changé de CIK, ainsi que de
banques sans dépôt annuel SEC et de sociétés nouvellement créées avant leur
premier rapport annuel. Les combler exige des intervalles prédécesseur sourcés
ou une source réglementaire officielle supplémentaire, pas un backfill
statique.

## Commandes de validation

```bash
.venv/bin/python \
  scripts/open_source/ingestion/build_historical_sector_history.py \
  --ticker-universe <predictions.parquet> \
  --general-reference-lineage \
    data/sec/output/lineage/general_reference_lineage.parquet \
  --start-year 2009 --end-year 2026 \
  --run-id 20261109_sector_history_resume4

.venv/bin/python \
  scripts/experiments/portfolio/run_sector_champion_allocation.py \
  --boosting-run-dir <boosting-run-dir> \
  --sector-history-release \
    data/warehouse/def/sector_history/sector_history_55fbc0c519ea \
  --market-prices <SP500Price.parquet> \
  --output-dir <audit-output>
```

Le rollback consiste à ignorer cette release. Aucun pointeur `latest.json`,
snapshot modèle, portefeuille publié ou résultat historique existant n'a été
modifié.
