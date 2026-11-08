# Replay sectoriel causal du 9 novembre 2026

## Verdict

La politique `boosting_sector_champions_top5_v1` fonctionne sans modifier les
scores : elle produit cinq positions équipondérées et cinq secteurs distincts
sur les sept décisions de mars à septembre 2026. Les 176 décisions antérieures
sont bloquées, car la première observation sectorielle conservée date du
28 mars 2026 ; elles ne sont pas remplies avec la classification actuelle.

Cette fenêtre est trop courte pour promouvoir la variante. Le Boosting natif
sectoriel perd 16,54 % sur les six mois réalisés, tandis que le Boosting tendance
sectoriel gagne 28,56 % contre 17,86 % pour SPY. Ces chiffres sont un diagnostic
avril-septembre 2026, pas un backtest long ni une recommandation.

## Entrées et conventions

| Élément | Valeur |
| --- | --- |
| prédictions | `replay_head3b17eb7/candidate/boosting_end_2026_09` |
| SHA-256 prédictions | `838e0246206940019e137c4e6b01a04fb38396f7949cbdbe370a9281a977da5f` |
| historique secteur | release `6c9a228b828bae35ea514388a259315f894a3e1748fb6b13a4e9e18ffeaabb9c` |
| SHA-256 Parquet secteur | `9e17835d37a555a4d492e740da6eb3bd0d89e4f13eacf73375d7133e1fc7c4aa` |
| prix calendrier | snapshot candidat `e0e6045a…ef95a` |
| période réalisée commune | holdings d'avril à septembre 2026, 6 mois |
| frais | 10 points de base multipliés par le turnover |
| benchmark | SPY total return contenu dans les mêmes prédictions |
| allocation | Top 5 après score, 20 % par titre, maximum un titre par secteur |
| secteur manquant | candidat exclu avant classement et listé dans l'audit |

La clôture de décision est la dernière séance SPY observée du mois à 16 h à
New York. Une classification est admissible seulement si
`max(observed_at, effective_at)` ne dépasse pas cet instant. Le registre des
événements terminaux v3 est appliqué avant le classement ; neuf candidats sont
bloqués sur l'historique complet.

## Couverture réelle

| Variante | Candidats | Mois sélectionnés | Mois bloqués | Positions |
| --- | ---: | ---: | ---: | ---: |
| Boosting | 89 944 | 7 | 176 | 35 |
| Boosting tendance | 41 005 | 7 | 176 | 35 |

En septembre, 497 des 500 candidats natifs possèdent un secteur causal et 159
des 161 candidats tendance. Les deux univers couvrent respectivement onze et
dix secteurs, donc le Top 5 contraint est réalisable. `FISV`, `P` et `BE` ne
sont pas complétés rétrospectivement pour cette décision.

## Portefeuille formé fin septembre pour octobre

Chaque ligne pèse 20 %. Le rendement d'octobre n'était pas mûr dans ce run.

| Rang | Boosting sectoriel | Secteur | Boosting tendance sectoriel | Secteur |
| ---: | --- | --- | --- | --- |
| 1 | MRNA | Healthcare | MRNA | Healthcare |
| 2 | LULU | Consumer Cyclical | DELL | Technology |
| 3 | DELL | Technology | MPC | Energy |
| 4 | CHTR | Communication Services | FDS | Financial Services |
| 5 | CSGP | Real Estate | META | Communication Services |

## Performance sur la seule fenêtre observable

| Stratégie | Rendement total | CAGR annualisé indicatif | Volatilité annualisée | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 5 natif | +20,12 % | +44,29 % | 51,18 % | -14,00 % |
| Boosting Top 5 sectoriel | -16,54 % | -30,35 % | 38,07 % | -21,85 % |
| Boosting tendance Top 5 | +105,97 % | +324,23 % | 82,98 % | -26,27 % |
| Boosting tendance sectoriel | +28,56 % | +65,27 % | 40,96 % | -18,65 % |
| SPY total return | +17,86 % | +38,91 % | 15,30 % | -1,03 % |

Les CAGR et volatilités annualisés sur six observations sont affichés pour
complétude mais ne sont pas stables. La contrainte réduit ici la volatilité de
la variante tendance et son drawdown, au prix d'une forte baisse du rendement ;
sur le Boosting natif, elle dégrade aussi le rendement et le drawdown. Aucun de
ces résultats ne justifie une promotion.

## Artefacts

- natif : `outputs/sector_champion_replay_20261109/native/`, holdings SHA-256
  `e40f1318effa106b6998ed69c845c2424be76153070ea5e33f5de345e08f73d1` ;
- tendance : `outputs/sector_champion_replay_20261109/trend/`, holdings SHA-256
  `121d54a4f45de6aa350ff0683980745b57383b6a6c18cfef9b336be56dd22a43` ;
- chaque dossier contient les positions, l'audit mensuel, les entrées terminales
  bloquées et un manifeste hashé.

Le rollback consiste à ignorer ces dossiers R&D. Aucun snapshot, profil public,
score ou portefeuille historique existant n'a été modifié.
