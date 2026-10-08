# Revue des mouvements extrêmes BE du 8 octobre 2026

## Décision

Les quatre alertes `BE.US` du run `20261008_072222` sont des séances de marché
réelles et non des ruptures d'échelle. Elles sont approuvées uniquement dans les
bornes étroites de `configs/data_quality/reviewed_extreme_price_moves.json`.
Toute autre date, valeur ou rendement reste bloquant. Cette revue ne publie ni
snapshot ni portefeuille.

## Preuve acquise

Le contrôle porte sur le RAW immuable
`data/open_source/official/runs/20261008_072222/raw/prices_yfinance.parquet`, de
SHA-256 `10f5f24665701773bd6baa11a29a54f82a530d9a15b4879af884b0d8bdabe967`.
Le fichier exhaustif des alertes a le SHA-256
`8b9ff9af6e734a81c0424e65d44f56b68f5b2d280dacb394be6b1a6566088260`.

| Séance | Clôture précédente | OHLC | Clôture ajustée | Volume | Rendement |
| --- | ---: | --- | ---: | ---: | ---: |
| 2019-08-13 | 8,00 | 5,98 / 6,11 / 4,54 / 4,60 | 4,60 | 19 436 600 | -42,50 % |
| 2020-03-19 | 3,07 | 3,19 / 4,59 / 3,00 / 4,37 | 4,37 | 5 259 100 | +42,35 % |
| 2020-03-24 | 3,89 | 4,27 / 5,97 / 4,24 / 5,62 | 5,62 | 6 189 700 | +44,47 % |
| 2024-11-15 | 13,28 | 20,95 / 22,50 / 17,80 / 21,14 | 21,14 | 64 722 200 | +59,19 % |

Pour les quatre lignes, `close == adjusted_close`. Les ouvertures, plus hauts,
plus bas, clôtures et volumes décrivent des séances négociées ; aucun facteur de
split ou changement d'unité n'apparaît.

## Contexte public, sans sur-attribution

- Le 13 août 2019 suit les résultats T2 et la conférence investisseurs publiés
  par Bloom après la séance précédente. Le dépôt SEC correspondant confirme
  l'identité de l'émetteur et la période.
- Les 19 et 24 mars 2020 appartiennent à la volatilité exceptionnelle de mars.
  Les résultats annuels avaient été publiés le 16 mars et le travail de Bloom
  sur les ventilateurs était rapporté le 23 mars. Ces éléments établissent le
  contexte ; la revue n'attribue pas artificiellement tout le rendement à une
  annonce unique.
- Le 15 novembre 2024 suit l'annonce officielle, après la séance précédente,
  d'un accord avec AEP portant jusqu'à un gigawatt de piles à combustible.

Sources :

- [résultats T2 2019 de Bloom](https://investor.bloomenergy.com/press-releases/press-release-details/2019/Bloom-Energy-Announces-Second-Quarter-2019-Financial-Results/default.aspx) ;
- [dépôt SEC T2 2019](https://www.sec.gov/Archives/edgar/data/1664703/000166470319000023/0001664703-19-000023-index.html) ;
- [résultats T4 et annuels 2019 publiés le 16 mars 2020](https://investor.bloomenergy.com/press-releases/press-release-details/2020/Bloom-Energy-Announces-Fourth-Quarter-2019-Financial-Results/default.aspx) ;
- [10-K 2019 de Bloom](https://www.sec.gov/Archives/edgar/data/1664703/000166470320000013/0001664703-20-000013-index.htm) ;
- [article du 23 mars 2020 sur le travail de Bloom sur les ventilateurs](https://www.latimes.com/business/story/2020-03-23/coronavirus-california-companies-medical-supplies) ;
- [annonce Bloom/AEP du 14 novembre 2024](https://investor.bloomenergy.com/press-releases/press-release-details/2024/Bloom-Energy-Announces-Gigawatt-Fuel-Cell-Procurement-Agreement-with-AEP-to-Power-AI-Data-Centers/default.aspx) ;
- [dépôt SEC du 14 novembre 2024](https://www.sec.gov/Archives/edgar/data/1664703/000162828024046303/0001628280-24-046303-index.htm).

## Garantie de contrôle

Le registre compare la clôture précédente, la clôture courante et le rendement
à des bornes propres à chaque date. Il ne constitue donc ni une exemption par
ticker ni un relâchement du seuil global. Une future révision Yahoo qui sort de
ces bornes redeviendra automatiquement bloquante. Le run acquis, son échec
initial et ses quatre lignes non revues restent conservés comme preuve.

La republication différée sans réseau dans
`outputs/data_refresh_replay_20261008/data038_price_preflight` passe : cinq
mouvements sur cinq sont approuvés (les quatre BE et le RDDT déjà revu), zéro
mouvement reste non revu, zéro transition ou révision canonique est détectée et
les 1 200 483 clés EODHD attendues sont présentes. Le manifeste de ce préflight
a le SHA-256
`ec5389843fc12887ffd2fd4dbe23f172b74b91e255bec30c9faeffd1dfd0815d`.
