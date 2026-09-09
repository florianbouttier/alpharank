# Refresh et replay causal du 9 septembre 2026

## Verdict

Le refresh complet acquis par le run `20260908_002341`, puis composé dans le
snapshot candidat `35f0244f39cc8afffaf1af08886d7f3c1ab8a2fdf4ba5043782118eba7978421`,
a été rejoué sur Legacy, Boosting, le moteur commun et la variante tendance.
Les replays baseline, prix-seuls, SEC-seuls et candidat complet utilisent le
même code `424a3b475950163c17e5582ed0c47fc8e2add9c1`, les mêmes configurations et
le même runtime.

Aucune contamination future n'est observée. Le portefeuille décidé le
31 juillet pour détention en août est identique sur les 80 lignes Legacy et
Boosting : zéro ajout, zéro retrait, zéro poids modifié et différence numérique
maximale nulle. Le portefeuille décidé fin août est présent dans le candidat et
dans le rapport de performance, mais il n'existait pas dans l'ancien snapshot :
il n'est donc pas présenté comme une comparaison historique ni assorti d'un
rendement de septembre inventé.

Le statut brut au cutoff mûr de juin est `unexplained_portfolio_drift`, comme
l'exige le fail-closed avant attribution. Les trois ablations le transforment
en `explained_data_drift` avec les contrôles Legacy, Boosting et additivité des
portefeuilles tous vrais. Ce statut exige encore une revue humaine et
`promotion_allowed=false` ; aucun pointeur de production n'a bougé.

## Périodes et fraîcheur

- dernière séance réellement acquise pour les prix et les filings :
  4 septembre 2026 ;
- raison : le run a démarré avant l'ouverture du 8 septembre et le 7 septembre
  était férié aux États-Unis ;
- cutoff historique mûr commun : décision du 1er juin, détention de juin ;
- dernier portefeuille déjà commun aux deux snapshots : décision du 1er
  juillet, formée fin juillet et détenue en août ;
- nouveau portefeuille courant : décision du 1er août, formée fin août et
  détenue en septembre.

La convention des replays encode le mois de décision au premier jour du mois
qui vient de finir. Ainsi `decision_month=2026-07-01` désigne le portefeuille
formé à la clôture de juillet, et non un portefeuille arrêté au 1er juillet.

## Durée observée

L'acquisition intégrale a duré environ 19 h 22, du 8 septembre 02:23:41 au
8 septembre 21:45:53, heure de Paris. Cette mesure est l'estimation réaliste
pour un nouveau téléchargement complet dans des conditions réseau semblables.
Les replays finaux séquentiels et les audits ont ensuite demandé environ
1 h 30. Une première tentative parallèle a saturé le disque de swap macOS ; ses
sorties incomplètes sont conservées mais n'entrent dans aucun résultat ci-dessous.

## Comparaison causale

| Scénario | Legacy | Scores Boosting | Portefeuilles communs | Lecture |
| --- | ---: | ---: | ---: | --- |
| baseline | référence | référence | référence | snapshot publié |
| prix candidats, SEC baseline | 4 entrées, 4 sorties, 4 poids | 13 scores modifiés | 10 entrées, 10 sorties, 4 poids | effet univers/prix |
| prix baseline, SEC candidat | aucune différence | aucune différence | 99 entrées, 99 sorties, zéro poids canonique | variantes R&D PE seulement |
| candidat complet | 4 entrées, 4 sorties, 4 poids | 13 scores modifiés | 109 entrées, 109 sorties, 4 poids | somme exacte des deux effets |

Les 1 077 lignes Boosting initialement signalées comme « modifiées » incluent
des colonnes descriptives et des effets de rang. Seuls 13 scores changent
réellement, soit 0,0147 % des 88 442 prédictions communes ; le scénario
SEC-seul en modifie zéro. Les 99 changements SEC-only du moteur commun se
limitent aux variantes de recherche `Legacy PE universe`, non aux stratégies
canoniques sans fondamentaux SEC.

Les changements canoniques de juin viennent du calendrier S&P point-in-time :
`ECHO`, `FLEX`, `HONA` et `MRVL` entrent, tandis que `SATS`, `CAG`, `CPB`,
`EPAM` et `POOL` sortent. Le registre local source montre que chaque événement
était observé avant sa date d'effet et avant la décision concernée : il s'agit
d'une correction causale de l'univers, pas d'une information future. Sur les
clés prix communes jusqu'au cutoff, aucune valeur action ou SPY n'a changé.

## Portefeuilles récents

Le contrôle machine du portefeuille fin juillet porte sur 80 positions :
30 lignes Legacy réparties entre `Combined_Equal` et `Combined_Frequency`, plus
50 lignes Boosting. Les 15 titres Legacy sont `AMAT`, `AMD`, `DDOG`, `DELL`,
`FLEX`, `FTNT`, `HPE`, `HUM`, `INTC`, `MRVL`, `MU`, `PANW`, `SNDK`, `STX` et
`WDC`. Les deux snapshots donnent exactement les mêmes titres et poids.

Le rapport de performance candidat expose en plus toutes les positions formées
fin août : 14 titres par allocation Legacy et 100 lignes Boosting couvrant les
Top 5/10/15/20 natifs et filtrés par tendance. Ses courbes réalisées restent
arrêtées en août et le portefeuille de septembre est clairement marqué comme
courant/non réalisé.

## Artefacts et hashes

- racine du replay :
  `outputs/data_refresh_replay_20260908/replay_20260909_head424a3b4` ;
- audit machine brut :
  `audit_common_cutoff_june/refresh_replay_report.json`, SHA-256
  `3805cff27e8c10a4398a37de7d339cda2f8e62aa0dd72885964f4b6067864b31` ;
- attribution finale :
  `audit_common_cutoff_june/refresh_replay_attribution.json`, SHA-256
  `b017a40d71dadf106fbfacdb93e2f7afeb32e53bf79df0d2afc1c54a36f19ce3` ;
- rapport HTML causal :
  `audit_common_cutoff_june/refresh_replay_report.html`, SHA-256
  `2df8fcc008f521d1fcae013ca322a46ce6f4e52d980a12bf22b1d27b078b3087` ;
- rapport de performance et portefeuille fin août :
  `outputs/performance_reports/data_refresh_20260909/backtest_performance_report.html`,
  SHA-256
  `632961099368de7f02f82a058bd10612f7bdfd70c530c0d5fe7d023519f3353d` ;
- manifeste du rapport de performance : SHA-256
  `887bf07257f99d7bfe485ea3287947d1297ee84205e43418b1a615730a6dc950`.

Le pointeur canonique `data/model_inputs/manifests/latest.json` est resté
byte-identique, SHA-256
`5c2d0ec0a6cd716543e03b3caa5662d8c0f096d048c371b1c29ef2ad411642e4`.

## Validations techniques

- 17 tests ciblés du drift et du rapport : tous verts ;
- 44 tests marqués `replay` : tous verts ;
- suite complète : 540 tests verts, 269 avertissements de dépréciation ;
- Ruff sur les fichiers modifiés, inventaires code/dossiers/data, documentation
  et liens Markdown : verts ;
- parse HTML, syntaxe JavaScript et absence de ressource réseau sur les deux
  rapports : verts.

La gate différentielle de taille ne trouve aucune régression dans les fichiers
de `REPLAY-008`. Elle reste globalement rouge sur deux dettes SEC hors de ce
diff (`_sec_explorer_html.py` et `build_sec_output_package.py`). Le périmètre
Mypy existant reste également rouge sur un typage de
`terminal_event_registry.py`, sans rapport avec les replays exécutés.

## Risque restant et décision de promotion

Le candidat conserve volontairement les révisions historiques SEC reçues. Elles
n'affectent ni Legacy canonique ni les scores Boosting EMA-only, mais elles
modifient les variantes R&D fondées sur l'univers PE. Elles doivent donc être
revues comme données historiques avant toute promotion générale du snapshot.
Le replay prouve l'absence de contamination non expliquée ; il ne transforme
pas automatiquement une révision fournisseur en vérité de production.
