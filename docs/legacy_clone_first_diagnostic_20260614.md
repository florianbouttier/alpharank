# Diagnostic clonage legacy - 2026-06-14

## Changement de priorite

Avant d'essayer de generaliser ou de battre legacy, il faut d'abord verifier qu'un modele ML sait reproduire ses signaux et ses trades.

J'ai donc relu le run des 7 variantes comme un probleme de clonage :

- pour chaque mois, compter le nombre exact de lignes legacy `K` ;
- prendre les `K` meilleurs tickers du modele ;
- mesurer combien de trades legacy sont retrouves.

Cette metrique est plus importante que l'AUC et meme plus importante que le rendement a ce stade.

## Run utilise

- run multi-modeles : `outputs/signal_copy_models_20260614_214711`
- diagnostic clone : `outputs/legacy_clone_diagnostics_20260614`
- source legacy : `outputs/2026-06-07/legacy_detailed_returns_polars.parquet`
- fenetre : holdings `2025-06` a `2026-04`

## Les bonnes metriques

Pour prouver qu'on copie legacy, les metriques principales doivent etre :

1. `recall@legacy_k` : part des vrais trades legacy retrouves quand le modele choisit le meme nombre de lignes.
2. `precision@legacy_k` : part des trades du modele qui sont bien des trades legacy. Ici elle est egale au recall car on selectionne exactement `K`.
3. `jaccard@legacy_k` : intersection / union des paniers.
4. `monthly stability` : recall mois par mois, pas seulement moyenne.
5. `rank of legacy names` : le modele met-il les actions legacy dans son haut de panier ?
6. `clone return` : sanity check, pas metrique primaire. Un clone peut temporairement surperformer sans copier correctement.

Seuils pragmatiques :

- `recall@legacy_k > 70%` : clone exploitable.
- `recall@legacy_k 50%-70%` : piste serieuse mais pas encore clone.
- `recall@legacy_k < 50%` : le modele capte un signal voisin, pas legacy.

## Resultats clone stricts

| modele | recall@legacy_k | jaccard | trades retrouves | clone return | actif compose | mois actifs gagnants |
|---|---:|---:|---:|---:|---:|---:|
| distill_legacy | 53.5% | 37.2% | 38 / 71 | +143.8% | +104.4% | 9 / 11 |
| blend_ema_distill | 50.7% | 34.8% | 36 / 71 | +142.2% | +103.1% | 8 / 11 |
| rank_pairwise | 16.4% | 9.4% | 12 / 71 | +48.9% | +22.6% | 8 / 11 |
| ema_residual_benchmark | 15.0% | 8.8% | 11 / 71 | +16.8% | -4.5% | 7 / 11 |
| two_stage_ema_full | 10.9% | 6.3% | 8 / 71 | +18.7% | -2.6% | 6 / 11 |
| gated_full_after_ema | 5.5% | 3.0% | 4 / 71 | +10.7% | -9.2% | 4 / 11 |
| regression_excess | 5.2% | 2.9% | 4 / 71 | +31.5% | +7.9% | 7 / 11 |
| monotone_ema_full | 4.2% | 2.4% | 3 / 71 | +11.9% | -8.3% | 3 / 11 |
| weighted_top_classifier | 0.0% | 0.0% | 0 / 71 | +9.5% | -10.4% | 4 / 11 |

Le gagnant clair est `distill_legacy`.

## Resultats top 10 allocation

Quand on prend top 10 fixe par mois :

| modele | total return | actif compose | overlap legacy moyen | hit-rate +5% |
|---|---:|---:|---:|---:|
| distill_legacy | +100.3% | +67.5% | 43.6% | 44.5% |
| blend_ema_distill | +99.1% | +66.5% | 46.4% | 45.5% |
| ema_residual_benchmark | +61.4% | +33.3% | 16.4% | 34.5% |
| rank_pairwise | +45.9% | +20.1% | 19.1% | 34.5% |
| regression_excess | +33.9% | +9.6% | 18.2% | 40.0% |
| two_stage_ema_full | +29.0% | +5.8% | 11.8% | 31.8% |
| monotone_ema_full | +20.7% | -0.7% | 9.1% | 34.5% |
| weighted_top_classifier | +16.8% | -4.1% | 0.9% | 36.4% |
| gated_full_after_ema | +14.8% | -5.7% | 9.1% | 32.7% |

Le modele qui copie le mieux legacy est aussi celui qui performe le mieux. C'est le signal important.

## Lecture

Le modele doit apprendre `legacy_selected`, pas seulement `future_excess_return > 5%`.

Le legacy contient un choix implicite de :

- momentum / EMA ;
- agressivite top-of-ranking ;
- filtres de valorisation/univers ;
- probablement contraintes sectorielles et nombre de votes modeles.

Quand on entraine explicitement une distillation legacy, le modele commence a recuperer les trades.

## Prochaine etape

Industrialiser le clonage legacy avant toute generalisation :

1. Construire un dataset `legacy_teacher_frame.parquet` avec :
   - `legacy_selected`;
   - `legacy_n_models`;
   - `legacy_weight_normalized`;
   - `legacy_rank_in_month`;
   - features decision-month.
2. Entrainer trois teachers :
   - classification `legacy_selected`;
   - regression/ranking `legacy_n_models`;
   - regression `legacy_weight_normalized`.
3. Evaluer en walk-forward avec :
   - `recall@legacy_k`;
   - `jaccard@legacy_k`;
   - erreur absolue de poids;
   - clone return.
4. Seulement ensuite ajouter la target de rendement futur pour generaliser.

Objectif court terme : passer `recall@legacy_k` de 53.5% a plus de 70%.
