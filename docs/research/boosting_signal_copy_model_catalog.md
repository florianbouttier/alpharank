# Catalogue Boosting / Legacy-copy

**Rôle : synthèse courante et index des journaux de recherche.**

La méthodologie active est définie dans
[`../legacy_boosting_methodology.md`](../legacy_boosting_methodology.md). Les
contrats de simulation et de publication restent dans
[`../common_portfolio_backtest_engine.md`](../common_portfolio_backtest_engine.md)
et [`../research_governance.md`](../research_governance.md). Cette page ne les
duplique pas : elle résume les décisions de recherche durables et permet de
retrouver chaque essai historique.

## Statut courant

- Legacy reste la méthode mensuelle de production.
- Boosting reste un challenger R&D ; un CAGR supérieur ne suffit pas à le
  promouvoir lorsque le risque, la concentration ou la preuve temporelle sont
  insuffisants.
- Legacy et Boosting génèrent leurs signaux séparément. Leurs portefeuilles ne
  sont comparés qu'après résolution d'un même snapshot, des mêmes exclusions,
  du même calendrier, des mêmes coûts et du moteur commun.
- Le diagnostic du 2026-08-20 conserve la clôture comme convention AlphaRank et
  la prochaine ouverture comme sensibilité. Sa publication économique reste
  suspendue jusqu'à la matérialisation runtime `METH-001` / `LEG-005` et à la
  séparation de l'identité SNDK sous `METH-002` / `LIVE-022`.

Les priorités et leur statut sont exclusivement dans
[`../../ROADMAP.md`](../../ROADMAP.md). Les identifiants détaillés et preuves
historiques restent dans
[`../../METHODOLOGY_AUDIT_ROADMAP.md`](../../METHODOLOGY_AUDIT_ROADMAP.md).

## Conclusions durables

### Diagnostic Legacy et allocation autonome sont deux questions différentes

La recomposition de Legacy mesure si une représentation retrouve les mêmes
tickers. Elle ne prouve ni la prédiction du rendement futur ni la qualité d'un
portefeuille autonome.

Les modèles `distill_legacy`, les features `legacy_atomic_*` et les variables
issues des décisions Legacy restent donc des teachers ou contrôles de plafond.
Ils ne sont pas des candidats finaux de trading.

### Le challenger doit rester du Boosting causal

Le candidat autonome apprend une cible future définie avant l'allocation, avec
des features disponibles au mois de décision et des folds chronologiques. Il ne
doit pas optimiser une cible Legacy ni consulter la disponibilité du rendement
réalisé avant le classement.

Les témoins EMA déterministes prouvent qu'un signal exploitable existe. Leur
supériorité éventuelle diagnostique un objectif Boosting mal aligné ; elle ne
permet pas de remplacer silencieusement la méthode demandée.

### Le portefeuille juge plus que le rendement brut

Chaque challenger est lu avec au minimum rendement, CAGR, volatilité, Sharpe,
drawdown, turnover, concentration, couverture et différence appariée contre
Legacy et SPY. Une variante proche de Legacy en rendement mais nettement plus
risquée reste diagnostique.

### La comparabilité est un contrat de données et de temps

Une parité mécanique du simulateur ne suffit pas. La comparaison doit prouver
le snapshot, les hashes, l'univers point-in-time, les exclusions, le cutoff, la
maturité des cibles, les mois réalisés, les frais et le benchmark SPY total
return. Les runs construits sur deux snapshots restent conservés, mais ne sont
pas agrégés sous un même résultat économique.

## Derniers jalons

| Date | Lecture durable | Journal |
| --- | --- | --- |
| 2026-08-20 | diagnostic clôture/prochaine ouverture, SHAP exhaustif et blocage d'identité SNDK | [`boosting_signal_copy_log_20260820.md`](boosting_signal_copy_log_20260820.md) |
| 2026-08-12 | replay avec mêmes hashes, exclusions et filtre de prix ; Boosting conservé comme challenger non promu | [`boosting_signal_copy_log_20260727_20260812.md`](boosting_signal_copy_log_20260727_20260812.md) |
| 2026-07-25–26 | correction de la fuite teacher, challenger verrouillé, audit d'identité et overlays de risque | [`boosting_signal_copy_log_20260725_20260726.md`](boosting_signal_copy_log_20260725_20260726.md) |
| 2026-06-14–27 | diagnostics de recomposition, familles de modèles et séparation entre teacher Legacy et allocation autonome | [`boosting_signal_copy_log_20260614_20260627.md`](boosting_signal_copy_log_20260614_20260627.md) |

## Règles de lecture des journaux

- Un run marqué invalide reste conservé comme preuve d'incident ; ses métriques
  ne redeviennent pas utilisables parce qu'elles figurent dans un journal.
- Un chiffre appartient au run, snapshot, calendrier, univers et coût indiqués
  dans son entrée. Il n'est pas automatiquement le chiffre courant.
- Une expérience datée peut expliquer une décision ; elle ne remplace pas les
  contrats méthodologiques canoniques.
- Les chemins `outputs/` sont des références de replay locales, pas des données
  à recopier dans Git.

## Ajouter une expérience

1. enregistrer hypothèse, configuration, snapshot, commande, run et résultat
   dans un journal daté ;
2. conserver les résultats négatifs et incidents avec leur statut explicite ;
3. modifier cette synthèse seulement si la conclusion durable ou la prochaine
   décision change ;
4. mettre à jour le contrat canonique dans la même tâche lorsqu'une méthode ou
   une règle de publication change.
