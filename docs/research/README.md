# Recherche reproductible

Ce dossier contient le registre chronologique Boosting et des packages
d'expérience datés. Il ne remplace pas les contrats méthodologiques courants.

- `boosting_signal_copy_model_catalog.md` : synthèse courante et index des
  décisions Boosting/Legacy-copy.
- `boosting_signal_copy_log_20260614_20260627.md` : premiers diagnostics,
  modèles teacher et allocation autonome.
- `boosting_signal_copy_log_20260725_20260726.md` : correction anti-fuite et
  challengers verrouillés.
- `boosting_signal_copy_log_20260727_20260812.md` : dashboards, moteur commun et
  replays alignés.
- `boosting_signal_copy_log_20260820.md` : convention de clôture, SHAP et
  anomalie d'identité SNDK.
- `data_refresh_replay_20260824_20260825.md` : refresh intégral bloqué sur les
  révisions Yahoo et preuve d'identité des replays Legacy/Boosting sur le
  snapshot publié.
- `data_refresh_replay_20260827.md` : refresh intégral jusqu'au 26 août,
  replays same-code et ablation qui attribue le blocage `CVC.US` au package
  SEC candidat, sans promotion.
- `sats_echo_replay_20260829.md` : continuité SATS/ECHO sans valeur manuelle,
  replay Legacy/Boosting/tendance complet et résolution causale de Top 15/20.
- `backtest_performance_report_20260829.md` : première exécution du standard de
  performance sur SATS/ECHO, hashes du rapport et preuve de publication site.
- `backtest_performance_report_20260830.md` : republication comparative avec
  onze stratégies en colonnes, multiselect et model cards cumulées/annuelles
  strictement bornées par la fenêtre choisie.
- `backtest_performance_report_20260830_composer.md` : correction des vues pour
  qu'elles suivent les courbes affichées, drawdown pleine largeur et
  laboratoire équipondéré de 1 023 combinaisons comparées au SPY.

- `exact_legacy_ema_20260725/` : reproduction exacte du signal EMA Legacy.
- `legacy_ema_data_integrity_audit_20260726/` : audit d'identité et de prix.
- `legacy_ema_risk_overlay_long_history_20260725/` : premier overlay risque.
- `legacy_ema_risk_overlay_long_history_clean_v2_20260726/` : variante nettoyée.
- `locked_challenger_confirmation_20260725/` : confirmation du challenger figé.
- `multihorizon_boosting_20260725/` : package initial du Boosting multi-horizon.

Ces packages servent à reproduire une étape de recherche. Le statut actif des
modèles doit être résumé dans `boosting_signal_copy_model_catalog.md`, chaque
nouvelle preuve va dans un journal daté et la méthodologie courante reste dans
`../legacy_boosting_methodology.md`.
