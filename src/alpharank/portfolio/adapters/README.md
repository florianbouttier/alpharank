# Adaptateurs de stratégies

- `legacy.py` convertit les portefeuilles Legacy finalisés.
- `boosting.py` convertit les rangs/scores Boosting et leur allocation.

Les adaptateurs ne modifient pas le signal. Ils normalisent uniquement le
schéma, les dates et les poids avant la simulation partagée.
