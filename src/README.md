# Code source

- `alpharank/` : package Python actif installé avec le layout `src/`.
- `alpharank.egg-info/` : métadonnées générées par l'installation éditable ; ne
  pas les modifier manuellement.

Les scripts importent toujours `alpharank.*`. La logique métier ne doit pas être
dupliquée dans `scripts/`.
