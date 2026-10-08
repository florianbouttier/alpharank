# Univers S&P 500 reconstruit au 8 octobre 2026

## Objet

Le refresh d'octobre ne pouvait pas partir du calendrier local précédent : le
dernier snapshot présent était septembre et le registre versionné s'arrêtait
aux événements du 18 août. `DATA-035` ajoute donc les changements officiels
connus avant le téléchargement des prix et des fondamentaux.

## Sources et événements

- annonce S&P du 4 septembre : BE, P et ILMN entrent le 21 septembre ; TAP,
  TTD et BLDR sortent ;
- annonce S&P du 1er octobre : VYLR est ajouté le 1er octobre ; CTVA et WBD
  sortent le 6 octobre et TWLO entre à cette date.

Sources officielles :

- <https://press.spglobal.com/2026-09-04-Bloom-Energy,-Illumina,-and-Everpure-Set-to-Join-S-P-500-Others-to-Join-S-P-100,-S-P-MidCap-400,-and-S-P-SmallCap-600>
- <https://press.spglobal.com/2026-10-01-Vylor-Added-to-the-S-P-500-Twilio-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600>

L'annonce VYLR ne donne pas d'heure de publication. Conformément au contrat de
lignée, `observed_at` est fixé prudemment à `2026-10-01T23:59:59-04:00` et
`effective_at` à ce même instant : aucune décision antérieure dans la journée
ne peut voir l'événement.

## Résultat

La commande canonique a reconstruit les mois d'avril à octobre depuis le
snapshot de base et le registre complet :

```text
run id : 20261008_091656
septembre : 503 titres
octobre : 503 titres
ajouts septembre : BE, P, ILMN
retraits septembre : TAP, TTD, BLDR
ajouts octobre : VYLR, TWLO
retraits octobre : CTVA, WBD
```

Le fichier `data/SP500_Constituents.csv` passe de
`b8c7b8a618e9c8d00a25cc9d56b8b5a6e1f14049c63b5e8ff1d380751135e209`
à `5dadfa090b08f786fed071b6c86fe3584b75f050f949ed643f9bd80dc0680aa1`.
Le registre versionné porte le hash
`7dc7bbed328832d3c5276f328c09ded460c0e31c783227f4158d88075c13aa56`.

Les preuves générées restent hors Git sous
`outputs/data_refresh_20261008/constituents/` :

| Artefact | SHA-256 |
| --- | --- |
| `constituent_refresh_manifest.json` | `982213423ef878e990722effcc134f6101b6a63965192a113614550b49cfa97d` |
| `html/constituent_refresh_audit.html` | `a036ad1729cc973929f2b0dbee83665d14f02e983605146099149d51f94c06d4` |

Cette tâche ne télécharge encore aucune donnée fournisseur et ne produit aucun
portefeuille. Elle corrige l'univers d'entrée requis par le refresh complet qui
suit.
