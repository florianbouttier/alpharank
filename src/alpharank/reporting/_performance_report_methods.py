"""Methodology cards shared by the standard performance-report profiles."""

from __future__ import annotations

from alpharank.reporting._performance_report_analytics import frontier_methodology_card
from alpharank.reporting._performance_report_sector import SectorReportData


def methodology_cards(sector: SectorReportData | None) -> list[dict[str, object]]:
    """Describe only the methods represented in the selected report profile."""

    cards = [_legacy_frequency_card(), _legacy_equal_card()]
    cards.extend(_boosting_cards())
    if sector is not None:
        cards.append(_sector_card(sector))
    return cards + [_spy_card(), _composer_card(), frontier_methodology_card()]


def _legacy_frequency_card() -> dict[str, object]:
    return {
        "title": "Legacy · Frequency",
        "status": "Méthode de production rejouée sur un snapshot candidat",
        "summary": "Trend-following relatif, quatre pistes walk-forward puis votes pondérés.",
        "pseudo_code": [
            "calculer les EMA relatives à SPY avec les prix connus à t",
            "optimiser chaque piste uniquement sur les mois antérieurs",
            "appliquer univers historique, liquidité et plafond sectoriel",
            "agréger les votes puis détenir le panier en t+1",
        ],
    }


def _legacy_equal_card() -> dict[str, object]:
    return {
        "title": "Legacy · Equal",
        "status": "Agrégation historique",
        "summary": "Même recherche Legacy, mais moyenne égale des quatre trajectoires.",
        "pseudo_code": [
            "produire les quatre portefeuilles Legacy causaux",
            "donner le même poids à chaque trajectoire",
            "agréger les poids par titre",
            "simuler avec le même calendrier et les mêmes frais",
        ],
    }


def _boosting_cards() -> list[dict[str, object]]:
    return [
        {
            "title": "Boosting natif",
            "status": "R&D comparable, non promu",
            "summary": "XGBoost classe les titres avec les EMA relatives causales de chaque fold.",
            "pseudo_code": [
                "geler les paires EMA disponibles au cutoff du fold",
                "entraîner, valider et calibrer sans toucher au test",
                "scorer toutes les lignes OOS du mois t",
                "prendre Top N avant de joindre le rendement de t+1",
            ],
        },
        {
            "title": "Boosting filtré par tendance",
            "status": "Diagnostic post-hoc, non promu",
            "summary": "Le score ne change pas ; une majorité stricte d’EMA positives borne l’univers.",
            "pseudo_code": [
                "reconstruire les ratios EMA causaux du fold",
                "exiger toutes les paires et une majorité strictement positive",
                "conserver le score Boosting natif inchangé",
                "classer puis prendre Top N dans l’univers filtré",
            ],
        },
    ]


def _sector_card(sector: SectorReportData) -> dict[str, object]:
    return {
        "title": "Boosting natif et tendance · Top 5 sectoriels",
        "status": "Recherche uniquement · pas de promotion",
        "summary": (
            "Exactement 5 actions à 20 % chacune, 1 maximum par secteur. "
            "Taxonomie AlphaRank issue des SIC SEC (pas GICS) : "
            + ", ".join(sector.allowed_sectors)
            + "."
        ),
        "pseudo_code": [
            "prendre le secteur SEC connu à la clôture de décision, sans backfill courant",
            "écarter les secteurs inconnus et appliquer le filtre tendance pour cette variante",
            "garder le meilleur score par secteur puis prendre les cinq meilleurs champions",
            "bloquer le mois s’il reste moins de cinq secteurs ; aucun remplissage avec un doublon",
        ],
    }


def _spy_card() -> dict[str, object]:
    return {
        "title": "SPY total return",
        "status": "Benchmark commun",
        "summary": "Benchmark dividendes réinvestis depuis adjusted_close, sans frais simulés.",
        "pseudo_code": [
            "résoudre SPY sur le même calendrier mensuel",
            "calculer le rendement total adjusted_close",
            "exclure le mois courant incomplet",
            "apparier chaque mois aux stratégies avant les KPI",
        ],
    }


def _composer_card() -> dict[str, object]:
    return {
        "title": "Portefeuille composé",
        "status": "Laboratoire post-hoc, non promu",
        "summary": "Équipondération mensuelle des stratégies cochées, comparée au SPY.",
        "pseudo_code": [
            "choisir au moins une poche de stratégie",
            "attribuer 1 / N du capital à chaque poche chaque mois",
            "moyenner leurs rendements nets sans coût inter-poche ajouté",
            "lire KPI, corrélations et richesse relative pré-calculés face au SPY",
        ],
    }
