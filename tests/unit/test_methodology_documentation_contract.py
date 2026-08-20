from __future__ import annotations

from pathlib import Path


def _load_markdown_documents(root: Path) -> dict[Path, str]:
    paths = (*root.glob("*.md"), *(root / "docs").rglob("*.md"))
    return {path: path.read_text(encoding="utf-8") for path in sorted(paths)}


def _require_unique_document(
    documents: dict[Path, str],
    *,
    label: str,
    fragments: tuple[str, ...],
) -> str:
    matches = [
        path
        for path, content in documents.items()
        if all(fragment in content for fragment in fragments)
    ]
    assert len(matches) == 1, f"{label}: expected one document, found {matches}"
    return documents[matches[0]]


def test_documentation_structure_maps_normative_rules_to_code_and_tests() -> None:
    root = Path(__file__).resolve().parents[2]
    documents = _load_markdown_documents(root)

    required_contract_fragments = (
        "Contrat normatif `v2-causal`",
        "Temps et cible (`BST-001`, `BST-003`, `QA-001`)",
        "Prix et vintages (`PRC-001` à `PRC-003`)",
        "Univers, secteurs et fondamentaux",
        "Événements terminaux (`BST-002`, `SIM-001`)",
        "Exécution (`LEG-003`, `LEG-005`, `SIM-004`)",
        "Allocation et coûts (`SIM-002`, `SIM-003`)",
        "Limites, promotion et statut",
        "Replay et provenance (`GOV-003`, `QA-002`, `QA-003`)",
        "missing_return_policy=raise",
        "reference_close_adjusted_close_v1",
        "next_session_open_v1",
        "sec-filing-availability-v1",
        "tests/replay/test_recomputable_replay.py::test_replay_recomputes_outputs_from_sealed_inputs",
    )
    _require_unique_document(
        documents,
        label="research governance contract",
        fragments=required_contract_fragments,
    )

    required_document_contracts = (
        (
            "Legacy and Boosting methodology",
            ("# Legacy And Boosting Methodologies", "source of truth for signal generation"),
        ),
        (
            "common portfolio engine",
            ("# Common Portfolio And Backtest Engine", "source of truth for the code shared"),
        ),
        (
            "SEC fundamentals",
            ("# SEC Fundamentals Contract", "une seule source de verite pour les fondamentaux"),
        ),
        (
            "SEC robustness",
            ("# SEC Data Robustness Plan", "non-replayable monthly"),
        ),
        (
            "monthly runbook",
            ("# Monthly Portfolio Runbook", "canonical runbook for the monthly"),
        ),
        (
            "methodology audit registry",
            (
                "# Registre détaillé de l'audit méthodologique AlphaRank",
                "registre conserve sans suppression",
            ),
        ),
    )
    for label, fragments in required_document_contracts:
        _require_unique_document(documents, label=label, fragments=fragments)

    for relative_path in (
        "src/alpharank/replay_validation.py",
        "src/alpharank/portfolio/simulation.py",
        "src/alpharank/portfolio/execution.py",
    ):
        assert (root / relative_path).is_file(), relative_path
