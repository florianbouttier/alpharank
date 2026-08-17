from __future__ import annotations

from pathlib import Path


def test_documentation_structure_maps_normative_rules_to_code_and_tests() -> None:
    root = Path(__file__).resolve().parents[1]
    governance = (root / "docs/research_governance.md").read_text(encoding="utf-8")

    required_contract_fragments = (
        "Contrat normatif `v2-causal`",
        "Temps et cible (`BST-001`, `BST-003`, `QA-001`)",
        "Prix et vintages (`PRC-001` à `PRC-003`)",
        "Univers, secteurs et fondamentaux",
        "Événements terminaux (`BST-002`, `SIM-001`)",
        "Exécution (`LEG-003`, `SIM-004`)",
        "Allocation et coûts (`SIM-002`, `SIM-003`)",
        "Limites, promotion et statut",
        "Replay et provenance (`GOV-003`, `QA-002`, `QA-003`)",
        "missing_return_policy=raise",
        "next_session_open_v1",
        "sec-filing-availability-v1",
        "tests/test_recomputable_replay.py::test_replay_recomputes_outputs_from_sealed_inputs",
    )
    for fragment in required_contract_fragments:
        assert fragment in governance

    for relative_path in (
        "docs/legacy_boosting_methodology.md",
        "docs/common_portfolio_backtest_engine.md",
        "docs/sec_fundamentals_contract.md",
        "docs/sec_data_robustness_plan.md",
        "docs/monthly_portfolio_runbook.md",
        "docs/methodology_audit_roadmap.md",
        "src/alpharank/replay_validation.py",
        "src/alpharank/portfolio/simulation.py",
        "src/alpharank/portfolio/execution.py",
    ):
        assert (root / relative_path).is_file(), relative_path
