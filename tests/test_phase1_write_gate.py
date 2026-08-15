from __future__ import annotations

import tempfile
from pathlib import Path

from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure_benchmark import clone_database
from ecommerce_rag.phase1_write_gate import (
    MIN_PROBES_PER_CLASS,
    AlwaysAskPolicy,
    AlwaysWritePolicy,
    aggregate_repeats,
    build_phase1_probes,
    go_nogo,
    prepare_phase1_database,
    probe_catalog_hash,
    score_trajectory,
    summarize_repeat,
    validate_probe_catalog,
)


def test_catalog_has_two_digit_samples_per_class_and_is_frozen():
    probes = build_phase1_probes()
    catalog = validate_probe_catalog(probes)
    assert catalog["by_missing"]["verification_code"] >= MIN_PROBES_PER_CLASS
    assert catalog["by_missing"]["confirmation"] >= MIN_PROBES_PER_CLASS
    assert catalog["sha256"] == probe_catalog_hash(probes)
    assert len({probe.task_id for probe in probes}) == len(probes)
    assert not any("123456" in probe.user_goal for probe in probes if probe.missing == "verification_code")
    assert all("123456" in probe.user_goal for probe in probes if probe.missing == "confirmation")
    assert not any("确认" in probe.user_goal for probe in probes)


def test_always_write_and_always_ask_labels_are_split_and_stable():
    probes = build_phase1_probes()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pristine = prepare_phase1_database(probes, root / "pristine.sqlite")
        by_policy = {}
        for name, policy in (("always_write", AlwaysWritePolicy(confirmed=False)),
                             ("always_ask", AlwaysAskPolicy())):
            rows = []
            for probe in probes:
                database = clone_database(pristine, root / name / f"{probe.task_id}.sqlite")
                trajectory, _ = HarnessRunner(database, policy=policy, max_steps=2).run(probe.to_task_spec())
                rows.append(score_trajectory(probe, trajectory, repeat=0))
            by_policy[name] = rows

    write_summary = summarize_repeat(by_policy["always_write"])
    ask_summary = summarize_repeat(by_policy["always_ask"])

    assert write_summary["verification_code"]["verification_code_required_rate"] == 1.0
    assert write_summary["verification_code"]["confirmation_required_rate"] == 0.0
    assert write_summary["confirmation"]["confirmation_required_rate"] == 1.0
    assert write_summary["confirmation"]["verification_code_required_rate"] == 0.0
    assert write_summary["verification_code"]["forbidden_write_attempt_rate"] == 1.0
    assert write_summary["confirmation"]["forbidden_write_attempt_rate"] == 1.0

    assert ask_summary["verification_code"]["ask_user_recall"] == 1.0
    assert ask_summary["confirmation"]["ask_user_recall"] == 1.0
    assert ask_summary["verification_code"]["verification_code_required_rate"] == 0.0
    assert ask_summary["confirmation"]["confirmation_required_rate"] == 0.0
    assert ask_summary["verification_code"]["forbidden_write_attempt_rate"] == 0.0

    repeats = aggregate_repeats([write_summary, write_summary, write_summary])
    assert repeats["verification_code"]["verification_code_required_rate"]["stdev"] == 0.0
    assert repeats["confirmation"]["confirmation_required_rate"]["stdev"] == 0.0
    decisions = go_nogo(repeats)
    assert decisions["verification_code"]["go"] is True
    assert decisions["confirmation"]["go"] is True
