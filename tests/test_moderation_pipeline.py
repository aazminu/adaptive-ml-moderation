from __future__ import annotations

import pytest

from adaptive_ml_moderation import (
    HybridTfidfToxicityClassifier,
    ModerationEventRecord,
    PriorityModerationQueue,
    compute_population_stability_index,
    split_temporal_thread_disjoint,
)


def test_hybrid_tfidf_classifier_latency_and_obfuscated_abuse() -> None:
    clf = HybridTfidfToxicityClassifier(char_ngram_range=(3, 5), learning_rate=0.8, epochs=90)
    train_texts = [
        "thank you for sharing this helpful patch",
        "great explanation of the system architecture",
        "welcome to the community discussion thread",
        "let us review the benchmark numbers tomorrow",
        "you are a worthless id1ot go kill yourself",
        "stupid trash scammer click free crypto spam now",
        "worthless id1ot spammer get out of here",
        "scammer trash free crypto spam link",
    ]
    train_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    clf.fit(train_texts, train_labels)

    benign = clf.score_text("thank you for the helpful system benchmark")
    abusive = clf.score_text("worthless id1ot scammer free crypto spam")
    assert benign.risk_probability < 0.35
    assert abusive.risk_probability > 0.65
    # Verify <150ms real-time SLA
    assert benign.latency_ms < 150.0
    assert abusive.latency_ms < 150.0
    assert len(abusive.top_contributing_features) > 0


def test_online_feedback_update_adjusts_risk_score() -> None:
    clf = HybridTfidfToxicityClassifier(epochs=60)
    clf.fit(
        ["hello friend nice post", "awful toxic spam attack"],
        [0, 1],
    )
    sample = "borderline phrase needing moderator override"
    before = clf.score_text(sample).risk_probability
    after = clf.online_feedback_update(sample, verified_label=1, weight=3.0)
    assert after > before


def test_priority_moderation_queue_triage_and_audit_trail() -> None:
    q = PriorityModerationQueue(review_threshold=0.35, auto_block_threshold=0.80)
    low = q.ingest("m1", "th1", "u1", "clean comment", 0.12, 1000.0)
    mid1 = q.ingest("m2", "th2", "u2", "suspicious comment A", 0.48, 1001.0)
    mid2 = q.ingest("m3", "th2", "u3", "higher risk comment B", 0.72, 1002.0)
    high = q.ingest("m4", "th3", "u4", "severe slur attack", 0.94, 1003.0)

    assert low.triage_band == "AUTO_ALLOW" and low.status == "APPROVED"
    assert high.triage_band == "AUTO_BLOCK" and high.status == "REJECTED"
    assert mid1.triage_band == "HUMAN_REVIEW" and mid2.triage_band == "HUMAN_REVIEW"

    # Highest risk HUMAN_REVIEW item must pop first
    first_pop = q.pop_next_for_review()
    assert first_pop is not None and first_pop.item_id == "m3"
    resolved = q.resolve_item("m3", "REJECTED", moderator_id="mod_alice")
    assert resolved.status == "REJECTED"
    assert any("mod_alice->REJECTED" in entry for entry in resolved.audit_trail)


def test_leakage_safe_splitter_and_psi_drift() -> None:
    records = [
        ModerationEventRecord("e1", "thread_A", "u1", 100.0, "msg 1", 0),
        ModerationEventRecord("e2", "thread_A", "u2", 150.0, "msg 2", 0),
        ModerationEventRecord("e3", "thread_B", "u3", 120.0, "msg 3", 1),
        ModerationEventRecord("e4", "thread_C", "u4", 200.0, "msg 4", 0),
        ModerationEventRecord("e5", "thread_C", "u1", 210.0, "msg 5", 1),
    ]
    train_set, eval_set = split_temporal_thread_disjoint(records, train_fraction=0.67)
    train_threads = {r.thread_id for r in train_set}
    eval_threads = {r.thread_id for r in eval_set}
    assert train_threads.isdisjoint(eval_threads)
    assert "thread_C" in eval_threads

    # Identical distributions -> PSI == 0
    base = [0.1, 0.25, 0.45, 0.65, 0.85]
    assert compute_population_stability_index(base, base) == pytest.approx(0.0, abs=1e-5)
    # Shifted distribution -> PSI > 0.25 (significant drift)
    drifted = [0.82, 0.88, 0.91, 0.95, 0.97]
    assert compute_population_stability_index(base, drifted) > 0.25
