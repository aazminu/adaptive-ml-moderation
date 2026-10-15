from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .leakage_safe_splitter import (
    ModerationEventRecord,
    compute_population_stability_index,
    split_temporal_thread_disjoint,
)
from .moderation_queue import PriorityModerationQueue
from .toxicity_classifier import HybridTfidfToxicityClassifier


def run_benchmark(seed: int = 42) -> List[Dict[str, Any]]:
    """Run reproducible hybrid TF-IDF toxicity classification, priority review queue, and leakage/PSI benchmarks."""
    _ = seed

    # 1. Hybrid char/word TF-IDF classifier & OOV online update
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
    risk_gap = abusive.risk_probability - benign.risk_probability

    oov_sample = "zzzxq99 novel obfuscated slur"
    oov_before = clf.score_text(oov_sample).risk_probability
    oov_after = clf.online_feedback_update(oov_sample, verified_label=1, weight=3.0)

    # 2. Priority moderation queue triage & human-validation safeguard
    q = PriorityModerationQueue(review_threshold=0.35, auto_block_threshold=0.80)
    q.ingest("m1", "th1", "u1", "clean comment", 0.12, 1000.0)
    q.ingest("m2", "th2", "u2", "suspicious comment A", 0.48, 1001.0)
    q.ingest("m3", "th2", "u3", "higher risk comment B", 0.72, 1002.0)
    q.ingest("m4", "th3", "u4", "severe slur attack", 0.94, 1003.0)
    top_review = q.pop_next_for_review()
    resolved = q.resolve_item("m3", "REJECTED", moderator_id="mod_alice")

    # 3. Leakage-safe thread-disjoint temporal split & PSI drift detection
    records = [
        ModerationEventRecord("e1", "thread_A", "u1", 100.0, "msg 1", 0),
        ModerationEventRecord("e2", "thread_A", "u2", 150.0, "msg 2", 0),
        ModerationEventRecord("e3", "thread_B", "u3", 120.0, "msg 3", 1),
        ModerationEventRecord("e4", "thread_C", "u4", 200.0, "msg 4", 0),
        ModerationEventRecord("e5", "thread_C", "u1", 210.0, "msg 5", 1),
    ]
    train_set, eval_set = split_temporal_thread_disjoint(records, train_fraction=0.67)
    disjoint = {r.thread_id for r in train_set}.isdisjoint({r.thread_id for r in eval_set})
    psi_val = compute_population_stability_index(
        [0.1, 0.25, 0.45, 0.65, 0.85],
        [0.82, 0.88, 0.91, 0.95, 0.97],
    )

    return [
        {
            "benchmark": "hybrid_tfidf_toxicity_classifier",
            "primary_metric": "abusive_benign_risk_gap",
            "score": round(risk_gap, 4),
            "secondary_metric": "abusive_risk_prob",
            "secondary_value": round(abusive.risk_probability, 4),
            "status": "SLA_OK" if abusive.latency_ms < 150.0 else "SLA_BREACH",
        },
        {
            "benchmark": "online_oov_feedback_adaptation",
            "primary_metric": "post_update_risk_prob",
            "score": round(oov_after, 4),
            "secondary_metric": "risk_delta",
            "secondary_value": round(oov_after - oov_before, 4),
            "status": "ADAPTED" if oov_after > oov_before else "STALE",
        },
        {
            "benchmark": "priority_moderation_queue_triage",
            "primary_metric": "top_popped_risk_score",
            "score": round(top_review.risk_score if top_review else 0.0, 4),
            "secondary_metric": "audit_trail_entries",
            "secondary_value": len(resolved.audit_trail),
            "status": resolved.status,
        },
        {
            "benchmark": "leakage_safe_split_and_psi_drift",
            "primary_metric": "population_stability_index",
            "score": round(psi_val, 4),
            "secondary_metric": "thread_disjoint",
            "secondary_value": int(disjoint),
            "status": "DRIFT_ALERT" if (psi_val > 0.25 and disjoint) else "STABLE",
        },
    ]


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "benchmark",
        "primary_metric",
        "score",
        "secondary_metric",
        "secondary_value",
        "status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible adaptive-ml-moderation classifier, queue, and drift benchmarks."
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write benchmark output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = run_benchmark(seed=args.seed)
    rendered = (
        json.dumps({"seed": args.seed, "benchmarks": rows}, indent=2)
        if args.format == "json"
        else format_markdown_table(rows)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
