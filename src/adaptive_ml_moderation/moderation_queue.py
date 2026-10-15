from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModerationItem:
    item_id: str
    thread_id: str
    author_id: str
    text: str
    risk_score: float
    created_at_epoch: float
    triage_band: str
    status: str = "PENDING"
    moderator_id: Optional[str] = None
    audit_trail: List[str] = field(default_factory=list)


class PriorityModerationQueue:
    """Three-band triage router (AUTO_ALLOW, HUMAN_REVIEW, AUTO_BLOCK) with max-risk priority heap."""

    def __init__(self, review_threshold: float = 0.38, auto_block_threshold: float = 0.82) -> None:
        if not (0.0 < review_threshold < auto_block_threshold < 1.0):
            raise ValueError("Thresholds must satisfy 0 < review_threshold < auto_block_threshold < 1")
        self.review_threshold = review_threshold
        self.auto_block_threshold = auto_block_threshold
        self._heap: List[tuple[float, int, str]] = []
        self._seq: int = 0
        self._items: Dict[str, ModerationItem] = {}

    def ingest(
        self,
        item_id: str,
        thread_id: str,
        author_id: str,
        text: str,
        risk_score: float,
        created_at_epoch: float,
    ) -> ModerationItem:
        if item_id in self._items:
            raise ValueError(f"Duplicate item_id: {item_id}")

        if risk_score >= self.auto_block_threshold:
            band = "AUTO_BLOCK"
            status = "REJECTED"
            trail = [f"auto_blocked(score={risk_score:.3f})"]
        elif risk_score >= self.review_threshold:
            band = "HUMAN_REVIEW"
            status = "PENDING"
            trail = [f"queued_for_review(score={risk_score:.3f})"]
        else:
            band = "AUTO_ALLOW"
            status = "APPROVED"
            trail = [f"auto_allowed(score={risk_score:.3f})"]

        item = ModerationItem(
            item_id=item_id,
            thread_id=thread_id,
            author_id=author_id,
            text=text,
            risk_score=risk_score,
            created_at_epoch=created_at_epoch,
            triage_band=band,
            status=status,
            audit_trail=trail,
        )
        self._items[item_id] = item
        if band == "HUMAN_REVIEW":
            self._seq += 1
            heapq.heappush(self._heap, (-risk_score, self._seq, item_id))
        return item

    def pop_next_for_review(self) -> Optional[ModerationItem]:
        while self._heap:
            _, _, item_id = heapq.heappop(self._heap)
            item = self._items[item_id]
            if item.status == "PENDING":
                return item
        return None

    def resolve_item(self, item_id: str, decision: str, moderator_id: str) -> ModerationItem:
        if decision not in ("APPROVED", "REJECTED", "ESCALATED"):
            raise ValueError(f"Unsupported decision: {decision}")
        if item_id not in self._items:
            raise KeyError(f"Unknown item_id: {item_id}")
        item = self._items[item_id]
        item.status = decision
        item.moderator_id = moderator_id
        item.audit_trail.append(f"moderator:{moderator_id}->{decision}")
        return item
