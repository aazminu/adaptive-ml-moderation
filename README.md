# Adaptive ML Content Moderation (`adaptive-ml-moderation`)

Real-time (`<150 ms` p99 latency) content moderation, priority triage queue, leakage-safe temporal/thread evaluation splitter, and human-in-the-loop active feedback retraining system.

## Architecture & Modules

1. **`toxicity_classifier.py`**:
   - Hybrid **Word n-gram (1–2) + Character n-gram (3–5)** sublinear TF-IDF feature extractor paired with an L2-regularized Logistic Regression classifier.
   - Captures adversarial obfuscation (leetspeak, punctuation injection) while maintaining sub-millisecond single-document inference latency.
2. **`moderation_queue.py`**:
   - Three-band policy router (`AUTO_ALLOW`, `HUMAN_REVIEW`, `AUTO_BLOCK`) backed by a max-risk priority heap and full state-machine audit log (`PENDING -> APPROVED | REJECTED | ESCALATED`).
3. **`leakage_safe_splitter.py`**:
   - Enforces strict **temporal cutoff** and **disjoint `thread_id` / `author_id` partitioning** between training and evaluation sets, preventing inflated offline metrics caused by thread-level data leakage.
4. **`feedback_loop.py`**:
   - Computes **Population Stability Index (PSI)** across score distributions to detect vocabulary/behavior drift and performs sample-weighted online gradient updates from verified moderator overrides.
