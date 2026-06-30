"""RewardBench scoring aggregation — a non-saturated, discriminating RM yardstick.

RewardBench (allenai/reward-bench) groups ~3k chosen/rejected pairs into subsets; the
headline metric is the mean of four CATEGORY accuracies (Chat / Chat Hard / Safety /
Reasoning), each the mean of its subsets. We report the unweighted category mean (a
faithful approximation of the official score, which weights a few subsets) plus the raw
micro-accuracy and the per-subset / per-category breakdown.
"""

from __future__ import annotations

from collections import defaultdict

# RewardBench v1 subset -> category map (allenai/reward-bench leaderboard groupings).
CATEGORIES = {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard",
             "mt-bench-easy", "mt-bench-med"],
    "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
                  "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
               "xstest-should-respond", "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js",
                  "hep-python", "hep-rust"],
}
_MAIN = ["Chat", "Chat Hard", "Safety", "Reasoning"]
_SUBSET2CAT = {s: c for c, subs in CATEGORIES.items() for s in subs}


def rewardbench_report(subsets, correct) -> dict:
    """Aggregate per-example correctness into the RewardBench scoreboard.

    ``subsets``: list of subset names; ``correct``: matching list of bool/int (chosen>rejected).
    Returns per-subset accuracy, per-category accuracy (subset-mean), the overall
    category-mean (the headline RewardBench number), and the raw micro-accuracy.
    """
    by_subset = defaultdict(list)
    for s, ok in zip(subsets, correct):
        by_subset[s].append(bool(ok))
    per_subset = {s: sum(v) / len(v) for s, v in by_subset.items()}

    cat_accs = defaultdict(list)
    for s, acc in per_subset.items():
        cat_accs[_SUBSET2CAT.get(s, "other")].append(acc)
    per_category = {c: sum(v) / len(v) for c, v in cat_accs.items()}

    present = [c for c in _MAIN if c in per_category]
    overall = sum(per_category[c] for c in present) / len(present) if present else 0.0
    n = len(list(correct))
    micro = sum(bool(x) for x in correct) / n if n else 0.0
    return {"overall": overall, "per_category": per_category, "per_subset": per_subset,
            "accuracy_micro": micro, "n": n}
