"""Shared validation rules for checkpoint selection and band comparisons."""

import math


# criterion: (score stream, target group, recall cutoff)
SELECTION_CRITERIA = {
    "first_visit_unseen_model_r100": ("model", "first_visit_unseen", "100"),
    "first_visit_unseen_ensemble_r100": ("ensemble", "first_visit_unseen", "100"),
    "first_visit_unseen_trained_r100": ("trained_combination", "first_visit_unseen", "100"),
    "overall_model_r10": ("model", "all", "10"),  # historical runner only
}
SELECTION_METRICS = set(SELECTION_CRITERIA) - {"overall_model_r10"}


def selection_score(metrics, criterion):
    if criterion not in SELECTION_CRITERIA:
        raise ValueError(f"Unknown selection criterion: {criterion}")
    stream, group, cutoff = SELECTION_CRITERIA[criterion]
    if stream == "trained_combination" and stream not in metrics:
        raise ValueError("Learned-weight selection requires --heuristic_loss.")
    row = metrics[stream][group]
    score = row["recall"][cutoff]
    if row["n"] < 1 or score is None or not math.isfinite(score):
        raise ValueError("Selection requires a nonempty validation group and finite recall")
    return score


def best_validation_epoch(history, criterion):
    """Validate recorded scores, then retain the earliest epoch in an exact tie."""
    if not history:
        raise ValueError("Checkpoint selection requires a nonempty validation history.")
    previous_epoch = 0
    for row in history:
        epoch = row["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= previous_epoch:
            raise ValueError("Validation epochs must be positive and strictly increasing.")
        previous_epoch = epoch
        if row["selection_metric"] != criterion:
            raise ValueError("Validation history uses a different selection criterion.")
        if row["selection_score"] != selection_score(row["metrics"], criterion):
            raise ValueError("Validation selection score disagrees with its recorded metrics.")
    return max(history, key=lambda row: row["selection_score"])
