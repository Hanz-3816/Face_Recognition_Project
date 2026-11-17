"""Evaluate LFW pair distances with 10-fold threshold selection and testing.

For each held-out fold, choose the distance threshold that maximizes accuracy
on the other nine folds, then report thresholded metrics on the held-out fold.
Distances at or below the threshold are predicted as genuine (same person).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS = {
    "VGG-Face": "vgg_face_distances.csv",
    "Facenet": "facenet_distances.csv",
    "ArcFace": "arcface_distances.csv",
}
METRICS = ("euclidean_distance", "cosine_distance")
REQUIRED_COLUMNS = {"pair_index", "fold", "label", "pair_type", *METRICS}
N_FOLDS = 10


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Distance file not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows")

    for column in ("pair_index", "fold", "label", *METRICS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise ValueError(f"{path.name}: {column} contains missing or non-finite values")
    if frame["pair_index"].duplicated().any():
        raise ValueError(f"{path.name}: pair_index values must be unique")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: label must contain only 0 (impostor) and 1 (genuine)")
    if (frame[list(METRICS)] < 0).any().any():
        raise ValueError(f"{path.name}: distances must be non-negative")
    if not set(frame["pair_type"].astype(str).str.lower().unique()).issubset({"genuine", "impostor"}):
        raise ValueError(f"{path.name}: pair_type must be genuine or impostor")
    expected = frame["pair_type"].astype(str).str.lower().map({"genuine": 1, "impostor": 0})
    if not np.array_equal(expected.to_numpy(), frame["label"].astype(int).to_numpy()):
        raise ValueError(f"{path.name}: pair_type and label disagree")

    folds = sorted(frame["fold"].astype(int).unique().tolist())
    if folds != list(range(1, N_FOLDS + 1)):
        raise ValueError(f"{path.name}: expected folds 1 through {N_FOLDS}; found {folds}")
    frame["fold"] = frame["fold"].astype(int)
    frame["label"] = frame["label"].astype(int)
    frame["pair_index"] = frame["pair_index"].astype(int)
    for fold, group in frame.groupby("fold"):
        if set(group["label"]) != {0, 1}:
            raise ValueError(f"{path.name}: fold {fold} must contain both classes")
    return frame.sort_values("pair_index").reset_index(drop=True)


def choose_threshold(distances: np.ndarray, labels: np.ndarray) -> float:
    """Maximize training accuracy; <= threshold means genuine.

    Candidate cutoffs lie between adjacent observed distances, so predictions
    change only when a real boundary is crossed. Ties select the lowest cutoff.
    """
    unique = np.unique(distances)
    if len(unique) == 1:
        candidates = np.array([np.nextafter(unique[0], -np.inf), unique[0]])
    else:
        middle = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
        candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)], middle, [unique[-1]]))
    accuracies = np.array([np.mean((distances <= t).astype(int) == labels) for t in candidates])
    best = np.flatnonzero(accuracies == accuracies.max())[0]
    return float(candidates[best])


def roc_statistics(labels: np.ndarray, distances: np.ndarray) -> tuple[float, float]:
    """Return test-fold EER and ROC-AUC for lower-is-more-genuine scores."""
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("ROC statistics require both genuine and impostor pairs")

    # Empirical ROC points from all observed thresholds, ordered by ascending distance.
    thresholds = np.concatenate(([np.nextafter(distances.min(), -np.inf)], np.unique(distances)))
    fars, frrs = [], []
    for threshold in thresholds:
        prediction = distances <= threshold
        fars.append(np.sum(prediction & (labels == 0)) / negatives)
        frrs.append(np.sum((~prediction) & (labels == 1)) / positives)
    fars = np.asarray(fars)
    frrs = np.asarray(frrs)
    tprs = 1.0 - frrs
    # Thresholds increase from strict to permissive, so FPR and TPR increase.
    auc = float(np.trapezoid(tprs, fars))
    gap = fars - frrs
    crossing = np.flatnonzero(gap[:-1] * gap[1:] <= 0)
    if len(crossing):
        i = crossing[0]
        denominator = gap[i + 1] - gap[i]
        fraction = 0.0 if denominator == 0 else -gap[i] / denominator
        eer = float((fars[i] + fraction * (fars[i + 1] - fars[i]) +
                     frrs[i] + fraction * (frrs[i + 1] - frrs[i])) / 2.0)
    else:
        i = int(np.argmin(np.abs(gap)))
        eer = float((fars[i] + frrs[i]) / 2.0)
    return eer, auc


def calculate_metrics(labels: np.ndarray, distances: np.ndarray, threshold: float) -> dict:
    predicted = (distances <= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    eer, auc = roc_statistics(labels, distances)
    return {
        "accuracy": (tp + tn) / len(labels), "precision": precision,
        "recall": recall, "f1": f1, "far": fp / (fp + tn),
        "frr": fn / (fn + tp), "eer": eer, "roc_auc": auc,
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = {model: load_and_validate(RESULTS_DIR / filename) for model, filename in MODELS.items()}
    reference = loaded["VGG-Face"]
    for model, frame in loaded.items():
        if not np.array_equal(frame["pair_index"].to_numpy(), reference["pair_index"].to_numpy()):
            raise ValueError(f"{model}: pair_index rows do not align with VGG-Face")
        if not np.array_equal(frame["fold"].to_numpy(), reference["fold"].to_numpy()) or not np.array_equal(frame["label"].to_numpy(), reference["label"].to_numpy()):
            raise ValueError(f"{model}: fold/label assignments do not align with VGG-Face")

    rows = []
    for model, frame in loaded.items():
        for metric in METRICS:
            distances = frame[metric].to_numpy(dtype=float)
            labels = frame["label"].to_numpy(dtype=int)
            folds = frame["fold"].to_numpy(dtype=int)
            for held_out in range(1, N_FOLDS + 1):
                train = folds != held_out
                test = ~train
                threshold = choose_threshold(distances[train], labels[train])
                row = {"model": model, "distance_metric": metric.removesuffix("_distance"),
                       "fold": held_out, "threshold": threshold}
                row.update(calculate_metrics(labels[test], distances[test], threshold))
                rows.append(row)
    fold_results = pd.DataFrame(rows)
    metric_columns = ["accuracy", "precision", "recall", "f1", "far", "frr", "eer", "roc_auc"]
    grouped = fold_results.groupby(["model", "distance_metric"], sort=False)
    means = grouped[metric_columns + ["threshold"]].mean().add_suffix("_mean")
    stds = grouped[metric_columns + ["threshold"]].std(ddof=1).add_suffix("_std")
    summary = means.join(stds).reset_index()
    return fold_results, summary


def main() -> int:
    try:
        folds, summary = evaluate()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Evaluation could not be completed: {error}", file=sys.stderr)
        return 1

    fold_path = RESULTS_DIR / "evaluation_fold_results.csv"
    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    folds.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    display = ["model", "distance_metric", "accuracy_mean", "precision_mean", "recall_mean",
               "f1_mean", "far_mean", "frr_mean", "eer_mean", "roc_auc_mean"]
    print("10-fold evaluation complete. Metrics below are means across held-out folds:")
    print(summary[display].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nFold-level results saved to: {fold_path}")
    print(f"Summary (mean and standard deviation) saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
