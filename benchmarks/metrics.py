"""EER and Accuracy metrics for speaker verification evaluation."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding threshold.

    EER is the point where the False Acceptance Rate (FAR) equals the
    False Rejection Rate (FRR). Linear interpolation is used when the
    curves do not cross exactly at a score boundary.

    Parameters
    ----------
    scores : np.ndarray
        Cosine similarity scores for each pair.
    labels : np.ndarray
        Ground-truth labels: ``1`` for genuine pairs, ``0`` for impostors.

    Returns
    -------
    tuple[float, float]
        ``(eer, threshold)`` where ``eer`` is in ``[0, 1]``.
    """
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        logger.warning("Cannot compute EER: missing genuine or impostor scores")
        return float("nan"), float("nan")

    thresholds = np.sort(np.unique(scores))
    far_list: list[float] = []
    frr_list: list[float] = []

    for t in thresholds:
        far = float(np.mean(impostor_scores >= t))
        frr = float(np.mean(genuine_scores < t))
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    diff = far_arr - frr_arr
    idx = int(np.argmin(np.abs(diff)))

    if idx > 0 and diff[idx - 1] * diff[idx] < 0:
        t0, t1 = thresholds[idx - 1], thresholds[idx]
        d0, d1 = diff[idx - 1], diff[idx]
        t_cross = t0 + (t1 - t0) * (-d0) / (d1 - d0)
        far_cross = far_arr[idx - 1] + (far_arr[idx] - far_arr[idx - 1]) * (
            (t_cross - t0) / (t1 - t0)
        )
        frr_cross = frr_arr[idx - 1] + (frr_arr[idx] - frr_arr[idx - 1]) * (
            (t_cross - t0) / (t1 - t0)
        )
        eer = (far_cross + frr_cross) / 2.0
        threshold = float(t_cross)
    else:
        eer = (far_arr[idx] + frr_arr[idx]) / 2.0
        threshold = float(thresholds[idx])

    logger.debug(f"EER={eer:.4f} at threshold={threshold:.4f}")
    return float(eer), threshold


def compute_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """Compute binary classification accuracy at a given threshold.

    Parameters
    ----------
    scores : np.ndarray
        Cosine similarity scores.
    labels : np.ndarray
        Ground-truth labels (``1`` genuine, ``0`` impostor).
    threshold : float
        Decision boundary.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    predictions = (scores >= threshold).astype(int)
    accuracy = float(np.mean(predictions == labels))
    logger.debug(f"Accuracy={accuracy:.4f} at threshold={threshold:.4f}")
    return accuracy


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    fixed_threshold: float | None = None,
) -> dict[str, float]:
    """Compute EER and accuracy, returning them as a single dict.

    Parameters
    ----------
    scores : np.ndarray
        Cosine similarity scores.
    labels : np.ndarray
        Ground-truth labels.
    fixed_threshold : float or None
        If provided, accuracy is computed at this threshold instead of the
        EER threshold.

    Returns
    -------
    dict[str, float]
        Keys: ``eer``, ``eer_threshold``, ``accuracy``.
    """
    eer, eer_threshold = compute_eer(scores, labels)
    threshold = fixed_threshold if fixed_threshold is not None else eer_threshold
    accuracy = compute_accuracy(scores, labels, threshold)
    return {"eer": eer, "eer_threshold": eer_threshold, "accuracy": accuracy}
