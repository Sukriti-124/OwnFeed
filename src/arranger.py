"""Arranger: pick the final page of articles from the finder's candidates so it matches the slider mix."""
import numpy as np
import torch

PAGE_SIZE = 20


def mix_to_slots(targets: np.ndarray, page_size: int = PAGE_SIZE) -> np.ndarray:
    """Slider percentages -> whole article counts per bucket that add up to page_size (largest remainder rounding).

    targets: (page views, buckets), each row sums to 1.
    """
    raw = targets * page_size
    slots = np.floor(raw).astype(int)
    leftover = page_size - slots.sum(1)
    by_remainder = np.argsort(-(raw - slots), axis=1)
    for row in np.flatnonzero(leftover):
        slots[row, by_remainder[row, : leftover[row]]] += 1
    return slots


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Rescale each row of candidate scores to 0..1 (1 = best candidate in the cart, 0 = worst)."""
    low, high = scores.min(1, keepdim=True).values, scores.max(1, keepdim=True).values
    return (scores - low) / (high - low).clamp(min=1e-9)


def arrange(relevance: torch.Tensor, candidate_buckets: torch.Tensor, slots: torch.Tensor, strictness: float,
            page_size: int = PAGE_SIZE) -> torch.Tensor:
    """Pick page_size candidates one at a time for many page views at once.

    Each pick maximizes relevance + strictness x (open slots left in the candidate's bucket).
    strictness = 0 is a plain top-k feed; a very large strictness fills the slots exactly.

    relevance: (N, C) in 0..1; candidate_buckets: (N, C) bucket index; slots: (N, buckets).
    Returns the chosen candidate positions, (N, page_size), best relevance first.
    """
    n = relevance.shape[0]
    rows = torch.arange(n, device=relevance.device)
    taken = torch.zeros_like(slots)
    available = torch.ones_like(relevance, dtype=torch.bool)
    picks = []
    for _ in range(page_size):
        open_slots = (slots - taken).gather(1, candidate_buckets).float()
        gain = (relevance + strictness * open_slots).masked_fill(~available, float("-inf"))
        pick = gain.argmax(1)
        picks.append(pick)
        available[rows, pick] = False
        taken[rows, candidate_buckets[rows, pick]] += 1
    chosen = torch.stack(picks, 1)
    order = relevance.gather(1, chosen).argsort(1, descending=True)
    return chosen.gather(1, order)
