"""Boundary scores and whole-turn dynamic programming (doc §4.3-4.5).

s = w_p * p + w_q * q + w_v * v for every word gap, with p relative to this speaker's typical pause.
The cheapest partition minimises  sum L(duration) + lambda * sum (1 - s)  with
L(d) = ((d - target) / target)^2 for lo <= d <= hi (default 2-15 s), infinite otherwise.
"""
import math
from statistics import median

INF = float("inf")


def typical_pause(gaps, min_gap=0.1, min_count=3, fallback=0.3):
    """Median of gaps longer than min_gap; fallback when fewer than min_count such gaps."""
    long_gaps = [g for g in gaps if g > min_gap]
    return median(long_gaps) if len(long_gaps) >= min_count else fallback


def boundary_score(gap, typical, q, v, punctuated=True, weights=(0.5, 0.3, 0.2), plain_weights=(0.7, 0.3)):
    """Score in [0, 1]. Without any punctuation in the transcript: s = 0.7 p + 0.3 v."""
    p = min(gap / (2 * typical), 1.0)
    if punctuated:
        w_p, w_q, w_v = weights
        return w_p * p + w_q * q + w_v * v, p
    w_p, w_v = plain_weights
    return w_p * p + w_v * v, p


def length_cost(d, target=7.0, lo=2.0, hi=15.0):
    if d < lo or d > hi:
        return INF                                  # not allowed
    return ((d - target) / target) ** 2


def best_partition(times, scores, lam=1.0, target=7.0, lo=2.0, hi=15.0):
    """times: [turn start, candidate cut times..., turn end] in order; scores: s per time (turn end = 1).
    Returns (indices of chosen cuts into times, total cost, length cost, cut cost), or None if no partition
    satisfies the length limits. Exact: best[j] is the cheapest way to cover times[0]..times[j] with a cut at j."""
    n = len(times)
    best = [0.0] + [INF] * (n - 1)
    prev = [None] * n
    for j in range(1, n):
        cut_cost = lam * (1.0 - scores[j])
        for i in range(j - 1, -1, -1):
            d = times[j] - times[i]
            if d > hi:
                break                               # only starts within the maximum length of j
            cost = best[i] + length_cost(d, target, lo, hi) + cut_cost
            if cost < best[j]:
                best[j], prev[j] = cost, i
    if math.isinf(best[-1]):
        return None
    cuts, j = [], prev[n - 1]
    while j:                                        # walk back to the start (index 0)
        cuts.append(j)
        j = prev[j]
    cuts.reverse()
    bounds = [0, *cuts, n - 1]
    length_total = sum(length_cost(times[b] - times[a], target, lo, hi) for a, b in zip(bounds, bounds[1:]))
    cut_total = sum(lam * (1.0 - scores[c]) for c in cuts)
    return cuts, best[-1], length_total, cut_total


def partition_cost(times, scores, cut_indices, lam=1.0, target=7.0, lo=2.0, hi=15.0):
    """(total, length cost, cut cost) of one given partition (for checks and the worked example)."""
    bounds = [0, *cut_indices, len(times) - 1]
    length_total = sum(length_cost(times[b] - times[a], target, lo, hi) for a, b in zip(bounds, bounds[1:]))
    cut_total = sum(lam * (1.0 - scores[c]) for c in cut_indices)
    return length_total + cut_total, length_total, cut_total


def assign_words(words, bounds):
    """Segment index of every word; bounds are the segment edges in seconds [b0, b1, ..., bK].
    Raises if a word crosses a cut, lies outside the turn span, or the order breaks."""
    assignment = []
    for word in words:
        inside = [k for k in range(len(bounds) - 1)
                  if bounds[k] - 1e-6 <= word.start and word.end <= bounds[k + 1] + 1e-6]
        if len(inside) != 1:
            raise AssertionError(f"word {word.text!r} [{word.start}, {word.end}] is not inside exactly one segment")
        assignment.append(inside[0])
    if assignment != sorted(assignment) or set(assignment) != set(range(len(bounds) - 1)):
        raise AssertionError("words are not in order or a segment has no words")
    return assignment
