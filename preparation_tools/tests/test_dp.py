"""Boundary scores and the dynamic programme against the worked examples of the method document (§4.3-4.7)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter.boundaries import (best_partition, boundary_score, length_cost, partition_cost,  # noqa: E402
                                  typical_pause)

# Doc §4.6: slow speaker, 40 s turn, typical pause 0.6 s. (time, gap, q, v, score in the doc table)
STRONG = [(7.4, 1.2, 1.0, 0.95, 0.99), (12.1, 0.6, 0.0, 0.90, 0.43), (15.3, 1.0, 0.5, 0.90, 0.75),
          (21.8, 1.4, 1.0, 0.20, 0.84), (23.0, 0.9, 1.0, 0.95, 0.87), (29.6, 1.3, 1.0, 0.95, 0.99),
          (34.2, 0.7, 0.5, 0.90, 0.62)]
# The other 18 ordinary 0.4-0.6 s pauses (scores about 0.2-0.45).
FILLER_TIMES = [1.9, 3.2, 4.5, 5.8, 9.0, 10.3, 11.0, 13.5, 17.0, 18.4, 19.9, 25.0, 26.3, 27.7, 31.2, 32.6, 35.8, 37.3]


def worked_example():
    candidates = [(t, boundary_score(g, 0.6, q, v)[0]) for t, g, q, v, _ in STRONG]
    for k, t in enumerate(FILLER_TIMES):
        gap, v = (0.4, 0.5, 0.6)[k % 3], (0.3, 0.6, 0.9)[k // 3 % 3]
        candidates.append((t, boundary_score(gap, 0.6, 0.0, v)[0]))
    candidates.sort()
    times = [0.0] + [t for t, _ in candidates] + [40.0]
    scores = [0.0] + [s for _, s in candidates] + [1.0]
    return times, scores


class BoundaryScore(unittest.TestCase):
    def test_doc_table_scores(self):
        for _, gap, q, v, expected in STRONG:
            self.assertAlmostEqual(boundary_score(gap, 0.6, q, v)[0], expected, delta=0.006)

    def test_relative_pause(self):
        self.assertEqual(boundary_score(1.2, 0.6, 0, 0)[1], 1.0)        # twice the typical pause: full marks
        self.assertAlmostEqual(boundary_score(0.6, 0.6, 0, 0)[1], 0.5)  # an ordinary pause of a slow speaker
        self.assertEqual(boundary_score(0.4, 0.2, 0, 0)[1], 1.0)        # fast speaker: full marks at 0.4 s

    def test_no_punctuation_weights(self):
        s, p = boundary_score(0.6, 0.6, q=1.0, v=0.9, punctuated=False)
        self.assertAlmostEqual(s, 0.7 * 0.5 + 0.3 * 0.9)                # q is ignored

    def test_typical_pause(self):
        self.assertAlmostEqual(typical_pause([0.05, 0.2, 0.6, 0.8, 0.0]), 0.6)
        self.assertEqual(typical_pause([0.05, 0.2, 0.6]), 0.3)          # fewer than three gaps > 0.1 s


class DynamicProgramme(unittest.TestCase):
    def test_length_cost_values(self):
        self.assertAlmostEqual(length_cost(7.4), 0.0033, places=4)
        self.assertAlmostEqual(length_cost(4.6), 0.1176, places=4)
        self.assertAlmostEqual(length_cost(12.0), 0.5102, places=4)
        self.assertAlmostEqual(length_cost(2.0), 0.5102, places=4)      # window edges 2 s and 15 s are allowed
        self.assertAlmostEqual(length_cost(15.0), 1.3061, places=4)
        self.assertEqual(length_cost(1.9), float("inf"))
        self.assertEqual(length_cost(15.1), float("inf"))

    def test_worked_example(self):
        times, scores = worked_example()
        self.assertEqual(len(times) - 2, 25)                             # 25 pauses
        cuts, total, length_total, cut_total = best_partition(times, scores, lo=2.0, hi=15.0)
        self.assertEqual([times[i] for i in cuts], [7.4, 15.3, 23.0, 29.6])   # partition A, as in the doc
        a = [times.index(t) for t in (7.4, 15.3, 23.0, 29.6)]
        b = a + [times.index(34.2)]
        total_a, length_a, cut_a = partition_cost(times, scores, a)
        total_b, length_b, cut_b = partition_cost(times, scores, b)
        self.assertAlmostEqual(total, total_a)
        self.assertAlmostEqual((length_total, cut_total), (length_a, cut_a))
        self.assertAlmostEqual(length_a, 0.269, delta=0.001)             # doc table: 0.27 + 0.41 = 0.68
        self.assertAlmostEqual(cut_a, 0.408, delta=0.001)
        self.assertAlmostEqual(total_a, 0.677, delta=0.001)
        self.assertAlmostEqual(length_b, 0.180, delta=0.001)             # doc table: 0.18 + 0.79 = 0.97
        self.assertAlmostEqual(cut_b, 0.787, delta=0.001)
        self.assertAlmostEqual(total_b, 0.967, delta=0.001)
        self.assertNotIn(times.index(21.8), cuts)                        # the gap with a sigh is skipped

    def test_fast_speaker_two_segments(self):
        times = [0.0] + [round(0.35 * k, 2) for k in range(1, 58)] + [20.0]
        scores = [0.0] + [0.2] * 57 + [1.0]
        cuts, *_ = best_partition(times, scores)
        self.assertEqual(len(cuts), 1)                                   # 20 s > 15 s: exactly one cut ...
        self.assertTrue(9.0 <= times[cuts[0]] <= 11.0)                   # ... near the middle
        sentence_end = times.index(9.1)
        scores[sentence_end] = 0.6                                       # a sentence end between 8 and 12 s
        cuts, *_ = best_partition(times, scores)
        self.assertEqual(cuts, [sentence_end])

    def test_ten_to_fifteen_seconds(self):
        self.assertEqual(best_partition([0.0, 5.75, 11.5], [0.0, 0.2, 1.0])[0], [])   # no good pause: whole
        self.assertEqual(best_partition([0.0, 5.75, 11.5], [0.0, 1.0, 1.0])[0], [1])  # perfect pause: cut
        self.assertEqual(best_partition([0.0, 6.5, 13.0], [0.0, 0.2, 1.0])[0], [])    # 13 s, poor pause: whole
        self.assertEqual(best_partition([0.0, 7.5, 15.0], [0.0, 0.2, 1.0])[0], [1])   # 15 s, poor pause: cut

    def test_infeasible(self):
        times, scores = [0.0, 3.0, 17.0, 30.0], [0.0, 0.9, 0.9, 1.0]
        self.assertEqual(best_partition(times, scores)[0], [1, 2])       # a 14 s stretch fits the window
        self.assertIsNone(best_partition([0.0, 5.0, 40.0], [0.0, 0.9, 1.0]))   # 35 s without a usable gap
        self.assertIsNone(best_partition([0.0, 1.0, 16.5], [0.0, 0.9, 1.0]))   # 1 s + 15.5 s, or 16.5 s whole

    def test_matches_brute_force(self):
        """Every partition whose segments all lie within 2-15 s (all others cost infinity), enumerated."""
        times, scores = worked_example()
        n, costs = len(times), []

        def extend(i, cost):
            for j in range(i + 1, n):
                d = times[j] - times[i]
                if d > 15.0:
                    break
                if d < 2.0:
                    continue
                step = length_cost(d) + (0.0 if j == n - 1 else 1.0 - scores[j])
                if j == n - 1:
                    costs.append(cost + step)
                else:
                    extend(j, cost + step)

        extend(0, 0.0)
        self.assertGreater(len(costs), 100_000)                          # 116,674 allowed partitions
        self.assertAlmostEqual(best_partition(times, scores)[1], min(costs))


if __name__ == "__main__":
    unittest.main()
