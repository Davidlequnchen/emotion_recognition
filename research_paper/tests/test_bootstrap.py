"""Synthetic checks of pooled cluster scores and paired confidence intervals.

No models, audio, APIs or source prediction artifacts are needed.
"""
from pathlib import Path
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
import consensus


def fixture_run(records, classes=("angry", "neutral")):
    """Build a saved-run equivalent; records are (path, gold, prediction)."""
    return consensus.Run(
        "fixture", "crema_d", Path("unused_predictions.json"), {}, list(classes), {},
        {path: {"gold": gold, "prediction": prediction} for path, gold, prediction in records}, [],
    )


class BootstrapScoreTests(unittest.TestCase):
    def test_unequal_clusters_use_pooled_clips_not_mean_speaker_scores(self):
        # One speaker has one correct clip; another has three wrong clips.
        paths = [f"data/crema_d/{i}.wav" for i in range(4)]
        run = fixture_run([
            (paths[0], "angry", "angry"),
            *[(path, "neutral", "angry") for path in paths[1:]],
            # Excluded from this cohort: paired comparisons must not leak it in.
            ("data/crema_d/not_in_cohort.wav", "neutral", "neutral"),
        ])
        counts = bootstrap.cluster_counts(run, paths, np.array([0, 1, 1, 1]), 2)
        expected = np.array([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [3, 0, 0]]])
        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(bootstrap.METRICS,
                         ("accuracy", "unweighted_accuracy", "macro_f1", "weighted_f1"))
        # Accuracy is 1/4, rather than (1 + 0)/2. F1_angry=2/(1+4).
        np.testing.assert_allclose(bootstrap.scores_from_counts(counts.sum(axis=0)),
                                   [1 / 4, 1 / 2, 1 / 5, 1 / 10])

    def test_repeated_clusters_repeat_all_their_clips(self):
        counts = np.array([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [3, 0, 0]]])
        repetitions = np.array([[2, 0], [1, 1], [0, 2]])
        pooled = np.einsum("bg,gij->bij", repetitions, counts)
        # Equal numbers of sampled speakers do not imply equal numbers of clips.
        np.testing.assert_array_equal(pooled.sum(axis=(1, 2)), [2, 4, 6])
        np.testing.assert_allclose(bootstrap.scores_from_counts(pooled),
                                   [[1, 1 / 2, 1 / 2, 1],
                                    [1 / 4, 1 / 2, 1 / 5, 1 / 10],
                                    [0, 0, 0, 0]])

    def test_invalid_answers_are_false_negatives_without_an_extra_macro_class(self):
        run = fixture_run([
            ("a.wav", "angry", "angry"), ("b.wav", "angry", None),
            ("c.wav", "neutral", "neutral"), ("d.wav", "neutral", None),
        ])
        counts = bootstrap.cluster_counts(run, list(run.rows), np.array([0, 0, 1, 1]), 2)
        pooled = counts.sum(axis=0)
        np.testing.assert_array_equal(pooled, [[1, 0, 1], [0, 1, 1]])
        # Each genuine class has TP=1, gold support=2 and predicted support=1.
        np.testing.assert_allclose(bootstrap.scores_from_counts(pooled),
                                   [1 / 2, 1 / 2, 2 / 3, 2 / 3])

    def test_missing_classes_keep_fixed_macro_denominator_and_zero_division(self):
        # A draw contains only correct angry clips, but the dataset still has K=3.
        counts = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_allclose(bootstrap.scores_from_counts(counts), [1, 1 / 3, 1 / 3, 1])

    def test_batched_scores_keep_each_draw_separate(self):
        perfect = np.array([[2, 0, 0], [0, 2, 0]])
        swapped = np.array([[0, 2, 0], [2, 0, 0]])
        batched = np.stack([np.stack([perfect, swapped]), np.stack([swapped, perfect])])
        scores = bootstrap.scores_from_counts(batched)
        self.assertEqual(scores.shape, (2, 2, 4))
        np.testing.assert_allclose(scores, [[[1] * 4, [0] * 4], [[0] * 4, [1] * 4]])


class BootstrapIntervalTests(unittest.TestCase):
    def test_percentile_endpoints_are_quantiles_of_draws(self):
        values = np.array([0, .25, .5, .75, 1], dtype=float)
        np.testing.assert_allclose(bootstrap.interval(values), [.025, .975])
        np.testing.assert_allclose(bootstrap.interval(values, confidence=.5), [.25, .75])

    def test_identical_paired_predictions_have_exactly_zero_difference_interval(self):
        draws = np.random.default_rng(18).uniform(size=(100, 4))
        lower, upper = bootstrap.paired_interval(draws, draws.copy())
        np.testing.assert_array_equal(lower, np.zeros(4))
        np.testing.assert_array_equal(upper, np.zeros(4))

    def test_paired_interval_uses_drawwise_difference_and_reverses_sign(self):
        # Independent marginal intervals cannot establish this matched difference.
        draws_b = np.array([[.2] * 4, [.4] * 4, [.6] * 4, [.8] * 4])
        scales = np.array([1, .5, -.5, -1])
        differences = np.array([-.1, .2, -.3, .1])[:, None] * scales
        draws_a = draws_b + differences
        forward = bootstrap.paired_interval(draws_a, draws_b)
        reverse = bootstrap.paired_interval(draws_b, draws_a)
        # Sorted first-column differences: [-.3, -.1, .1, .2].
        np.testing.assert_allclose(forward,
                                   [[-.285, -.1425, -.09625, -.1925],
                                    [.1925, .09625, .1425, .285]], atol=1e-15)
        np.testing.assert_allclose(reverse[0], -np.asarray(forward[1]), atol=1e-15)
        np.testing.assert_allclose(reverse[1], -np.asarray(forward[0]), atol=1e-15)


class BootstrapSamplingTests(unittest.TestCase):
    def test_seed_and_cohort_are_reproducible_without_python_hash_state(self):
        arguments = (4, 128, 20261111, "crema_d", "same-audio-cohort")
        first = bootstrap.bootstrap_weights(*arguments)
        np.testing.assert_array_equal(first, bootstrap.bootstrap_weights(*arguments))
        self.assertEqual(first.shape, (128, 4))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        self.assertTrue(np.all(first >= 0))
        np.testing.assert_array_equal(first.sum(axis=1), np.full(128, 4))
        for seed, dataset, cohort in [(20261112, "crema_d", "same-audio-cohort"),
                                      (20261111, "ravdess", "same-audio-cohort"),
                                      (20261111, "crema_d", "new-audio-cohort")]:
            with self.subTest(seed=seed, dataset=dataset, cohort=cohort):
                self.assertFalse(np.array_equal(
                    first, bootstrap.bootstrap_weights(4, 128, seed, dataset, cohort)))

    def test_uniform_group_resampling_can_repeat_right_or_wrong_groups(self):
        weights = bootstrap.bootstrap_weights(2, 128, 20261111, "crema_d", "two-speakers")
        counts = np.array([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [3, 0, 0]]])
        repeated_right = np.flatnonzero(np.all(weights == [2, 0], axis=1))
        repeated_wrong = np.flatnonzero(np.all(weights == [0, 2], axis=1))
        self.assertGreater(len(repeated_right), 0)
        self.assertGreater(len(repeated_wrong), 0)
        pooled = np.einsum("bg,gij->bij", weights, counts)
        scores = bootstrap.scores_from_counts(pooled)
        np.testing.assert_array_equal(pooled[repeated_right].sum(axis=(1, 2)),
                                      np.full(len(repeated_right), 2))
        np.testing.assert_array_equal(pooled[repeated_wrong].sum(axis=(1, 2)),
                                      np.full(len(repeated_wrong), 6))
        np.testing.assert_array_equal(scores[repeated_right, 0], np.ones(len(repeated_right)))
        np.testing.assert_array_equal(scores[repeated_wrong, 0], np.zeros(len(repeated_wrong)))


class BootstrapIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="research_bootstrap_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "emotion_recognition"
        self.paper = Path(temporary.name) / "research_paper"
        (self.root / "config").mkdir(parents=True)
        registry = {f"fake{i}": {"name": f"Fake {i}"} for i in range(3)}
        (self.root / "config/models.yaml").write_text(yaml.safe_dump(registry))
        self.paths = [f"data/crema_d/AudioWAV/{actor}_DFA_{emotion}_XX.wav"
                      for actor in (1001, 1002, 1003) for emotion in ("ANG", "NEU")]
        self.golds = ["angry", "neutral"] * 3

    def write_run(self, model, predictions, *, golds=None, total=None):
        folder = self.root / "outputs" / model
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "crema_d_predictions.json"
        golds = self.golds[:len(predictions)] if golds is None else golds
        rows = [{"audio_path": audio, "acted_emotion": gold, "prediction": prediction}
                for audio, gold, prediction in zip(self.paths, golds, predictions)]
        with path.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "acted_emotion", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        path.write_text(json.dumps({
            "key": "crema_d", "dataset": "CREMA-D", "model": model, "clips": len(rows),
            "total_clips": len(rows) if total is None else total,
            "classes": ["angry", "neutral"], "ground_truth": "acted_emotion",
            "metrics": {"accuracy": sum(g == p for g, p in zip(golds, predictions)) / len(rows)},
        }))
        return path

    def analyze(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return bootstrap.run(self.root, self.paper, datasets=("crema_d",), n_resamples=200)

    def test_identical_inputs_reproduce_files_and_new_model_preserves_existing_comparisons(self):
        self.write_run("fake0", self.golds)
        self.write_run("fake1", ["angry"] * 6)
        rows, _ = self.analyze()
        files = {path.relative_to(self.paper): path.read_bytes()
                 for path in self.paper.rglob("*") if path.is_file()}
        self.assertEqual(len(files), 4)
        self.analyze()
        self.assertTrue(all((self.paper / path).read_bytes() == data for path, data in files.items()))
        self.write_run("fake2", list(reversed(self.golds)))
        updated, manifest = self.analyze()
        original_rows = [row for row in updated if row["model_a"] != "fake2" and row["model_b"] != "fake2"]
        self.assertEqual(rows, original_rows)
        self.assertEqual(manifest["results"]["single_model_runs"], 3)
        self.assertEqual(manifest["results"]["model_pairs"], 3)

    def test_partial_run_pair_uses_common_clips_and_reports_its_own_scope(self):
        self.write_run("fake0", ["angry", "neutral", "angry", "neutral", "neutral", "angry"])
        self.write_run("fake1", ["angry", "angry", "neutral", "neutral"], total=6)
        rows, _ = self.analyze()
        pair = next(row for row in rows if row["kind"] == "paired_difference" and row["metric"] == "accuracy")
        self.assertEqual((pair["n_samples"], pair["n_clusters"]), (4, 2))
        self.assertEqual((pair["score_a"], pair["score_b"], pair["estimate"]), (1, .5, .5))
        self.assertAlmostEqual(pair["common_fraction_a"], 2 / 3)
        self.assertEqual(pair["common_fraction_b"], 1)
        self.assertEqual(pair["run_scope_b"], "partial_declared_scope")
        index = {("crema_d", "fake0", "fake1", "accuracy"): pair}
        reversed_row = bootstrap.oriented_row(index, "crema_d", "fake1", "fake0", "accuracy")
        self.assertEqual(reversed_row["estimate"], -.5)
        self.assertEqual(reversed_row["run_samples_a"], 4)
        self.assertEqual(reversed_row["run_samples_b"], 6)
        self.assertEqual(reversed_row["common_fraction_a"], 1)
        self.assertEqual(reversed_row["run_scope_a"], "partial_declared_scope")

    def test_mismatched_gold_fails_before_any_output_is_written(self):
        self.write_run("fake0", self.golds)
        self.write_run("fake1", self.golds, golds=["neutral", *self.golds[1:]])
        with self.assertRaisesRegex(ValueError, "ground-truth mismatch"):
            self.analyze()
        self.assertFalse(self.paper.exists())

    def test_source_mutation_during_analysis_fails_before_any_output_is_written(self):
        source = self.write_run("fake0", self.golds).with_suffix(".csv")
        original = bootstrap.distribution

        def changing_source(*args):
            source.write_text(source.read_text() + "\n")
            return original(*args)

        with mock.patch.object(bootstrap, "distribution", side_effect=changing_source), \
                self.assertRaisesRegex(RuntimeError, "Source predictions changed"):
            self.analyze()
        self.assertFalse(self.paper.exists())


if __name__ == "__main__":
    unittest.main()
