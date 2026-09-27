"""Independent decoding oracle, score alignment and component-fold isolation."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import takeover_eval as ev


def world(left, right, label, prob, n, *, truth=None, component=None, folds=None, population=None):
    left, right, label = np.asarray(left, np.int64), np.asarray(right, np.int64), np.asarray(label, bool)
    if truth is None:
        truth = np.bincount(left[label], minlength=n)
    population = np.ones(n, bool) if population is None else np.asarray(population, bool)
    f = np.arange(n) % 5 if folds is None else np.asarray(folds)
    f = np.where(population, f, -1)
    return ev.World(left, right, label, np.asarray(prob, np.float32), np.asarray(truth, np.int64), population,
                    np.arange(n) if component is None else np.asarray(component), np.arange(n) % 2,
                    population.astype(np.int8), f)


def oracle(w, p, thresholds=ev.BASE_THRESHOLD):
    """Deliberately plain Python; does not call any evaluator grouping helper."""
    p = np.asarray(p, dtype=w.baseline_p.dtype)
    threshold = np.broadcast_to(np.asarray(thresholds, dtype=p.dtype), p.shape)
    owners = {}
    for row in range(len(p)):
        if p[row] >= threshold[row]:
            key = (-float(p[row]), int(w.left[row]), row)
            record = int(w.right[row])
            if record not in owners or key < owners[record][0]:
                owners[record] = (key, row)
    lists = [[] for _ in range(len(w.population))]
    for _, row in owners.values():
        lists[w.left[row]].append(row)
    accepted, scores, counts, tp = [], [], [], []
    for s1, rows in enumerate(lists):
        rows.sort(key=lambda r: (-float(p[r]), r))
        chosen = rows[:11]
        accepted.extend(chosen)
        k, t, hit = len(chosen), int(w.true_count[s1]), sum(bool(w.label[r]) for r in chosen)
        scores.append(1.0 if k == t == 0 else 5 * hit / (4 * k + t))
        counts.append(k)
        tp.append(hit)
    return np.sort(accepted).astype(np.int64), np.asarray(scores), np.asarray(counts), np.asarray(tp)


def random_world(seed=11, n=70):
    rng = np.random.default_rng(seed)
    pairs = sorted(set(zip(rng.integers(0, n, n * 15), rng.integers(0, n * 6, n * 15))))
    rng.shuffle(pairs)
    left, right = np.asarray(pairs).T
    labels, claimed = np.zeros(len(pairs), bool), set()
    for i, rec in enumerate(right):
        if rec not in claimed and rng.random() < .3:
            labels[i] = True
            claimed.add(rec)
    prob = rng.choice([0., .001, .1, .5, .7, .725, .8, .9, .999, 1.], len(left))
    truth = np.bincount(left[labels], minlength=n) + rng.integers(0, 3, n)
    return world(left, right, labels, prob, n, truth=truth)


def learning_world():
    rng = np.random.default_rng(9)
    n, per = 60, 4
    left, right = np.repeat(np.arange(n), per), np.arange(n * per)
    label = rng.random(len(left)) < .45
    prob = np.clip(np.where(label, .7, .4) + rng.normal(0, .22, len(left)), .01, .99)
    # Pairs of S1s share component AND fold. Ten S1s remain fixed competitors.
    components = np.arange(n) // 2
    folds = components % 5
    w = world(left, right, label, prob, n, component=components, folds=folds,
              truth=np.bincount(left[label], minlength=n) + (np.arange(n) % 7 == 0))
    selection = np.arange(50)
    rows = np.flatnonzero(w.left < 50)
    features = np.where(label[rows], 1., -1.) + rng.normal(0, 1, len(rows))
    scores = ev.Scores(rows, features[:, None], ["model:logit"], [0], {})
    return w, selection, scores


def keys(w, rows):
    rows = np.asarray(rows, np.int64)
    return {"pair_row": rows, "s1_id": [f"s{w.left[r]}" for r in rows],
            "rec_id": [f"r{w.right[r]}" for r in rows], "left_idx": w.left[rows], "right_idx": w.right[rows]}


class DecodeTests(unittest.TestCase):
    def test_singletons_and_unretrieved_truth_are_preserved(self):
        w = world([0, 1, 1], [0, 1, 2], [1, 0, 0], [.9, .1, .8], 4, truth=[2, 0, 3, 0])
        accepted = w.decode()
        counts = w.counts(accepted)
        np.testing.assert_allclose(ev.entity_f05(counts[1], counts[0], w.true_count), [5 / 6, 0, 0, 1])
        report = ev.with_countries(w, np.arange(4), *counts)
        self.assertEqual(report["s1"], 4)
        self.assertEqual(report["missed_true_links"], 4)
        self.assertEqual(report["s1_without_candidates"], 2)
        self.assertEqual(report["singleton_accuracy"], .5)
        self.assertEqual(report["false_links"], 1)
        self.assertAlmostEqual(report["micro_recall"], 1 / 5)
        self.assertEqual(set(report["by_country"]), {"India", "US"})

    def test_retrieval_oracle_does_not_inject_edges(self):
        w = world([0, 0, 1], [0, 1, 2], [1, 0, 1], [1., 0., 1.], 4, truth=[2, 1, 0, 1])
        result = ev.summary(w, np.arange(4), *w.counts(w.decode()))
        self.assertAlmostEqual(result["macro_f05"], (5 / 6 + 1 + 1 + 0) / 4)
        self.assertEqual(result["macro_f05"], result["retrieval_oracle_macro_f05"])
        self.assertEqual(len(w.left), 3)

    def test_owner_tie_and_cap_without_backfill(self):
        # Row 0 belongs to the higher S1; equal-score row 1 must win record 7.
        w = world([1, 0], [7, 7], [0, 1], [.8, .8], 2)
        np.testing.assert_array_equal(w.decode(), [1])
        # S1 0 owns all 12 ties, but the final record is capped, not given to S1 1.
        left, right = [0] * 12 + [1], list(range(12)) + [11]
        w = world(left, right, [1] * 12 + [0], [.8] * 13, 2)
        np.testing.assert_array_equal(w.decode(), np.arange(11))
        decoder = ev.ScreenDecoder(w, np.array([0]))
        local = decoder.decode(np.array([.1]))
        np.testing.assert_array_equal(decoder.materialize(local), np.arange(1, 12))

    def test_dominated_fixed_competitor_can_become_owner(self):
        w = world([0, 1], [1, 1], [1, 0], [.9, .8], 2)
        decoder = ev.ScreenDecoder(w, np.array([0]))
        np.testing.assert_array_equal(decoder.baseline, [0])
        result = decoder.decode(np.array([.1]))
        np.testing.assert_array_equal(decoder.materialize(result), [1])
        count, _ = decoder.counts_for(result, np.array([0, 1]))
        np.testing.assert_array_equal(count, [0, 1])

    def test_joint_owner_uses_unselected_competitor(self):
        w = world([0, 1], [1, 1], [0, 1], [.7, .9], 3, truth=[0, 1, 0])
        decoder = ev.ScreenDecoder(w, np.array([0]))
        np.testing.assert_array_equal(decoder.materialize(decoder.decode([.85])), [1])
        local = decoder.decode([.95])
        np.testing.assert_array_equal(decoder.materialize(local), [0])
        count, tp = decoder.counts_for(local, np.arange(3))
        np.testing.assert_array_equal(ev.entity_f05(tp, count, w.true_count), [0, 0, 1])

    def test_random_full_and_reduced_decoders_match_independent_oracle(self):
        w = random_world()
        rng = np.random.default_rng(17)
        mutable = np.sort(rng.choice(len(w.left), 130, replace=False))
        decoder = ev.ScreenDecoder(w, mutable)
        for _ in range(15):
            p = w.baseline_p.copy()
            p[mutable] = rng.choice([0., .5, .725, .8, .9, 1.], len(mutable))
            ts = np.full(len(p), ev.BASE_THRESHOLD, np.float32)
            ts[mutable] = rng.choice([.5, .725, .9], len(mutable))
            expected, scores, count, tp = oracle(w, p, ts)
            np.testing.assert_array_equal(w.decode(p, ts), expected)
            local = decoder.decode(p[mutable], ts[mutable])
            np.testing.assert_array_equal(decoder.materialize(local), expected)
            actual_count, actual_tp = decoder.counts_for(local, np.arange(len(w.population)))
            np.testing.assert_array_equal(actual_count, count)
            np.testing.assert_array_equal(actual_tp, tp)
            np.testing.assert_allclose(ev.entity_f05(actual_tp, actual_count, w.true_count), scores)

    def test_no_mutable_or_no_candidate_rows(self):
        for w in (random_world(), world([], [], [], [], 5, truth=[0, 1, 2, 0, 1])):
            decoder = ev.ScreenDecoder(w, np.empty(0, np.int64))
            local = decoder.decode(np.empty(0))
            np.testing.assert_array_equal(decoder.materialize(local), w.decode())

    def test_bad_shapes_truth_duplicates_and_nonfinite_baseline_rejected(self):
        w = world([0, 1], [1, 2], [1, 0], [.9, .2], 3)
        cases = [dict(left=np.array([0, 3])), dict(right=np.array([-1, 2])),
                 dict(baseline_p=np.array([.9, np.nan])), dict(baseline_p=np.array([.9, 1.1])),
                 dict(baseline_p=np.ones((2, 1))), dict(true_count=np.zeros(3, int)),
                 dict(left=np.array([0, 0]), right=np.array([1, 1])), dict(label=np.array([1, 2]))]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(w, **changes)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.w = world([0, 0, 1, 1, 2, 3], [0, 1, 2, 3, 4, 5], [1, 0, 1, 0, 1, 0],
                       [.8, .001, .7, .999, .8, .6], 5)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = self.write("ids", keys(self.w, np.arange(6)))

    def write(self, name, values):
        p = self.root / (name + ".parquet")
        pq.write_table(pa.table(values), p)
        return p

    def test_global_rows_reordered_scores_and_bounded_complete_case(self):
        p = self.write("scores", {**keys(self.w, [4, 2, 0, 1, 3, 5]), "logit": [2., 0., 1., 8., 9., 3.]})
        scores = ev.load_scores([p], self.w, np.arange(3), self.ids)
        np.testing.assert_array_equal(scores.rows, [0, 2, 4])
        np.testing.assert_array_equal(scores.features[:, 0], [1, 0, 2])
        self.assertEqual(scores.provenance["outside_band"], 2)
        self.assertEqual(scores.provenance["outside_selected_s1"], 1)
        self.assertIn("checked against", scores.provenance["key_verification"])

    def test_wrong_keys_local_rows_duplicates_nulls_and_bounds_rejected(self):
        valid = {**keys(self.w, [0, 2]), "prob": [.8, .4]}
        changes = [dict(pair_row=np.array([0, 0])), dict(pair_row=np.array([-1, 2])),
                   dict(pair_row=np.array([0, 6])), dict(pair_row=np.array([0., 2.])),
                   dict(pair_row=np.array([0, 1])), dict(s1_id=["wrong", "s1"]),
                   dict(rec_id=["r0", None]), dict(right_idx=np.array([0, 5])), dict(prob=[1.1, .2])]
        for i, change in enumerate(changes):
            p = self.write(str(i), {**valid, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                ev.load_scores([p], self.w, np.arange(5), self.ids)

    def test_cross_file_keys_and_duplicate_file_rejected(self):
        a = self.write("a", {**keys(self.w, [0]), "logit": [1.]})
        b = self.write("b", {**keys(self.w, [0]), "s1_id": ["wrong"], "logit": [2.]})
        for paths in ([a, b], [a, a]):
            with self.assertRaises(ValueError):
                ev.load_scores(paths, self.w, np.arange(5))

    def test_missing_nonfinite_and_partial_scores_fall_back_to_baseline(self):
        a = self.write("a", {**keys(self.w, [0, 2, 4]), "logit": [np.nan, np.inf, 2.], "prob": [.5, np.nan, ev.expit(2.)]})
        b = self.write("b", {**keys(self.w, [0, 2]), "logit": [-1., 3.]})
        result = ev.load_scores([a, b], self.w, np.arange(5), self.ids)
        np.testing.assert_array_equal(result.rows, [0])
        np.testing.assert_array_equal(result.features, [[0., -1.]])
        self.assertEqual(result.provenance["unchanged_selected_band_rows"], 3)
        self.assertEqual(result.provenance["incomplete_rows_in_selected_band"], 2)

    def test_byt5_ratio_features_and_invalid_token_count(self):
        values = {**keys(self.w, [0, 2]), "copy_nll": [2., 4.], "nonmatch_nll": [8., 7.],
                  "token_count": [3., 0.], "log_ratio": [6., 3.], "log_ratio_normalized": [2., np.inf]}
        p = self.write("byte", values)
        result = ev.load_scores([p], self.w, np.arange(5), self.ids)
        np.testing.assert_array_equal(result.rows, [0])
        np.testing.assert_array_equal(result.features, [[2, 8, 3, 6, 2]])
        self.assertEqual(result.primary_columns, [4])
        p = self.write("badbyte", {**values, "log_ratio": [4., 3.]})
        with self.assertRaisesRegex(ValueError, "log-ratio"):
            ev.load_scores([p], self.w, np.arange(5))

    def test_contradictory_probability_and_logit_rejected(self):
        p = self.write("bad", {**keys(self.w, [0]), "prob": [.1], "logit": [5.]})
        with self.assertRaisesRegex(ValueError, "disagree"):
            ev.load_scores([p], self.w, np.arange(5))


class CrossfitTests(unittest.TestCase):
    def test_component_fold_leak_and_selection_rejected(self):
        with self.assertRaisesRegex(ValueError, "Component fold leak"):
            world([0, 1], [0, 1], [1, 0], [.8, .6], 3, component=[7, 7, 8], folds=[0, 1, 2])
        w, selection, scores = learning_world()
        with self.assertRaisesRegex(ValueError, "selected-S1"):
            ev.crossfit_screen(w, scores, selection[:10])
        with self.assertRaisesRegex(ValueError, "three"):
            subset = selection[w.folds[selection] < 2]
            m = np.isin(w.left[scores.rows], subset)
            ev.crossfit_screen(w, replace(scores, rows=scores.rows[m], features=scores.features[m]), subset)

    def test_outer_labels_cannot_choose_its_model_threshold_or_blend(self):
        w, selection, scores = learning_world()
        held = w.folds[w.left] == 0
        changed = w.label.copy()
        changed[held] = ~changed[held]
        w2 = replace(w, label=changed, true_count=np.bincount(w.left[changed], minlength=len(w.population)) + 1)
        # Preserve non-held truth counts; only held labels/counts may differ.
        w2.true_count[w.folds != 0] = w.true_count[w.folds != 0]
        with patch.multiple(ev, THRESHOLDS=(.5, .725), BLENDS=(0., .25)):
            first, audit1 = ev.crossfit_screen(w, scores, selection)
            second, audit2 = ev.crossfit_screen(w2, scores, selection)
        self.assertEqual(audit1["per_outer_fold"]["0"], audit2["per_outer_fold"]["0"])
        for key, value in audit1["model_packs"].items():
            self.assertFalse(set(value["excluded_folds"]) & set(value["train_folds"]))
            if 0 in value["excluded_folds"]:
                self.assertEqual(value, audit2["model_packs"][key])
        take = w.folds[w.left[scores.rows]] == 0
        for method in ev.METHODS:
            np.testing.assert_array_equal(first[method]["prob"][take], second[method]["prob"][take])
            np.testing.assert_array_equal(first[method]["threshold"][take], second[method]["threshold"][take])

    def test_identical_new_score_has_identical_matched_blend_control(self):
        w, selection, scores = learning_world()
        scores.features[:, 0] = ev.logit(w.baseline_p[scores.rows])
        with patch.multiple(ev, THRESHOLDS=(.5, .725), BLENDS=(0., .25)):
            result, _ = ev.crossfit_screen(w, scores, selection)
        for key in ("prob", "threshold"):
            np.testing.assert_array_equal(result["convex_blend"][key], result["blend_baseline_control"][key])

    def test_small_models_have_finite_empty_and_single_class_fallbacks(self):
        for labels in (np.array([], bool), np.zeros(4, bool), np.ones(4, bool)):
            x = np.ones((len(labels), 2))
            model = ev.fit_logistic(x, labels)
            prediction = model.predict(np.array([[0., 100.]]), np.array([.8]))
            self.assertTrue(np.isfinite(prediction).all())
            if not len(labels):
                np.testing.assert_array_equal(prediction, [.8])


class ReportingTests(unittest.TestCase):
    def test_bootstrap_resamples_components_and_weights_s1(self):
        a, b, component = [0., 0., 0.], [1., 1., 0.], [8, 8, 9]
        result = ev.component_bootstrap(a, b, component, n=1000, seed=19)
        self.assertAlmostEqual(result["delta"], 2 / 3)
        self.assertEqual(result["components"], 2)
        self.assertEqual((result["ci_low"], result["ci_high"]), (0, 1))
        self.assertEqual(result, ev.component_bootstrap(a, b, component, n=1000, seed=19))
        one = ev.component_bootstrap(a, b, [8, 8, 8], n=1000)
        self.assertEqual(one["ci_low"], 2 / 3)
        self.assertEqual(one["ci_high"], 2 / 3)
        with self.assertRaises(ValueError):
            ev.component_bootstrap(a, b, component, n=999)

    def test_manifest_hash_and_baseline_cli(self):
        w = random_world(n=20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "brest_world.npz"
            names = ("left", "right", "label", "baseline_p", "true_count", "population", "component",
                     "country_code", "partition", "folds")
            np.savez(path, **{k: getattr(w, k) for k in names})
            marker = root / "brest_complete.json"
            manifest = {"state": "complete", "world_sha256": ev.digest(path), "holdout2_rows_read": 0,
                        "rows": len(w.left), "band_low": .001, "band_high": .999}
            marker.write_text(json.dumps(manifest))
            with redirect_stdout(io.StringIO()):
                ev.main(["baseline", "--world", str(path), "--output", str(root / "baseline")])
            report = json.loads((root / "baseline/report.json").read_text())
            _, expected, _, _ = oracle(w, w.baseline_p)
            self.assertAlmostEqual(report["metrics"]["baseline_fixed"]["macro_f05"], expected.mean())
            self.assertEqual(report["paired_reference_minus_fixed"]["macro_f05_component_bootstrap"]["replicates"], 1000)
            manifest["world_sha256"] = "wrong"
            marker.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ev.load_world(path)

    def test_screen_reports_complete_population_replay_controls_and_hashes(self):
        w, selection, scores = learning_world()
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(ev, THRESHOLDS=(.5, .725), BLENDS=(0., .25)):
            root = Path(tmp) / "result"
            report = ev.run_screen(w, scores, selection, root)
            self.assertEqual(report["metrics"]["baseline_fixed"]["s1"], len(selection))
            self.assertEqual(report["fixed_competitors"]["baseline_fixed"]["s1"], 10)
            self.assertFalse(report["promotion_ready"])
            self.assertIn("not a fully nested", report["limitations"])
            for path, sha in report["artifacts"].items():
                self.assertEqual(ev.digest(root / path), sha)
            baseline = w.decode()
            with np.load(root / "per_s1.npz") as vectors, np.load(root / "screen_scores.npz") as replay, \
                    np.load(root / "decision_changes.npz") as changes:
                np.testing.assert_array_equal(vectors["s1_rows"], np.arange(60))
                self.assertEqual(int(vectors["selected"].sum()), 50)
                for method in ev.METHODS:
                    p = w.baseline_p.copy()
                    threshold = np.full(len(p), ev.BASE_THRESHOLD, p.dtype)
                    p[replay["pair_row"]] = replay[method + "_prob"]
                    threshold[replay["pair_row"]] = replay[method + "_threshold"]
                    accepted, expected, _, _ = oracle(w, p, threshold)
                    np.testing.assert_allclose(vectors[method + "_f05"], expected)
                    np.testing.assert_array_equal(changes[method + "_added_rows"], np.setdiff1d(accepted, baseline))
                    np.testing.assert_array_equal(changes[method + "_dropped_rows"], np.setdiff1d(baseline, accepted))
                for method, control in ev.CONTROLS.items():
                    result = report["paired"][method + "_minus_" + control]
                    self.assertEqual(result["macro_f05_component_bootstrap"]["s1"], 50)
            # JSON artifacts must use null, not NaN/Infinity, for unavailable metrics.
            def reject_constant(value):
                raise AssertionError(value)
            json.loads((root / "report.json").read_text(), parse_constant=reject_constant)
            with self.assertRaises(FileExistsError):
                ev._output_dir(root)


if __name__ == "__main__":
    unittest.main()
