"""P5: full-S1 singleton coverage, one-to-one conflicts and no test/TRAIN ID mixing."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ber import v2_ce_train as ce
from ber import v2_test_adaptation as p5


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def raw_table(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def fixture(root: Path) -> dict:
    raw = root / "test"
    raw.mkdir()
    references = pd.DataFrame([
        ("S1-U1", "Rose Bakery", "10 High St", "US"),
        ("S1-U2", "Other Bakery", "11 High St", "US"),
        ("S1-F1", "École", "1 Rue Victor Hugo", "France"),
        ("S1-F2", "Lycée", "2 Rue A", "France"),
        ("S1-I1", "दुकान", "3 Main Road", "India"),
        ("S1-I2", "दुकान", "4 Main Road", "India"),
    ], columns=p5.RECORD_COLUMNS)
    source2 = pd.DataFrame([
        ("S2-US", "Rose Bakery", "10 High St", "US"),
        ("S2-I", "दुकान", "3 Main Road", "India"),
        ("S2-TWIN", "Duplicate US", "99 Elm", "US"),
    ], columns=p5.RECORD_COLUMNS)
    source3 = pd.DataFrame([
        ("S3-F", "École", "1 Rue Victor Hugo", "France"),
        ("S3-TWIN", "Duplicate US", "99 Elm", "US"),
    ], columns=p5.RECORD_COLUMNS)
    pool = pd.concat([source2, source3], ignore_index=True)
    for source, frame in (("source1", references), ("source2", source2), ("source3", source3)):
        raw_table(raw / f"test_{source}.tsv", frame)
    inventory = {"files": [{"path": f"data/raw/dataset/test/test_{source}.tsv",
                            "bytes": (raw / f"test_{source}.tsv").stat().st_size,
                            "sha256": p5.sha256(raw / f"test_{source}.tsv")}
                           for source in ("source1", "source2", "source3")]}
    integrity = root / "source_integrity.json"
    write_json(integrity, inventory)
    selected = root / "scored"
    selected.mkdir()
    references.to_parquet(selected / "reference.parquet", index=False)
    pool.to_parquet(selected / "pool.parquet", index=False)
    # Target 0 gets conflicting >=.98 claims; target 3/4 form an exact-text cluster.
    pairs = pd.DataFrame({"left_idx": [0, 0, 0, 1, 1, 2, 4],
                          "right_idx": [0, 2, 4, 0, 4, 3, 1]})
    scores = np.array([.99, .019, .01, .985, .005, .999, .015], dtype=np.float32)
    pairs.to_parquet(selected / "pairs.parquet", index=False, row_group_size=2)
    np.save(selected / "probabilities.npy", scores)
    retrieval = {"state": "complete", "config": {"positive_candidate_injection": False,
                                                     "labels_accessed": False},
                 "metadata": {"mining_full_pool_records": 5, "candidate_pairs": len(pairs),
                              "reference_sha256": p5.sha256(selected / "reference.parquet"),
                              "pool_sha256": p5.sha256(selected / "pool.parquet"),
                              "candidate_pairs_sha256": p5.sha256(selected / "pairs.parquet")}}
    write_json(selected / "retrieval_complete.json", retrieval)
    tune = root / "tune.json"
    write_json(tune, {"state": "completed_fixed_TUNE", "labels_from_test": False})
    calibration = root / "calibration.json"
    write_json(calibration, {"state": "frozen", "fit": "TUNE_only", "France": "pooled_fallback"})
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"toy BGE weights - identity guard, not a real model")
    write_json(checkpoint / "training_config.json", {"state": "TRAIN/TUNE only"})
    write_json(checkpoint / "config.json", {"num_labels": 1})
    write_json(checkpoint / "tokenizer.json", {})
    write_json(checkpoint / "complete.json", {"state": "complete", "files_sha256": {
        name: p5.sha256(checkpoint / name) for name in
        ("model.safetensors", "training_config.json", "config.json", "tokenizer.json")}})
    allowlist = root / "inventory.yaml"
    allowlist.write_text("budget_total_params_b: 8.0\nmodels:\n"
                         "  - {id: BAAI/bge-reranker-v2-m3, licence: apache-2.0, params_b: 0.568}\n"
                         "  - {id: intfloat/multilingual-e5-small, licence: mit, params_b: 0.118}\n"
                         "  - {id: big, licence: mit, params_b: 7.5}\n", encoding="utf-8")
    frozen = root / "frozen_v2.json"
    write_json(frozen, {"schema": 1, "version": "v2", "state": "frozen", "selected_on": "TUNE",
                        "holdout2_labels_used": False, "test_labels_used": False,
                        "positive_candidate_injection": False, "ce_checkpoint_path": str(checkpoint),
                        "ce_checkpoint_complete_sha256": p5.sha256(checkpoint / "complete.json"),
                        "ce_model_id": "BAAI/bge-reranker-v2-m3",
                        "model_ids": ["intfloat/multilingual-e5-small", "BAAI/bge-reranker-v2-m3"],
                        "allowlist_sha256": p5.sha256(allowlist),
                        "calibration": {"path": str(calibration), "sha256": p5.sha256(calibration),
                                        "fit_scope": "TUNE_US_India_only",
                                        "france_policy": "pooled_TUNE_no_France_labels"},
                        "tune": {"references": 12_000, "id_sha256": p5.TUNE_ID_SHA256,
                                 "labels_used_only_for_selection": True, "report_path": str(tune),
                                 "report_sha256": p5.sha256(tune)}})
    verified = p5.checked_frozen(frozen, allowlist=allowlist)
    source_ids = {name: p5.sha256(raw / f"test_{name}.tsv") for name in ("source1", "source2", "source3")}
    files = {name: {"path": f"{filename}", "sha256": p5.sha256(selected / filename), "rows": rows}
             for name, filename, rows in (("reference", "reference.parquet", 6), ("pool", "pool.parquet", 5),
                                          ("pairs", "pairs.parquet", len(pairs)),
                                          ("probabilities", "probabilities.npy", len(pairs)),
                                          ("retrieval_complete", "retrieval_complete.json", 1))}
    marker = selected / "score_manifest.json"
    write_json(marker, {"schema": 1, "state": "complete", "population": "official_test",
                        "labels_read": False, "positive_candidate_injection": False,
                        "score_semantics": "TUNE_calibrated_match_probability",
                        "all_candidates_calibrated": True,
                        "frozen_v2_sha256": verified["sha256"],
                        "calibration_sha256": verified["calibration_sha256"],
                        "ce_checkpoint_complete_sha256": verified["checkpoint_complete_sha256"],
                        "full_target_pool_rows": 5, "reference_count": 6, "candidate_pairs": len(pairs),
                        "test_input_sha256": source_ids, "retrieved_pair_sha256": files["pairs"]["sha256"],
                        "files": files})
    inputs = p5._test_inputs(raw, integrity)
    scored = p5.checked_scored_v2(marker, verified, inputs, nreference=6, full_targets=5)
    return {"raw": raw, "integrity": integrity, "references": references, "pool": pool,
            "pairs": pairs, "scores": scores, "scored": scored, "marker": marker,
            "frozen": frozen, "checked_frozen": verified, "allowlist": allowlist, "root": root}


class V2TestAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_balanced_score_and_cluster_diagnostics_have_all_six_references(self):
        x = self.data
        result = p5.diagnose_scores(x["scored"], x["raw"], x["integrity"], self.root / "diagnose",
                                    sample_per_country=2, batch_size=2)
        self.assertEqual(result["full_references"], 6)
        self.assertEqual(result["full_scored_pairs"], 7)
        self.assertEqual([result["by_country"][c]["sampled_references"] for c in p5.COUNTRIES], [2] * 3)
        self.assertEqual(result["by_country"]["France"]["full_zero_candidate_references"], 1)
        self.assertEqual(result["by_country"]["India"]["full_zero_candidate_references"], 1)
        self.assertEqual(result["by_country"]["US"]["candidate_pair_scores"]["count"], 5)
        self.assertEqual(result["balanced_text_diagnostics"]["by_country"]["US"]["references_with_two_observed_same_cluster_candidates"], 1)
        self.assertIsNotNone(result["france_vs_india_us"]["max_score_ks"])
        self.assertTrue(result["no_french_threshold_selection"])
        self.assertFalse(result["labels_read"])
        self.assertTrue((self.root / "diagnose/diagnostics.json").exists())

    def test_conflicting_high_claims_all_dropped_and_empty_singletons_retained(self):
        x = self.data
        report = p5.select_pseudo(x["scored"], x["raw"], x["integrity"], x["checked_frozen"],
                                  self.root / "selected", batch_size=2)
        self.assertEqual(report["conflicting_targets"], 1)
        self.assertEqual(report["counts"]["high_raw"], 3)
        self.assertEqual(report["counts"]["high_discarded_conflict"], 2)
        self.assertEqual(report["counts"]["pseudo_positive"], 1)
        self.assertEqual(report["counts"]["pseudo_negative"], 4)
        self.assertEqual(report["counts"]["references_without_candidates"], 2)
        selected, pairs = p5.checked_selection(self.root / "selected", x["scored"], x["checked_frozen"])
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs.loc[pairs.pseudo_label, ["left_idx", "right_idx"]].values.tolist(), [[2, 3]])
        self.assertEqual(sorted(pairs.loc[~pairs.pseudo_label, "left_idx"]), [0, 0, 1, 4])
        audit = pd.read_parquet(self.root / "selected/entity_summary.parquet")
        self.assertEqual(len(audit), 6)
        self.assertEqual(audit.entity_id.tolist(), x["references"].entity_id.tolist())
        self.assertEqual(audit.loc[3, "candidate_pairs"], 0)
        self.assertEqual(audit.loc[5, "pseudo_positives"], 0)
        self.assertTrue(pairs.weight.le(.3).all())
        self.assertFalse(selected["positive_candidate_injection"])

    def test_selection_rejects_a_tampered_edge_score_or_missing_singleton_audit(self):
        x = self.data
        directory = self.root / "selected"
        p5.select_pseudo(x["scored"], x["raw"], x["integrity"], x["checked_frozen"],
                         directory, batch_size=2)
        marker_path = directory / "selection.json"
        pair_path = directory / "pseudo_pairs.parquet"
        pairs = pd.read_parquet(pair_path)
        pairs.loc[0, "p"] = .018  # Still within <=.02, but no longer the frozen candidate's score.
        pairs.to_parquet(pair_path, index=False)
        modified = json.loads(marker_path.read_text())
        modified["outputs"]["pseudo_pairs"]["sha256"] = p5.sha256(pair_path)
        write_json(marker_path, modified)
        with self.assertRaisesRegex(ValueError, "not in frozen test retrieval"):
            p5.checked_selection(directory, x["scored"], x["checked_frozen"])
        pairs.loc[0, "p"] = float(x["scores"][1])  # first selected edge is (0,2)
        pairs.to_parquet(pair_path, index=False)
        modified["outputs"]["pseudo_pairs"]["sha256"] = p5.sha256(pair_path)
        audit_path = directory / "entity_summary.parquet"
        audit = pd.read_parquet(audit_path)
        audit.loc[5, "left_idx"] = 4  # Same row count, but the zero-candidate S1 is lost.
        audit.to_parquet(audit_path, index=False)
        modified["outputs"]["entity_summary"]["sha256"] = p5.sha256(audit_path)
        write_json(marker_path, modified)
        with self.assertRaisesRegex(ValueError, "singleton"):
            p5.checked_selection(directory, x["scored"], x["checked_frozen"])

    def test_pool_duplicate_target_ids_and_nonfrozen_fit_are_rejected(self):
        x = self.data
        pool = x["pool"].copy()
        pool.loc[4, "entity_id"] = pool.entity_id.iat[2]
        path = self.root / "duplicate_target_ids.parquet"
        pool.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            p5._pool_codes(path, len(pool))
        with self.assertRaisesRegex(ValueError, "launch-frozen-fit"):
            p5.fit(SimpleNamespace(launch_frozen_fit=False, steps=5_000, device="cuda"),
                   x["scored"], x["checked_frozen"])
        with self.assertRaisesRegex(ValueError, "fixed decision adapter"):
            p5.fit_preflight(SimpleNamespace(selection=self.root / "missing"),
                             x["scored"], x["checked_frozen"])

    def test_fixed_tune_decision_keeps_singletons_and_refuses_incompatible_v2_rule(self):
        tune = SimpleNamespace(reference=pd.DataFrame({"entity_id": ["S1-T1", "S1-T2"],
                                                         "business_name": ["A", "B"],
                                                         "business_address": ["1 Main St", "2 Main St"],
                                                         "country": ["US", "US"]}),
                               pairs=pd.DataFrame({"left_idx": [0], "right_idx": [0]}),
                               labels=np.array([True]),
                               truth={"S1-T1": {"S2-T"}, "S1-T2": set()},
                               base=np.array([.5], dtype=np.float32))
        decision = {"kind": "tree_ce_blend_threshold", "tree_gate": .01,
                    "ce_weight": .75, "threshold": .75}
        report, frame, values = p5.fixed_tune_decision(decision, tune, np.array([.9], dtype=np.float32))
        self.assertEqual(report["reference_count"], 2)
        self.assertEqual(report["macro_f05"], 1.)
        self.assertEqual(report["singleton_accuracy"], 1.)
        self.assertEqual(len(frame), 2)
        self.assertGreater(float(values[0]), decision["threshold"])
        with self.assertRaisesRegex(ValueError, "decision adapter"):
            p5.fixed_tune_decision({**decision, "kind": "expected_f05_plus_assignment"},
                                   tune, np.array([.9], dtype=np.float32))

    def test_completed_generated_roundtrip_preserves_raw_door_digits(self):
        x = self.data
        checkpoint = self.root / "denoiser_checkpoint_complete.json"
        write_json(checkpoint, {"state": "complete", "model_id": "google/byt5-small"})
        frame = pd.DataFrame({"entity_id": x["references"].entity_id,
                              "generated": [f"{n} | {a}" for n, a in x["references"][
                                  ["business_name", "business_address"]].itertuples(index=False, name=None)],
                              "input_truncated": [False] * 6, "output_truncated": [False] * 6})
        frame.loc[2, "generated"] = "École | 2 Rue Victor Hugo"  # France digit hallucination -> fallback.
        probes = self.root / "generated.parquet"
        frame.to_parquet(probes, index=False)
        marker = self.root / "generated_complete.json"
        write_json(marker, {"state": "complete", "labels_read": False,
                            "positive_candidate_injection": False,
                            "denoiser_checkpoint_complete_path": str(checkpoint),
                            "denoiser_checkpoint_complete_sha256": p5.sha256(checkpoint),
                            "test_input_sha256": {name: p5.sha256(x["raw"] / f"test_{name}.tsv")
                                                  for name in ("source1", "source2", "source3")},
                            "files": {"probes": {"path": str(probes), "sha256": p5.sha256(probes), "rows": 6}}})
        report = p5.generated_roundtrip(marker, x["raw"], x["integrity"],
                                        sample_per_country=2, nreference=6)
        self.assertEqual(report["by_country"]["France"]["reason/address_digits_changed"], 1)
        self.assertEqual(report["by_country"]["France"]["digit_preserving"], 2)
        self.assertEqual(report["by_country"]["France"]["references"], 2)
        self.assertIn("not denoising accuracy", report["limitation"])

    def test_duplicate_pair_across_row_batches_or_country_crossing_is_rejected(self):
        x = self.data
        ref_codes = np.asarray([p5.COUNTRY_CODE[x["references"].country.iat[i]] for i in range(6)], dtype=np.uint8)
        pool_codes = np.asarray([p5.COUNTRY_CODE[x["pool"].country.iat[i]] for i in range(5)], dtype=np.uint8)
        pairs = x["pairs"].copy()
        pairs.loc[2, "right_idx"] = 2  # Duplicate (0,2), spanning a 2-row batch boundary.
        path = self.root / "duplicate_pairs.parquet"
        pairs.to_parquet(path, index=False, row_group_size=2)
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            list(p5._score_batches(replace(x["scored"], pairs=path), ref_codes, pool_codes, batch_size=2))
        pairs = x["pairs"].copy()
        pairs.loc[0, "right_idx"] = 1  # India target under a US S1.
        pairs.to_parquet(path, index=False, row_group_size=2)
        with self.assertRaisesRegex(ValueError, "country"):
            list(p5._score_batches(replace(x["scored"], pairs=path), ref_codes, pool_codes, batch_size=2))

    def test_frozen_tune_license_param_cap_and_injected_retrieval_gates(self):
        x = self.data
        saved = json.loads(x["frozen"].read_text())
        for field in ("state", "selected_on", "holdout2_labels_used", "test_labels_used"):
            with self.subTest(field=field):
                changed = dict(saved)
                changed[field] = "unfinished" if field in ("state", "selected_on") else True
                write_json(x["frozen"], changed)
                with self.assertRaises(ValueError):
                    p5.checked_frozen(x["frozen"], allowlist=x["allowlist"])
        write_json(x["frozen"], {**saved, "model_ids": saved["model_ids"] + ["big"]})
        with self.assertRaisesRegex(ValueError, "parameter cap"):
            p5.checked_frozen(x["frozen"], allowlist=x["allowlist"])
        write_json(x["frozen"], saved)
        marker = json.loads(x["marker"].read_text())
        write_json(x["marker"], {**marker, "positive_candidate_injection": True})
        with self.assertRaisesRegex(ValueError, "injection"):
            p5.checked_scored_v2(x["marker"], x["checked_frozen"],
                                 p5._test_inputs(x["raw"], x["integrity"]), nreference=6, full_targets=5)
        write_json(x["marker"], marker)
        retrieval = x["marker"].parent / "retrieval_complete.json"
        original = json.loads(retrieval.read_text())
        original["config"]["positive_candidate_injection"] = True
        write_json(retrieval, original)
        marker["files"]["retrieval_complete"]["sha256"] = p5.sha256(retrieval)
        write_json(x["marker"], marker)
        with self.assertRaisesRegex(ValueError, "never truth-injected"):
            p5.checked_scored_v2(x["marker"], x["checked_frozen"],
                                 p5._test_inputs(x["raw"], x["integrity"]), nreference=6, full_targets=5)

    def test_train_and_test_ids_must_never_share_a_population(self):
        x = self.data
        chosen = pd.DataFrame({"left_idx": [2, 0], "right_idx": [3, 2],
                               "p": [.999, .01], "pseudo_label": [True, False], "weight": [.3, .3]})
        train_ref = pd.DataFrame({"entity_id": ["S1-TRAIN"]})
        train_pool = pd.DataFrame({"entity_id": ["S2-TRAIN"]})
        tune_ref = pd.DataFrame({"entity_id": ["S1-TUNE"]})
        check = p5.assert_disjoint_populations(train_ref, train_pool, tune_ref,
                                               x["references"], x["pool"], chosen)
        self.assertTrue(check["no_shared_population_ids"])
        for modified_ref, modified_pool in ((pd.DataFrame({"entity_id": ["S1-F1"]}), train_pool),
                                            (train_ref, pd.DataFrame({"entity_id": ["S2-TWIN"]}))):
            with self.assertRaisesRegex(ValueError, "mixing"):
                p5.assert_disjoint_populations(modified_ref, modified_pool, tune_ref,
                                               x["references"], x["pool"], chosen)
        with self.assertRaisesRegex(ValueError, "country"):
            other = chosen.copy()
            other.loc[0, "right_idx"] = 1  # India target for France.
            p5.assert_disjoint_populations(train_ref, train_pool, tune_ref,
                                           x["references"], x["pool"], other)

    def test_balanced_real_and_pseudo_ce_slots_and_weight_cap(self):
        positives = np.array([[0, 0], [1, 1], [2, 2]])
        negatives = np.zeros(3, dtype=ce.PAIR_DTYPE)
        negatives["left"] = [0, 1, 2]
        negatives["right"] = [2, 0, 1]
        negatives["channels"] = [7, 7, 7]
        pseudo = pd.DataFrame({"left_idx": [0, 1], "right_idx": [0, 1],
                               "pseudo_label": [True, False]})
        plan = p5.mixed_batch_plan(positives, negatives, pseudo, ce.PositiveCycle(3, 7),
                                   ce.hard_negative_indices(negatives), np.random.default_rng(8))
        self.assertEqual(len(plan), 256)
        self.assertEqual(sum(row[3] for row in plan), 128)
        self.assertEqual(sum(row[0] == "test" for row in plan), 64)
        self.assertEqual(sum(row[0] == "test" and row[3] == 1 for row in plan), 32)
        self.assertEqual({row[4] for row in plan if row[0] == "train"}, {1.})
        self.assertEqual({row[4] for row in plan if row[0] == "test"}, {.3})
        with self.assertRaisesRegex(ValueError, "capped"):
            p5.mixed_batch_plan(positives, negatives, pseudo, ce.PositiveCycle(3, 7),
                                ce.hard_negative_indices(negatives), np.random.default_rng(8),
                                pseudo_fraction=.31)

    def test_validated_v1_screen_is_label_free_and_rejects_missing_s1(self):
        x = self.data
        release_dir = self.root / "docs"
        release_dir.mkdir()
        output = self.root / "output/submission_v1"
        output.mkdir(parents=True)
        ids = x["references"].entity_id.tolist()
        rows = ["source1_entity_id\tmatched_entity_ids",
                ids[0] + "\tS2-TWIN,S3-TWIN", ids[1] + "\t", ids[2] + "\tS3-F",
                ids[3] + "\t", ids[4] + "\tS2-I", ids[5] + "\t"]
        matching = output / "matching_results.tsv"
        matching.write_text("\r\n".join(rows) + "\r\n", encoding="utf-8")
        write_json(output / "report.json", {"accepted_pairs": 4})
        release = release_dir / "submission_v1_release.json"
        write_json(release, {"source1_rows": 6, "matching_results": {"path": "output/submission_v1/matching_results.tsv",
                         "sha256": p5.sha256(matching), "bytes": matching.stat().st_size}})
        result = p5.screen_v1(x["raw"], x["integrity"], release, self.root / "screen_v1",
                              sample_per_country=2, expected_references=6)
        self.assertEqual(result["full_reference_count"], 6)
        self.assertEqual(result["full_by_country"]["US"]["accepted_links"], 2)
        self.assertEqual(result["full_by_country"]["India"]["empty_predictions"], 1)
        self.assertEqual(result["balanced_text_diagnostics"]["by_country"]["US"]["targets_in_observed_exact_duplicate_group"], 2)
        self.assertEqual(result["score_distribution_shift"]["status"],
                         "requires aligned saved score array; binary v1 TSV is not a probability")
        self.assertFalse(result["pseudo_labels_created"])
        matching.write_text("\r\n".join(rows[:-1]) + "\r\n", encoding="utf-8")
        release_doc = json.loads(release.read_text())
        release_doc["matching_results"].update(sha256=p5.sha256(matching), bytes=matching.stat().st_size)
        write_json(release, release_doc)
        with self.assertRaisesRegex(ValueError, "omitted"):
            p5.screen_v1(x["raw"], x["integrity"], release, self.root / "missing",
                         sample_per_country=2, expected_references=6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Run real BCE derivative test on the GPU pod")
    def test_actual_ce_bce_objective_rejects_overweight_test_pairs(self):
        import torch

        logits = torch.tensor([0., 0.], requires_grad=True)
        labels = torch.tensor([1., 0.])
        weights = torch.tensor([1., .3])
        loss = p5.weighted_binary_ce(logits, labels, weights, ["train", "test"])
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), float(torch.nn.functional.softplus(torch.tensor(0.))), places=6)
        self.assertAlmostEqual(float(logits.grad[0] / logits.grad[1]), -1 / .3, places=6)
        with self.assertRaisesRegex(ValueError, "<=.3"):
            p5.weighted_binary_ce(logits.detach(), labels, torch.tensor([1., .31]), ["train", "test"])


if __name__ == "__main__":
    unittest.main()
