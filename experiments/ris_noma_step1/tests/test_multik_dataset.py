"""7b: real bank/constraint checks, data leakage guards and resumable exports."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ris_noma.channels import build_bank, public_order
from ris_noma.config import load_config
from ris_noma.learning_data import read_json, sha256
from ris_noma.multik_core import assess_allocation
from ris_noma import multik_dataset as data
from ris_noma.multik_data_core import (STREAMS, at_target, audit_group, choose_label,
                                      context, geometry, raw_assessment, stream)
from ris_noma.pso import PSOSettings

ROOT = Path(__file__).resolve().parents[1]


def fixture(folder):
    """Mock only prior-step inspection; generation and all exports run for real."""
    previous = Path(folder) / "previous"
    previous.mkdir()
    source = {"base_config": load_config(ROOT / "configs/legacy_physics_k2_pilot.json"),
              "experiment": {"position_half_width_m": {"bob": 5., "willie": 4.}}}
    data.write_json(previous / "source_physics_plan.json", source)
    for name in ("plan.json", "summary.json", "completion.json"):
        data.write_json(previous / name, {"unit_test_fixture": name})
    # Windows-style source bytes must survive copying on any host platform.
    for path in previous.iterdir():
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    c = read_json(ROOT / "configs/multik_dataset_7b.json")
    c.update(training_public_counts=[2], unseen_public_counts=[3],
             splits={"train": [1200], "validation": [1201], "test": [1202]},
             power_dbm=[35.], xi=[.75, .95], search_samples=1000,
             selection_samples=1000, heldout_samples=1000,
             pso_evaluations=30, pso_seeds=[101])
    jobs = data.make_jobs(c)
    plan = {"step": "7b", "format": data.FORMAT, "graph_schema": data.SCHEMA_VERSION,
            "config": c, "jobs": jobs, "streams": STREAMS,
            "pso_settings": asdict(PSOSettings(swarm_size=30, sweeps=1)),
            "planned_counts": {"train": 2, "validation": 2, "test": 2, "test_unseen_k": 2},
            "planned_cases": 8, "max_pso_evaluations": 240,
            "previous_sha256": {name: sha256(previous / name) for name in ("plan.json", "summary.json", "completion.json", "source_physics_plan.json")},
            "code_sha256": {}, "versions": {"purpose": "unit_test_fixture"}}
    return previous, plan, source


def fingerprints(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in Path(folder).rglob("*") if p.is_file()}


class DatasetPolicyTests(unittest.TestCase):
    def test_main_plan_counts_and_unseen_k_never_crosses_split(self):
        c = data.validate_settings(read_json(ROOT / "configs/multik_dataset_7b.json"))
        jobs = data.make_jobs(c)
        self.assertEqual(len(jobs), 249)
        self.assertEqual(sum(len(v) for v in c["splits"].values()), 60)
        counts = {name: sum(data.role(j) == name for j in jobs) * 15 for name in ("train", "validation", "test", "test_unseen_k")}
        self.assertEqual(counts, {"train": 2520, "validation": 540, "test": 540, "test_unseen_k": 135})
        self.assertTrue(all(j["split"] == "test" for j in jobs if j["regime"] == "unseen_k"))

    def test_invalid_config_leakage_and_nonfinite_values_rejected(self):
        original = read_json(ROOT / "configs/multik_dataset_7b.json")
        for changes in ({"training_public_counts": [2, True]}, {"unseen_public_counts": [5]},
                        {"power_dbm": [float("nan")]}, {"xi": [.95, .75]},
                        {"search_margin": .05}, {"alpha": False}, {"pso_evaluations": 31},
                        {"search_samples": 999}, {"pso_seeds": [7, 7]},
                        {"splits": {"train": [1000], "validation": [1000], "test": [1001]}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                data.validate_settings(original | changes)

    def test_common_pool_can_select_another_origin_dep(self):
        raw = {"bob_rate": 3., "dep_gamma": .97, "dep_lower": .96,
               "point_checks_without_dep": {"qos": True}, "bound_checks_without_dep": {"qos_bound": True}}
        proposals = [{"candidate_id": "a", "origin_xi": .95, "search_rate": 3.,
                      "search_feasible_by_xi": [True, True], "selection": raw},
                     {"candidate_id": "b", "origin_xi": .75, "search_rate": 4.,
                      "search_feasible_by_xi": [True, False], "selection": raw | {"bob_rate": 4.}}]
        self.assertEqual(data.choose_label(proposals, 0, .75)["candidate_id"], "b")
        self.assertEqual(data.choose_label(proposals, 1, .95)["origin_xi"], .95)
        failed = deepcopy(proposals)
        failed[0]["selection"]["dep_lower"] = .94
        self.assertIsNone(data.choose_label(failed, 1, .95))

    def test_non_dep_failure_cannot_pass_at_a_looser_dep(self):
        raw = {"bob_rate": 8., "dep_gamma": .99, "dep_lower": .98,
               "point_checks_without_dep": {"qos": False}, "bound_checks_without_dep": {"outage": True}}
        self.assertFalse(at_target(raw, .75)["passed"])
        self.assertEqual(at_target(raw, .75)["gated_bob_rate"], 0.)


class DatasetPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        _, self.plan, self.source = fixture(self.temp.name)
        self.job = self.plan["jobs"][0]

    def test_geometry_prefix_is_fixed_across_k_and_banks(self):
        c2, a = geometry(self.plan, self.source, self.job, 1000)
        c5, b = geometry(self.plan, self.source, self.job | {"n_public": 5}, 1000)
        np.testing.assert_array_equal(a.public_xy, b.public_xy[:2])
        np.testing.assert_array_equal(a.receiver_xy[[0, -1]], b.receiver_xy[[0, -1]])
        self.assertEqual(c2["positions"], c5["positions"])
        self.assertEqual(self.source["base_config"]["n_public"], 2)

    def test_bank_streams_distinct_reproducible_and_do_not_depend_on_power(self):
        values = [stream(self.plan, self.source, self.job, name).random(12) for name in ("search", "selection", "heldout")]
        for i in range(len(values)):
            for j in range(i):
                self.assertFalse(np.array_equal(values[i], values[j]))
        np.testing.assert_array_equal(values[0], stream(self.plan, self.source, self.job, "search").random(12))
        self.assertNotEqual(STREAMS["search"], 72)

    def test_compact_bank_and_retarget_match_unchanged_full_evaluator(self):
        physical, layout, compact = context(self.plan, self.source, self.job, "selection")
        full = build_bank(physical, layout, stream(self.plan, self.source, self.job, "selection"))
        order = public_order(full)
        allocation = [.12, .40, .48, .5]
        raw = raw_assessment(physical, layout, compact, order, allocation, 4, .01)
        for xi in (.75, .85, .95):
            info, screen = assess_allocation(physical, layout, full, order, allocation, xi, 4, .01)
            self.assertEqual(at_target(raw, xi)["passed"], screen["accepted"])
            self.assertAlmostEqual(raw["bob_rate"], info["bob_mean_rate_bit_s_hz"])
            self.assertAlmostEqual(raw["dep_lower"], screen["dep_uniform_lower"])
            np.testing.assert_array_equal(raw["public_rate_lower_sic"], screen["public_rate_lower_sic_order"])
        self.assertFalse(hasattr(compact, "h_ru"))

    def test_heldout_failure_never_changes_selected_label(self):
        physical, layout, bank = context(self.plan, self.source, self.job, "search")
        order = public_order(bank).tolist()
        from ris_noma.multik_graph import statistical_graph
        graph = statistical_graph(physical, layout, bank, np.array(order), .75)
        allocation = [.12, .4, .48, .5]
        raw = {"bob_rate": 3., "dep_gamma": .96, "dep_lower": .70,
               "point_checks_without_dep": {"qos": True}, "bound_checks_without_dep": {"qos_bound": True}}
        choice = {"xi": .75, "label_available": True, "selected_candidate_id": "a", "selected_origin_xi": .75,
                  "allocation": allocation, "no_label_reason": None}
        selected = self.job | {"order": order, "geometry_valid": True, "layout_eligible": True,
                              "powers": [{"power_dbm": 35., "choices": [choice],
                                          "candidates": [{"candidate_id": "a", "search_rate": 3., "selection": raw}]}]}
        search = {"geometry_valid": True, "powers": [{"graphs": [graph]}]}
        with patch("ris_noma.multik_data_core.raw_assessment", return_value=raw):
            result = audit_group(self.plan, self.source, search, selected, "fixed_selection_hash")
        row = result["rows"][0]
        self.assertTrue(row["label_available"])
        self.assertFalse(row["heldout"]["passed"])
        self.assertEqual(row["allocation"], allocation)
        self.assertFalse(row["heldout_used_for_label_filter"])


class DatasetRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.previous, self.plan, self.source = fixture(self.folder)
        self.output = self.folder / "data"

    def generate(self, output=None, workers=1, resume=False):
        with patch("ris_noma.multik_dataset.inspect", return_value=(self.plan, self.source)):
            return data.run(self.previous, self.output if output is None else output, self.plan["config"], workers, resume)

    def test_end_to_end_exports_missing_cases_and_replays_banks(self):
        result = self.generate()
        self.assertTrue(result["complete"])
        self.assertEqual(result["planned_cases"], 8)
        self.assertEqual(result["pso_evaluations"], 240)
        checked = data.check(self.output, replay_groups=1)
        self.assertEqual(checked["replayed_groups"], 1)
        self.assertEqual(len(data.read_jsonl(self.output / "records.jsonl")), 8)
        self.assertFalse((self.output / "normalization.json").exists())
        self.assertFalse((self.output / "models").exists())
        self.assertEqual((self.output / "source_physics_plan.json").read_bytes(), (self.previous / "source_physics_plan.json").read_bytes())
        train = data.read_jsonl(self.output / "graphs/train.jsonl")
        unseen = data.read_jsonl(self.output / "graphs/test_unseen_k.jsonl")
        self.assertTrue(all(row["n_public"] == 2 for row in train))
        self.assertTrue(all(row["n_public"] == 3 and row["split"] == "test" for row in unseen))

    def test_existing_and_changed_output_rejected_without_mutation(self):
        self.generate()
        before = fingerprints(self.output)
        with self.assertRaises(ValueError):
            self.generate()
        changed = deepcopy(self.plan)
        changed["config"]["search_margin"] = .02
        with patch("ris_noma.multik_dataset.inspect", return_value=(changed, self.source)), self.assertRaises(ValueError):
            data.run(self.previous, self.output, changed["config"], resume=True)
        self.assertEqual(before, fingerprints(self.output))
        self.assertTrue(self.generate(resume=True)["complete"])
        self.assertEqual(before, fingerprints(self.output))

    def test_partial_resume_keeps_search_and_recomputes_no_pso(self):
        actual = data.run_phase
        def interrupt(output, phase, jobs, workers):
            if phase == "selection":
                raise RuntimeError("simulated interruption after search freeze")
            return actual(output, phase, jobs, workers)
        with patch("ris_noma.multik_dataset.run_phase", side_effect=interrupt), self.assertRaises(RuntimeError):
            self.generate()
        self.assertEqual(data.status(self.output)["committed_groups"]["search"], 4)
        search_before = fingerprints(self.output / "search")
        with patch("ris_noma.multik_data_core.optimize", side_effect=AssertionError("completed PSO must not rerun")):
            result = self.generate(resume=True)
        self.assertTrue(result["complete"])
        self.assertEqual(search_before, fingerprints(self.output / "search"))

    def test_uncommitted_file_is_recomputed_and_corrupt_receipt_rejected(self):
        self.output.mkdir()
        data.write_json(self.output / "plan.json", self.plan)
        data.write_json(self.output / "source_physics_plan.json", self.source)
        job = self.plan["jobs"][0]
        path = self.output / "search" / data.group_name(job)
        data.write_json(path, {"partial": True})
        data.task((str(self.output), "search", job))
        self.assertIsNotNone(data.read_committed(path))
        path.write_text('{}', encoding="utf-8")
        with self.assertRaises(ValueError):
            data.read_committed(path)

    def test_spawn_parallel_and_sequential_exports_are_byte_identical(self):
        self.generate(workers=1)
        other = self.folder / "parallel"
        self.generate(output=other, workers=2)
        self.assertEqual(fingerprints(self.output), fingerprints(other))


if __name__ == "__main__":
    unittest.main()
