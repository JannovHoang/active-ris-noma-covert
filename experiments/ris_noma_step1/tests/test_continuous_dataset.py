"""7e scientific invariants: continuous queries, split isolation and frozen labels."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ris_noma.channels import build_bank, public_order
from ris_noma.config import load_config
from ris_noma.continuous_data import (MarginObjective, jobs_for, parameter_cases, role,
    validate_query, validate_settings)
from ris_noma.continuous_dataset import export, read_lines, run
from ris_noma.continuous_workers import audit_job, choose_label, select_job
from ris_noma.learning_data import read_json
from ris_noma.multik_core import MultiKObjective, original_allocation
from ris_noma.multik_data_core import geometry
from ris_noma.multik_dataset import write_json
from ris_noma.multik_graph import fit_normalizer, statistical_graph

ROOT = Path(__file__).resolve().parents[1]


def config():
    return read_json(ROOT / 'configs/continuous_pilot_7e.json')


def physical_fixture():
    source = {'base_config': load_config(ROOT / 'configs/legacy_physics_k2_pilot.json'),
              'experiment': {'position_half_width_m': {'bob': 5., 'willie': 4.}}}
    c, layout = geometry({'config': {'power_dbm': [35.]}}, source,
                         {'layout_id': 1400, 'n_public': 2}, 256)
    bank = build_bank(c, layout, np.random.default_rng(2))
    return c, layout, bank, public_order(bank)


class ContinuousSamplingTests(unittest.TestCase):
    def test_latin_hypercube_is_continuous_stratified_and_reproducible(self):
        c = config(); values = parameter_cases(c, 7, 2000)
        self.assertEqual(values, parameter_cases(c, 7, 2000))
        self.assertNotEqual(values, parameter_cases(c, 7, 2001))
        n = c['cases_per_layout']
        for field, domain in [('power_dbm', 'power_dbm_range'), ('xi', 'dep_range')]:
            lo, hi = c[domain]; unit = (np.array([v[field] for v in values]) - lo) / (hi - lo)
            self.assertEqual(sorted(np.floor(n * unit).astype(int)), list(range(n)))
            self.assertTrue(np.all((unit > 0) & (unit < 1)))
        self.assertTrue(all(v['power_dbm'] not in (30., 35., 40.) for v in values))
        self.assertTrue(all(v['xi'] not in (.75, .8, .85, .9, .95) for v in values))

    def test_same_layout_shares_parameters_across_k_without_crossing_splits(self):
        c = validate_settings(config()); jobs = jobs_for(c, 7)
        self.assertEqual(len(jobs), 50)
        counts = {name: sum(len(j['cases']) for j in jobs if role(j) == name)
                  for name in ('train', 'validation', 'test', 'test_unseen_k')}
        self.assertEqual(counts, {'train': 256, 'validation': 64, 'test': 64, 'test_unseen_k': 16})
        for lid in {j['layout_id'] for j in jobs}:
            part = [j for j in jobs if j['layout_id'] == lid]
            self.assertEqual(len({j['split'] for j in part}), 1)
            self.assertTrue(all(j['cases'] == part[0]['cases'] for j in part))
        self.assertTrue(all(j['n_public'] != 6 for j in jobs if j['split'] != 'test'))

    def test_query_preserves_exact_parameters_and_is_not_training_data(self):
        c = config(); q = {'layout_id': 2300, 'n_public': 3, 'power_dbm': 32.7, 'xi': .873}
        validated = validate_query(c, q, {2000})
        j, = jobs_for(c, 7, validated)
        self.assertEqual(j['cases'], [{'case_index': 0, 'power_dbm': 32.7, 'xi': .873}])
        self.assertEqual(role(j), 'query')
        self.assertEqual(q, validated)

    def test_query_rejects_leaked_layouts_and_outside_domain_without_clipping(self):
        c = config(); q = {'layout_id': 2300, 'n_public': 3, 'power_dbm': 32.7, 'xi': .873}
        for overrides in ({'layout_id': 2000}, {'power_dbm': 29.9}, {'power_dbm': 40.1},
                          {'xi': .97}, {'xi': float('nan')}, {'n_public': 7}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_query(c, dict(q, **overrides), {2000})

    def test_invalid_configs_do_not_silently_change_the_experiment(self):
        updates = [{'dep_range': [.75, .98]}, {'pso_evaluations': 61}, {'pso_evaluations': True},
                   {'pso_seeds': [101, 101]}, {'search_samples': 100}, {'search_qos_margin': -.01},
                   {'power_dbm_range': [40., 30.]}, {'unseen_public_counts': [5]},
                   {'parameter_seed': True}, {'profile': 'legacy'}]
        for updates_ in updates:
            with self.subTest(updates=updates_), self.assertRaises(ValueError):
                validate_settings(dict(config(), **updates_))
        c = config(); c['splits']['test'] = [c['splits']['train'][0]]
        with self.assertRaises(ValueError): validate_settings(c)


class SearchPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c, cls.layout, cls.bank, cls.order = physical_fixture()

    def test_zero_margin_reproduces_original_objective_exactly(self):
        old = MultiKObjective(self.c, self.layout, self.bank, self.order)
        same = MarginObjective(self.c, self.layout, self.bank, self.order, 0.)
        for z in ([.2, .5, .5], [.05, .9, .3], [.6, .2, .8]):
            self.assertEqual(old.evaluate(z, .85), same.evaluate(z, .85))

    def test_qos_margin_affects_admission_not_rate_or_dep_equations(self):
        c = self.c; snapshot = deepcopy(c)
        old = MultiKObjective(c, self.layout, self.bank, self.order)
        safe = MarginObjective(c, self.layout, self.bank, self.order, .01)
        mb = .2; cap = np.expm1(np.log(2) * .205)
        p_last = cap * (mb + c['kappa'] ** 2)
        _, z = original_allocation(mb, [1 - mb - p_last, p_last], .5, self.order, c['bounds'])
        a, b = old.evaluate(z, .75), safe.evaluate(z, .75)
        self.assertEqual(c, snapshot)
        self.assertEqual(a.score, b.score)
        for key in ('dep_min', 'public_mean_rate_bit_s_hz', 'bob_outage_estimate', 'lambda_star_w'):
            self.assertEqual(a.metrics[key], b.metrics[key])
        self.assertEqual(a.metrics['normalized_violations']['qos'], 0.)
        self.assertGreater(b.metrics['normalized_violations']['qos'], 0.)

    def test_graph_carries_actual_qos_and_continuous_inputs_not_search_targets(self):
        c = dict(self.c, total_dbm=32.7)
        safe = MarginObjective(c, self.layout, self.bank, self.order, .01)
        g = statistical_graph(c, self.layout, self.bank, self.order, .873)
        self.assertEqual(g['global_raw'][0], .873)
        self.assertAlmostEqual(g['global_raw'][1], (32.7 - 30) / 10)
        self.assertEqual([r[6] for r in g['node_raw'][1:]], [.2, .2])
        np.testing.assert_allclose(safe.qos, [.21, .21])


class SelectionAndAuditTests(unittest.TestCase):
    def test_selection_uses_independent_rate_only_among_admissible_candidates(self):
        candidates = [
            {'candidate_id': 'fast-invalid', 'search_feasible': True, 'passed': False, 'physical': {'bob_rate': 9.}},
            {'candidate_id': 'search-invalid', 'search_feasible': False, 'passed': True, 'physical': {'bob_rate': 8.}},
            {'candidate_id': 'a', 'search_feasible': True, 'passed': True, 'search_rate': 7., 'physical': {'bob_rate': 4.}},
            {'candidate_id': 'b', 'search_feasible': True, 'passed': True, 'search_rate': 6., 'physical': {'bob_rate': 5.}}]
        self.assertEqual(choose_label(candidates)['candidate_id'], 'b')
        self.assertIsNone(choose_label(candidates[:2]))

    def test_bank_generation_waits_for_global_freeze_barriers(self):
        with tempfile.TemporaryDirectory() as folder, patch('ris_noma.continuous_workers.physical_context') as bank:
            with self.assertRaisesRegex(ValueError, 'Freeze every search'):
                select_job((folder, {'id': 'test'}))
            with self.assertRaisesRegex(ValueError, 'Freeze every selection'):
                audit_job((folder, {'id': 'test'}))
            bank.assert_not_called()

    def test_export_keeps_audit_failure_and_missing_case_in_denominators(self):
        c, layout, bank, order = physical_fixture()
        graph = statistical_graph(c, layout, bank, order, .873)
        changed = deepcopy(graph); changed['global_raw'][0] = .8
        allocation = [.2, .4, .4, .5]
        final = {'passed': False, 'gated_bob_rate': 0., 'bob_rate': 4., 'dep_lower': .85,
                 'qos_lower_margin': .02, 'failed_checks': ['empirical_dep_lower']}
        base = {'n_public': 2, 'regime': 'seen_k', 'power_dbm': 35., 'xi': .873,
                'graph': graph, 'allocation': allocation, 'selected_candidate_id': 's101:e1',
                'missing_reason': None, 'selection': {}, 'audit': final,
                'audit_used_for_label_filter': False, 'label_available': True}
        train = dict(base, case_id='train', layout_id=2000, split='train', role='train')
        val = dict(base, case_id='val', layout_id=2008, split='validation', role='validation', graph=changed)
        missing = dict(base, case_id='test', layout_id=2010, split='test', role='test',
                       label_available=False, allocation=None, selected_candidate_id=None,
                       missing_reason='no_candidate_passed_selection', audit=None)
        ledger = [train, val, missing]
        exports = {name: [] for name in ('train', 'validation', 'test', 'test_unseen_k', 'query')}
        keys = ('case_id', 'layout_id', 'n_public', 'split', 'regime', 'power_dbm', 'xi', 'graph',
                'allocation', 'selected_candidate_id', 'audit_used_for_label_filter')
        exports['train'] = [{k: train[k] for k in keys}]
        exports['validation'] = [{k: val[k] for k in keys}]
        plan = {'config': config(), 'mode': 'dataset', 'jobs': [{}, {}, {}],
                'planned_counts': {name: sum(r['role'] == name for r in ledger) for name in exports},
                'max_objective_evaluations': 10800}
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder) / 'plan.json', plan)
            with patch('ris_noma.continuous_dataset.collect', return_value=(plan, ledger, exports, 10800)):
                summary = export(folder)
            self.assertEqual(summary['planned_cases'], 3)
            self.assertEqual(summary['available_labels'], 2)
            self.assertEqual(summary['audit_passed'], 0)
            self.assertEqual(len(read_lines(Path(folder) / 'ledger.jsonl')), 3)
            exported = read_lines(Path(folder) / 'graphs/train.jsonl')
            self.assertEqual(len(exported), 1)
            self.assertNotIn('audit', exported[0]); self.assertNotIn('selection', exported[0])
            norm = read_json(Path(folder) / 'graphs/normalization.json')
            self.assertEqual(norm['parameters'], fit_normalizer([graph]))
            self.assertEqual(norm['training_graph_count'], 1)

    def test_nonempty_and_changed_resume_are_rejected_before_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder) / 'plan.json', {'old': True})
            with patch('ris_noma.continuous_dataset.inspect', return_value={'new': True}), \
                 patch('ris_noma.continuous_dataset.execute_stage') as stage:
                with self.assertRaisesRegex(ValueError, 'nonempty'): run('source', {}, folder)
                with self.assertRaisesRegex(ValueError, 'Changed'): run('source', {}, folder, resume=True)
                stage.assert_not_called()


if __name__ == '__main__': unittest.main()
