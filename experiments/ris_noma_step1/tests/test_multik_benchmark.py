"""7d invariants: geometric phase, unchanged physics, budgets and bank barriers."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ris_noma.channels import build_bank, public_order
from ris_noma.config import load_config
from ris_noma.geometry_los import (SPEED_OF_LIGHT, STREAMS, apply_profile, array_elements,
    channel_context, relative_los_phases)
from ris_noma.learning_data import read_json, sha256
from ris_noma.metrics import power_budget
from ris_noma.multik_core import MultiKObjective, decode_allocation, original_allocation
from ris_noma.multik_data_core import geometry, raw_assessment
from ris_noma.multik_dataset import write_json, receipt_path
from ris_noma.multik_benchmark import make_jobs, validate_settings
from ris_noma.multik_benchmark_core import (lookup_allocation, search_pool, select_job, audit_job, table_policy)
from ris_noma.multik_benchmark_analysis import paired_bootstrap
from ris_noma.pso import PSOSettings, optimize
from ris_noma.stable_assessment import assert_power_identity, raw_assessment as stable_assessment

ROOT = Path(__file__).resolve().parents[1]


def fixture(k=2, samples=256):
    source = {"base_config": load_config(ROOT / "configs/legacy_physics_k2_pilot.json"),
              "experiment": {"position_half_width_m": {"bob": 5., "willie": 4.}}}
    dataset = {"config": {"power_dbm": [35.]}}
    settings = read_json(ROOT / "configs/multik_benchmark_7d.json")
    job = {"id": "test", "stage": "benchmark", "layout_id": 1400, "n_public": k,
           "profile": "geometry", "deployment": settings["benchmark_deployments"][0]}
    c, layout = geometry(dataset, source, job, samples)
    return c, layout, source, dataset, settings, job


def commit(path, value):
    write_json(path, value); write_json(receipt_path(path), {"sha256": sha256(path)})


class GeometricLoSTests(unittest.TestCase):
    def test_exact_distance_phase_and_unit_modulus(self):
        center = [95., 5.]; points = np.array([[0., 0.], [110., 1.]])
        wavelength = SPEED_OF_LIGHT / 3.5e9
        elements = array_elements(center, 32, 3.5e9, .5, [0., 1.])
        actual = np.exp(1j * relative_los_phases(points, center, elements, wavelength))
        propagation = np.exp(-2j * np.pi * np.linalg.norm(points[:, None] - elements[None, :], axis=-1) / wavelength)
        reference = np.exp(-2j * np.pi * np.linalg.norm(points - center, axis=-1) / wavelength)
        np.testing.assert_allclose(actual, propagation / reference[:, None], atol=3e-12)
        np.testing.assert_allclose(np.abs(actual), 1., atol=1e-15)

    def test_translation_invariance_of_relative_geometry(self):
        elements = array_elements([95., 5.], 32, 3.5e9, .5, [0., 1.])
        points = np.array([[0., 0.], [110., 1.]])
        move = np.array([123., -43.])
        a = relative_los_phases(points, [95., 5.], elements, .085)
        b = relative_los_phases(points + move, np.array([95., 5.]) + move, elements + move, .085)
        np.testing.assert_allclose(np.exp(1j*a), np.exp(1j*b), atol=4e-12)

    def test_axis_normalization_and_centered_array(self):
        a = array_elements([0., 0.], 32, 3.5e9, .5, [0., 1.])
        b = array_elements([0., 0.], 32, 3.5e9, .5, [0., 7.])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a.mean(0), 0., atol=1e-15)

    def test_exact_distance_converges_to_plane_wave_with_correct_sign(self):
        wavelength=SPEED_OF_LIGHT/3.5e9
        elements=array_elements([0.,0.],32,3.5e9,.5,[0.,1.])
        direction=np.array([.6,.8]);point=1e6*direction
        actual=np.exp(1j*relative_los_phases(point[None,:],[0.,0.],elements,wavelength)[0])
        expected=np.exp(2j*np.pi*(elements@direction)/wavelength)
        np.testing.assert_allclose(actual,expected,atol=1e-5)

    def test_invalid_carrier_axis_and_spacing_rejected(self):
        for carrier, spacing, axis in [(0., .5, [0, 1]), (3.5e9, .5, [0, 0]), (3.5e9, 1., [0, 1])]:
            with self.assertRaises(ValueError): array_elements([0., 0.], 32, carrier, spacing, axis)

    def test_bob_move_changes_his_los_but_not_alice_ris(self):
        c, layout, *_rest, settings, job = fixture()
        a, _, _ = apply_profile(c, layout, "geometry", job["deployment"], settings)
        from dataclasses import replace
        moved = layout.receiver_xy.copy(); moved[0] += [1., 2.]
        b, _, _ = apply_profile(c, replace(layout, receiver_xy=moved), "geometry", job["deployment"], settings)
        self.assertEqual(a["los"]["ar_phase_rad"], b["los"]["ar_phase_rad"])
        x = np.exp(1j*np.array(a["los"]["ru_phase_rad"][0])); y = np.exp(1j*np.array(b["los"]["ru_phase_rad"][0]))
        self.assertLess(abs(np.vdot(x, y))/32, .999)

    def test_legacy_profile_is_numerically_identical(self):
        c, layout, *_rest, settings, job = fixture()
        new, next_layout, _ = apply_profile(c, layout, "legacy", job["deployment"], settings)
        self.assertEqual(c, new)
        old = build_bank(c, layout, np.random.default_rng(9)); actual = build_bank(new, next_layout, np.random.default_rng(9))
        np.testing.assert_array_equal(old.signal_gain_tilde, actual.signal_gain_tilde)
        np.testing.assert_array_equal(old.noise_gain_tilde, actual.noise_gain_tilde)

    def test_friis_reference_changes_loss_and_recomputes_gain_budget(self):
        c, layout, *_rest, settings, job = fixture()
        before_config=deepcopy(c)
        changed, next_layout, _ = apply_profile(c, layout, "geometry_friis", job["deployment"], settings)
        self.assertEqual(c,before_config)
        expected = (SPEED_OF_LIGHT / settings["carrier_hz"] / (4*np.pi))**2
        self.assertAlmostEqual(changed["pathloss"]["beta0_ar"], expected)
        self.assertGreater(next_layout.loss_ar, layout.loss_ar)
        before, after = power_budget(c, layout.loss_ar, .5), power_budget(changed, next_layout.loss_ar, .5)
        self.assertEqual(before["p0_w"], after["p0_w"])
        self.assertNotEqual(before["rho2"], after["rho2"])
        self.assertEqual(c["noise_dbm"], changed["noise_dbm"])

    def test_same_nlos_draws_across_profiles_and_new_draws_across_bank_roles(self):
        c, layout, source, dataset, settings, job = fixture()
        settings.update(search_samples=1000, selection_samples=1000, audit_samples=1000)
        plan = {"config": settings}
        outputs = []
        for profile in ("legacy", "geometry"):
            _, _, bank, meta = channel_context(plan, source, dataset, dict(job, profile=profile), "search")
            los = np.exp(1j*np.asarray(meta["los"]["ar_phase_rad"]))
            outputs.append(bank.h_ar - np.sqrt(.5)*los)
        np.testing.assert_allclose(*outputs, atol=4e-16)
        legacy,legacy_layout,_=apply_profile(c,layout,'legacy',job['deployment'],settings)
        geometric,geometric_layout,_=apply_profile(c,layout,'geometry',job['deployment'],settings)
        first=build_bank(legacy,legacy_layout,np.random.default_rng(11))
        second=build_bank(geometric,geometric_layout,np.random.default_rng(11))
        first_nlos=first.h_ru-np.sqrt(.5)*np.exp(1j*np.asarray(legacy['los']['ru_phase_rad']))
        second_nlos=second.h_ru-np.sqrt(.5)*np.exp(1j*np.asarray(geometric['los']['ru_phase_rad']))
        np.testing.assert_allclose(first_nlos,second_nlos,atol=5e-16)
        _, _, bank, search_meta = channel_context(plan, source, dataset, job, "search")
        _, _, other, selection_meta = channel_context(plan, source, dataset, job, "selection")
        self.assertEqual(search_meta["los_sha256"], selection_meta["los_sha256"])
        self.assertFalse(np.array_equal(bank.h_ar, other.h_ar))


class BudgetAndPolicyTests(unittest.TestCase):
    def setUp(self): self.c = read_json(ROOT / "configs/multik_benchmark_7d.json")

    def test_main_factorial_counts_include_all_seeds_and_selectors(self):
        c = validate_settings(self.c); jobs = make_jobs(c, [2,3,4,5])
        self.assertEqual(len(jobs), 72)
        self.assertEqual(sum(j["planned_search_pools"] for j in jobs), 13446)
        self.assertEqual(sum(j["audit_family"] for j in jobs), 26892)
        self.assertEqual(sum(j["planned_objective_evaluations"] for j in jobs), 4179600)
        self.assertEqual({j["selection_family"] for j in jobs}, {45,2520})

    def test_overlapping_development_layouts_rejected(self):
        self.c["benchmark_layout_ids"] = self.c["calibration_layout_ids"]
        with self.assertRaises(ValueError): validate_settings(self.c)

    def test_invalid_target_plus_margin_rejected(self):
        self.c["search_margins"] = [0., .06]
        with self.assertRaises(ValueError): validate_settings(self.c)

    def test_nonintegral_or_unfair_budget_rejected(self):
        for value in (61, 0, 1800):
            c = deepcopy(self.c); c["short_evaluations"] = value
            with self.assertRaises(ValueError): validate_settings(c)

    def test_new_physical_regime_cannot_be_claimed_as_fair_frozen_benchmark(self):
        self.c["benchmark_profiles"] = ["geometry_friis"]
        with self.assertRaises(ValueError): validate_settings(self.c)

    def test_seed_replicates_must_be_paired(self):
        self.c["pso_seeds"] = [101]
        with self.assertRaises(ValueError): validate_settings(self.c)

    def test_lookup_is_train_only_and_supports_unseen_k(self):
        train = [{"power_dbm":35., "xi":.85, "allocation":[.2,.4,.4,.5]},
                 {"power_dbm":35., "xi":.85, "allocation":[.3,.35,.35,.6]}]
        policy = table_policy(train, {}, .01)
        self.assertEqual(policy["fit_role"], "train")
        for k in (2,5,6):
            c, *_ = fixture(k); order = np.arange(k)[::-1]
            y = lookup_allocation(policy,c,order,35.,.85)
            self.assertAlmostEqual(sum(y[:-1]),1.)
            powers = y[1:-1][order]
            for j in range(k):
                rate = np.log2(1+powers[j]/(y[0]+powers[j+1:].sum()+.1**2))
                self.assertGreaterEqual(rate,.21-1e-12)

    def test_sic_ceiling_rule_does_not_guarantee_finite_snr_qos(self):
        c, layout, *_ = fixture(6)
        c["noise_dbm"]["bob"] = 30.; c["noise_dbm"]["public"] = [30.]*6
        bank = build_bank(c,layout,np.random.default_rng(2)); order = public_order(bank)
        policy = table_policy([{"power_dbm":35.,"xi":.85,"allocation":[.2,.8,.5]}],{},.01)
        y = lookup_allocation(policy,c,order,35.,.85)
        result = stable_assessment(c,layout,bank,order,y,1,.05)
        self.assertFalse(result["point_checks_without_dep"]["public_average_qos"])

    def test_archive_preserves_warm_start_and_exact_pso_evaluation_count(self):
        for k in (2,6):
            c,layout,*_ = fixture(k)
            bank = build_bank(c,layout,np.random.default_rng(3)); order=public_order(bank)
            objective=MultiKObjective(c,layout,bank,order)
            initial=np.full((1,k+1),.5)
            settings=PSOSettings(swarm_size=30,sweeps=3)
            result=search_pool(objective,c,order,.85,settings,np.random.default_rng(4),initial)
            self.assertEqual(result["objective_evaluations"],90)
            self.assertLessEqual(len(result["candidates"]),5)
            self.assertTrue(any(p["evaluation"]==1 for p in result["candidates"]))
            plain=optimize(lambda z:objective.evaluate(z,.85),k+1,settings,np.random.default_rng(4),initial)
            best=next(v for v in result["candidates"] if v["candidate_id"]==result["raw_candidate_id"])
            self.assertEqual(best["search_rate"],plain.outcome.score)


class SelectionAndAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.job={"id":"fixture","selection_family":10,"audit_family":4}

    def setup_pool(self, all_fail=False):
        write_json(self.root/'plan.json',{"config":{"alpha":.05},"jobs":[self.job]})
        for name in ('source_physics_plan.json','source_dataset_plan.json','search_frozen.json'):write_json(self.root/name,{})
        pool={"power_dbm":35.,"xi":.85,"search_margin":.03,"method":"random_short_pso",
            "training_seed":7,"pso_seed":101,"objective_evaluations":60,"search_return_seconds_cached_bank":.001,
            "search_bank_seconds":.002,"raw_candidate_id":"bad","candidates":[
                {"candidate_id":"bad","allocation":[.3,.3,.4,.5]},
                {"candidate_id":"good","allocation":[.1,.4,.5,.5]}]}
        commit(self.root/'search/fixture.json',{"geometry_valid":True,"order":[0,1],"pools":[pool],"plan_sha256":sha256(self.root/'plan.json')})
        def fake(c,layout,bank,order,allocation,family,alpha):
            dep=.5 if all_fail or allocation[0]>.2 else .9
            return {"bob_rate":10. if allocation[0]>.2 else 5.,"dep_gamma":dep,"dep_lower":dep,
                "point_checks_without_dep":{},"bound_checks_without_dep":{},"constants":{"pool_size":family}}
        return fake

    def test_selection_barrier_before_channel_generation(self):
        with patch('ris_noma.multik_benchmark_core.channel_context') as context:
            with self.assertRaisesRegex(ValueError,'Freeze every search'):select_job((str(self.root),self.job))
            context.assert_not_called()

    def test_audit_barrier_before_channel_generation(self):
        with patch('ris_noma.multik_benchmark_core.channel_context') as context:
            with self.assertRaisesRegex(ValueError,'Freeze every selection'):audit_job((str(self.root),self.job))
            context.assert_not_called()

    def test_independent_selection_rejects_fast_but_infeasible_candidate(self):
        fake=self.setup_pool()
        with patch('ris_noma.multik_benchmark_core.channel_context',return_value=({},None,None,{"bank_generation_seconds":.1})),patch('ris_noma.multik_benchmark_core.raw_assessment',side_effect=fake):
            select_job((str(self.root),self.job))
        rows=read_json(self.root/'selection/fixture.json')['rows']
        self.assertEqual(rows[0]['candidate_id'],'bad');self.assertEqual(rows[1]['candidate_id'],'good')
        self.assertGreater(rows[1]['return_seconds_with_bank_generation'],rows[0]['return_seconds_with_bank_generation'])

    def test_no_feasible_candidate_is_retained_as_failed_no_return(self):
        fake=self.setup_pool(all_fail=True)
        with patch('ris_noma.multik_benchmark_core.channel_context',return_value=({},None,None,{"bank_generation_seconds":.1})),patch('ris_noma.multik_benchmark_core.raw_assessment',side_effect=fake):
            select_job((str(self.root),self.job));write_json(self.root/'selection_frozen.json',{})
            audit_job((str(self.root),self.job))
        row=read_json(self.root/'audit/fixture.json')['rows'][1]
        self.assertFalse(row['returned']);self.assertFalse(row['passed']);self.assertEqual(row['gated_bob_rate'],0.)
        self.assertEqual(row['failed_checks'],['no_candidate_returned'])

    def test_tampered_committed_search_detected(self):
        self.setup_pool();path=self.root/'search/fixture.json'
        path.write_text(path.read_text(encoding="utf-8")+' ', encoding="utf-8")
        with self.assertRaisesRegex(ValueError,'missing/changed'):select_job((str(self.root),self.job))

    def test_committed_group_cannot_be_reused_with_another_plan(self):
        self.setup_pool()
        plan=read_json(self.root/'plan.json');plan['changed_config']=True
        write_json(self.root/'plan.json',plan)
        with patch('ris_noma.multik_benchmark_core.channel_context') as context:
            with self.assertRaisesRegex(ValueError,'another frozen plan'):select_job((str(self.root),self.job))
            context.assert_not_called()


class ClusteredAnalysisTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for lid in (1,2,3):
            for method in ('gnn_short_pso','random_short_pso','mlp_short_pso','lookup_sic_short_pso','full_pso'):
                for xi in (.75,.95):
                    rows.append({'profile':'legacy','deployment_id':'reference','search_margin':.03,'selector':'search_best',
                        'method':method,'layout_id':lid,'n_public':2,'power_dbm':35.,'xi':xi,'training_seed':7,'pso_seed':101,
                        'passed':True,'gated_bob_rate':1.,'return_seconds_cached_banks':.001})
        return rows

    def test_identical_methods_have_zero_paired_difference_and_three_clusters(self):
        result=paired_bootstrap(self.rows(),100)
        self.assertEqual(len(result),4)
        for row in result:
            self.assertEqual(row['layout_clusters'],3);self.assertEqual(row['paired_case_rows'],6)
            self.assertEqual(row['gated_rate_difference_high'],0.)

    def test_missing_paired_case_is_an_error(self):
        rows=self.rows();rows.pop()
        with self.assertRaisesRegex(ValueError,'Unpaired'):paired_bootstrap(rows,100)


class FloatingPointIdentityTests(unittest.TestCase):
    def test_noise_dominated_subtraction_is_bounded_by_machine_roundoff(self):
        v0=np.array([1e-10]);delta=np.array([1e-28]);v1=v0+delta
        self.assertFalse(np.allclose(v1-v0,delta,rtol=1e-9,atol=0))
        assert_power_identity(v0,v1,delta)

    def test_genuine_power_identity_violation_is_rejected(self):
        with self.assertRaises(ArithmeticError):assert_power_identity([1.],[.9],[.1])
        with self.assertRaises(ArithmeticError):assert_power_identity([1.],[1.3],[.1])

    def test_valid_legacy_metrics_are_exactly_preserved(self):
        for k in (2,6):
            c,layout,*_rest,settings,job=fixture(k,samples=1000)
            for profile in ('legacy','geometry','geometry_friis'):
                physical,new_layout,_=apply_profile(c,layout,profile,job['deployment'],settings)
                bank=build_bank(physical,new_layout,np.random.default_rng(5));order=public_order(bank)
                y=[.2,*([.8/k]*k),.5]
                old=raw_assessment(physical,new_layout,bank,order,y,10,.01)
                new=stable_assessment(physical,new_layout,bank,order,y,10,.01)
                self.assertEqual(old,new)

    def test_only_internal_consistency_assertion_changed_in_metric_function(self):
        import ast
        def node(name):
            tree=ast.parse((ROOT/'ris_noma'/name).read_text(encoding="utf-8"))
            return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_candidate')
        old,new=node('metrics.py'),node('stable_assessment.py')
        old.body=[n for n in old.body if not (isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='Công suất Willie vi phạm (9.10)' for x in ast.walk(n)))]
        new.body=[n for n in new.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='assert_power_identity')]
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))


class FrozenSearchReuseTests(unittest.TestCase):
    def test_search_function_fingerprint_ignores_format_but_not_math(self):
        from ris_noma.multik_benchmark_cache import function_hashes,FUNCTIONS
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.py';b=root/'b.py';changed=root/'changed.py'
            a.write_text('\n'.join(f'def {name}():\n    return 1\n' for name in FUNCTIONS), encoding="utf-8")
            b.write_text('\n\n# comments\n'.join(f'def {name}():\n    return 1\n' for name in FUNCTIONS), encoding="utf-8")
            changed.write_text(a.read_text(encoding="utf-8").replace('return 1','return 2',1), encoding="utf-8")
            self.assertEqual(function_hashes(a),function_hashes(b))
            self.assertNotEqual(function_hashes(a),function_hashes(changed))

    def test_changed_factorial_config_rejected_before_cache_use(self):
        from ris_noma.multik_benchmark_cache import validate_cache
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_json(root/'plan.json',{'format':'v1','config':{'seed':1}})
            write_json(root/'search_frozen.json',{})
            with self.assertRaisesRegex(ValueError,'different config'):
                validate_cache(root,{'format':'v1','config':{'seed':2}},ROOT)


if __name__=='__main__':unittest.main()
