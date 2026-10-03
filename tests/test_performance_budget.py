import copy

import pytest

from tools import benchmark_versions
from tools import performance_budget as budget


def evidence(ratio=1.0, stage_ratio=None):
    rows = []
    for index, demo in enumerate(budget.REFERENCES):
        for mode in ('timing', 'stage'):
            for rnd in range(8):
                for label in ('reference', 'candidate'):
                    factor = (stage_ratio if mode == 'stage' and stage_ratio is not None
                              else ratio) if label == 'candidate' else 1.0
                    rows.append({'demo': demo, 'mode': mode, 'round': rnd, 'label': label,
                                     'order': ['reference', 'candidate'] if (rnd + index) % 2 == 0
                                     else ['candidate', 'reference'],
                                     'blend_sha256': 'a' * 64, 'settings': {'voxel_size': .03},
                                     'warp': '1.16.0' if label == 'reference' else '1.17.0',
                                     'device': 'RTX test', 'profiler_sync': mode == 'stage', 'particles': 1000,
                                     'records': [{'frame': i, 'step_ms': (100000 if i == 1 else 100 * factor),
                                                   'solver_frame': i, 'sim_time': i / 24,
                                                   'particles': 1000, 'substeps': 8,
                                                   'stages': {'p2g': 10 * factor, 'solver': 50 * factor}}
                                              for i in range(1, 49)]})
    return rows


def test_real_regression_fails():
    result = budget.evaluate(evidence(1.12))
    assert result['verdict'] == 'FAIL'
    assert all(d['verdict'] == 'FAIL' for d in result['demos'].values())


def test_frame_one_and_one_noisy_pair_do_not_decide():
    rows = evidence()
    for row in rows:
        if row['label'] == 'candidate' and row['round'] == 0:
            for frame in row['records'][1:]:
                frame['step_ms'] *= 3
    assert budget.evaluate(rows)['verdict'] == 'PASS'


def test_whole_frame_noise_alone_is_inconclusive():
    result = budget.evaluate(evidence(1.3, stage_ratio=1.0))
    assert result['verdict'] == 'INCONCLUSIVE'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nan', 'settings', 'order', 'stage'])
def test_incomplete_or_incomparable_evidence_cannot_pass(mutation):
    rows = evidence()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'nan':
        rows[0]['records'][3]['step_ms'] = float('nan')
    elif mutation == 'settings':
        rows[1]['settings']['voxel_size'] = .1
    elif mutation == 'order':
        rows[0]['order'] = ['candidate', 'reference']
    else:
        next(r for r in rows if r['mode'] == 'stage')['records'][3]['stages'] = {}
    with pytest.raises(ValueError):
        budget.evaluate(rows)


def test_existing_runner_rotates_order_and_retains_rounds():
    calls = []
    def trial(label, root, demo, rnd):
        calls.append((rnd, label))
        return {'hostMsMedian': 1.0}
    runs = benchmark_versions.run_paired_trials(
        [('reference', 'a'), ('candidate', 'b')], ['dam_break'], 8, trial)
    assert calls[:4] == [(0, 'reference'), (0, 'candidate'), (1, 'candidate'), (1, 'reference')]
    assert len(runs) == 16
    assert runs[-1]['round'] == 7


def test_strict_trial_count_and_order_independence():
    rows = [r for r in evidence() if r['round'] < 6]
    with pytest.raises(ValueError, match='eight'):
        budget.evaluate(rows)
    rows = evidence(1.01)
    rows.reverse()
    assert budget.evaluate(rows)['verdict'] == 'PASS'


def test_gate_regression_cannot_reach_success_marker(monkeypatch, capsys):
    import subprocess

    from tools import run_production_gate as gate
    monkeypatch.setattr(gate, '_clear_gate_manifest', lambda: None)
    monkeypatch.setattr(gate, '_print_gate_metadata', lambda *a: None)
    monkeypatch.setattr(gate, '_check_no_tracked_npz', lambda: None)
    monkeypatch.setattr(gate, '_check_no_tracked_ignored_files', lambda: None)
    monkeypatch.setattr(gate, '_check_certified_warp', lambda *a: 'warp')
    def run(label, command, **kwargs):
        if label == 'paired D7 performance budget':
            assert command[command.index('--trials') + 1] == '8'
            assert kwargs['expect_marker'] == 'PERFORMANCE-BUDGET-PASS'
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(gate, '_run', run)
    with pytest.raises(subprocess.CalledProcessError):
        gate.main([])
    assert 'GATE-PASS' not in capsys.readouterr().out


def test_workload_cannot_drift_for_both_versions_between_rounds():
    rows = evidence()
    for row in rows:
        if row['round'] == 2:
            row['settings']['voxel_size'] = .1
    with pytest.raises(ValueError, match='changed'):
        budget.evaluate(rows)


def test_added_settings_preserve_implicit_legacy_work():
    a = {'voxel_size': .03}
    b = {'voxel_size': .03, 'export_attributes': True,
         'species_enabled': False, 'species_mixing_rate': 8.0}
    assert budget.comparable_settings(a) == budget.comparable_settings(b)
    b['export_attributes'] = False
    assert budget.comparable_settings(a) != budget.comparable_settings(b)


def test_ordering_effect_alone_cannot_prove_regression():
    rows = evidence()
    for row in rows:
        if row['label'] == 'candidate':
            ratio = 1.6 if row['order'][0] == 'reference' else 1 / 1.6
            for frame in row['records'][1:]:
                frame['step_ms'] *= ratio
                frame['stages'] = {k: v * ratio for k, v in frame['stages'].items()}
    result = budget.evaluate(rows)
    assert result['verdict'] == 'INCONCLUSIVE'
    assert all(d['verdict'] != 'FAIL' for d in result['demos'].values())


def test_bridge_adapter_preserves_authored_physics_and_substeps():
    from types import SimpleNamespace

    from tools.budget_scene_compat import restore_legacy_domain
    fields = {'voxel_size': .03, 'min_substeps': 1, 'max_substeps': 16, 'ww_churn_gain': 2.0}
    owner = SimpleNamespace(bl_rna=SimpleNamespace(properties={
        name: SimpleNamespace(type='FLOAT') for name in fields}))
    restore_legacy_domain(owner, fields)
    assert {key: getattr(owner, key) for key in fields} == fields
    with pytest.raises(ValueError, match='cannot read'):
        restore_legacy_domain(owner, {'unknown_physics': 1})


def test_inactive_default_differences_are_not_physics_overrides():
    a = {'viscosity_enabled': False, 'viscosity_iterations': 40,
         'spacetime_enabled': False, 'spacetime_variable_projection': True}
    b = {'viscosity_enabled': False, 'viscosity_iterations': 64,
         'spacetime_enabled': False, 'spacetime_variable_projection': False}
    assert budget.comparable_settings(a) == budget.comparable_settings(b)
    a['viscosity_enabled'] = b['viscosity_enabled'] = True
    assert budget.comparable_settings(a) != budget.comparable_settings(b)


def test_missing_runtime_replaces_stale_pass_evidence(monkeypatch, tmp_path):
    import json

    from tools import run_performance_budget as runner
    report = tmp_path / 'budget.json'
    report.write_text('{"verdict":"PASS"}')
    monkeypatch.delenv('FLUIDSIM_PERF_10255_ROOT', raising=False)
    assert runner.main(['--blender', 'unused', '--out-dir', str(tmp_path)]) == 1
    result = json.loads(report.read_text())
    assert result['verdict'] == 'ERROR'
    assert 'FLUIDSIM_PERF_10255_ROOT' in result['error']


def test_vortex_uses_the_recorded_bridge_resave_with_110_code():
    assert budget.REFERENCES['chromatic_vortex'] == 'c5919d668f469942200c58274eca13337d3aa624'
    assert budget.SCENE_REFS['chromatic_vortex'] == 'ad769d23c3c4d7e6eaa5ac0e0a3b29a46ce07689'
    assert budget.SCENE_REFS['dam_break'] == budget.SCENE_REF


@pytest.mark.parametrize('field,value', [('solver_frame', 1), ('sim_time', 0),
                                        ('particles', 0), ('substeps', 0)])
def test_nonadvancing_or_empty_work_cannot_pass(field, value):
    rows = evidence(.8)
    rows[1]['records'][4][field] = value
    with pytest.raises(ValueError):
        budget.evaluate(rows)


def test_particle_loss_cannot_buy_a_pass():
    rows = evidence(.8)
    rows[1]['records'][4]['particles'] = 900
    with pytest.raises(ValueError, match='particle'):
        budget.evaluate(rows)


def test_reduced_adaptive_work_requires_investigation():
    rows = evidence(.8)
    for row in rows:
        if row['label'] == 'candidate':
            for frame in row['records']:
                frame['substeps'] = 7
    result = budget.evaluate(rows)
    assert result['verdict'] == 'INCONCLUSIVE'
    assert all(d['work_ratios']['timing'] == .875 for d in result['demos'].values())


def test_small_adaptive_variation_keeps_authored_schedule():
    rows = evidence(.98)
    for row in rows:
        if row['label'] == 'candidate':
            row['records'][3]['substeps'] = 7
    assert budget.evaluate(rows)['verdict'] == 'PASS'
