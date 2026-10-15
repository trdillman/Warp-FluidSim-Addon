"""CPU regressions for preserved evidence and independently checked claims."""
import json
import subprocess
from pathlib import Path

import pytest
from test_performance_budget import _campaign

from tools import benchmark_versions as bench
from tools import run_production_gate as gate
from tools.performance_budget import evaluate_budget


def test_equal_but_shortened_duration_cannot_override_authored_settings():
    runs = _campaign()
    for run in runs:
        run['effectiveSettings'].update(time_scale=1.0, frame_rate=24)
        for record in run['records']:
            for field in ('frameDt', 'simTimeBefore', 'simTimeAfter'):
                record[field] *= .01
    assert not evaluate_budget(runs)['passed']


@pytest.mark.parametrize('mutation', ['missing', 'zero_p2g', 'device'])
def test_incomplete_stage_or_device_evidence_cannot_pass(mutation):
    runs = _campaign()
    if mutation == 'missing':
        runs[0]['records'][4]['timers'].pop('compute_pressure')
    elif mutation == 'zero_p2g':
        for run in runs:
            for record in run['records']:
                record['timers']['compute_p2g'] = 0
    else:
        runs[1]['device'] = 'cuda:9'
    assert not evaluate_budget(runs)['passed']


@pytest.mark.parametrize('mutation', ['missing', 'zero'])
def test_mandatory_pressure_stage_cannot_disappear_from_entire_campaign(mutation):
    runs = _campaign()
    for run in runs:
        for record in run['records']:
            if mutation == 'missing':
                record['timers'].pop('compute_pressure')
            else:
                record['timers']['compute_pressure'] = 0
    assert not evaluate_budget(runs)['passed']


def test_timeout_keeps_captured_child_output(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('child', 900, output=b'partial stdout', stderr=b'partial stderr')
    monkeypatch.setattr(bench.subprocess, 'run', timeout)
    out = tmp_path / 'trial.json'
    with pytest.raises(subprocess.TimeoutExpired):
        bench._run_trial(['child'], out)
    assert out.with_suffix('.log').read_text() == 'partial stdoutpartial stderr'


@pytest.mark.parametrize('mutation', ['status', 'demos', 'referenceCommits'])
def test_gate_rejects_altered_stored_summary(monkeypatch, tmp_path, mutation):
    runs = _campaign()
    report = evaluate_budget(runs)
    report.update(runs=runs, gitHead='a' * 40, candidateInventorySha256=bench._inventory(tmp_path),
                  referenceCommits=bench.REFERENCE_COMMITS)
    report[mutation] = 'altered'
    path = tmp_path / 'artifacts/performance_budget/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate, '_git_text', lambda *args: 'a' * 40)
    with pytest.raises(RuntimeError):
        gate._validate_performance_evidence()


def _stub_campaign(monkeypatch, tmp_path):
    source = tmp_path / 'source'
    for directory in ('fluidSim_warp_blender51', 'tools', 'demos'):
        (source / directory).mkdir(parents=True)
    (source / 'fluidSim_warp_blender51/version.py').write_text('VERSION_STRING = "2.0.0"')
    for demo in ('crown_splash', 'dam_break', 'paddle_stir', 'chromatic_vortex'):
        (source / f'demos/{demo}.blend').write_bytes(b'saved demo')
    for name in ('repo_addon_loader.py', 'benchmark_demo_trial.py', 'benchmark_versions.py',
                 'performance_budget.py', 'run_production_gate.py'):
        (source / f'tools/{name}').write_text('# source')
    monkeypatch.setattr(bench.subprocess, 'check_output', lambda *a, **kw: 'a' * 40)
    import io
    import tarfile
    def archive(command, **kwargs):
        data = b'VERSION_STRING = "1.0.255"'
        member = tarfile.TarInfo('fluidSim_warp_blender51/version.py')
        member.size = len(data)
        with tarfile.open(command[-2].split('=', 1)[1], 'w') as bundle:
            bundle.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(bench.subprocess, 'run', archive)
    runs = iter(_campaign())
    def child(command, out, **kwargs):
        args = command[command.index('--') + 1:]
        if args[0] == '--extract':
            Path(out).write_text('{}')
            return {}
        root = Path(args[0])
        row = next(runs)
        row.update(root=str(root), addon=str(root / 'fluidSim_warp_blender51/__init__.py'),
                   rootInventorySha256=bench._inventory(root), processId=123,
                   version=args[4] if row['label'] == 'baseline' else '2.0.0')
        return row
    return source, child


def test_interruption_preserves_completed_trial_metadata(monkeypatch, tmp_path):
    source, child = _stub_campaign(monkeypatch, tmp_path)
    timed = 0
    def interrupted(command, out, **kwargs):
        nonlocal timed
        if command[command.index('--') + 1] != '--extract':
            timed += 1
            if timed == 2:
                raise KeyboardInterrupt()
        return child(command, out, **kwargs)
    monkeypatch.setattr(bench, '_run_trial', interrupted)
    out = tmp_path / 'evidence'
    with pytest.raises(KeyboardInterrupt):
        bench.demo_budget_main(['--blender', 'fake', '--candidate-root', str(source), '--out-dir', str(out)])
    report = json.loads((out / 'summary.json').read_text())
    assert report['passed'] is False and len(report['runs']) == 1
    assert report['runs'][0]['slot'] == 0


@pytest.mark.parametrize('mutation', ['root', 'addon', 'inventory'])
def test_runner_rejects_wrong_observed_child_identity(monkeypatch, tmp_path, mutation):
    source, child = _stub_campaign(monkeypatch, tmp_path)
    def wrong(command, out, **kwargs):
        row = child(command, out, **kwargs)
        if row:
            row[{'root': 'root', 'addon': 'addon', 'inventory': 'rootInventorySha256'}[mutation]] = 'wrong'
        return row
    monkeypatch.setattr(bench, '_run_trial', wrong)
    out = tmp_path / 'evidence'
    assert bench.demo_budget_main(['--blender', 'fake', '--candidate-root', str(source), '--out-dir', str(out)]) == 1
    assert json.loads((out / 'summary.json').read_text())['passed'] is False


def test_trial_rejects_wrong_actual_process_id(monkeypatch, tmp_path):
    out = tmp_path / 'trial.json'
    class Process:
        pid = 123
        returncode = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def communicate(self, timeout=None):
            out.write_text(json.dumps({'processId': 999}))
            return 'DEMO-PERFORMANCE-TRIAL-PASS', ''
    monkeypatch.setattr(bench.subprocess, 'Popen', lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match='process'):
        bench._run_trial(['child'], out, expect_marker='DEMO-PERFORMANCE-TRIAL-PASS')


def test_cancelled_trial_kills_only_owned_child_and_keeps_output(monkeypatch, tmp_path):
    out = tmp_path / 'trial.json'
    class Process:
        pid = 123
        killed = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def communicate(self, timeout=None):
            if timeout is not None:
                raise KeyboardInterrupt()
            assert self.killed
            return 'interrupted stdout', 'interrupted stderr'
        def kill(self):
            self.killed = True
    child = Process()
    monkeypatch.setattr(bench.subprocess, 'Popen', lambda *args, **kwargs: child)
    with pytest.raises(KeyboardInterrupt):
        bench._run_trial(['child'], out, expect_marker='DEMO-PERFORMANCE-TRIAL-PASS')
    assert child.killed
    assert out.with_suffix('.log').read_text() == 'interrupted stdoutinterrupted stderr'


def _bound_report(root):
    runs = _campaign()
    inventory = bench._inventory(root)
    for run in runs:
        selected = Path(run['root'])
        run.update(addon=str(selected / 'fluidSim_warp_blender51/__init__.py'), processId=123,
                   rootInventorySha256=inventory if run['label'] == 'candidate' else 'b' * 64,
                   gitHead='a' * 40 if run['label'] == 'candidate' else bench.REFERENCE_COMMITS[run['referenceVersion']],
                   candidateInventorySha256=inventory)
    report = evaluate_budget(runs)
    report.update(runs=runs, gitHead='a' * 40, candidateInventorySha256=inventory,
                  referenceCommits=bench.REFERENCE_COMMITS, frames=48, trials=8)
    return report


@pytest.mark.parametrize('field,value', [('gitHead', 'wrong'), ('candidateInventorySha256', 'wrong'),
                                       ('rootInventorySha256', 'wrong'), ('processId', 0), ('addon', 'wrong')])
def test_gate_rejects_unbound_trial_provenance(monkeypatch, tmp_path, field, value):
    report = _bound_report(tmp_path)
    candidate = next(run for run in report['runs'] if run['label'] == 'candidate')
    candidate[field] = value
    path = tmp_path / 'artifacts/performance_budget/summary.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate, '_git_text', lambda *args: 'a' * 40)
    with pytest.raises(RuntimeError):
        gate._validate_performance_evidence()


def test_runner_measures_only_selected_demos_and_carries_the_rest(monkeypatch, tmp_path):
    import re
    source, _ = _stub_campaign(monkeypatch, tmp_path)
    fixtures = _campaign()
    measured = []
    def child(command, out, **kwargs):
        args = command[command.index('--') + 1:]
        if args[0] == '--extract':
            Path(out).write_text('{}')
            return {}
        root, blend, _, frames, reference, payload = args
        demo = Path(blend).stem
        measured.append(demo)
        block, slot = map(int, re.search(r'_b(\d+)_s(\d+)_', Path(out).stem).groups())
        row = next(dict(run) for run in fixtures
                   if run['workload'] == demo and run['round'] == block and run['slot'] == slot)
        row.update(root=root, addon=str(Path(root) / 'fluidSim_warp_blender51/__init__.py'),
                   rootInventorySha256=bench._inventory(Path(root)), processId=123,
                   version=reference if row['label'] == 'baseline' else '2.0.0')
        return row
    monkeypatch.setattr(bench, '_run_trial', child)
    prior = _bound_report(tmp_path / 'earlier')
    prior['gitHead'] = 'c' * 40
    for run in prior['runs']:
        if run['label'] == 'candidate':
            run['gitHead'] = 'c' * 40
    carried = tmp_path / 'carried.json'
    carried.write_text(json.dumps(prior))
    out = tmp_path / 'evidence'
    assert bench.demo_budget_main(['--blender', 'fake', '--candidate-root', str(source),
                                   '--out-dir', str(out), '--demos', 'paddle_stir',
                                   '--carry-summary', str(carried)]) == 0
    report = json.loads((out / 'summary.json').read_text())
    assert set(measured) == {'paddle_stir'}
    assert len(report['runs']) == 64 and report['passed'] is True
    assert report['demoMeasuredAt'] == {'crown_splash': 'c' * 40, 'dam_break': 'c' * 40,
                                        'paddle_stir': 'a' * 40, 'chromatic_vortex': 'c' * 40}
    assert report['demoInventory']['paddle_stir'] == bench._inventory(source)
    assert report['demoInventory']['dam_break'] == prior['candidateInventorySha256']


def test_runner_refuses_a_demo_subset_without_carried_evidence(tmp_path):
    with pytest.raises(SystemExit):
        bench.demo_budget_main(['--blender', 'fake', '--out-dir', str(tmp_path),
                                '--demos', 'paddle_stir'])
