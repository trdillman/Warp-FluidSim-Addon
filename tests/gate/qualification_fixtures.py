"""Synthetic observed-runtime receipts for CPU orchestration tests only."""
from pathlib import Path

from tools.gate import qualification as q


def context(root, head='a' * 40):
    observed = {'warp_version': '1.19.0', 'gpu_uuid': 'GPU-test', 'driver': '600.1',
                'blender_sha256': 'b' * 64, 'platform': 'test'}
    return {'run_id': 'test-measurement', 'gitStatus': [],
            'source_fingerprint': 'a' * 64, 'verifier_fingerprint': 'b' * 64,
            'environment_fingerprint': q.fingerprint(observed), 'observed_environment': observed,
            'generated_semantics': {}, 'environment_request': str(root / 'request.json'),
            'frozen_source': {'git_head': head, 'files': {}, 'file_mtimes': {}, 'source_fingerprint': 'a' * 64,
                              'verifier_fingerprint': 'b' * 64}}


def bind_report(report, root, binding=None):
    binding = context(root) if binding is None else binding
    report.update(binding)
    report['retained_summary'] = str(root / 'retained-summary.json')
    for index, run in enumerate(report['runs']):
        run.update({key: value for key, value in binding.items()
                    if key in (*q.BINDING_FIELDS, 'observed_environment', 'run_id', 'gitStatus')})
        run['warp'] = binding['observed_environment']['warp_version']
        run['source_fingerprint'] = q.measurement_sources(root, run['workload'], binding['generated_semantics'])
        retain(run, root / 'raw-trials' / f'{index}.json')
    q.atomic_json(report["retained_summary"], report)
    return binding


def retain(run, path=None):
    path = Path(path or run['trial_artifact'])
    run.pop('trial_artifact', None)
    run.pop('trial_sha256', None)
    q.atomic_json(path, run)
    run.update(trial_artifact=str(path), trial_sha256=q.sha256(path))


def stub_measurement(monkeypatch, bench, root, head='a' * 40):
    binding = context(root, head)
    (root / 'tools/gate').mkdir(parents=True, exist_ok=True)
    (root / 'tools/gate/qualification.py').write_text('# snapshot fixture')
    monkeypatch.setattr(bench, '_measurement_context', lambda *args: binding)
    return binding
