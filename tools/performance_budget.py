"""D7 demo budget: paired medians, frozen scenes and independent stage evidence.

The raw evidence is authoritative; summary scores are always recomputed.
"""
from __future__ import annotations

import math
import statistics

REFERENCES = {
    'dam_break': '9c0d99e62adda14bb7b075a447c01fefe7187da3',
    'crown_splash': '9c0d99e62adda14bb7b075a447c01fefe7187da3',
    'paddle_stir': '9c0d99e62adda14bb7b075a447c01fefe7187da3',
    'chromatic_vortex': 'c5919d668f469942200c58274eca13337d3aa624',
}
SCENE_REF = 'c5919d668f469942200c58274eca13337d3aa624'
SCENE_REFS = {demo: SCENE_REF for demo in REFERENCES}
SCENE_REFS['chromatic_vortex'] = 'ad769d23c3c4d7e6eaa5ac0e0a3b29a46ce07689'
LIMIT = 1.05
MIN_WORK_RATIO = .95
MIN_TRIALS = 8
FRAMES = 48


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('timing must be positive and finite')
    return value


def comparable_settings(settings):
    result = dict(settings)
    # Only the three added fields have an explicit compatibility meaning.
    # Old versions always export attributes and have no color mixing.
    result.setdefault('export_attributes', True)
    result.setdefault('species_enabled', False)
    if not result['species_enabled']:
        result.pop('species_mixing_rate', None)
    # These two defaults changed between releases but execute no work while
    # their parent feature is disabled. Active values remain strict.
    if result.get('viscosity_enabled') is False:
        result.pop('viscosity_iterations', None)
    if result.get('spacetime_enabled') is False:
        result.pop('spacetime_variable_projection', None)
    return result


def evaluate(runs):
    """PASS needs both timing and synced-stage solver medians within 5%.

    A disagreement is INCONCLUSIVE and blocks the gate. A noisy wall-clock
    series alone can never label a change a proven performance regression.
    """
    demos = {}
    expected = set(REFERENCES)
    if {r.get('demo') for r in runs} != expected:
        raise ValueError('all four D7 demos are required')
    if any(r.get('mode') not in ('timing', 'stage') or r.get('label') not in ('reference', 'candidate')
           or type(r.get('round')) is not int for r in runs):
        raise ValueError('invalid trial identity')
    keys = [(r['demo'], r['mode'], r['round'], r['label']) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate trial')
    for index, demo in enumerate(REFERENCES):
        ratios = {}
        order_ratios = {}
        work_ratios = {}
        stages = {}
        cohort = [r for r in runs if r['demo'] == demo]
        first = cohort[0]
        for row in cohort:
            if (row.get('blend_sha256') != first.get('blend_sha256')
                    or row.get('device') != first.get('device')
                    or comparable_settings(row.get('settings', {})) != comparable_settings(first.get('settings', {}))):
                raise ValueError('workload/device changed between trials or measurement modes')
        if {r['round'] for r in cohort if r['mode'] == 'timing'} != {r['round'] for r in cohort if r['mode'] == 'stage'}:
            raise ValueError('timing/stage rounds differ')
        for mode in ('timing', 'stage'):
            selected = [r for r in runs if r['demo'] == demo and r['mode'] == mode]
            rounds = sorted({r['round'] for r in selected})
            if len(rounds) < MIN_TRIALS or len(rounds) % 2 or rounds != list(range(len(rounds))):
                raise ValueError('at least eight complete even paired rounds required')
            paired = []
            by_order = {'reference-first': [], 'candidate-first': []}
            stage_pairs = {}
            work_pairs = []
            for rnd in rounds:
                pair = {r['label']: r for r in selected if r['round'] == rnd}
                if set(pair) != {'reference', 'candidate'}:
                    raise ValueError('incomplete pair')
                a, b = pair['reference'], pair['candidate']
                order = (['reference', 'candidate'] if (rnd + index) % 2 == 0
                         else ['candidate', 'reference'])
                if any(r['order'] != order for r in pair.values()):
                    raise ValueError('non-rotating order')
                for field in ('blend_sha256', 'device'):
                    if not a.get(field) or a[field] != b.get(field):
                        raise ValueError(f'incomparable {field}')
                if not a.get('settings') or comparable_settings(a['settings']) != comparable_settings(b.get('settings', {})):
                    raise ValueError('incomparable physics/workload settings')
                if a.get('warp') != '1.16.0' or b.get('warp') != '1.17.0':
                    raise ValueError('reference Warp 1.16.0 and candidate Warp 1.17.0 required')
                medians = []
                for row in (a, b):
                    if row['profiler_sync'] != (mode == 'stage'):
                        raise ValueError('stage evidence must synchronize the GPU')
                    records = row['records']
                    if [f['frame'] for f in records] != list(range(1, FRAMES + 1)):
                        raise ValueError('frames 1-48 required; frame 1 excluded from scores')
                    previous_time = 0.0
                    for frame in records:
                        if frame.get('solver_frame') != frame['frame']:
                            raise ValueError('solver must advance exactly once per frame')
                        current_time = _positive(frame.get('sim_time'))
                        if current_time <= previous_time:
                            raise ValueError('simulation time must advance')
                        previous_time = current_time
                        for field in ('particles', 'substeps'):
                            if type(frame.get(field)) is not int or frame[field] <= 0:
                                raise ValueError(f'nonempty {field} evidence required')
                    if row.get('particles') != records[-1]['particles']:
                        raise ValueError('final particle evidence differs')
                    values = [_positive(f['step_ms']) for f in records[1:]]
                    medians.append(statistics.median(values))
                    if mode == 'stage':
                        names = set(records[1]['stages'])
                        if not {'solver', 'p2g'}.issubset(names):
                            raise ValueError('missing solver/P2G stage evidence')
                        for frame in records[1:]:
                            if set(frame['stages']) != names:
                                raise ValueError('incomplete stage breakdown')
                            for value in frame['stages'].values():
                                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                                    raise ValueError('invalid stage time')
                if [f['particles'] for f in a['records']] != [f['particles'] for f in b['records']]:
                    raise ValueError('incomparable per-frame particle workload')
                # Adaptive CFL schedules can differ as trajectories evolve.
                # Never credit a material reduction in actual work as speedup.
                work_pairs.append(sum(f['substeps'] for f in b['records'][1:]) /
                                  sum(f['substeps'] for f in a['records'][1:]))
                if mode == 'stage':
                    if set(a['records'][1]['stages']) != set(b['records'][1]['stages']):
                        raise ValueError('stage inventory differs')
                    for name in a['records'][1]['stages']:
                        av = statistics.median(f['stages'][name] for f in a['records'][1:])
                        bv = statistics.median(f['stages'][name] for f in b['records'][1:])
                        stage_pairs.setdefault(name, []).append((av, bv))
                    av, bv = stage_pairs['solver'][-1]
                    paired.append(_positive(bv) / _positive(av))
                else:
                    paired.append(medians[1] / medians[0])
                by_order['reference-first' if order[0] == 'reference' else 'candidate-first'].append(paired[-1])
            ratios[mode] = statistics.median(paired)
            work_ratios[mode] = statistics.median(work_pairs)
            order_ratios[mode] = {key: statistics.median(values) for key, values in by_order.items()}
            if mode == 'stage':
                stages = {name: {'reference_ms': statistics.median(a for a, _ in pairs),
                                 'candidate_ms': statistics.median(b for _, b in pairs)}
                          for name, pairs in stage_pairs.items()}
        over = [ratios[m] > LIMIT for m in ('timing', 'stage')]
        confirmed = all(value > LIMIT for groups in order_ratios.values() for value in groups.values())
        verdict = 'FAIL' if all(over) and confirmed else 'INCONCLUSIVE' if any(over) else 'PASS'
        if verdict == 'PASS' and any(value < MIN_WORK_RATIO for value in work_ratios.values()):
            verdict = 'INCONCLUSIVE'
        demos[demo] = {'reference_commit': REFERENCES[demo], 'ratios': ratios,
                       'stages': stages, 'order_ratios': order_ratios,
                       'work_ratios': work_ratios, 'verdict': verdict}
    verdicts = {d['verdict'] for d in demos.values()}
    return {'schema': 'fluidsim-d7-budget-v1', 'limit': LIMIT, 'demos': demos,
            'verdict': 'FAIL' if 'FAIL' in verdicts else
            'INCONCLUSIVE' if 'INCONCLUSIVE' in verdicts else 'PASS'}
