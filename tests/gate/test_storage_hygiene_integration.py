"""Production orchestration proof with DCC/GPU execution replaced at its seams."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import run_production_gate as gate
from tools.storage_hygiene import integration


class ProductionStorageTests(unittest.TestCase):
    def test_agent_guidance_directs_normal_workflow_to_hooked_entrypoint(self):
        guidance = (gate.ROOT / 'AGENTS.md').read_text()
        self.assertIn('Active local development workflow', guidance)
        self.assertIn('`python tools/run_production_gate.py`', guidance)
        self.assertIn('mandatory storage regressions', guidance)
        self.assertIn('docs/RELEASING.md', guidance)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        output = root / 'completed-output'
        output.mkdir()
        (output / 'cache.bin').write_bytes(b'owned disposable fixture')
        self.policy = root / 'policy.json'
        self.policy.write_text(json.dumps(dict(
            version=1, drive=str(root.anchor), free_warn_bytes=10**18,
            max_entries=100, max_seconds=5, never_probe=[], records=[],
            roots=[dict(path=str(output), budget_bytes=1, stale_days=0,
                        project='liquid', owner='integration-test', state='completed',
                        disposable=True, retention_reason='fixture only', protected=[])])))

    def test_complete_production_main_dispatches_reports_and_real_hygiene_tests(self):
        calls = []
        def record(label, command, **kwargs):
            calls.append((label, command, kwargs))
        stream = io.StringIO()
        with patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': str(self.policy)}), \
             patch.object(gate, '_run', side_effect=record), \
             patch.object(gate, '_print_gate_metadata'), \
             patch.object(gate, '_clear_gate_manifest'), \
             patch.object(gate, '_check_no_tracked_npz'), \
             patch.object(gate, '_check_no_tracked_ignored_files'), \
             patch.object(gate, '_check_certified_warp', return_value='warp-root'), \
             patch.object(gate, '_validate_performance_evidence', return_value={}), \
             patch.object(gate, '_reusable_budget_demos', return_value=frozenset()), \
             contextlib.redirect_stderr(stream), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main([]), 0)
        report = stream.getvalue()
        self.assertIn('phase=pre-production', report)
        self.assertIn('phase=end-task', report)
        for code in ('low_free_space', 'budget', 'stale'):
            self.assertIn(code, report)
        self.assertEqual(calls[0][0], 'git whitespace')
        self.assertIn('paired demo performance budget', [row[0] for row in calls])
        self.assertTrue((Path(json.loads(self.policy.read_text())['roots'][0]['path']) / 'cache.bin').exists())

    def test_production_failure_preserves_exception_exit_and_post_report(self):
        stream = io.StringIO()
        failure = subprocess.CalledProcessError(37, ['production-fixture'])
        with patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': str(self.policy)}), \
             patch.object(gate, '_production_main', side_effect=failure), \
             contextlib.redirect_stderr(stream):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                gate.main([])
        self.assertIs(caught.exception, failure)
        self.assertEqual(caught.exception.returncode, 37)
        self.assertIn('phase=end-task', stream.getvalue())

    def test_hygiene_failure_prevents_production_and_still_reports_end(self):
        stream = io.StringIO()
        with patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': str(self.policy)}), \
             patch.object(integration.subprocess, 'run', side_effect=subprocess.CalledProcessError(9, ['tests'])), \
             patch.object(gate, '_production_main') as production, \
             contextlib.redirect_stderr(stream):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                gate.main([])
        production.assert_not_called()
        self.assertEqual(caught.exception.returncode, 9)
        self.assertIn('phase=end-task', stream.getvalue())

    def test_invalid_policy_diagnostic_does_not_replace_production_failure(self):
        self.policy.write_text('{}')
        stream = io.StringIO()
        with patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': str(self.policy)}), \
             patch.object(gate, '_production_main', side_effect=ValueError('production error')), \
             contextlib.redirect_stderr(stream):
            with self.assertRaisesRegex(ValueError, 'production error'):
                gate.main([])
        self.assertEqual(stream.getvalue().count('STORAGE-HYGIENE-DIAGNOSTIC'), 3)

    def test_override_cannot_remove_denied_or_source_protections(self):
        with patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': str(self.policy)}):
            config = integration.policy(gate.ROOT)
        required = json.loads((integration.HERE / 'policy.json').read_text())
        self.assertTrue(set(required['never_probe']).issubset(config['never_probe']))
        for root in required['roots']:
            self.assertTrue(set(root.get('protected', [])).issubset(config['never_probe']))
        self.assertIn(str(gate.ROOT.resolve()), config['never_probe'])
        template = dict(config['roots'][0])
        config['roots'] = [dict(template, path=required['never_probe'][0]),
                           dict(template, path=str(gate.ROOT.resolve()))]
        real_lstat = integration.checker.os.lstat
        def guarded(path):
            if any(integration.checker.within(path, hold) for hold in config['never_probe']):
                raise AssertionError('protected path probed')
            return real_lstat(path)
        with patch.object(integration.checker.os, 'lstat', side_effect=guarded):
            report = integration.checker.check(config, free_bytes=100)
        self.assertEqual(len(report['roots']), 2)
        self.assertTrue(all(not row['complete'] and row['logical_bytes'] is None
                            for row in report['roots']))


if __name__ == '__main__':
    unittest.main()
