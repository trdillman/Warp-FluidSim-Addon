import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import storage_hygiene as h


class HygieneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.file = self.output / 'cache.bin'
        self.file.write_bytes(b'1234567890')
        self.now = time.time()
        self.cfg = dict(version=1, free_warn_bytes=100, max_entries=1000,
                        max_seconds=5, roots=[dict(path=str(self.output),
                        budget_bytes=100, stale_days=7, project='test',
                        owner='test-owner', state='completed', disposable=True,
                        retention_reason='regenerable cache', protected=[])])

    def check(self):
        return h.check(self.cfg, now=self.now, free_bytes=1000)

    def codes(self, result):
        return {w['code'] for w in result['warnings']}

    def test_normal(self):
        result = self.check()
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['roots'][0]['logical_bytes'], 10)
        self.assertTrue(result['roots'][0]['complete'])

    def test_windows_policy_paths_validate_on_every_host(self):
        self.cfg['roots'][0]['path'] = r'C:\fixture'
        self.cfg['roots'][0]['protected'] = [r'C:\fixture\hold']
        self.cfg['never_probe'] = [r'C:\denied']
        self.cfg['records'] = [dict(path=r'C:\fixture\cache')]
        h.validate(self.cfg)
        self.assertTrue(h.within(r'c:\FIXTURE\hold', r'C:\fixture'))
        self.assertFalse(h.within(r'C:\fixture-other', r'C:\fixture'))
        self.cfg['roots'].append(dict(self.cfg['roots'][0], path=r'c:\FIXTURE\child'))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            h.validate(self.cfg)

    def test_windows_drive_relative_paths_remain_invalid(self):
        self.cfg['never_probe'] = [r'C:relative']
        with self.assertRaisesRegex(ValueError, 'absolute paths'):
            h.validate(self.cfg)

    @unittest.skipIf(os.name == 'nt', 'requires a non-Windows host')
    def test_foreign_windows_roots_are_never_probed(self):
        for path in (r'C:\fixture', r'\\server\share\fixture'):
            with self.subTest(path=path):
                self.cfg['roots'][0]['path'] = path
                with patch.object(h.os, 'lstat', side_effect=AssertionError('foreign probe')), \
                     patch.object(h.os, 'scandir', side_effect=AssertionError('foreign scan')):
                    result = self.check()
                root = result['roots'][0]
                self.assertFalse(root['complete'])
                self.assertIsNone(root['logical_bytes'])
                self.assertEqual(root['budget_assessment'], 'unknown')
                self.assertIn('foreign_platform', self.codes(result))
                self.assertIn('coverage_incomplete', self.codes(result))

    @unittest.skipIf(os.name == 'nt', 'requires a non-Windows host')
    def test_foreign_windows_drive_is_never_probed(self):
        self.cfg['drive'] = 'C:\\'
        self.cfg['roots'] = []
        with patch.object(h.shutil, 'disk_usage', side_effect=AssertionError('foreign drive probe')):
            result = h.check(self.cfg)
        self.assertIsNone(result['free_bytes'])
        self.assertIn('free_space_unavailable', self.codes(result))

    def test_oversized_is_warning_with_owner_and_safe_guidance(self):
        self.cfg['roots'][0]['budget_bytes'] = 5
        result = self.check()
        self.assertIn('budget', self.codes(result))
        warning = result['warnings'][0]
        self.assertEqual(warning['owner'], 'test-owner')
        self.assertIn('recoverable', warning['guidance'])
        self.assertEqual(self.file.read_bytes(), b'1234567890')

    def test_stale_owned_disposable(self):
        os.utime(self.file, (self.now - 8*86400, self.now - 8*86400))
        self.assertIn('stale', self.codes(self.check()))

    def test_active_not_stale_cleanup_candidate(self):
        os.utime(self.file, (self.now - 8*86400, self.now - 8*86400))
        self.cfg['roots'][0]['state'] = 'active'
        self.assertNotIn('stale', self.codes(self.check()))

    def test_low_free_space_separate_from_budget(self):
        self.assertIn('low_free_space', self.codes(h.check(self.cfg, free_bytes=50)))

    def test_inaccessible_not_empty(self):
        with patch.object(h.os, 'scandir', side_effect=PermissionError('denied')):
            result = self.check()
        self.assertIn('inaccessible', self.codes(result))
        self.assertFalse(result['roots'][0]['complete'])
        self.assertIsNone(result['roots'][0]['logical_bytes'])

    def test_reparse_is_not_followed(self):
        with patch.object(h, 'is_reparse', return_value=True):
            result = self.check()
        self.assertIn('reparse_skipped', self.codes(result))
        self.assertFalse(result['roots'][0]['complete'])
        self.assertIsNone(result['roots'][0]['logical_bytes'])

    def test_hardlinks_logical_and_unique_are_distinct(self):
        os.link(self.file, self.output / 'cache2.bin')
        measure = self.check()['roots'][0]
        self.assertEqual(measure['logical_bytes'], 20)
        self.assertEqual(measure['unique_file_bytes'], 10)

    def test_protected_evidence_not_read_or_marked_stale(self):
        evidence = self.output / 'failure-proof'
        evidence.mkdir()
        (evidence / 'keep.bin').write_bytes(b'x'*1000)
        self.cfg['roots'][0]['protected'] = [str(evidence)]
        result = self.check()
        measure = result['roots'][0]
        self.assertEqual(measure['logical_bytes'], 10)
        self.assertFalse(measure['complete'])
        self.assertIn('protected_unmeasured', self.codes(result))
        self.assertEqual((evidence / 'keep.bin').stat().st_size, 1000)

    def test_missing_registration_is_warning_never_disposable(self):
        self.cfg['roots'][0].pop('owner')
        self.cfg['roots'][0].pop('retention_reason')
        self.assertIn('registration', self.codes(self.check()))

    def test_bounded_scan_is_explicit_lower_bound(self):
        self.cfg['max_entries'] = 0
        result = self.check()
        self.assertIn('scan_limit', self.codes(result))
        self.assertFalse(result['roots'][0]['complete'])

    def test_missing_path_not_claimed_empty(self):
        self.file.unlink()
        self.output.rmdir()
        result = self.check()
        self.assertIn('missing', self.codes(result))
        self.assertIsNone(result['roots'][0]['logical_bytes'])

    def test_overlap_rejected_to_avoid_double_count(self):
        self.cfg['roots'].append(dict(self.cfg['roots'][0], path=str(self.output / 'nested')))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.check()

    def test_wrapper_runs_checks_before_and_after_and_preserves_exit(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.cfg))
        proc = subprocess.run([sys.executable, str(Path(h.__file__)),
            '--config', str(config), 'run', '--project', 'test', '--',
            sys.executable, '-c', 'raise SystemExit(7)'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 7)
        self.assertIn('phase=pre-production', proc.stderr)
        self.assertIn('phase=end-task', proc.stderr)

    def test_warnings_do_not_fail_production(self):
        self.cfg['roots'][0]['budget_bytes'] = 1
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.cfg))
        proc = subprocess.run([sys.executable, str(Path(h.__file__)),
            '--config', str(config), 'run', '--project', 'test', '--',
            sys.executable, '-c', 'print("production ran")'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('production ran', proc.stdout)
        self.assertIn('budget', proc.stderr)

    def test_invalid_config_is_diagnostic_error(self):
        self.cfg['max_entries'] = -1
        with self.assertRaises(ValueError):
            self.check()

    def test_unregistered_children_warn_without_assuming_ownership(self):
        self.cfg['roots'][0]['require_child_records'] = True
        (self.output / 'unknown-job').mkdir()
        result = self.check()
        self.assertIn('unregistered_children', self.codes(result))

    def test_registered_children_not_reported_unknown(self):
        self.cfg['roots'][0]['require_child_records'] = True
        child = self.output / 'known-job'
        child.mkdir()
        self.cfg['records'] = [dict(path=str(child), project='test', owner='owner',
             state='active', retention_reason='Current production input')]
        self.assertNotIn('unregistered_children', self.codes(self.check()))

    def test_nested_link_ancestor_is_not_scanned(self):
        real_lstat = h.os.lstat
        with patch.object(h, 'is_reparse', side_effect=lambda st: st.st_ino == real_lstat(self.output).st_ino):
            self.cfg['roots'][0]['path'] = str(self.output / 'nested')
            result = self.check()
        self.assertIn('reparse_skipped', self.codes(result))
        self.assertIsNone(result['roots'][0]['logical_bytes'])

    def test_bad_root_object_is_diagnostic_error(self):
        self.cfg['roots'] = ['bad']
        with self.assertRaises(ValueError):
            self.check()

    def test_bad_never_probe_is_diagnostic_error(self):
        self.cfg['never_probe'] = 'C:\\bad'
        with self.assertRaises(ValueError):
            self.check()

    def test_queued_directory_replaced_by_reparse_is_skipped(self):
        child = self.output / 'queued'
        child.mkdir()
        (child / 'must-not-read').write_bytes(b'x'*100)
        real_lstat, real_scandir = h.os.lstat, h.os.scandir
        queued_stat = real_lstat(child)
        def replacement_stat(path):
            if normalized_for_test(path) == normalized_for_test(child):
                values = list(queued_stat)
                values[0] = h.stat.S_IFLNK | 0o777
                return os.stat_result(values)
            return real_lstat(path)
        def guard_scandir(path):
            self.assertNotEqual(normalized_for_test(path), normalized_for_test(child))
            return real_scandir(path)
        with patch.object(h.os, 'lstat', side_effect=replacement_stat), patch.object(h.os, 'scandir', side_effect=guard_scandir):
            result = self.check()
        self.assertIn('reparse_skipped', self.codes(result))
        self.assertEqual(result['roots'][0]['logical_bytes'], 10)
        self.assertFalse(result['roots'][0]['complete'])

    def test_protected_path_is_never_statted(self):
        child = self.output / 'protected'
        child.mkdir()
        self.cfg['roots'][0]['protected'] = [str(child)]
        original_stat = h.os.stat
        def guard_stat(path, **kwargs):
            self.assertFalse(h.within(path, child))
            return original_stat(path, **kwargs)
        with patch.object(h.os, 'stat', side_effect=guard_stat):
            result = self.check()
        self.assertIn('protected_unmeasured', self.codes(result))

    def test_partial_access_failure_retains_lower_bound(self):
        child = self.output / 'denied'
        child.mkdir()
        original = h.os.scandir
        def denied(path):
            if normalized_for_test(path) == normalized_for_test(child):
                raise PermissionError('denied')
            return original(path)
        with patch.object(h.os, 'scandir', side_effect=denied):
            result = self.check()
        self.assertIn('inaccessible', self.codes(result))
        self.assertEqual(result['roots'][0]['logical_bytes'], 10)
        self.assertFalse(result['roots'][0]['complete'])

    def test_invalid_policy_does_not_replace_production_failure(self):
        cfg = self.root / 'bad.json'
        cfg.write_text('{"version": 9}')
        proc = subprocess.run([sys.executable, str(Path(h.__file__)), '--config', str(cfg),
            'run', '--project', 'test', '--', sys.executable, '-c', 'raise SystemExit(7)'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 7)
        self.assertIn('STORAGE-HYGIENE-DIAGNOSTIC', proc.stderr)

    def test_self_test_failure_prevents_production_start(self):
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(self.cfg))
        with patch.object(h.subprocess, 'call', return_value=3) as launch, contextlib.redirect_stderr(io.StringIO()) as errors:
            code = h.main(['--config', str(cfg), 'run', '--project', 'test', '--self-test', '--', 'production-exe'])
        self.assertEqual(code, 3)
        self.assertEqual(launch.call_count, 1)
        self.assertIn('STORAGE-HYGIENE-TEST-FAIL', errors.getvalue())

    def test_retained_old_files_warn_age_without_cleanup_authority(self):
        os.utime(self.file, (self.now - 8*86400, self.now - 8*86400))
        self.cfg['roots'][0].update(state='retained', disposable=False)
        result = self.check()
        self.assertIn('aging', self.codes(result))
        self.assertNotIn('stale', self.codes(result))
        warning = next(w for w in result['warnings'] if w['code'] == 'aging')
        self.assertFalse(warning['cleanup_authorized'])
        self.assertEqual(warning['retention_state'], 'retained')
        self.assertEqual(warning['observed_old_file_bytes'], 10)
        self.assertEqual(self.file.read_bytes(), b'1234567890')

    def test_unknown_owner_can_receive_age_observation(self):
        os.utime(self.file, (self.now - 8*86400, self.now - 8*86400))
        self.cfg['roots'][0].pop('owner')
        result = self.check()
        self.assertIn('aging', self.codes(result))
        self.assertIn('registration', self.codes(result))
        self.assertNotIn('stale', self.codes(result))

    def test_fresh_file_does_not_hide_old_observed_file(self):
        os.utime(self.file, (self.now - 8*86400, self.now - 8*86400))
        (self.output / 'fresh.bin').write_bytes(b'x')
        self.assertIn('aging', self.codes(self.check()))

    def test_excluded_coverage_warns_unknown_budget_and_sample_age(self):
        excluded = self.output / 'retained-proof'
        excluded.mkdir()
        (excluded / 'unmeasured').write_bytes(b'x'*1000)
        self.cfg['roots'][0]['protected'] = [str(excluded)]
        os.utime(self.file, (self.now - 8*86400, self.now - 8*86400))
        result = self.check()
        self.assertIn('coverage_incomplete', self.codes(result))
        warning = next(w for w in result['warnings'] if w['code'] == 'coverage_incomplete')
        self.assertEqual(warning['budget_assessment'], 'unknown')
        self.assertEqual(warning['observed_logical_bytes'], 10)
        aging = next(w for w in result['warnings'] if w['code'] == 'aging')
        self.assertFalse(aging['complete'])
        self.assertFalse(aging['cleanup_authorized'])
        self.assertNotIn('stale', self.codes(result))

    def test_entire_protected_root_has_unknown_coverage_not_zero(self):
        self.cfg['roots'][0]['protected'] = [str(self.output)]
        result = self.check()
        warning = next(w for w in result['warnings'] if w['code'] == 'coverage_incomplete')
        self.assertIsNone(warning['observed_logical_bytes'])
        self.assertEqual(warning['budget_assessment'], 'unknown')
        self.assertNotIn('aging', self.codes(result))

    def test_bounded_scan_is_unknown_budget_not_within_budget(self):
        self.cfg['max_entries'] = 0
        result = self.check()
        self.assertEqual(result['roots'][0]['budget_assessment'], 'unknown')
        self.assertIn('coverage_incomplete', self.codes(result))

    def test_all_observed_denials_are_recorded_for_later_phase_exclusion(self):
        children = [self.output / name for name in ('denied-one', 'denied-two')]
        for child in children:
            child.mkdir()
        original = h.os.scandir
        def denied(path):
            if normalized_for_test(path) in {normalized_for_test(p) for p in children}:
                raise PermissionError('synthetic denial')
            return original(path)
        with patch.object(h.os, 'scandir', side_effect=denied):
            result = self.check()
        self.assertEqual(set(result['roots'][0]['inaccessible_paths']), {str(p) for p in children})
        warnings = [w for w in result['warnings'] if w['code'] == 'inaccessible']
        self.assertEqual(len(warnings), 1)
        self.assertIn(warnings[0]['affected_path'], {str(p) for p in children})

    def test_observed_denial_excludes_exact_paths_without_hiding_container(self):
        child = str(self.output / 'denied')
        result = {'roots': [{'inaccessible_paths': [child]}], 'warnings': [
            {'code': 'inaccessible', 'path': str(self.output), 'affected_path': child}]}
        self.assertEqual(h.exclude_inaccessible(self.cfg, result), [child])
        self.assertIn(child, self.cfg['never_probe'])
        self.assertNotIn(str(self.output), self.cfg['never_probe'])
        self.assertEqual(h.exclude_inaccessible(self.cfg, result), [])


def normalized_for_test(path):
    return os.path.normcase(os.path.abspath(path))


if __name__ == '__main__':
    unittest.main()
