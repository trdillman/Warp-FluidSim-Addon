"""Integration test proving the Preferences Install Warp button in real Blender 5.2.

Covers:
1. Offline refusal (--offline-mode and use_online_access=False).
2. Online install from PyPI via bpy.ops.fluidsim.install_warp():
   - The locked Warp version lands strictly under the add-on's own extension path.
   - Blender's shared extensions/.local and user site-packages are never touched.
   - Runtime probe sees the locked Warp version.
   - Second press is idempotent and safe (no re-download, no rewrite).
3. Foreign Warp in shared extensions/.local:
   - Does not count as owned or already installed for FluidSim.
   - Refuses to overwrite or change foreign Warp.
4. Install from File with online access off (needs FLUIDSIM_TEST_WARP_WHEEL):
   - A wheel that is not the locked one is refused before anything is written.
   - The locked wheel installs into the add-on's own extension path.

Optional FLUIDSIM_TEST_WARP_WHEEL reuses a locally retained locked wheel for
installation/repair qualification. That mode does not qualify network transfer;
the default still downloads from PyPI. Every reuse verifies the pin and bytes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import json

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS_RELEASE = ROOT / "tools" / "release"
ADDON_ROOT = ROOT / "fluidSim_warp_blender51"
# The Warp version the add-on installs; the Blender scripts read it from the env.
LOCKED_WARP = json.loads(
    (ADDON_ROOT / "warp_wheels.lock.json").read_text(encoding="utf-8"))["warp_version"]


def _run_blender(*args, **kwargs):
    """Keep background qualification processes hidden on Windows."""
    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(*args, check=kwargs.pop("check", False), **kwargs)


@lru_cache(maxsize=1)
def _find_blender() -> Path | None:
    configured = os.environ.get("BLENDER")
    if configured and not Path(configured).is_file():
        pytest.fail(f"BLENDER executable does not exist: {configured}")
    candidates = (configured,) if configured else (
        shutil.which("blender"),
        "/usr/local/bin/blender",
        "/opt/blender-5.2.0-linux-x64/blender",
    )
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        result = _run_blender(
            [str(candidate), "--version"], capture_output=True, text=True,
            check=False, timeout=30,
        )
        match = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", result.stdout)
        if result.returncode == 0 and match and match.groups() == ("5", "2", "0"):
            return Path(candidate).resolve()
        if configured:
            pytest.fail(f"BLENDER must identify certified Blender 5.2.0: {result.stdout}")
    return None


def test_explicit_missing_blender_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER", str(tmp_path / "missing-blender"))
    _find_blender.cache_clear()
    try:
        with pytest.raises(pytest.fail.Exception, match="BLENDER executable does not exist"):
            _find_blender()
    finally:
        _find_blender.cache_clear()


@pytest.fixture(scope="module")
def built_package(tmp_path_factory) -> Path:
    blender = _find_blender()
    if blender is None:
        pytest.skip("Blender not found")

    sys.path.insert(0, str(TOOLS_RELEASE))
    try:
        import build_extension
    finally:
        sys.path.pop(0)

    dist_dir = tmp_path_factory.mktemp("real_blender_dist")
    packages = build_extension.build(
        ADDON_ROOT,
        blender,
        dist_dir,
        run_python_checks=False,
        run_lint=False,
    )
    platform = "windows_x64" if sys.platform == "win32" else "linux_x64"
    host_packages = [p for p in packages if platform in p.name]
    assert len(host_packages) == 1, f"Expected 1 {platform} package, got {packages}"
    return host_packages[0]


def _setup_isolated_blender(base_dir: Path, package: Path, repo_id: str = "test_repo") -> dict[str, str]:
    blender = _find_blender()
    assert blender is not None

    repo_dir = base_dir / "repo"
    config_dir = base_dir / "config"
    scripts_dir = base_dir / "scripts"
    extensions_dir = base_dir / "extensions"
    datafiles_dir = base_dir / "datafiles"
    temp_dir = base_dir / "temp"

    for d in (repo_dir, config_dir, scripts_dir, extensions_dir, datafiles_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "FLUIDSIM_ALLOW_EXTERNAL_WARP",
                "FLUIDSIM_TEST_EXTERNAL_RUNTIME"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env.update({
        "BLENDER_USER_CONFIG": str(config_dir),
        "BLENDER_USER_SCRIPTS": str(scripts_dir),
        "BLENDER_USER_EXTENSIONS": str(extensions_dir),
        "BLENDER_USER_DATAFILES": str(datafiles_dir),
        "PYTHONUSERBASE": str(base_dir / "python-user-base"),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "FLUIDSIM_TEST_REPOSITORY": str(repo_dir),
        "FLUIDSIM_TEST_LOCKED_WARP": LOCKED_WARP,
    })

    _run_blender(
        [
            str(blender),
            "--command",
            "extension",
            "repo-add",
            repo_id,
            "--name",
            "Test Repo",
            "--directory",
            str(repo_dir),
            "--clear-all",
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    _run_blender(
        [
            str(blender),
            "--command",
            "extension",
            "install-file",
            "-r",
            repo_id,
            "-e",
            str(package),
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    return env


def _factory_script(script: str) -> str:
    """Enable only the installed test extension after factory startup."""
    setup = """
import addon_utils
import bpy
import os

bpy.context.preferences.extensions.repos.new(
    name='Test Repo', module='test_repo',
    custom_directory=os.environ['FLUIDSIM_TEST_REPOSITORY'],
)
addon_utils.enable('bl_ext.test_repo.fluidsim_warp', default_set=True)
assert 'bl_ext.test_repo.fluidsim_warp' in bpy.context.preferences.addons

cached_wheel = os.environ.get('FLUIDSIM_TEST_WARP_WHEEL')
if cached_wheel:
    import importlib
    import shutil
    from pathlib import Path
    installer = importlib.import_module('bl_ext.test_repo.fluidsim_warp.warp_installer')
    locked = installer.get_wheel_info_for_platform()
    payload = Path(cached_wheel)
    assert locked is not None and installer._wheel_matches_lock(payload, locked)
    def copy_locked_wheel(url, destination, *, expected_sha256, expected_size,
                          progress_callback=None):
        assert url == locked.url
        assert expected_sha256 == locked.sha256 and expected_size == locked.size
        assert installer._wheel_matches_lock(payload, locked)
        shutil.copyfile(payload, destination)
        assert installer._wheel_matches_lock(destination, locked)
        if progress_callback:
            progress_callback(locked.size, locked.size)
    installer._download_wheel = copy_locked_wheel
    print('LOCKED-WHEEL-REUSE: installation checks; network transfer excluded')
"""
    return setup + script


def test_offline_refusal_in_real_blender(tmp_path: Path, built_package: Path):
    """Verify that bpy.ops.fluidsim.install_warp() refuses before download when offline."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)

    script = """
import bpy
from pathlib import Path

assert bpy.app.online_access is False
module = 'bl_ext.test_repo.fluidsim_warp'
owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))

try:
    res = bpy.ops.fluidsim.install_warp()
    raise AssertionError(f'Expected operator error/cancellation, got {res}')
except RuntimeError as exc:
    assert 'Online access required' in str(exc), f'Unexpected message: {exc}'

assert not (owned_dir / 'warp').exists(), 'Warp directory was created during offline refusal!'
print('OFFLINE-REFUSE-VERIFIED')
"""
    result = _run_blender(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--offline-mode",
            "--python-exit-code",
            "1",
            "--python-expr",
            _factory_script(script),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "OFFLINE-REFUSE-VERIFIED" in result.stdout


def test_install_from_file_offline_in_real_blender(tmp_path: Path, built_package: Path):
    """Install from File works with online access off and refuses a wrong wheel."""
    wheel = os.environ.get("FLUIDSIM_TEST_WARP_WHEEL")
    if not wheel:
        pytest.skip("FLUIDSIM_TEST_WARP_WHEEL names no locally retained locked wheel")
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)
    wrong = tmp_path / Path(wheel).name
    wrong.write_bytes(b"not the certified wheel")
    env["FLUIDSIM_TEST_WRONG_WHEEL"] = str(wrong)

    script = """
import bpy
import importlib
import os
from pathlib import Path

assert bpy.app.online_access is False
module = 'bl_ext.test_repo.fluidsim_warp'
owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))

try:
    bpy.ops.fluidsim.install_warp_from_file(filepath=os.environ['FLUIDSIM_TEST_WRONG_WHEEL'])
    raise AssertionError('Expected refusal of a wheel that is not the locked one')
except RuntimeError as exc:
    assert 'is not the certified Warp wheel' in str(exc), f'Unexpected message: {exc}'
assert not (owned_dir / 'warp').exists(), 'A refused wheel left Warp behind'

res = bpy.ops.fluidsim.install_warp_from_file(filepath=os.environ['FLUIDSIM_TEST_WARP_WHEEL'])
assert res == {'FINISHED'}, f'Expected FINISHED, got {res}'
warp_init = owned_dir / 'warp' / '__init__.py'
assert warp_init.is_file(), f'Expected Warp installed at {warp_init}'
assert not (Path(bpy.utils.user_resource('EXTENSIONS')) / '.local' / 'warp').exists()

runtime = importlib.import_module(module + '.runtime.warp_runtime')
probe = runtime.probe_runtime(force=True)
assert probe.warp_version == os.environ['FLUIDSIM_TEST_LOCKED_WARP'], probe
assert Path(probe.module_file).resolve() == warp_init.resolve()
print('INSTALL-FROM-FILE-OFFLINE-VERIFIED')
"""
    result = _run_blender(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--offline-mode",
            "--python-exit-code",
            "1",
            "--python-expr",
            _factory_script(script),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "INSTALL-FROM-FILE-OFFLINE-VERIFIED" in result.stdout


def test_online_install_and_idempotency_in_real_blender(tmp_path: Path, built_package: Path):
    """Verify online install, destination isolation, probe detection, and second-press idempotency."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)

    script = """
import bpy
import os
import site
import importlib
from pathlib import Path

bpy.context.preferences.system.use_online_access = True
assert bpy.app.online_access is True

module = 'bl_ext.test_repo.fluidsim_warp'
owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
shared_extensions = Path(bpy.utils.user_resource('EXTENSIONS'))
shared_local = shared_extensions / '.local'
user_site = Path(site.getusersitepackages())

assert not (owned_dir / 'warp').exists()

# 1. First press: online install from PyPI
res = bpy.ops.fluidsim.install_warp()
assert res == {'FINISHED'}, f'Expected FINISHED, got {res}'

warp_init = owned_dir / 'warp' / '__init__.py'
assert warp_init.is_file(), f'Expected Warp installed at {warp_init}'

# Shared extensions/.local and user site-packages must NEVER be touched
assert not (shared_local / 'warp').exists(), f'Shared {shared_local} was touched!'
assert not (user_site / 'warp').exists(), f'User site {user_site} was touched!'

# 2. Runtime probe detection
res_probe = bpy.ops.fluidsim.refresh_runtime_probe()
assert res_probe == {'FINISHED'}

runtime = importlib.import_module(f'{module}.runtime.warp_runtime')
probe = runtime.probe_runtime()
assert probe.warp_version == os.environ['FLUIDSIM_TEST_LOCKED_WARP'], f'Expected the locked Warp, got {probe.warp_version}'
assert Path(probe.module_file).resolve() == warp_init.resolve()

# 3. Second press: idempotent, no rewrite or foreign changes
stat_before = warp_init.stat()
res2 = bpy.ops.fluidsim.install_warp()
assert res2 == {'FINISHED'}, f'Expected FINISHED, got {res2}'
stat_after = warp_init.stat()

assert stat_before.st_mtime == stat_after.st_mtime, 'Warp was rewritten on second press!'
assert not (shared_local / 'warp').exists()
assert not (user_site / 'warp').exists()

# A forced repair while Warp's native DLL is loaded must not mutate its tree.
assert bpy.ops.fluidsim.install_warp(force_reinstall=True) == {'FINISHED'}
assert warp_init.stat().st_mtime == stat_before.st_mtime

# Missing version metadata must still stage a loaded runtime for restart.
import shutil
installer = importlib.import_module(f'{module}.warp_installer')
runtime_files = {
    str(path.relative_to(owned_dir)): (path.stat().st_size, path.stat().st_mtime_ns)
    for path in (owned_dir / 'warp').rglob('*') if path.is_file()
}
metadata = list(owned_dir.glob('warp_lang-*.dist-info'))
assert len(metadata) == 1
assert metadata[0].resolve().parent == owned_dir.resolve()
shutil.rmtree(installer._filesystem_path(metadata[0]))
assert installer._owned_distribution_version(owned_dir) is None
assert bpy.ops.fluidsim.install_warp(force_reinstall=True) == {'FINISHED'}
assert installer._pending_repair_path(owned_dir).is_file()
for relative, state in runtime_files.items():
    path = owned_dir / relative
    assert (path.stat().st_size, path.stat().st_mtime_ns) == state, relative
print('ONLINE-INSTALL-AND-IDEMPOTENCY-VERIFIED')
"""
    result = _run_blender(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python-expr",
            _factory_script(script),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "ONLINE-INSTALL-AND-IDEMPOTENCY-VERIFIED" in result.stdout
    verify_restart = """
import bpy
import os
import importlib
from pathlib import Path
module = 'bl_ext.test_repo.fluidsim_warp'
runtime = importlib.import_module(module + '.runtime.warp_runtime')
probe = runtime.probe_runtime(force=True)
owned = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
assert probe.warp_version == os.environ['FLUIDSIM_TEST_LOCKED_WARP'], probe
assert Path(probe.module_file).resolve() == (owned / 'warp' / '__init__.py').resolve()
print('LOADED-NATIVE-REPAIR-AFTER-RESTART-VERIFIED')
"""
    result = _run_blender(
        [str(blender), '--background', '--factory-startup',
         '--python-exit-code', '1', '--python-expr', _factory_script(verify_restart)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert 'LOADED-NATIVE-REPAIR-AFTER-RESTART-VERIFIED' in result.stdout


def test_shared_extensions_local_foreign_warp_refusal_in_real_blender(tmp_path: Path, built_package: Path):
    """Verify that Warp in shared extensions/.local is not treated as owned and is not overwritten."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)

    # Seed Warp into shared extensions/.local
    shared_local = Path(env["BLENDER_USER_EXTENSIONS"]) / ".local"
    warp_dir = shared_local / "warp"
    warp_dir.mkdir(parents=True, exist_ok=True)
    (warp_dir / "__init__.py").write_text(f"__version__ = '{LOCKED_WARP}'\n", encoding="utf-8")
    dist_info = shared_local / f"warp_lang-{LOCKED_WARP}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(f"Name: warp-lang\nVersion: {LOCKED_WARP}\n", encoding="utf-8")

    script = """
import bpy
import sys
from pathlib import Path

bpy.context.preferences.system.use_online_access = True
module = 'bl_ext.test_repo.fluidsim_warp'
installer = __import__(f'{module}.warp_installer', fromlist=['warp_installer'])

shared_local = Path(bpy.utils.user_resource('EXTENSIONS')) / '.local'
assert (shared_local / 'warp' / '__init__.py').is_file()

owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
assert not (owned_dir / 'warp').exists()

# Shared extensions/.local must NOT be in owned candidates or satisfy installed check
assert shared_local not in installer._owned_candidates()
assert installer.owned_install_status() is None
assert installer.is_certified_warp_installed() is False

# If foreign Warp was imported from shared_local, install_warp must refuse
if str(shared_local) not in sys.path:
    sys.path.insert(0, str(shared_local))
import warp as wp
assert wp.__file__ == str(shared_local / 'warp' / '__init__.py')

try:
    bpy.ops.fluidsim.install_warp()
    raise AssertionError('Expected refusal when foreign Warp is loaded')
except RuntimeError as exc:
    assert 'which FluidSim does not own' in str(exc)
    assert str(shared_local) in str(exc)

print('SHARED-EXTENSIONS-LOCAL-REFUSAL-VERIFIED')
"""
    result = _run_blender(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python-expr",
            _factory_script(script),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "SHARED-EXTENSIONS-LOCAL-REFUSAL-VERIFIED" in result.stdout


def test_corrupted_owned_install_can_be_repaired_in_real_blender(tmp_path: Path, built_package: Path):
    """Repair a certified owned copy after a fresh process sees import failure."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)
    seed = """
import bpy
from pathlib import Path
bpy.context.preferences.system.use_online_access = True
assert bpy.ops.fluidsim.install_warp() == {'FINISHED'}
owned = Path(bpy.utils.extension_path_user(
    'bl_ext.test_repo.fluidsim_warp', path='.local', create=False))
(owned / 'warp' / '__init__.py').write_text(
    "raise ImportError('deliberately corrupted owned test install')\\n",
    encoding='utf-8')
print('OWNED-CORRUPTION-SEEDED')
"""
    repair = """
import bpy
import importlib
from pathlib import Path
module = 'bl_ext.test_repo.fluidsim_warp'
runtime = importlib.import_module(module + '.runtime.warp_runtime')
broken = runtime.probe_runtime(force=True)
assert not broken.available
assert 'deliberately corrupted' in broken.last_error, broken.last_error
bpy.context.preferences.system.use_online_access = True
assert bpy.ops.fluidsim.install_warp(force_reinstall=True) == {'FINISHED'}
owned = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
assert 'deliberately corrupted' in (owned / 'warp' / '__init__.py').read_text(encoding='utf-8')
print('CORRUPTED-OWNED-INSTALL-REPAIR-STAGED')
"""
    verify_restart = """
import bpy
import os
import importlib
from pathlib import Path
module = 'bl_ext.test_repo.fluidsim_warp'
runtime = importlib.import_module(module + '.runtime.warp_runtime')
probe = runtime.probe_runtime(force=True)
owned = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
assert probe.warp_version == os.environ['FLUIDSIM_TEST_LOCKED_WARP'], probe
assert Path(probe.module_file).resolve() == (owned / 'warp' / '__init__.py').resolve()
assert not (Path(bpy.utils.user_resource('EXTENSIONS')) / '.local' / 'warp').exists()
print('CORRUPTED-OWNED-INSTALL-REPAIR-VERIFIED')
"""
    for script, marker in (
        (seed, 'OWNED-CORRUPTION-SEEDED'),
        (repair, 'CORRUPTED-OWNED-INSTALL-REPAIR-STAGED'),
        (verify_restart, 'CORRUPTED-OWNED-INSTALL-REPAIR-VERIFIED'),
    ):
        result = _run_blender(
            [str(blender), '--background', '--factory-startup',
             '--python-exit-code', '1', '--python-expr', _factory_script(script)],
            env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
        assert marker in result.stdout


def test_explicit_local_helper_provisions_owned_warp_in_real_blender(tmp_path: Path, built_package: Path):
    """Exercise the helper's actual Blender script and online-state restoration."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)
    script = f"""
import bpy
import os
import importlib
import sys
from pathlib import Path
sys.path.insert(0, {str(TOOLS_RELEASE)!r})
import install_local
bpy.context.preferences.system.use_online_access = False
exec(compile(install_local.INSTALL_WARP_SCRIPT, '<explicit-local-warp-install>', 'exec'))
assert bpy.context.preferences.system.use_online_access is False
module = 'bl_ext.test_repo.fluidsim_warp'
runtime = importlib.import_module(module + '.runtime.warp_runtime')
probe = runtime.probe_runtime(force=True)
owned = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
assert probe.warp_version == os.environ['FLUIDSIM_TEST_LOCKED_WARP'], probe
assert Path(probe.module_file).resolve() == (owned / 'warp' / '__init__.py').resolve()
assert not (Path(bpy.utils.user_resource('EXTENSIONS')) / '.local' / 'warp').exists()
print('EXPLICIT-LOCAL-HELPER-INSTALL-VERIFIED')
"""
    result = _run_blender(
        [str(blender), '--background', '--factory-startup',
         '--python-exit-code', '1', '--python-expr', _factory_script(script)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert 'EXPLICIT-LOCAL-HELPER-INSTALL-VERIFIED' in result.stdout


def test_authorized_external_override_keeps_controls_in_real_blender(tmp_path, built_package):
    """A configured external copy stays visible without becoming installer-owned."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)
    seed = """
import bpy
bpy.context.preferences.system.use_online_access = True
assert bpy.ops.fluidsim.install_warp() == {'FINISHED'}
print('OVERRIDE-SEED-INSTALLED')
"""
    result = _run_blender(
        [str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
         '--python-expr', _factory_script(seed)],
        env=env, capture_output=True, text=True, check=False, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OVERRIDE-SEED-INSTALLED' in result.stdout
    # Move only this disposable fixture's owned installation, never user files.
    roots = list(Path(env["BLENDER_USER_EXTENSIONS"]).rglob("warp/__init__.py"))
    assert len(roots) == 1
    external = tmp_path / "external-runtime"
    shutil.move(str(roots[0].parents[1]), str(external))
    env["FLUIDSIM_ALLOW_EXTERNAL_WARP"] = "1"
    env["FLUIDSIM_TEST_EXTERNAL_RUNTIME"] = str(external)
    script = """
import bpy, importlib, os
from types import SimpleNamespace
module = 'bl_ext.test_repo.fluidsim_warp'
preferences = importlib.import_module(module + '.preferences')
installer = importlib.import_module(module + '.warp_installer')
runtime = importlib.import_module(module + '.runtime.warp_runtime')
prefs = bpy.context.preferences.addons[module].preferences
prefs.developer_warp_path = os.environ['FLUIDSIM_TEST_EXTERNAL_RUNTIME']
assert bpy.ops.fluidsim.refresh_runtime_probe() == {'FINISHED'}
probe = runtime.probe_runtime()
assert probe.warp_version == os.environ['FLUIDSIM_TEST_LOCKED_WARP'], probe
assert installer.owned_install_status() is None
assert 'does not own' in installer.foreign_warp_block_reason()
calls = []
class Layout:
    def box(self): return self
    def row(self, **kwargs): return self
    def label(self, **kwargs): calls.append(('label', kwargs.get('text')))
    def prop(self, owner, name, **kwargs): calls.append(('prop', name))
    def operator(self, name, **kwargs):
        calls.append(('operator', name))
        return SimpleNamespace(force_reinstall=False)
view = SimpleNamespace(layout=Layout(), developer_warp_path=prefs.developer_warp_path)
preferences.FLUIDSIM_AP_preferences.draw(view, bpy.context)
assert ('prop', 'device') in calls, calls
assert ('operator', 'fluidsim.refresh_runtime_probe') in calls, calls
assert ('label', 'Remove the other Warp copy first') not in calls, calls

# The UI exception never relaxes the native runtime version admission gate.
import warp
version = warp.__version__
warp.__version__ = '1.16.0'
rejected = runtime.probe_runtime(force=True)
assert not rejected.available and 'unsupported Warp version' in rejected.last_error, rejected
warp.__version__ = version
runtime.probe_runtime(force=True)

# Revoking authorization immediately restores the normal foreign-copy UI.
os.environ.pop('FLUIDSIM_ALLOW_EXTERNAL_WARP')
calls.clear()
preferences.FLUIDSIM_AP_preferences.draw(view, bpy.context)
assert ('label', 'Remove the other Warp copy first') in calls, calls
assert ('prop', 'device') not in calls, calls
print('AUTHORIZED-EXTERNAL-OVERRIDE-CONTROLS-VERIFIED')
"""
    result = _run_blender(
        [str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
         '--python-expr', _factory_script(script)],
        env=env, capture_output=True, text=True, check=False, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'AUTHORIZED-EXTERNAL-OVERRIDE-CONTROLS-VERIFIED' in result.stdout
