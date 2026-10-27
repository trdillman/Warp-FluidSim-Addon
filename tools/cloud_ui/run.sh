#!/usr/bin/env bash
# Linux cloud desktop only. Reuses repository canonical profile-safe launcher.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${FLUIDSIM_SOURCE_ROOT:-$(cd "$HERE/../.." && pwd)}"
SOURCE="$(realpath -m "$SOURCE")"
WORK="${CLOUD_UI_WORK_ROOT:-$SOURCE/artifacts/cloud_ui}"
WORK="$(realpath -m "$WORK")"
RUN="${CLOUD_UI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe run ID' >&2; exit 2; }
mkdir -p "$WORK"
exec 9>"$WORK/.cloud-ui.lock"
flock -n 9 || { echo 'Another cloud UI lane is active' >&2; exit 2; }
export CLOUD_UI_OUTPUT="$WORK/$RUN"
[[ ! -e "$CLOUD_UI_OUTPUT" ]] || { echo 'Run already exists; choose a fresh run ID' >&2; exit 2; }
mkdir -p "$CLOUD_UI_OUTPUT"
RUNTIME="${CLOUD_UI_RUNTIME_ROOT:-$WORK/runtime}"
RUNTIME="$(realpath -m "$RUNTIME")"
export BLENDER_EXECUTABLE="${BLENDER_EXECUTABLE:-$RUNTIME/blender-5.2.2-linux-x64/blender}"
BLENDER_EXECUTABLE="$(realpath -m "$BLENDER_EXECUTABLE")"
if [[ ! -f "$BLENDER_EXECUTABLE" ]]; then
    [[ "${1:-}" == '--download' ]] || { echo 'Runtime absent. Use --download for official Blender 5.2.2.' >&2; exit 2; }
    mkdir -p "$RUNTIME"
    curl --fail --location --proto '=https' -o "$RUNTIME/blender-5.2.2-linux-x64.tar.xz" https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz
    (cd "$RUNTIME"; echo '84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168  blender-5.2.2-linux-x64.tar.xz' | sha256sum -c -; tar -xf blender-5.2.2-linux-x64.tar.xz)
fi
printf '%s  %s\n' bafec183e960819c76fe213c000cdb2284c53a87ef6791812f8728b51f07145b "$BLENDER_EXECUTABLE" | sha256sum -c -
"$BLENDER_EXECUTABLE" --version > "$CLOUD_UI_OUTPUT/blender-version.txt"
grep -q '^Blender 5.2.2 LTS$' "$CLOUD_UI_OUTPUT/blender-version.txt"
grep -q 'build hash: d13f752e3b9c' "$CLOUD_UI_OUTPUT/blender-version.txt"
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] || { echo 'Run from the cloud desktop terminal; no display is attached to this shell.' >&2; exit 2; }
[[ -f "$SOURCE/tools/gate/launch_profile_safe_addon_review.py" ]] || { echo 'Canonical repository launcher absent' >&2; exit 2; }
export BLENDER_USER_RESOURCES="$CLOUD_UI_OUTPUT/profile/resources"
export BLENDER_USER_CONFIG="$CLOUD_UI_OUTPUT/profile/config"
export BLENDER_USER_SCRIPTS="$CLOUD_UI_OUTPUT/profile/scripts"
export BLENDER_USER_EXTENSIONS="$CLOUD_UI_OUTPUT/profile/extensions"
export BLENDER_USER_DATAFILES="$CLOUD_UI_OUTPUT/profile/datafiles"
export XDG_CACHE_HOME="$CLOUD_UI_OUTPUT/profile/cache"
export TMPDIR="$CLOUD_UI_OUTPUT/profile/tmp"
export CUDA_CACHE_PATH="$CLOUD_UI_OUTPUT/profile/cuda-cache"
export WARP_CACHE_PATH="$CLOUD_UI_OUTPUT/profile/warp-cache"
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
unset PYTHONPATH PYTHONHOME
mkdir -p "$BLENDER_USER_RESOURCES" "$BLENDER_USER_CONFIG" "$BLENDER_USER_SCRIPTS" "$BLENDER_USER_EXTENSIONS" "$BLENDER_USER_DATAFILES" "$XDG_CACHE_HOME" "$TMPDIR" "$CUDA_CACHE_PATH" "$WARP_CACHE_PATH"
export CLOUD_UI_SOURCE="$SOURCE"
"$BLENDER_EXECUTABLE" -b --factory-startup --python-exit-code 1 --python "$HERE/create_fixture.py" > "$CLOUD_UI_OUTPUT/fixture.log" 2>&1
python3 "$SOURCE/tools/gate/launch_profile_safe_addon_review.py" --blender "$BLENDER_EXECUTABLE" --blend "$CLOUD_UI_OUTPUT/ui-fixture.blend" --report "$CLOUD_UI_OUTPUT/profile-safe-review.json" --review-script "$HERE/foreground_probe.py" --startup-timeout 180 2>&1 | tee "$CLOUD_UI_OUTPUT/launcher.log"
"$BLENDER_EXECUTABLE" -b --factory-startup "$CLOUD_UI_OUTPUT/ui-fixture.blend" --python-exit-code 1 --python "$HERE/verify_saved.py" > "$CLOUD_UI_OUTPUT/presentation-reopen.log" 2>&1
