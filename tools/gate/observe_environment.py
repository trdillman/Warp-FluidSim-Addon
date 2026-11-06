"""Blender-side observations; never substitute requested versions for loaded ones."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location('_qualification', Path(__file__).with_name('qualification.py'))
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


def observe(request):
    import bpy
    import warp as wp

    wheel = Path(request['wheel'])
    loaded = Path(wp.__file__).resolve()
    # Check source/native payloads, never mutable bytecode or cache directories.
    payloads = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.parts[0] != 'warp' or path.suffix not in ('.py', '.so', '.dll', '.pyd', '.dylib'):
                continue
            actual = loaded.parent.parent / path
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if q.sha256(actual) != expected:
                raise RuntimeError(f'loaded Warp payload differs from locked wheel: {name}')
            payloads[name] = expected
    if not payloads:
        raise RuntimeError('wheel contains no verifiable Warp payload')
    wp.init()
    device = wp.get_device('cuda:0')
    if not device.is_cuda or not device.uuid:
        raise RuntimeError('qualification requires an observed CUDA GPU UUID')
    driver_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                           '--format=csv,noheader'], text=True)
    drivers = dict(tuple(value.strip() for value in line.split(',', 1))
                   for line in driver_rows.splitlines())
    return {'blender_version': list(bpy.app.version[:3]),
            'blender_build': bpy.app.build_hash.decode(),
            'blender_sha256': q.sha256(bpy.app.binary_path),
            'python_version': sys.version, 'python_prefix': str(Path(sys.prefix).resolve()),
            'python_executable': str(Path(request['python']).resolve()),
            'python_sha256': q.sha256(request['python']),
            'warp_version': str(wp.__version__),
            'warp_distribution': importlib.metadata.version('warp-lang'),
            'warp_wheel_sha256': q.sha256(wheel), 'warp_loaded_path': str(loaded),
            'warp_payload_fingerprint': q.fingerprint(payloads),
            'platform': platform.system(), 'architecture': platform.machine(),
            'gpu_uuid': device.uuid, 'gpu_model': device.name,
            'driver': drivers[device.uuid], 'cuda_driver': list(wp.get_cuda_driver_version())}


def scene_semantics(blend, root):
    """Conservative authored-data comparison for the two rebuilt demo files.

    Include ID properties, RNA scalars/arrays, nested collections and all referenced
    datablocks. Exclude only editor state and Blender's transient ID bookkeeping.
    Unsupported values fail closed instead of being stringified or omitted.
    """
    import bpy
    sys.path.insert(0, str(root))
    from tools.repo_addon_loader import enable_repo_addon
    enable_repo_addon(str(root), default_set=False, persistent=False)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    ignored = {'rna_type', 'name_full', 'users', 'is_updated', 'is_updated_data',
               'is_embedded_data', 'is_evaluated', 'original', 'session_uid',
               'is_runtime_data', 'tag', 'library', 'override_library', 'preview',
               'id_data', 'is_library_indirect', 'is_missing', 'is_editable', 'use_fake_user',
               'use_extra_user', 'asset_data'}
    def encode(value, ancestry=(), expand_id=False):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, bpy.types.ID) and not expand_id:
            return {'id_type': value.bl_rna.identifier, 'name': value.name,
                    'library': value.library.filepath if value.library else None}
        if hasattr(value, 'bl_rna'):
            pointer = value.as_pointer()
            if pointer in ancestry:
                raise RuntimeError('recursive non-ID authored data cannot be compared')
            result = {}
            for prop in value.bl_rna.properties:
                name = prop.identifier
                if name in ignored:
                    continue
                child = getattr(value, name)
                if prop.type == 'COLLECTION':
                    result[name] = [encode(item, (*ancestry, pointer)) for item in child]
                elif getattr(prop, 'is_array', False):
                    result[name] = list(child)
                else:
                    result[name] = encode(child, (*ancestry, pointer))
            if hasattr(value, 'keys'):
                result['id_properties'] = {key: encode(value[key], (*ancestry, pointer))
                                           for key in value.keys() if key != '_RNA_UI'}
            return result
        if hasattr(value, 'to_dict'):
            return {key: encode(item, ancestry) for key, item in value.to_dict().items()}
        if hasattr(value, 'to_list'):
            return value.to_list()
        raise RuntimeError(f'unsupported authored value: {type(value).__name__}')
    ignored_collections = {'screens', 'window_managers', 'workspaces'}
    values = {prop.identifier: [encode(item, expand_id=True)
                               for item in getattr(bpy.data, prop.identifier)]
              for prop in bpy.data.bl_rna.properties
              if prop.type == 'COLLECTION' and prop.identifier not in ignored_collections}
    return q.fingerprint(values)


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    if args[0] == '--scene':
        _, root, blend, report = args
        q.atomic_json(report, {'semantic_fingerprint': scene_semantics(blend, root)})
    else:
        request, report = args
        q.atomic_json(report, observe(json.loads(Path(request).read_text())))


if __name__ == '__main__':
    main()
