"""Write generated demo metadata without rewriting identical source inputs."""

import hashlib
import json
from pathlib import Path


def demo_object_identity(object_name, mesh_name):
    """Give canonical demo inputs stable identities before solver consumption."""
    basis = json.dumps(["canonical-demo-object", object_name, mesh_name], separators=(",", ":"))
    return "object:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()


def assign_object_identity(obj):
    obj["fluidsim_persistent_id"] = demo_object_identity(
        obj.name_full, getattr(obj.data, "name_full", ""))
    obj["fluidsim_persistent_owner"] = obj.name_full


def assign_input_identities(scene):
    """Assign authored input identities before any solver consumes them."""
    for obj in scene.objects:
        settings = getattr(obj, "fluidsim", None)
        if settings is not None and (
            settings.role in {"SOURCE", "COLLIDER", "SINK", "FIELD", "FORCE"}
            or getattr(settings, "auto_inject_seed", False)
        ):
            assign_object_identity(obj)


def prepare_scene_for_save(scene, name):
    """Keep the saved demo independent of checkout and observed step timing."""
    scene.render.filepath = f"//{name}_preview.png"
    scene.fluidsim.stat_step_ms = 0.0


def write_metadata(path, metadata):
    path = Path(path)
    content = json.dumps(metadata, indent=2) + "\n"
    try:
        if path.read_text(encoding="utf-8") == content:
            return
    except FileNotFoundError:
        pass
    path.write_text(content, encoding="utf-8", newline="\n")
