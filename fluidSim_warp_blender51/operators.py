"""Ordered public registration facade for FluidSim operators."""

from __future__ import annotations

import bpy

from .operator_domains import common as _operator_runtime
from .operator_domains.common import _FLUIDSIM_OT_session_base, _bridge
from .operator_domains.session import (
    FLUIDSIM_OT_initialize,
    FLUIDSIM_OT_release,
    FLUIDSIM_OT_clear_error,
    FLUIDSIM_OT_apply_capacity_suggestion,
    FLUIDSIM_OT_apply_safe_settings,
    FLUIDSIM_OT_reset,
    FLUIDSIM_OT_apply_pending_reinitialize,
    FLUIDSIM_OT_apply_pending_reset,
    FLUIDSIM_OT_revert_pending,
    FLUIDSIM_OT_step,
    FLUIDSIM_OT_toggle_live,
    FLUIDSIM_OT_preview_start,
    FLUIDSIM_OT_preview_pause,
    FLUIDSIM_OT_cache_preview_start,
    FLUIDSIM_OT_apply_changes,
    FLUIDSIM_OT_restart_simulation,
    FLUIDSIM_OT_cache_playback_start,
    FLUIDSIM_OT_cache_playback_stop,
)
from .operator_domains.source_roles import (
    FLUIDSIM_OT_setup_scene,
    FLUIDSIM_OT_tag_object,
    FLUIDSIM_OT_tag_liquid_emitter,
    FLUIDSIM_OT_tag_liquid_sink,
    FLUIDSIM_OT_tag_collider,
    FLUIDSIM_OT_tag_force_region,
    FLUIDSIM_OT_tag_field_source,
    FLUIDSIM_OT_clear_object_role,
    FLUIDSIM_OT_inject_particles,
)
from .operator_domains.diagnostics import (
    FLUIDSIM_OT_run_validation,
    FLUIDSIM_OT_run_benchmark,
    FLUIDSIM_OT_write_report,
)
from .operator_domains.export import (
    FLUIDSIM_OT_export_frame,
    FLUIDSIM_OT_export_cache_sequence,
)
from .operator_domains.presets import (
    FLUIDSIM_OT_apply_product_preset,
    FLUIDSIM_OT_save_product_preset,
)
from .operator_domains.cache import (
    FLUIDSIM_OT_bake_start,
    FLUIDSIM_OT_bake_resume,
    FLUIDSIM_OT_bake_cancel,
    FLUIDSIM_OT_cache_integrity_scan,
    FLUIDSIM_OT_cache_remove_orphans,
    FLUIDSIM_OT_cache_free,
    FLUIDSIM_OT_cache_move,
    FLUIDSIM_OT_cache_rename,
    FLUIDSIM_OT_cache_open_folder,
)
from .operator_domains.rendering import (
    FLUIDSIM_OT_use_active_as_domain,
    FLUIDSIM_OT_clear_domain_assignment,
    FLUIDSIM_OT_remove_selected_output,
    FLUIDSIM_OT_install_render_nodes,
    FLUIDSIM_OT_copy_bake_command,
)


CLASSES = (
    FLUIDSIM_OT_initialize,
    FLUIDSIM_OT_release,
    FLUIDSIM_OT_clear_error,
    FLUIDSIM_OT_apply_capacity_suggestion,
    FLUIDSIM_OT_apply_safe_settings,
    FLUIDSIM_OT_reset,
    FLUIDSIM_OT_apply_pending_reinitialize,
    FLUIDSIM_OT_apply_pending_reset,
    FLUIDSIM_OT_revert_pending,
    FLUIDSIM_OT_step,
    FLUIDSIM_OT_toggle_live,
    FLUIDSIM_OT_preview_start,
    FLUIDSIM_OT_preview_pause,
    FLUIDSIM_OT_cache_preview_start,
    FLUIDSIM_OT_apply_changes,
    FLUIDSIM_OT_restart_simulation,
    FLUIDSIM_OT_cache_playback_start,
    FLUIDSIM_OT_cache_playback_stop,
    FLUIDSIM_OT_setup_scene,
    FLUIDSIM_OT_apply_product_preset,
    FLUIDSIM_OT_save_product_preset,
    FLUIDSIM_OT_bake_start,
    FLUIDSIM_OT_bake_resume,
    FLUIDSIM_OT_bake_cancel,
    FLUIDSIM_OT_cache_integrity_scan,
    FLUIDSIM_OT_cache_remove_orphans,
    FLUIDSIM_OT_cache_free,
    FLUIDSIM_OT_cache_move,
    FLUIDSIM_OT_cache_rename,
    FLUIDSIM_OT_cache_open_folder,
    FLUIDSIM_OT_copy_bake_command,
    FLUIDSIM_OT_use_active_as_domain,
    FLUIDSIM_OT_clear_domain_assignment,
    FLUIDSIM_OT_remove_selected_output,
    FLUIDSIM_OT_install_render_nodes,
    FLUIDSIM_OT_tag_object,
    FLUIDSIM_OT_tag_liquid_emitter,
    FLUIDSIM_OT_tag_liquid_sink,
    FLUIDSIM_OT_tag_collider,
    FLUIDSIM_OT_tag_force_region,
    FLUIDSIM_OT_tag_field_source,
    FLUIDSIM_OT_clear_object_role,
    FLUIDSIM_OT_inject_particles,
    FLUIDSIM_OT_run_validation,
    FLUIDSIM_OT_run_benchmark,
    FLUIDSIM_OT_write_report,
    FLUIDSIM_OT_export_frame,
    FLUIDSIM_OT_export_cache_sequence,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
