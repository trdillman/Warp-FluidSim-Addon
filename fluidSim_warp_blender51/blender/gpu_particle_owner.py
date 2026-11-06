"""Lifecycle owner for a live GPU particle transport and renderer."""

_restart_required = False


class ParticleOwner:
    def __init__(self, transport_factory):
        self.transport_factory = transport_factory
        self.transport = None
        self._capacity = None
        self._step = None

    def publish(self, particles, step, count):
        if _restart_required:
            raise RuntimeError('GPU transport retirement failed; restart Blender')
        step = int(step)
        if step < 1:
            raise ValueError('display step must be positive')
        capacity = (int(particles.capacity) + step - 1) // step
        if capacity <= 0:
            self.close()
            return
        if self.transport is None or (capacity, step) != (self._capacity, self._step):
            self.close()
            self.transport = self.transport_factory(capacity)
            self._capacity, self._step = capacity, step
        try:
            return self.transport.publish(particles.position, int(count), step=step)
        except Exception:
            self.close()
            raise

    def close(self):
        global _restart_required
        transport = self.transport
        if transport is not None:
            try:
                result = transport.close()
            except Exception:
                _restart_required = True
                raise
            if result != 'completed':
                _restart_required = True
                raise RuntimeError('GPU transport retirement failed; restart Blender')
        self.transport = None
        self._capacity = self._step = None


_owner = None
_renderer = None
_handle = None
_hidden = None
_error = None
_retire_pending = False
_owner_window = None
_owner_scene = None
_pending_frame = None
_retired = []
_quarantined = []
_owner_solver = None
_saving = False


def _particles_visible(scene):
    from . import liquid_display, output_publication as output
    if liquid_display._HELD:
        return False
    settings = output._domain_settings(scene)
    surface, particles = liquid_display.viewport_families(
        scene, getattr(settings, 'surface_enabled', False))
    if surface:
        obj = output.output_object(scene, output.SURFACE_NAME)
        particles = obj is None or not len(getattr(getattr(obj, 'data', None), 'vertices', (1,)))
    return particles


def _preview_particles_visible(scene):
    # Older scenes without the representation policy retain their authored flag.
    if getattr(scene.fluidsim, 'liquid_display_mode', None) is None:
        return getattr(scene.fluidsim, 'show_particles', False)
    return _particles_visible(scene)


def _restore_output():
    global _hidden
    if _hidden is not None:
        obj, view_layer, was_hidden = _hidden
        try:
            if _owner_scene is not None and getattr(_owner_scene.fluidsim, 'liquid_display_mode', None) is not None:
                from . import liquid_display
                if liquid_display._HELD:
                    liquid_display.apply_viewport(_owner_scene)
                else:
                    obj.hide_set(not _particles_visible(_owner_scene), view_layer=view_layer)
            else:
                obj.hide_set(was_hidden, view_layer=view_layer)
        except (ReferenceError, RuntimeError):
            pass
        _hidden = None


def reset_runtime():
    """Detach immediately and retain transport/solver until timer retirement."""
    global _owner, _renderer, _pending_frame, _owner_scene, _owner_solver, _retire_pending
    _restore_output()
    _pending_frame = None
    if _owner is not None or _renderer is not None:
        _retired.append((_owner, _renderer, _owner_solver))
        _owner = _renderer = _owner_solver = None
        _retire_pending = True
        import bpy
        if not bpy.app.timers.is_registered(_retire_from_timer):
            bpy.app.timers.register(_retire_from_timer, first_interval=0.1, persistent=True)
    _owner_scene = None


def _retire_from_timer():
    """Retire retained owners in the qualified main-window timer context."""
    global _retire_pending, _owner_window, _error, _restart_required
    while _retired:
        owner, renderer, solver = _retired.pop(0)
        try:
            if owner is not None:
                owner.close()
            if renderer is not None:
                renderer.close()
        except Exception as exc:
            _restart_required = True
            _quarantined.append((owner, renderer, solver))
            _error = str(exc)
    _retire_pending = False
    _owner_window = None
    return None


def disable_preview(context=None, scene=None):
    """Restore output now; retain GPU resources until timer retirement."""
    if scene is not None and _owner_scene is not None and scene != _owner_scene:
        return
    reset_runtime()
    manager = getattr(context, 'window_manager', None)
    for window in getattr(manager, 'windows', ()):
        for area in getattr(getattr(window, 'screen', None), 'areas', ()):
            if getattr(area, 'type', None) == 'VIEW_3D':
                area.tag_redraw()


def _publish_now(scene, session, count):
    global _owner, _error, _owner_scene, _owner_solver
    fs = scene.fluidsim
    if _retire_pending:
        return
    # Non-flat modes need a GPU attribute provider; keep the authoritative
    # datablock presentation until that provider exists.
    if (not getattr(fs, 'gpu_particle_preview', False)
            or not _preview_particles_visible(scene)
            or getattr(fs, 'particle_attribute_mode', 'NONE') != 'NONE'
            or getattr(fs, 'particle_material', None) is not None):
        reset_runtime()
        return
    try:
        if _restart_required:
            raise RuntimeError('GPU transport retirement failed; restart Blender')
        if _owner is None:
            from prk.gpudraw import ExternalTexturePoints
            if not callable(ExternalTexturePoints):
                raise RuntimeError("adapted PRK renderer unavailable")
            from .sdk_interop import SDKTransport
            _owner = ParticleOwner(SDKTransport)
            _owner_scene = scene
            _owner_solver = session.solver
        fraction = max(0.01, float(fs.particle_display_fraction))
        step = max(1, round(1.0 / fraction))
        _owner.publish(session.solver.particles, step, count)
        _error = None
    except Exception as exc:
        reset_runtime()
        _error = str(exc)
        fs.status = f'{fs.status} - GPU preview unavailable: {_error}'



def _publish_from_timer():
    global _pending_frame
    pending, _pending_frame = _pending_frame, None
    if pending is None or _retire_pending:
        return None
    scene, session, solver, count = pending
    if session.solver is solver and session.owns_scene(scene):
        _publish_now(scene, session, count)
    return None


def publish_scene(scene, session, count):
    """Coalesce live positions; publish from the window timer GPU context."""
    global _pending_frame
    if _retire_pending:
        return
    if not getattr(scene.fluidsim, "gpu_particle_preview", False):
        if _owner is not None or _renderer is not None:
            reset_runtime()
        return
    _pending_frame = (scene, session, session.solver, int(count))
    import bpy
    if not bpy.app.timers.is_registered(_publish_from_timer):
        bpy.app.timers.register(_publish_from_timer, first_interval=0.1)


def _draw():
    global _renderer, _hidden, _error, _owner_window
    import bpy
    if _retire_pending or _saving:
        return
    if _owner is None or _owner.transport is None:
        return
    import gpu
    from . import scene_bridge, output_publication
    from .gpu_particle_renderer import PointRenderer, prepare_packet
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    if _hidden is not None and _hidden[1] != view_layer:
        return
    session = scene_bridge.session()
    fs = scene.fluidsim
    if (not session.owns_scene(scene) or not getattr(fs, 'gpu_particle_preview', False)
            or not _preview_particles_visible(scene)
            or getattr(fs, 'particle_attribute_mode', 'NONE') != 'NONE'
            or getattr(fs, 'particle_material', None) is not None):
        _restore_output()
        return
    try:
        if _owner_window is not None and _owner_window != bpy.context.window:
            return
        frame = _owner.transport.frame()
        if frame is None:
            return
        _owner_window = bpy.context.window
        obj = output_publication.output_object(scene, output_publication.PARTICLES_NAME)
        if _hidden is not None and _hidden[0] != obj:
            _restore_output()
        if obj is None or (_hidden is None and obj.hide_get(view_layer=view_layer)):
            return
        color = tuple(obj.color)
        radius = float(session.solver.settings.voxel_size) * 0.25 * float(scene.fluidsim.particle_display_scale)
        packet = prepare_packet(frame, width=frame.width, height=frame.height,
                                radius=radius, color=color)
        if _renderer is None:
            _renderer = PointRenderer()
        _renderer.draw(packet)
        if _hidden is None:
            _hidden = (obj, view_layer, obj.hide_get(view_layer=view_layer))
            obj.hide_set(True, view_layer=view_layer)
    except Exception as exc:
        disable_preview(bpy.context, scene)
        _error = str(exc)
        scene.fluidsim.status = f'GPU preview unavailable: {_error}'




def _on_gpu_preview_save_pre(_filepath):
    global _saving
    _saving = True
    _restore_output()
    # Blender versions without save_post_fail still resume after a failed save.
    try:
        import bpy
        timers = bpy.app.timers
        if not timers.is_registered(_save_completion_fallback):
            timers.register(_save_completion_fallback, first_interval=0.1)
    except (AttributeError, ModuleNotFoundError):
        pass


def _save_completion_fallback():
    if _saving:
        _on_gpu_preview_save_post(None)
    return None


def _on_gpu_preview_save_post(_filepath):
    global _saving
    _saving = False
    try:
        import bpy
        manager = getattr(bpy.context, 'window_manager', None)
        for window in getattr(manager, 'windows', ()):
            for area in getattr(getattr(window, 'screen', None), 'areas', ()):
                if getattr(area, 'type', None) == 'VIEW_3D':
                    area.tag_redraw()
    except (AttributeError, ReferenceError, ModuleNotFoundError):
        pass


def _on_gpu_preview_save_post_fail(filepath):
    _on_gpu_preview_save_post(filepath)


def _on_gpu_preview_load_pre(_filepath):
    """Retire before Blender destroys the currently qualified window context."""
    global _saving
    _saving = False
    reset_runtime()
    if _retired:
        _retire_from_timer()
    import bpy
    timers = bpy.app.timers
    if timers.is_registered(_retire_from_timer):
        timers.unregister(_retire_from_timer)


def _handler_entries(handlers):
    return ((getattr(handlers, 'load_pre', None), _on_gpu_preview_load_pre),
            (getattr(handlers, 'save_pre', None), _on_gpu_preview_save_pre),
            (getattr(handlers, 'save_post', None), _on_gpu_preview_save_post),
            (getattr(handlers, 'save_post_fail', None), _on_gpu_preview_save_post_fail))


def register():
    global _handle
    import bpy
    handlers = bpy.app.handlers
    installed = []
    try:
        for target, callback in _handler_entries(handlers):
            if target is None:
                continue
            handlers.persistent(callback)
            if callback not in target:
                target.append(callback)
                installed.append((target, callback))
        if _handle is None:
            _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    except Exception:
        for target, callback in reversed(installed):
            target.remove(callback)
        raise


def unregister():
    global _handle, _saving
    _saving = False
    reset_runtime()
    import bpy
    for target, callback in _handler_entries(bpy.app.handlers):
        if target is not None:
            while callback in target:
                target.remove(callback)
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
