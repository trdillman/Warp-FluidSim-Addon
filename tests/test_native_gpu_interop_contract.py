import ctypes
import inspect
import sys
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import native_interop


def _descriptor(**overrides):
    descriptor = {
        "api_version": 1,
        "generation": 1,
        "backend": "VULKAN",
        "memory_handle_type": "OPAQUE_WIN32",
        "semaphore_handle_type": "OPAQUE_WIN32",
        "device_uuid": bytes.fromhex("5d09f77ba936a6d9054d0324aef67adb"),
        "device_luid": bytes(8),
        "device_node_mask": 1,
        "memory_handle": 1,
        "memory_size": 4096,
        "draw_vertex_count_offset": 512,
        "pointcloud_bounds_offset": 528,
        "release_semaphore_handle": 2,
        "completion_semaphore_handle": 3,
        "memory_exportable": True,
        "memory_importable": True,
        "memory_dedicated_only": False,
        "semaphore_exportable": True,
        "semaphore_importable": True,
        "max_timeline_value_difference": 2**31 - 1,
    }
    descriptor.update(overrides)
    return descriptor


def test_cuda_driver_abi_layout_matches_cuda_12_9_x64():
    assert ctypes.sizeof(native_interop._ExternalMemoryHandleDesc) == 104
    assert ctypes.sizeof(native_interop._ExternalMemoryBufferDesc) == 88
    assert ctypes.sizeof(native_interop._ExternalSemaphoreHandleDesc) == 96
    assert ctypes.sizeof(native_interop._ExternalSemaphoreSignalParams) == 144
    assert ctypes.sizeof(native_interop._ExternalSemaphoreWaitParams) == 144


def test_cuda_uuid_parser_matches_blender_bytes():
    value = "GPU-5d09f77b-a936-a6d9-054d-0324aef67adb"
    assert native_interop._uuid_bytes(value) == bytes.fromhex(
        "5d09f77ba936a6d9054d0324aef67adb")


def test_native_publish_uses_device_kernel_not_numpy_materialization():
    source = inspect.getsource(native_interop.CudaExternalBuffer.publish)
    assert ".launch(" in source
    assert ".numpy(" not in source


def test_height_materialization_samples_one_scalar_without_position_readback():
    launches = []

    class Scratch:
        def __getitem__(self, item):
            assert item == slice(None, 2)
            return self

        def numpy(self):
            return [1.0, 2.0]

    scratch = Scratch()
    wp = SimpleNamespace(
        empty=lambda *_args, **_kwargs: scratch,
        launch=lambda kernel, dim, inputs, **_kwargs: launches.append(
            (kernel, dim, inputs)),
    )
    presenter = native_interop.NativeParticlePresenter.__new__(
        native_interop.NativeParticlePresenter)
    presenter.capacity = 8
    presenter.bridge = SimpleNamespace(wp=wp, device=object(), stream=object())
    presenter._display_values = None
    presenter._display_kernels = {"HEIGHT": "height"}
    presenter.materialization_stats = {
        "reads": 0,
        "bytes": 0,
        "reason": "",
    }
    particles = SimpleNamespace(position="positions")

    values = presenter.materialize_display_values(
        particles, "HEIGHT", count=2, step=3, frame_id=7)
    cached = presenter.materialize_display_values(
        particles, "HEIGHT", count=2, step=3, frame_id=7)

    assert values == [1.0, 2.0]
    assert cached == values
    assert launches == [("height", 2, ["positions", scratch, 3])]
    assert presenter.materialization_stats == {
        "reads": 1,
        "bytes": 8,
        "reason": "particle_attribute:HEIGHT",
    }


def test_publish_writes_active_range_and_clears_only_newly_inactive_tail():
    launches = []

    class Driver:
        def push(self, _context):
            pass

        def pop(self):
            pass

        def wait(self, *_args):
            pass

        def signal(self, *_args):
            pass

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(
        launch=lambda kernel, dim, inputs, **_kwargs: launches.append(
            (kernel, dim, inputs)),
    )
    bridge.device = object()
    bridge.stream = SimpleNamespace(cuda_stream=7)
    bridge.driver = Driver()
    bridge.context = object()
    bridge.positions = object()
    bridge.publish_kernel = "publish"
    bridge.clear_kernel = "clear"
    bridge.draw_vertex_count_kernel = "draw-count"
    bridge.draw_vertex_count = "draw-command"
    bridge.bounds_init_kernel = "bounds-init"
    bridge.bounds_reduce_kernel = "bounds-reduce"
    bridge.bounds_encode_kernel = "bounds-encode"
    bridge.bounds_min_max = "bounds-min-max"
    bridge.pointcloud_bounds = "pointcloud-bounds"
    bridge.release = object()
    bridge.completion = object()
    bridge.capacity = 100
    bridge.visible_count = None
    bridge.context_pushed = False
    bridge.stats = {"published_frames": 0, "published_bytes": 0}

    bridge.publish(object(), count=20, value=1, radius=0.1, step=4)
    bridge.publish(object(), count=30, value=2, radius=0.1)
    bridge.publish(object(), count=10, value=3, radius=0.1)

    assert [(kernel, dim) for kernel, dim, _inputs in launches] == [
        ("publish", 20),
        ("clear", 80),
        ("draw-count", 1),
        ("bounds-init", 1),
        ("bounds-reduce", 20),
        ("bounds-encode", 1),
        ("publish", 30),
        ("draw-count", 1),
        ("bounds-init", 1),
        ("bounds-reduce", 30),
        ("bounds-encode", 1),
        ("publish", 10),
        ("clear", 20),
        ("draw-count", 1),
        ("bounds-init", 1),
        ("bounds-reduce", 10),
        ("bounds-encode", 1),
    ]
    assert bridge.visible_count == 10
    assert launches[0][2][-1] == 4
    assert bridge.stats["published_bytes"] == (20 + 80 + 30 + 10 + 20) * 16


def test_device_count_publish_masks_capacity_without_host_count_readback():
    launches = []

    class Driver:
        push = lambda *_args: None
        pop = lambda *_args: None
        wait = lambda *_args: None
        signal = lambda *_args: None

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(
        launch=lambda kernel, dim, inputs, **_kwargs: launches.append(
            (kernel, dim, inputs)))
    bridge.device = object()
    bridge.stream = SimpleNamespace(cuda_stream=7)
    bridge.driver = Driver()
    bridge.context = object()
    bridge.positions = "target"
    bridge.counted_publish_kernel = "counted"
    bridge.draw_vertex_count = "draw-count"
    bridge.device_count_draw_kernel = "device-count-draw"
    bridge.bounds_init_kernel = "bounds-init"
    bridge.bounds_reduce_kernel = "bounds-reduce"
    bridge.bounds_encode_kernel = "bounds-encode"
    bridge.bounds_min_max = "bounds-min-max"
    bridge.pointcloud_bounds = "pointcloud-bounds"
    bridge.release = object()
    bridge.completion = object()
    bridge.context_pushed = False
    bridge.stats = {"published_frames": 0, "published_bytes": 0}

    bridge.publish_device_count(
        "source", "device-count", output_count=25, value=3, radius=0.1, step=4)

    assert launches == [
        ("counted", 25, ["source", "device-count", "target", 0.1, 4]),
        ("device-count-draw", 1, ["device-count", "draw-count"]),
        ("bounds-init", 1, ["bounds-min-max"]),
        ("bounds-reduce", 25, ["target", "draw-count", "bounds-min-max"]),
        ("bounds-encode", 1, ["bounds-min-max",
         "draw-count", "pointcloud-bounds", 0.1]),
    ]
    assert bridge.stats == {"published_frames": 1, "published_bytes": 400}


def test_device_count_no_repack_keeps_sync_and_metadata_publication():
    launches = []
    calls = []

    class Driver:
        push = lambda *_args: calls.append("push")
        pop = lambda *_args: calls.append("pop")
        wait = lambda *_args: calls.append("wait")
        signal = lambda *_args: calls.append("signal")

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(
        launch=lambda kernel, dim, inputs, **_kwargs: launches.append(
            (kernel, dim, inputs)))
    bridge.device = object()
    bridge.stream = SimpleNamespace(cuda_stream=7)
    bridge.driver = Driver()
    bridge.context = object()
    bridge.positions = "target"
    bridge.counted_publish_kernel = "counted"
    bridge.draw_vertex_count = "draw-count"
    bridge.device_count_draw_kernel = "device-count-draw"
    bridge.bounds_init_kernel = "bounds-init"
    bridge.bounds_reduce_kernel = "bounds-reduce"
    bridge.bounds_encode_kernel = "bounds-encode"
    bridge.bounds_min_max = "bounds-min-max"
    bridge.pointcloud_bounds = "pointcloud-bounds"
    bridge.release = object()
    bridge.completion = object()
    bridge.context_pushed = False
    bridge.stats = {"published_frames": 0, "published_bytes": 0}

    bridge.publish_device_count(
        "source",
        "device-count",
        output_count=25,
        value=3,
        radius=0.1,
        step=4,
        repack=False,
    )

    assert calls == ["push", "wait", "signal", "pop"]
    assert [kernel for kernel, _dim, _inputs in launches] == [
        "device-count-draw",
        "bounds-init",
        "bounds-reduce",
        "bounds-encode",
    ]
    assert bridge.stats == {"published_frames": 1, "published_bytes": 0}


def test_publication_stage_profiling_is_explicit_and_reports_cuda_stages():
    sequence = []

    class Event:
        next_id = 0

        def __init__(self, _device, enable_timing=False):
            assert enable_timing is True
            self.event_id = Event.next_id
            Event.next_id += 1

    class Stream:
        cuda_stream = 7

        @staticmethod
        def record_event(event):
            sequence.append(f"event-{event.event_id}")

    class Driver:
        push = lambda *_args: None
        pop = lambda *_args: None
        wait = lambda *_args: sequence.append("wait")
        signal = lambda *_args: sequence.append("signal")

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.device = object()
    bridge.stream = Stream()
    bridge.wp = SimpleNamespace(
        Event=Event,
        launch=lambda kernel, **_kwargs: sequence.append(kernel),
        get_event_elapsed_time=lambda start, end, synchronize=True: (
            assert_timing(start, end, synchronize)),
    )
    bridge.driver = Driver()
    bridge.context = object()
    bridge.positions = "target"
    bridge.counted_publish_kernel = "counted"
    bridge.draw_vertex_count = "draw-count"
    bridge.device_count_draw_kernel = "device-count-draw"
    bridge.bounds_init_kernel = "bounds-init"
    bridge.bounds_reduce_kernel = "bounds-reduce"
    bridge.bounds_encode_kernel = "bounds-encode"
    bridge.bounds_min_max = "bounds-min-max"
    bridge.pointcloud_bounds = "pointcloud-bounds"
    bridge.release = object()
    bridge.completion = object()
    bridge.context_pushed = False
    bridge.stats = {"published_frames": 0, "published_bytes": 0}
    bridge._publication_profile_events = None
    bridge.publication_profile = None

    def assert_timing(start, end, synchronize):
        assert synchronize is True
        return float(end.event_id - start.event_id)

    assert bridge.publication_profile is None
    bridge.publication_profiling_enable()
    bridge.publish_device_count(
        "source", "device-count", output_count=25, value=3, radius=0.1)
    bridge._publication_profile_collect()

    assert sequence == [
        "event-0",
        "wait",
        "event-1",
        "counted",
        "event-2",
        "device-count-draw",
        "bounds-init",
        "bounds-reduce",
        "bounds-encode",
        "event-3",
        "signal",
        "event-4",
    ]
    assert bridge.publication_profile == {
        "external_wait_ms": 1.0,
        "position_pack_ms": 1.0,
        "metadata_ms": 1.0,
        "cuda_publication_ms": 4.0,
    }


def test_profile_collection_runs_after_lease_transaction_is_submitted():
    calls = []
    slot = native_interop._NativeParticleSlot.__new__(
        native_interop._NativeParticleSlot)
    slot.value = 0
    slot.active_count = 0
    slot.lease = SimpleNamespace(
        release=lambda value: calls.append(("release", value)),
        acquire=lambda value: calls.append(("acquire", value)),
        submit=lambda value: calls.append(("submit", value)),
    )
    slot.vbo = SimpleNamespace(
        _pointcloud_position_count_set=lambda count: calls.append(("count", count)))
    slot.bridge = SimpleNamespace(
        _publication_profile_events=(object(),),
        stats={"published_bytes": 0},
        publish_device_count=lambda *_args: calls.append("publish"),
        _publication_profile_collect=lambda: calls.append("collect"),
    )

    slot.publish_device_count(
        "positions", "active-count", output_count=25, radius=0.1)

    assert calls == [
        ("release", 1),
        "publish",
        ("acquire", 1),
        ("count", 25),
        ("submit", 1),
        "collect",
    ]


def test_zero_particle_publish_clears_previous_visible_tail():
    presenter = native_interop.NativeParticlePresenter.__new__(
        native_interop.NativeParticlePresenter)
    presenter.active_count = 12
    calls = []
    slot = SimpleNamespace(
        vbo=SimpleNamespace(
            _pointcloud_position_count_set=lambda _count: None),
        bridge=SimpleNamespace(positions=[None] * 4, stream=object()),
        bind_pointcloud=lambda _pointcloud: calls.append("bind"),
        publish=lambda _positions, count, _radius, _step: calls.append(
            ("publish", count)),
    )
    presenter._slots = [slot]
    presenter._slot_index = -1
    presenter.pointcloud = object()
    presenter.radius = 0.2
    presenter.bridge = SimpleNamespace(
        positions=[None] * 4,
        stats={"published_frames": 0, "published_bytes": 0},
    )
    assert presenter.publish(object(), 0) is True
    assert presenter.active_count == 0
    assert calls == ["bind", ("publish", 0)]


def test_presenter_alternates_two_slots_with_independent_timeline_values(monkeypatch):
    calls = []

    class VertBuf:
        next_id = 0

        def __init__(self, **_kwargs):
            self.slot = VertBuf.next_id
            VertBuf.next_id += 1

        def external_lease_acquire(self):
            owner = self

            class Lease:
                generation = owner.slot + 1

                def descriptor_get(self):
                    return {"slot": owner.slot, "generation": self.generation}

                def release(self, value):
                    calls.append(("release", owner.slot, value))

                def acquire(self, value):
                    calls.append(("acquire", owner.slot, value))

                def submit(self, value):
                    calls.append(("submit", owner.slot, value))

                def close(self):
                    calls.append(("lease_close", owner.slot))

            return Lease()

        def _pointcloud_position_bind(self, _pointcloud):
            calls.append(("bind", self.slot))

        def _pointcloud_position_count_set(self, count):
            calls.append(("count", self.slot, count))

        def _pointcloud_position_unbind(self):
            calls.append(("unbind", self.slot))

    gpu = SimpleNamespace(
        types=SimpleNamespace(
            GPUVertFormat=lambda: SimpleNamespace(
                attr_add=lambda **_kwargs: None),
            GPUVertBuf=VertBuf,
        ))

    class Bridge:
        def __init__(self, descriptor, _capacity):
            self.slot = descriptor["slot"]
            self.device = object()
            self.stream = object()
            self.wp = object()
            self.positions = [None] * 4
            self.stats = {
                "imports": 3,
                "maps": 1,
                "published_frames": 0,
                "published_bytes": 0,
                "bulk_host_position_bytes": 0,
            }

        def publish(self, _positions, count, value, _radius, step):
            calls.append(("publish", self.slot, count, value))
            calls.append(("step", self.slot, step))
            self.stats["published_frames"] += 1
            self.stats["published_bytes"] += count * 16

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setattr(native_interop, "CudaExternalBuffer", Bridge)

    presenter = native_interop.NativeParticlePresenter(
        capacity=4, pointcloud=object(), radius=0.2)
    presenter.publish(object(), 3)
    presenter.publish(object(), 2)
    presenter.publish(object(), 1)

    assert len(presenter._slots) == 2
    assert ("publish", 0, 3, 1) in calls
    assert ("publish", 1, 2, 1) in calls
    assert ("publish", 0, 1, 2) in calls
    assert presenter.bridge.stats["imports"] == 6
    assert presenter.bridge.stats["maps"] == 2


def test_capacity_replacement_publishes_new_generation_before_retiring_old(monkeypatch):
    calls = []

    class Presenter:
        def __init__(self, capacity, pointcloud, radius):
            calls.append(("create", capacity, pointcloud, radius))

        def publish(self, positions, count, step):
            calls.append(("publish", positions, count, step))

        def close(self):
            calls.append("close-old")

    old = SimpleNamespace(close=lambda: calls.append("close-old"))
    monkeypatch.setattr(native_interop, "NativeParticlePresenter", Presenter)

    replacement = native_interop.replace_native_particle_presenter(
        old,
        capacity=8,
        pointcloud="pointcloud",
        radius=0.25,
        positions="positions",
        count=5,
    )

    assert isinstance(replacement, Presenter)
    assert calls == [
        ("create", 8, "pointcloud", 0.25),
        ("publish", "positions", 5, 1),
        "close-old",
    ]


def test_failed_capacity_replacement_keeps_previous_generation(monkeypatch):
    calls = []

    class Presenter:
        def __init__(self, *_args):
            pass

        def publish(self, *_args):
            raise RuntimeError("publish failed")

        def close(self):
            calls.append("close-new")

    old = SimpleNamespace(
        close=lambda: calls.append("close-old"),
        _restore_active_binding=lambda: calls.append("restore-old"),
    )
    monkeypatch.setattr(native_interop, "NativeParticlePresenter", Presenter)

    with pytest.raises(RuntimeError, match="publish failed"):
        native_interop.replace_native_particle_presenter(
            old,
            capacity=8,
            pointcloud=object(),
            radius=0.25,
            positions=object(),
            count=5,
        )

    assert calls == ["close-new", "restore-old"]


def test_capacity_replacement_keeps_new_generation_when_old_retirement_fails(monkeypatch):
    calls = []

    class Presenter:
        def __init__(self, *_args):
            pass

        def publish(self, *_args):
            calls.append("publish-new")

        def close(self):
            calls.append("close-new")

    def close_old():
        calls.append("close-old")
        raise RuntimeError("old close failed")

    monkeypatch.setattr(native_interop, "NativeParticlePresenter", Presenter)

    replacement = native_interop.replace_native_particle_presenter(
        SimpleNamespace(close=close_old),
        capacity=8,
        pointcloud=object(),
        radius=0.25,
        positions=object(),
        count=5,
    )

    assert isinstance(replacement, Presenter)
    assert calls == ["publish-new", "close-old"]


def test_presenter_constructor_closes_slot_when_initial_bind_fails(monkeypatch):
    closed = []

    class Slot:
        next_id = 0

        def __init__(self, _capacity):
            self.slot = Slot.next_id
            Slot.next_id += 1

        def bind_pointcloud(self, _pointcloud):
            raise RuntimeError("bind failed")

        def close(self):
            closed.append(self.slot)

    gpu = SimpleNamespace(
        types=SimpleNamespace(
            GPUVertBuf=SimpleNamespace(_pointcloud_position_bind=object())))
    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setattr(native_interop, "_NativeParticleSlot", Slot)
    monkeypatch.setattr(
        native_interop, "_PresenterBridgeView", lambda _slots: object())

    with pytest.raises(RuntimeError, match="bind failed"):
        native_interop.NativeParticlePresenter(
            capacity=4, pointcloud=object(), radius=0.2)

    assert closed == [0, 1]


def test_slot_close_attempts_cuda_cleanup_when_unbind_fails_and_can_retry():
    calls = []
    fail_unbind = [True]

    class Vbo:
        def _pointcloud_position_unbind(self):
            calls.append("unbind")
            if fail_unbind[0]:
                fail_unbind[0] = False
                raise RuntimeError("unbind failed")

    slot = native_interop._NativeParticleSlot.__new__(
        native_interop._NativeParticleSlot)
    slot.vbo = Vbo()
    slot.bridge = SimpleNamespace(close=lambda: calls.append("bridge_close"))
    slot.lease = SimpleNamespace(close=lambda: calls.append("lease_close"))
    slot.pointcloud = object()

    with pytest.raises(RuntimeError, match="unbind failed"):
        slot.close()

    assert calls == ["unbind", "bridge_close", "lease_close"]
    assert slot.vbo is not None

    slot.close()
    assert slot.vbo is None
    assert slot.lease is None
    assert slot.pointcloud is None


def test_presenter_close_retains_only_failed_slots_for_retry():
    calls = []
    fail_first = [True]

    class Slot:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(self.name)
            if self.name == "first" and fail_first[0]:
                fail_first[0] = False
                raise RuntimeError("first close failed")

    first = Slot("first")
    second = Slot("second")
    presenter = native_interop.NativeParticlePresenter.__new__(
        native_interop.NativeParticlePresenter)
    presenter._slots = [first, second]
    presenter.pointcloud = object()

    with pytest.raises(RuntimeError, match="first close failed"):
        presenter.close()

    assert presenter._slots == [first]
    assert presenter.pointcloud is not None

    presenter.close()
    assert presenter._slots == []
    assert presenter.pointcloud is None
    assert calls == ["first", "second", "first"]


def test_failed_slot_publish_restores_previous_binding_and_slot_index():
    calls = []

    class Vbo:
        def __init__(self, name):
            self.name = name

        def _pointcloud_position_unbind(self):
            calls.append(("unbind", self.name))

    class Slot:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail
            self.vbo = Vbo(name)

        def bind_pointcloud(self, _pointcloud):
            calls.append(("bind", self.name))

        def publish(self, *_args):
            calls.append(("publish", self.name))
            if self.fail:
                raise RuntimeError("publish failed")

    previous = Slot("previous")
    failing = Slot("failing", fail=True)
    presenter = native_interop.NativeParticlePresenter.__new__(
        native_interop.NativeParticlePresenter)
    presenter._slots = [previous, failing]
    presenter._slot_index = 0
    presenter.pointcloud = object()
    presenter.radius = 0.2
    presenter.active_count = 3
    presenter.bridge = SimpleNamespace(stats={})

    with pytest.raises(RuntimeError, match="publish failed"):
        presenter.publish(object(), 2)

    assert presenter._slot_index == 0
    assert calls == [
        ("bind", "failing"),
        ("publish", "failing"),
        ("unbind", "failing"),
        ("bind", "previous"),
    ]


def test_failed_single_slot_publish_remains_unbound_for_cpu_fallback():
    calls = []

    class Slot:
        def __init__(self):
            self.vbo = SimpleNamespace(
                _pointcloud_position_unbind=lambda: calls.append("unbind"))

        def bind_pointcloud(self, _pointcloud):
            calls.append("bind")

        def publish(self, *_args):
            calls.append("publish")
            raise RuntimeError("publish failed")

    slot = Slot()
    presenter = native_interop.NativeParticlePresenter.__new__(
        native_interop.NativeParticlePresenter)
    presenter._slots = [slot]
    presenter._slot_index = 0
    presenter.pointcloud = object()
    presenter.radius = 0.2
    presenter.active_count = 3
    presenter.bridge = SimpleNamespace(stats={})

    with pytest.raises(RuntimeError, match="publish failed"):
        presenter.publish(object(), 2)

    assert presenter._slot_index == 0
    assert calls == ["bind", "publish", "unbind"]


def test_constructor_rollback_retries_first_cleanup_failure(monkeypatch):
    close_calls = []

    class Slot:
        def __init__(self, _capacity):
            self.vbo = SimpleNamespace(
                _pointcloud_position_count_set=lambda _count: None)

        def bind_pointcloud(self, _pointcloud):
            raise RuntimeError("bind failed")

        def close(self):
            close_calls.append("close")
            if len(close_calls) == 1:
                raise RuntimeError("close failed")

    gpu = SimpleNamespace(
        types=SimpleNamespace(
            GPUVertBuf=SimpleNamespace(_pointcloud_position_bind=object())))
    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setattr(native_interop, "_NativeParticleSlot", Slot)
    monkeypatch.setattr(
        native_interop, "_PresenterBridgeView", lambda _slots: object())

    with pytest.raises(RuntimeError, match="bind failed"):
        native_interop.NativeParticlePresenter(
            capacity=4, pointcloud=object(), radius=0.2)

    assert close_calls == ["close", "close", "close"]


def test_constructor_preserves_repeated_cleanup_failure_for_later_retry(monkeypatch):
    native_interop._DEFERRED_SLOT_CLEANUP.clear()
    allow_close = [False]

    class Slot:
        def __init__(self, _capacity):
            self.vbo = SimpleNamespace(
                _pointcloud_position_count_set=lambda _count: None)

        def bind_pointcloud(self, _pointcloud):
            raise RuntimeError("bind failed")

        def close(self):
            if not allow_close[0]:
                raise RuntimeError("close failed")

    gpu = SimpleNamespace(
        types=SimpleNamespace(
            GPUVertBuf=SimpleNamespace(_pointcloud_position_bind=object())))
    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setattr(native_interop, "_NativeParticleSlot", Slot)
    monkeypatch.setattr(
        native_interop, "_PresenterBridgeView", lambda _slots: object())

    with pytest.raises(RuntimeError, match="bind failed"):
        native_interop.NativeParticlePresenter(
            capacity=4, pointcloud=object(), radius=0.2)

    assert len(native_interop._DEFERRED_SLOT_CLEANUP) == 2
    allow_close[0] = True
    native_interop._retry_deferred_slot_cleanup()
    assert native_interop._DEFERRED_SLOT_CLEANUP == []


def test_descriptor_contract_is_validated_before_cuda_import(monkeypatch):
    warp = SimpleNamespace(
        get_device=lambda _name: (_ for _ in ()).throw(
            AssertionError("CUDA discovery ran before descriptor validation")))
    monkeypatch.setitem(sys.modules, "warp", warp)

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="API version"):
        native_interop.CudaExternalBuffer(
            _descriptor(api_version=2), capacity=32)


def test_failed_partial_import_releases_every_acquired_cuda_resource(monkeypatch):
    calls = []
    device = SimpleNamespace(
        uuid="GPU-5d09f77b-a936-a6d9-054d-0324aef67adb",
        context=object(),
        stream=SimpleNamespace(cuda_stream=7),
    )
    warp = SimpleNamespace(
        get_device=lambda _name: device,
        synchronize_stream=lambda _stream: calls.append("synchronize"),
    )

    class Driver:
        timeline_imports = 0

        def push(self, _context):
            calls.append("push")

        def pop(self):
            calls.append("pop")

        def import_external_memory(self, _handle, _size):
            return ctypes.c_void_p(11), 0x1234

        def import_timeline(self, _handle):
            self.timeline_imports += 1
            if self.timeline_imports == 2:
                raise native_interop.NativeInteropUnavailable(
                    "completion import failed")
            return ctypes.c_void_p(22)

        def destroy_semaphore(self, semaphore):
            calls.append(("destroy_semaphore", semaphore.value))
            return 0

        def free_memory(self, pointer):
            calls.append(("free_memory", pointer))
            return 0

        def destroy_memory(self, memory):
            calls.append(("destroy_memory", memory.value))
            return 0

        @staticmethod
        def check(result, _operation):
            assert result == 0

    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setattr(native_interop, "_CudaDriver", Driver)
    descriptor = _descriptor()

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="completion import failed"):
        native_interop.CudaExternalBuffer(descriptor, capacity=32)

    assert ("destroy_semaphore", 22) in calls
    assert ("free_memory", 0x1234) in calls
    assert ("destroy_memory", 11) in calls


def test_failed_memory_mapping_destroys_the_imported_external_memory():
    calls = []
    driver = native_interop._CudaDriver.__new__(native_interop._CudaDriver)

    def import_memory(memory, _descriptor):
        memory._obj.value = 11
        return 0

    driver.import_memory = import_memory
    driver.map_memory = lambda *_args: 7
    driver.destroy_memory = lambda memory: calls.append(memory.value) or 0

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="cuExternalMemoryGetMappedBuffer"):
        driver.import_external_memory(handle=1, size=4096)

    assert calls == [11]


def test_failed_warp_array_wrap_releases_all_imported_resources(monkeypatch):
    calls = []
    device = SimpleNamespace(
        uuid="GPU-5d09f77b-a936-a6d9-054d-0324aef67adb",
        context=object(),
        stream=SimpleNamespace(cuda_stream=7),
    )
    warp = SimpleNamespace(
        get_device=lambda _name: device,
        synchronize_stream=lambda _stream: calls.append("synchronize"),
        array=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("array wrap failed")),
        vec3=object(),
        vec4=object(),
    )

    class Driver:
        def __init__(self):
            self.timeline = 20

        def push(self, _context):
            calls.append("push")

        def pop(self):
            calls.append("pop")

        def import_external_memory(self, _handle, _size):
            return ctypes.c_void_p(11), 0x1234

        def import_timeline(self, _handle):
            self.timeline += 1
            return ctypes.c_void_p(self.timeline)

        def destroy_semaphore(self, semaphore):
            calls.append(("destroy_semaphore", semaphore.value))
            return 0

        def free_memory(self, pointer):
            calls.append(("free_memory", pointer))
            return 0

        def destroy_memory(self, memory):
            calls.append(("destroy_memory", memory.value))
            return 0

        @staticmethod
        def check(result, _operation):
            assert result == 0

    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setattr(native_interop, "_CudaDriver", Driver)
    descriptor = _descriptor()

    with pytest.raises(RuntimeError, match="array wrap failed"):
        native_interop.CudaExternalBuffer(descriptor, capacity=32)

    assert ("destroy_semaphore", 22) in calls
    assert ("destroy_semaphore", 21) in calls
    assert ("free_memory", 0x1234) in calls
    assert ("destroy_memory", 11) in calls


def test_close_attempts_all_cleanup_and_retries_only_failed_resources():
    calls = []
    fail_completion = [True]

    class Driver:
        def push(self, _context):
            calls.append("push")

        def pop(self):
            calls.append("pop")

        def destroy_semaphore(self, semaphore):
            calls.append(("destroy_semaphore", semaphore.value))
            if semaphore.value == 33 and fail_completion[0]:
                fail_completion[0] = False
                return 7
            return 0

        def free_memory(self, pointer):
            calls.append(("free_memory", pointer))
            return 0

        def destroy_memory(self, memory):
            calls.append(("destroy_memory", memory.value))
            return 0

        @staticmethod
        def check(result, operation):
            if result:
                raise native_interop.NativeInteropUnavailable(operation)

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(
        synchronize_stream=lambda _stream: calls.append("synchronize"))
    bridge.stream = object()
    bridge.driver = Driver()
    bridge.context = object()
    bridge.completion = ctypes.c_void_p(33)
    bridge.release = ctypes.c_void_p(22)
    bridge.mapped_pointer = 0x1234
    bridge.memory = ctypes.c_void_p(11)
    bridge.closed = False

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="cuDestroyExternalSemaphore"):
        bridge.close()

    assert ("destroy_semaphore", 22) in calls
    assert ("free_memory", 0x1234) in calls
    assert ("destroy_memory", 11) in calls
    assert not bridge.closed

    bridge.close()
    calls_after_success = list(calls)
    bridge.close()

    assert calls.count(("destroy_semaphore", 33)) == 2
    assert calls.count(("destroy_semaphore", 22)) == 1
    assert calls.count(("free_memory", 0x1234)) == 1
    assert calls.count(("destroy_memory", 11)) == 1
    assert calls == calls_after_success
    assert bridge.closed


def test_close_retains_all_cuda_imports_when_stream_synchronization_fails():
    calls = []
    fail_sync = [True]

    class Driver:
        def push(self, _context):
            calls.append("push")

        def pop(self):
            calls.append("pop")

        def destroy_semaphore(self, semaphore):
            calls.append(("destroy_semaphore", semaphore.value))
            return 0

        def free_memory(self, pointer):
            calls.append(("free_memory", pointer))
            return 0

        def destroy_memory(self, memory):
            calls.append(("destroy_memory", memory.value))
            return 0

        @staticmethod
        def check(result, operation):
            if result:
                raise native_interop.NativeInteropUnavailable(operation)

    def synchronize(_stream):
        calls.append("synchronize")
        if fail_sync[0]:
            raise RuntimeError("stream synchronization failed")

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(synchronize_stream=synchronize)
    bridge.stream = object()
    bridge.driver = Driver()
    bridge.context = object()
    bridge.completion = ctypes.c_void_p(33)
    bridge.release = ctypes.c_void_p(22)
    bridge.mapped_pointer = 0x1234
    bridge.memory = ctypes.c_void_p(11)
    bridge.context_pushed = False
    bridge.closed = False

    with pytest.raises(RuntimeError, match="stream synchronization failed"):
        bridge.close()

    assert calls == ["synchronize"]
    assert bridge.completion.value == 33
    assert bridge.release.value == 22
    assert bridge.mapped_pointer == 0x1234
    assert bridge.memory.value == 11
    assert not bridge.closed

    fail_sync[0] = False
    bridge.close()
    assert bridge.closed
    assert ("destroy_semaphore", 33) in calls
    assert ("destroy_semaphore", 22) in calls
    assert ("free_memory", 0x1234) in calls
    assert ("destroy_memory", 11) in calls


def test_slot_defers_lease_close_when_constructor_cuda_cleanup_fails(monkeypatch):
    native_interop._DEFERRED_SLOT_CLEANUP.clear()
    calls = []
    allow_bridge_close = [False]

    class Lease:
        def descriptor_get(self):
            return _descriptor()

        def close(self):
            calls.append("lease_close")

    class VBO:
        def __init__(self, **_kwargs):
            self.lease = Lease()

        def external_lease_acquire(self):
            return self.lease

        def _pointcloud_position_unbind(self):
            calls.append("unbind")

    class Format:
        def attr_add(self, **_kwargs):
            pass

    class PartialBridge:
        def close(self):
            calls.append("bridge_close")
            if not allow_bridge_close[0]:
                raise RuntimeError("CUDA cleanup failed")

    partial = PartialBridge()

    def fail_cuda_construction(_descriptor_value, _capacity):
        error = RuntimeError("CUDA construction failed")
        error._native_interop_bridge = partial
        raise error

    gpu = SimpleNamespace(
        types=SimpleNamespace(GPUVertFormat=Format, GPUVertBuf=VBO))
    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setattr(
        native_interop, "CudaExternalBuffer", fail_cuda_construction)

    with pytest.raises(RuntimeError, match="CUDA construction failed"):
        native_interop._NativeParticleSlot(32)

    assert "lease_close" not in calls
    assert len(native_interop._DEFERRED_SLOT_CLEANUP) == 1

    allow_bridge_close[0] = True
    native_interop._retry_deferred_slot_cleanup()
    assert calls[-2:] == ["bridge_close", "lease_close"]
    assert native_interop._DEFERRED_SLOT_CLEANUP == []


def test_import_pop_failure_reuses_the_still_current_cuda_context(monkeypatch):
    calls = []
    device = SimpleNamespace(
        uuid="GPU-5d09f77b-a936-a6d9-054d-0324aef67adb",
        context=object(),
        stream=SimpleNamespace(cuda_stream=7),
    )
    warp = SimpleNamespace(
        get_device=lambda _name: device,
        synchronize_stream=lambda _stream: calls.append("synchronize"),
        vec3=object(),
    )

    class Driver:
        def __init__(self):
            self.timeline = 20
            self.pop_count = 0

        def push(self, _context):
            calls.append("push")

        def pop(self):
            self.pop_count += 1
            calls.append("pop")
            if self.pop_count == 1:
                raise native_interop.NativeInteropUnavailable("pop failed")

        def import_external_memory(self, _handle, _size):
            return ctypes.c_void_p(11), 0x1234

        def import_timeline(self, _handle):
            self.timeline += 1
            return ctypes.c_void_p(self.timeline)

        def destroy_semaphore(self, _semaphore):
            return 0

        def free_memory(self, _pointer):
            return 0

        def destroy_memory(self, _memory):
            return 0

        @staticmethod
        def check(result, _operation):
            assert result == 0

    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setattr(native_interop, "_CudaDriver", Driver)
    descriptor = _descriptor()

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="pop failed"):
        native_interop.CudaExternalBuffer(descriptor, capacity=32)

    assert calls.count("push") == 1
    assert calls.count("pop") == 2


def test_publish_pop_failure_is_recovered_without_a_duplicate_context_push():
    calls = []

    class Driver:
        def __init__(self):
            self.pop_count = 0

        def push(self, _context):
            calls.append("push")

        def pop(self):
            self.pop_count += 1
            calls.append("pop")
            if self.pop_count == 1:
                raise native_interop.NativeInteropUnavailable("pop failed")

        def wait(self, *_args):
            pass

        def signal(self, *_args):
            pass

        def destroy_semaphore(self, _semaphore):
            return 0

        def free_memory(self, _pointer):
            return 0

        def destroy_memory(self, _memory):
            return 0

        @staticmethod
        def check(result, _operation):
            assert result == 0

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(
        launch=lambda *_args, **_kwargs: None,
        synchronize_stream=lambda _stream: None,
    )
    bridge.device = object()
    bridge.stream = SimpleNamespace(cuda_stream=7)
    bridge.driver = Driver()
    bridge.context = object()
    bridge.positions = object()
    bridge.publish_kernel = object()
    bridge.clear_kernel = object()
    bridge.completion = ctypes.c_void_p(33)
    bridge.release = ctypes.c_void_p(22)
    bridge.mapped_pointer = 0x1234
    bridge.memory = ctypes.c_void_p(11)
    bridge.context_pushed = False
    bridge.closed = False
    bridge.capacity = 1
    bridge.visible_count = None
    bridge.stats = {"published_frames": 0, "published_bytes": 0}

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="pop failed"):
        bridge.publish(source=object(), count=1, value=1, radius=0.1)
    bridge.close()

    assert calls.count("push") == 1
    assert calls.count("pop") == 2


def test_close_keeps_parent_memory_alive_until_mapped_pointer_is_freed():
    calls = []
    fail_free = [True]

    class Driver:
        def push(self, _context):
            pass

        def pop(self):
            pass

        def free_memory(self, pointer):
            calls.append(("free_memory", pointer))
            if fail_free[0]:
                fail_free[0] = False
                return 7
            return 0

        def destroy_memory(self, memory):
            calls.append(("destroy_memory", memory.value))
            return 0

        @staticmethod
        def check(result, operation):
            if result:
                raise native_interop.NativeInteropUnavailable(operation)

    bridge = native_interop.CudaExternalBuffer.__new__(
        native_interop.CudaExternalBuffer)
    bridge.wp = SimpleNamespace(synchronize_stream=lambda _stream: None)
    bridge.stream = object()
    bridge.driver = Driver()
    bridge.context = object()
    bridge.completion = ctypes.c_void_p()
    bridge.release = ctypes.c_void_p()
    bridge.mapped_pointer = 0x1234
    bridge.memory = ctypes.c_void_p(11)
    bridge.context_pushed = False
    bridge.closed = False

    with pytest.raises(
            native_interop.NativeInteropUnavailable,
            match="cuMemFree"):
        bridge.close()
    assert ("destroy_memory", 11) not in calls
    assert not bridge.closed

    bridge.close()

    assert calls.count(("free_memory", 0x1234)) == 2
    assert calls.count(("destroy_memory", 11)) == 1
    assert bridge.closed
