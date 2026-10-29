# Allocation accounting

Admission and capacity suggestions use `sim/memory_safety.py`; transactional
particle growth uses `sim/particle_capacity.py`. These are conservative demand
estimates, not measurements of all device allocations or allocator reservation.

Pure owner declarations in `particle_storage_layout.py`,
`whitewater_storage_layout.py`, and `grid_storage_layout.py` supply the names,
extents and element widths consumed by both allocation owners and costing.
The current owned payload costs include:

- Whitewater live state: 92 bytes per slot; liquid emission candidates: 12 bytes
  per liquid slot (flag, offset and type), with at least one candidate slot.
- Optional liquid channels: both live and compaction buffers. Color costs 32
  bytes per slot; source ID and UID each cost 8. Retained APIC matrices cost 72.
- Grid topology: six int32 cell tables (24 bytes per cell), two vec3i coordinate
  arrays (24 bytes per leaf), plus the simulation's old table (4 bytes per cell).
  Secondary grids conservatively receive the same table allowance.
- Capillary normals and scalar scratch: 32 bytes per voxel; propagation remains
  a separate 28-byte allowance.

The existing base, base-particle, field, dense-surface and narrow-band reserves
remain conservative allowances. The budget policy remains 85 percent of total
and 90 percent of currently free device memory.

Growth charges the complete new particle pool and emission workspace while old
storage remains alive, plus replacement voxel-bin counts/starts. It uses channels
and affine buffers actually retained by the particle owner, since current
authored settings can differ from allocated state. Copy, synchronization, swap
and rollback keep their existing transactional ordering.

Capacity candidates retain a detached settings snapshot with candidate overrides.
The deliberate scene-signature projection stays separate from estimator inputs;
new estimator fields do not require another planner field list. Resolved quality
is recomputed for candidate bounds and resolution.

`tests/runtime/test_allocation_accounting.py` compares costs against real Warp
CPU arrays and replacement transactions. The production gate includes
`tests/acceptance/blender_allocation_accounting_acceptance.py` for bounded CUDA
initial/growth allocations across base, optional/APIC, capillary/narrow-band and
secondary-surface configurations. Its report separates reachable array payload,
memory-pool observations and free-device memory. It does not certify frame
advancement, every lazy backend allocation, performance or whole-process VRAM.
