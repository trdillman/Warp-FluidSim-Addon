# Project Directory Manifest

## Included clean synthesis package

```text
fluidSim_warp_blender51_synth_package/
├── AGENTS.md
├── CLAUDE.md
├── PROJECT_MANIFEST.md
├── PROMPT.md
├── docs/
│   ├── KERNEL_GROUPS.md
│   ├── PORTING_BRIEF.md
│   └── SOURCE_SYNTHESIS.md
└── fluidSim_synth/
    ├── __init__.py
    ├── blender_addon_blueprint.py
    ├── custom_kernel_contract.py
    ├── dispatch_contract.py
    ├── enums.py
    ├── fluid_meta.py
    ├── kernel_manifest.py
    ├── profiler_contract.py
    ├── ui_contract.py
    └── validation_plan.py
```

## Excluded by design

- Original OpenCL source files.
- Original Houdini Python source files.
- Original dispatch extraction source file.
- Any Houdini `hou` or PySide implementation code.

## Purpose of synthesized modules

- `fluid_meta.py`: FluidSim metadata, target backend, and source facts.
- `enums.py`: required enum/value tables for solver properties and UI controls.
- `kernel_manifest.py`: kernel identity inventory and phase grouping without source code.
- `dispatch_contract.py`: OpenCL-to-Warp runtime dispatch model.
- `custom_kernel_contract.py`: advanced user snippet/extension surface translated into Warp terms.
- `ui_contract.py`: Blender UI panels and data-integration behavior.
- `profiler_contract.py`: profile/benchmark schema and timer-track expectations.
- `blender_addon_blueprint.py`: target package structure and performance policy.
- `validation_plan.py`: validation and comparison targets.

## Official demo pipeline

- `tools/build_demo_scenes.py`: builds the canonical demo scenes:
  `dam_break`, `crown_splash`, `slosh_tank`, `tidal_lock`, and
  `maelstrom_basin`.
- `tools/analyze_demos.py`: analyzes the generated demo evidence and now
  treats maelstrom metadata as a first-class demo lane.
- `tools/check_demo_blends.py`: opens all canonical `.blend` files, including
  `demos/maelstrom_basin.blend`.
- `tests/test_demo_analysis_maelstrom.py`: focused regression coverage for the
  maelstrom-specific spin/churn/playback contract.
