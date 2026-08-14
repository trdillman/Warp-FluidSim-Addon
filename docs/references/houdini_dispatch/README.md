# Houdini Dispatch Reference

This directory preserves a tracked extraction from the Paradigm/FluidSim 1.0.226 Houdini archive. The evidence was handed into this repository by commit `a6965e6`.

It is reference and provenance material, not Blender add-on runtime source:

- `paradigm_dispatch_enqueue_design.md` explains the observed three-layer OpenCL architecture and records dispatch/enqueue pseudocode and diagrams.
- `paradigm_enqueue_dispatch.py` captures the extracted host dispatch API, diagnostics, kernel inventory helpers, and custom-kernel surface in executable Python form.
- `paradigm_kernel_names.txt` records the extracted kernel-name inventory.
- `paradigm_mapping_verify.txt` records the kernel-to-host-wrapper completeness check and mapping table.

The evidence establishes the generic compile, bind, enqueue, synchronize, and custom-kernel surfaces. It does **not** establish the concrete order in which every solver phase was dispatched; generic dispatch pseudocode must not be cited as proof of a specific phase sequence. Current source, runtime traces, and executable tests outrank this historical extraction when they disagree.
