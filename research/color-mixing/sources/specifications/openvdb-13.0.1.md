# Primary format reference clip: OpenVDB 13.0.1

- Originals: https://www.openvdb.org/documentation/doxygen/overview.html and https://www.openvdb.org/documentation/doxygen/points.html
- Title/owner: OpenVDB documentation / Academy Software Foundation OpenVDB project
- Version: OpenVDB 13.0.1
- Retrieved: 2026-08-22 UTC
- License: Mozilla Public License 2.0 for OpenVDB code; documentation attribution retained.

Primary-source excerpt (overview, 16 words):

> A VDB file can contain multiple grids, each identified by a unique name and associated metadata.

Recorded facts: OpenVDB supports multiple named grids and per-grid/file metadata; PointDataGrid supports typed point attributes. It is suitable as an optional cache/interchange target only. Its hierarchical runtime storage conflicts with the mandated flat 8x8x8 leaf table and is therefore rejected for live fields.
