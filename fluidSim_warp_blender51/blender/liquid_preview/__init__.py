"""Liquid Preview: screen-space fluid rendering of the published particles.

``handler`` owns draw-handler and hotkey registration, ``passes`` the GPU
passes and their pure math, ``shaders`` the create-info shader sources, and
``reference`` the deterministic payload, PNG and SSIM helpers used by the
offscreen reference test. Nothing here touches ``bpy.data`` or the solver: the
only input is :class:`..output_publication.PublishedPoints`.
"""
