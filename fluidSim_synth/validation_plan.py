"""Validation targets for the synthesized FluidSim Warp port."""

VALIDATION_TESTS = [
    "sparse tiled leaf-grid coordinate/index roundtrip",
    "leaf-local voxel addressing and global start-vector reconstruction",
    "field precision parity for 16-bit and 32-bit modes",
    "particle-to-grid count/average/active masks",
    "grid-to-particle velocity/attribute sampling",
    "source shape coverage and source particle attribute transfer",
    "collision/sink voxel-type bit behavior",
    "pressure projection divergence reduction",
    "Jacobi and PCG residual convergence",
    "surface SDF build/filter parity",
    "viscosity solve stability",
    "whitewater churn/emission/advection behavior",
    "benchmark profile schema compatibility",
]

NUMERICAL_COMPARISON = {
    "strict": ["active voxel counts", "particle allocation counts", "voxel type bit masks"],
    "tolerant": ["pressure", "velocity", "viscosity", "surface SDF", "whitewater positions"],
    "visual": ["surface continuity", "whitewater density", "collision response", "source behavior"],
}
