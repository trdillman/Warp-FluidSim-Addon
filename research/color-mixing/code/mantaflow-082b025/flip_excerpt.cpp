// Source: https://projects.blender.org/archive/blender-archive/src/commit/082b0253a24781ea34b69cd58e8aabb05194c212/extern/mantaflow/preprocessed/plugin/flip.cpp
// Immutable revision: 082b0253a24781ea34b69cd58e8aabb05194c212
// Narrow excerpt from mapPartsToMAC, mapLinearRealHelper/mapPartsToGrid, and knMapLinearMACGridToVec3_FLIP::op.

void mapPartsToMAC(const FlagGrid &flags, MACGrid &vel, MACGrid &velOld,
                   const BasicParticleSystem &parts,
                   const ParticleDataImpl<Vec3> &partVel,
                   Grid<Vec3> *weight = nullptr)
{
  vel.clear();
  knMapLinearVec3ToMACGrid(parts, flags, vel, *weight, partVel, nullptr, 0);
  weight->stomp(Vec3(VECTOR_EPSILON));
  vel.safeDivide(*weight);
  velOld.copyFrom(vel);
}

template<class T>
void mapLinearRealHelper(const FlagGrid &flags, Grid<T> &target,
                         const BasicParticleSystem &parts,
                         const ParticleDataImpl<T> &source)
{
  Grid<Real> tmp(flags.getParent());
  target.clear();
  knMapLinear<T>(parts, flags, target, tmp, source);
  knSafeDivReal<T>(target, tmp);
}

Vec3 v = vel.getInterpolated(p[idx].pos);
Vec3 delta = v - oldVel.getInterpolated(p[idx].pos);
pvel[idx] = flipRatio * (pvel[idx] + delta) + (1.0 - flipRatio) * v;
