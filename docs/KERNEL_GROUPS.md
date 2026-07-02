# Kernel Groups

## core_buffer_field_ops (47)

- `coreClearScalarInt32` — source `coreClearScalarInt32`, original line 5475
- `coreClearCoupleInt32` — source `coreClearCoupleInt32`, original line 5487
- `coreClearScalarInt16` — source `coreClearScalarInt16`, original line 5501
- `coreClearScalarInt8` — source `coreClearScalarInt8`, original line 5513
- `coreClearScalarVoxelType` — source `coreClearScalarVoxelType`, original line 5525
- `coreClearScalarFloat32` — source `coreClearScalarFloat32`, original line 5537
- `coreClearScalarField` — source `coreClearScalarField`, original line 5549
- `coreClearVectorField` — source `coreClearVectorField`, original line 5561
- `coreConvertFieldVectorToMagnitude` — source `coreConvertFieldVectorToMagnitude`, original line 5624
- `coreConvertFieldStaggeredVectorToMagnitude` — source `coreConvertFieldStaggeredVectorToMagnitude`, original line 5647
- `coreSyncScalarInt32` — source `coreSyncScalarInt32`, original line 5822
- `coreSyncScalarInt16` — source `coreSyncScalarInt16`, original line 5835
- `coreSyncScalarVoxelType` — source `coreSyncScalarVoxelType`, original line 5848
- `coreSyncScalarFloat32` — source `coreSyncScalarFloat32`, original line 5861
- `coreSyncScalarFloat16` — source `coreSyncScalarFloat16`, original line 5874
- `coreSyncScalarField` — source `coreSyncScalarField`, original line 5889
- `coreSyncCoupleInt32` — source `coreSyncCoupleInt32`, original line 5902
- `coreSyncVectorFloat32` — source `coreSyncVectorFloat32`, original line 5921
- `coreSyncCoupleField` — source `coreSyncCoupleField`, original line 5944
- `coreSyncVectorField` — source `coreSyncVectorField`, original line 5963
- `coreReduceScalarFieldDotPartialLinear128` — source `coreReduceScalarFieldDotPartialLinear128`, original line 5988
- `coreReduceScalarFieldDotPartialLinear256` — source `coreReduceScalarFieldDotPartialLinear256`, original line 6066
- `coreReduceScalarFieldDotPartialLinear512` — source `coreReduceScalarFieldDotPartialLinear512`, original line 6144
- `coreReduceScalarFieldDotPartialLinear1024` — source `coreReduceScalarFieldDotPartialLinear1024`, original line 6222
- `coreReduceVectorFieldDotPartialLinear128` — source `coreReduceVectorFieldDotPartialLinear128`, original line 6301
- `coreReduceVectorFieldDotPartialLinear256` — source `coreReduceVectorFieldDotPartialLinear256`, original line 6399
- `coreReduceVectorFieldDotPartialLinear512` — source `coreReduceVectorFieldDotPartialLinear512`, original line 6497
- `coreReduceVectorFieldDotPartialLinear1024` — source `coreReduceVectorFieldDotPartialLinear1024`, original line 6595
- `coreReduceSum128` — source `coreReduceSum128`, original line 6694
- `coreReduceSum256` — source `coreReduceSum256`, original line 6763
- `coreReduceSum512` — source `coreReduceSum512`, original line 6832
- `coreReduceSum1024` — source `coreReduceSum1024`, original line 6901
- `coreConvertFloat32ToFloat16` — source `coreConvertFloat32ToFloat16`, original line 7596
- `coreConvertFloat16ToFloat32` — source `coreConvertFloat16ToFloat32`, original line 7617
- `coreConvertInt32ToInt16` — source `coreConvertInt32ToInt16`, original line 7638
- `coreConvertInt16ToInt32` — source `coreConvertInt16ToInt32`, original line 7651
- `coreClampScalarField` — source `coreClampScalarField`, original line 7666
- `coreRemapScalarField` — source `coreRemapScalarField`, original line 7685
- `coreSortScalarInt32` — source `coreSortScalarInt32`, original line 7704
- `coreSortScalarInt16` — source `coreSortScalarInt16`, original line 7718
- `coreSortScalarVoxelType` — source `coreSortScalarVoxelType`, original line 7732
- `coreSortScalarFloat16` — source `coreSortScalarFloat16`, original line 7746
- `coreSortScalarFloat32` — source `coreSortScalarFloat32`, original line 7762
- `coreSortVectorFloat32` — source `coreSortVectorFloat32`, original line 7776
- `coreSortScalarField` — source `coreSortScalarField`, original line 7802
- `coreSortCoupleField` — source `coreSortCoupleField`, original line 7816
- `coreSortVectorField` — source `coreSortVectorField`, original line 7838

## core_misc (3)

- `coreComputeMaxActiveValue` — source `coreComputeMaxActiveValue`, original line 5579
- `coreResetScaleField` — source `coreResetScaleField`, original line 5686
- `coreRampToScaleFieldVolume` — source `coreRampToScaleFieldVolume`, original line 5700

## core_particle_grid_transfer (5)

- `coreRampToScaleFieldParticle` — source `coreRampToScaleFieldParticle`, original line 5764
- `coreGridToParticleFieldAttribute` — source `coreGridToParticleFieldAttribute`, original line 7866
- `coreParticleToGridCollect` — source `coreParticleToGridCollect`, original line 7925
- `coreGridExpand` — source `coreGridExpand`, original line 8012
- `coreParticleToGridAverage` — source `coreParticleToGridAverage`, original line 8072

## core_sdf_collision (10)

- `coreCreateSDFAxisX` — source `coreCreateSDFAxisX`, original line 6973
- `coreCreateSDFAxisY` — source `coreCreateSDFAxisY`, original line 7041
- `coreCreateSDFAxisZ` — source `coreCreateSDFAxisZ`, original line 7128
- `coreSmoothSDF` — source `coreSmoothSDF`, original line 7219
- `coreRenormalizeSdfFastSweep` — source `coreRenormalizeSdfFastSweep`, original line 7312
- `coreRecomputeCollisionBitFromSdf` — source `coreRecomputeCollisionBitFromSdf`, original line 7447
- `coreSDFToGradient` — source `coreSDFToGradient`, original line 7474
- `coreSourceNanoVDBTypeSDF` — source `coreSourceNanoVDBTypeSDF`, original line 8351
- `coreSourceNanoVDBTypeSDFWithTemperature` — source `coreSourceNanoVDBTypeSDFWithTemperature`, original line 8460
- `coreSourceNanoVDBMaskSDF` — source `coreSourceNanoVDBMaskSDF`, original line 8663

## core_sourcing_vdb_volume_particle (15)

- `coreSourceNanoVDBTypeFog` — source `coreSourceNanoVDBTypeFog`, original line 8130
- `coreSourceNanoVDBTypeFogWithTemperature` — source `coreSourceNanoVDBTypeFogWithTemperature`, original line 8239
- `coreSourceNanoVDBMaskFog` — source `coreSourceNanoVDBMaskFog`, original line 8572
- `coreSourceNanoVDBFloat` — source `coreSourceNanoVDBFloat`, original line 8754
- `coreSourceNanoVDBVector` — source `coreSourceNanoVDBVector`, original line 8856
- `coreSourceOpenVDBMask` — source `coreSourceOpenVDBMask`, original line 8955
- `coreSourceOpenVDBType` — source `coreSourceOpenVDBType`, original line 8989
- `coreSourceOpenVDBTypeWithTemperature` — source `coreSourceOpenVDBTypeWithTemperature`, original line 9037
- `coreSourceOpenVDBFloat` — source `coreSourceOpenVDBFloat`, original line 9088
- `coreSourceOpenVDBVectorComponent` — source `coreSourceOpenVDBVectorComponent`, original line 9129
- `coreSourceFloat` — source `coreSourceFloat`, original line 9155
- `coreSourceVector` — source `coreSourceVector`, original line 9210
- `coreSourceStaggeredVector` — source `coreSourceStaggeredVector`, original line 9306
- `coreSourceFloatParticleAttribute` — source `coreSourceFloatParticleAttribute`, original line 9405
- `coreSourceVectorParticleAttribute` — source `coreSourceVectorParticleAttribute`, original line 9503

## fluid_advection (2)

- `fluidSimAdvectParticles` — source `paradigmAdvectParticles`, original line 10243
- `fluidSimAdvectWhiteWater` — source `paradigmAdvectWhiteWater`, original line 10690

## fluid_forces_attraction (10)

- `fluidSimForcesGravity` — source `paradigmForcesGravity`, original line 12614
- `fluidSimForcesDirectionalForce` — source `paradigmForcesDirectionalForce`, original line 12643
- `fluidSimForcesOmnidirectionalForce` — source `paradigmForcesOmnidirectionalForce`, original line 12669
- `fluidSimForcesSpeedLimit` — source `paradigmForcesSpeedLimit`, original line 12706
- `fluidSimForcesDiffusion` — source `paradigmForcesDiffusion`, original line 12748
- `fluidSimForcesDissipateParticleAttribute` — source `paradigmForcesDissipateParticleAttribute`, original line 12855
- `fluidSimAttractMaskToField` — source `paradigmAttractMaskToField`, original line 12883
- `fluidSimAttractSmoothField` — source `paradigmAttractSmoothField`, original line 12906
- `fluidSimAttractDirection` — source `paradigmAttractDirection`, original line 12981
- `fluidSimAttractSdfGradientDirection` — source `paradigmAttractSdfGradientDirection`, original line 13101

## fluid_misc (3)

- `fluidSimComputeMaxActiveValue` — source `paradigmComputeMaxActiveValue`, original line 17593
- `fluidSimClearJustInitializedVoxelTypeTracked` — source `paradigmClearJustInitializedVoxelTypeTracked`, original line 17793
- `fluidSimExpandFluidMask` — source `paradigmExpandFluidMask`, original line 18047

## fluid_particle_indexing_sorting (8)

- `fluidSimParticleToVoxelCountAndUpdateVoxelIndex` — source `paradigmParticleToVoxelCountAndUpdateVoxelIndex`, original line 9645
- `fluidSimVoxelCountToVoxelParticleArrayStart` — source `paradigmVoxelCountToVoxelParticleArrayStart`, original line 9714
- `fluidSimParticleToVoxelParticleArray` — source `paradigmParticleToVoxelParticleArray`, original line 9758
- `fluidSimVoxelParticleArraySort` — source `paradigmVoxelParticleArraySort`, original line 9803
- `fluidSimLeafSortVoxelTypeTracked` — source `paradigmLeafSortVoxelTypeTracked`, original line 17821
- `fluidSimSortParticlesRandomMutex` — source `paradigmSortParticlesRandomMutex`, original line 18334
- `fluidSimParticleLeafCountToLeafArrayStart` — source `paradigmParticleLeafCountToLeafArrayStart`, original line 18411
- `fluidSimSortParticlesLeafArrayMutex` — source `paradigmSortParticlesLeafArrayMutex`, original line 18455

## fluid_particles_lifecycle_transfer (10)

- `fluidSimParticlesRelax` — source `paradigmParticlesRelax`, original line 17646
- `fluidSimUpdateParticleVoxelIndex` — source `paradigmUpdateParticleVoxelIndex`, original line 17768
- `fluidSimParticleAge` — source `paradigmParticleAge`, original line 17874
- `fluidSimParticleLifespan` — source `paradigmParticleLifespan`, original line 17893
- `fluidSimParticleInSinkCheck` — source `paradigmParticleInSinkCheck`, original line 18271
- `fluidSimParticleToLeafCount` — source `paradigmParticleToLeafCount`, original line 18366
- `fluidSimParticleSetGridData` — source `paradigmParticleSetGridData`, original line 18530
- `fluidSimParticleToCount` — source `paradigmParticleToCount`, original line 18608
- `fluidSimStaggeredVectorFieldToParticle` — source `paradigmStaggeredVectorFieldToParticle`, original line 18900
- `fluidSimParticleToGridActive` — source `paradigmParticleToGridActive`, original line 18987

## fluid_pressure_projection (40)

- `fluidSimProjectNonDivergentCollisionPreConditionerNew` — source `paradigmProjectNonDivergentCollisionPreConditionerNew`, original line 13262
- `fluidSimProjectNonDivergentComputeDivergence_old` — source `paradigmProjectNonDivergentComputeDivergence_old`, original line 13310
- `fluidSimProjectNonDivergentComputeDivergence` — source `paradigmProjectNonDivergentComputeDivergence`, original line 13475
- `fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates8` — source `paradigmProjectNonDivergentElectrostaticBuildLeafAggregates8`, original line 13676
- `fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates16` — source `paradigmProjectNonDivergentElectrostaticBuildLeafAggregates16`, original line 13760
- `fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates32` — source `paradigmProjectNonDivergentElectrostaticBuildLeafAggregates32`, original line 13844
- `fluidSimProjectNonDivergentElectrostaticSolvePotential8` — source `paradigmProjectNonDivergentElectrostaticSolvePotential8`, original line 13929
- `fluidSimProjectNonDivergentElectrostaticSolvePotential16` — source `paradigmProjectNonDivergentElectrostaticSolvePotential16`, original line 14114
- `fluidSimProjectNonDivergentElectrostaticSolvePotential32` — source `paradigmProjectNonDivergentElectrostaticSolvePotential32`, original line 14299
- `fluidSimProjectNonDivergentProject` — source `paradigmProjectNonDivergentProject`, original line 14485
- `fluidSimProjectNonDivergentExpandVelocityIntoCollision` — source `paradigmProjectNonDivergentExpandVelocityIntoCollision`, original line 14601
- `fluidSimProjectNonDivergentJacobi2_original` — source `paradigmProjectNonDivergentJacobi2_original`, original line 14730
- `fluidSimProjectNonDivergentJacobi4_original` — source `paradigmProjectNonDivergentJacobi4_original`, original line 14840
- `fluidSimProjectNonDivergentJacobi8_original` — source `paradigmProjectNonDivergentJacobi8_original`, original line 14950
- `fluidSimProjectNonDivergentJacobi16_original` — source `paradigmProjectNonDivergentJacobi16_original`, original line 15060
- `fluidSimProjectNonDivergentJacobi32_original` — source `paradigmProjectNonDivergentJacobi32_original`, original line 15170
- `fluidSimProjectNonDivergentJacobi2_optimizedUnroll` — source `paradigmProjectNonDivergentJacobi2_optimizedUnroll`, original line 15281
- `fluidSimProjectNonDivergentJacobi4_optimizedUnroll` — source `paradigmProjectNonDivergentJacobi4_optimizedUnroll`, original line 15529
- `fluidSimProjectNonDivergentJacobi8_optimizedUnroll` — source `paradigmProjectNonDivergentJacobi8_optimizedUnroll`, original line 15777
- `fluidSimProjectNonDivergentJacobi16_optimizedUnroll` — source `paradigmProjectNonDivergentJacobi16_optimizedUnroll`, original line 16025
- `fluidSimProjectNonDivergentJacobi32_optimizedUnroll` — source `paradigmProjectNonDivergentJacobi32_optimizedUnroll`, original line 16273
- `fluidSimProjectNonDivergentSampleDown` — source `paradigmProjectNonDivergentSampleDown`, original line 16522
- `fluidSimProjectNonDivergentSampleUp` — source `paradigmProjectNonDivergentSampleUp`, original line 16594
- `fluidSimProjectNonDivergentComputeInverseDiagonal8` — source `paradigmProjectNonDivergentComputeInverseDiagonal8`, original line 16665
- `fluidSimProjectNonDivergentComputeInverseDiagonal16` — source `paradigmProjectNonDivergentComputeInverseDiagonal16`, original line 16748
- `fluidSimProjectNonDivergentComputeInverseDiagonal32` — source `paradigmProjectNonDivergentComputeInverseDiagonal32`, original line 16831
- `fluidSimProjectNonDivergentInitializePcg` — source `paradigmProjectNonDivergentInitializePcg`, original line 16915
- `fluidSimProjectNonDivergentApplyMatrix8` — source `paradigmProjectNonDivergentApplyMatrix8`, original line 16955
- `fluidSimProjectNonDivergentApplyMatrix16` — source `paradigmProjectNonDivergentApplyMatrix16`, original line 17125
- `fluidSimProjectNonDivergentApplyMatrix32` — source `paradigmProjectNonDivergentApplyMatrix32`, original line 17295
- `fluidSimProjectNonDivergentComputeAlpha` — source `paradigmProjectNonDivergentComputeAlpha`, original line 17466
- `fluidSimProjectNonDivergentUpdateXR` — source `paradigmProjectNonDivergentUpdateXR`, original line 17484
- `fluidSimProjectNonDivergentApplyPreConditioner` — source `paradigmProjectNonDivergentApplyPreConditioner`, original line 17511
- `fluidSimProjectNonDivergentComputeBeta` — source `paradigmProjectNonDivergentComputeBeta`, original line 17542
- `fluidSimProjectNonDivergentUpdateP` — source `paradigmProjectNonDivergentUpdateP`, original line 17562
- `fluidSimParticleToCountPressure` — source `paradigmParticleToCountPressure`, original line 18675
- `fluidSimParticleCountToPressure` — source `paradigmParticleCountToPressure`, original line 18750
- `fluidSimPressureThresholdCompute` — source `paradigmPressureThresholdCompute`, original line 18796
- `fluidSimPressureThresholdApply` — source `paradigmPressureThresholdApply`, original line 18816
- `fluidSimPressureScale` — source `paradigmPressureScale`, original line 18833

## fluid_sourcing (11)

- `fluidSimSourceShapes` — source `paradigmSourceShapes`, original line 11184
- `fluidSimSourceShapesParticlesAttributes` — source `paradigmSourceShapesParticlesAttributes`, original line 11849
- `fluidSimSourceHeight` — source `paradigmSourceHeight`, original line 12293
- `fluidSimSourceResolveType` — source `paradigmSourceResolveType`, original line 12385
- `fluidSimSourceParticles` — source `paradigmSourceParticles`, original line 12413
- `fluidSimSourceParticleVectorFieldAttribute` — source `paradigmSourceParticleVectorFieldAttribute`, original line 12482
- `fluidSimSourceParticleScalarFieldAttribute` — source `paradigmSourceParticleScalarFieldAttribute`, original line 12516
- `fluidSimSourceParticleScalarIntegerAttribute` — source `paradigmSourceParticleScalarIntegerAttribute`, original line 12540
- `fluidSimSourceParticleScalarFloatAttribute` — source `paradigmSourceParticleScalarFloatAttribute`, original line 12562
- `fluidSimSourceParticleScalarHalfAttribute` — source `paradigmSourceParticleScalarHalfAttribute`, original line 12586
- `fluidSimResetSourceShapeTrackedVoxelType` — source `paradigmResetSourceShapeTrackedVoxelType`, original line 17855

## fluid_surface_collision (7)

- `fluidSimBuildSurfaceSdf` — source `paradigmBuildSurfaceSdf`, original line 19068
- `fluidSimSmoothSurfaceSdfDialate` — source `paradigmSmoothSurfaceSdfDialate`, original line 19174
- `fluidSimSmoothSurfaceSdfErode` — source `paradigmSmoothSurfaceSdfErode`, original line 19189
- `fluidSimSmoothSurfaceSdfBox` — source `paradigmSmoothSurfaceSdfBox`, original line 19204
- `fluidSimSmoothSurfaceSdfMeanCurvatureFlow` — source `paradigmSmoothSurfaceSdfMeanCurvatureFlow`, original line 19265
- `fluidSimSdfToAxisCollision` — source `paradigmSdfToAxisCollision`, original line 20460
- `fluidSimExpandAndShiftAxisVelocityInColliderAxis` — source `paradigmExpandAndShiftAxisVelocityInColliderAxis`, original line 20537

## fluid_viscosity (14)

- `fluidSimForceViscositySmooth` — source `paradigmForceViscositySmooth`, original line 19448
- `fluidSimForceViscosityCounterIncrement` — source `paradigmForceViscosityCounterIncrement`, original line 19747
- `fluidSimForceViscosityBuildRhs` — source `paradigmForceViscosityBuildRhs`, original line 19759
- `fluidSimForceViscosityComputeInverseDiagonal` — source `paradigmForceViscosityComputeInverseDiagonal`, original line 19923
- `fluidSimForceViscosityApplyMatrix` — source `paradigmForceViscosityApplyMatrix`, original line 20016
- `fluidSimForceViscosityInitializePcg` — source `paradigmForceViscosityInitializePcg`, original line 20199
- `fluidSimForceViscosityFinalizeInitialResidual` — source `paradigmForceViscosityFinalizeInitialResidual`, original line 20226
- `fluidSimForceViscosityInitializeDirectionFromPreconditioned` — source `paradigmForceViscosityInitializeDirectionFromPreconditioned`, original line 20265
- `fluidSimForceViscosityComputeAlpha` — source `paradigmForceViscosityComputeAlpha`, original line 20288
- `fluidSimForceViscosityUpdateXR` — source `paradigmForceViscosityUpdateXR`, original line 20307
- `fluidSimForceViscosityApplyPreConditioner` — source `paradigmForceViscosityApplyPreConditioner`, original line 20339
- `fluidSimForceViscosityComputeBeta` — source `paradigmForceViscosityComputeBeta`, original line 20368
- `fluidSimForceViscosityUpdateP` — source `paradigmForceViscosityUpdateP`, original line 20389
- `fluidSimForceViscosityApplyCollision` — source `paradigmForceViscosityApplyCollision`, original line 20419

## fluid_whitewater (3)

- `fluidSimComputeChurn` — source `paradigmComputeChurn`, original line 17918
- `fluidSimEmitWhiteWater` — source `paradigmEmitWhiteWater`, original line 18107
- `fluidSimWhiteWaterSetGridData` — source `paradigmWhiteWaterSetGridData`, original line 18572
