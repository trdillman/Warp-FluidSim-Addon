// Source: https://develop.openfoam.com/Development/OpenFOAM-plus/-/blob/244109d2f834ae99bf2311e69c27d567416e39f7/src/functionObjects/solvers/scalarTransport/scalarTransport.H
// Immutable revision: 244109d2f834ae99bf2311e69c27d567416e39f7

enum class diffusivityType
{
    none,
    constant,
    viscosity
};

scalar D_;
scalar alphal_;
scalar alphat_;
volScalarField s_;
bool MULES_;

tmp<volScalarField> D() const;
void subCycleMULES();
void solveMULES();
