# Twin-Roll Adhesive Calender: viscosity A/B

Both lanes run simultaneously under identical FLUID_FLOW shear bands and DRAG. The divider prevents viscosity averaging between lanes. Only the per-particle viscosity differs.

- Low lane viscosity: `0.002`
- High lane viscosity: `0.5`
- Evaluation frame: `25`
- Low core coupling: `0.215036`
- High core coupling: `0.440776`
- Low profile-gradient RMS: `0.260111`
- High profile-gradient RMS: `0.031931`
- Measured viscosity effect: `True`

Whitewater calibration is tied to voxel-normalized turbulent displacement and the authored roll target speed. The production build first emitted at frame `29`, peaked at `27,259` particles (`2.726%` of liquid), and dropped `0` candidates.

The viscosity values are solver coefficients in scene units; this demo is a comparative shear-transfer test, not a calibrated material certificate.
