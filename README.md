# pymddrive

A convenient personal package for molecular dynamics simulation.

## Known issues
- Cannot recover correct dynamics for Quasi-Floquet MD in the adiabatic representation WHEN then the time independent part of the Hamiltonian $H_0(R)$ is diagonal ($H(t, R) = H_0(R) + H_1(t, R; \Omega)$).
