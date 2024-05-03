# `pymddrive`

Convenient `python` package for non-adiabatic molecular dynamics simulation.


## Key algorithms implemented
- (mean-field) Ehrenfest dynamics
- Fewest-switches surface hopping
- Floquet space versions of the above

## TODO:
- [ ] Implement the Miao-Bellonzi-Subotnik 2019 Ansatz to deal with complex-valued Hamiltonians for surface hopping
- [ ] Implement code to calculate dipole-dipole correlation functions


## Installation
```bash
# Clone the repository
git clone git@github.com:ruihao69/pymddrive.git

# Install third-party dependencies (c++ and python dependencies)
git submodule update --init --recursive # c++ dependencies
pip install -r requirements.txt # python dependencies

# install the package
pip install -e . # based on pyproject.toml
```
## Known issues
- Cannot recover correct dynamics for Quasi-Floquet MD in the adiabatic representation WHEN then the time independent part of the Hamiltonian $H_0(R)$ is diagonal ($H(t, R) = H_0(R) + H_1(t, R; \Omega)$).
- The populations for Floquet space Ehrenfest dynamics is not correct
