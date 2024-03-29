// pybind11 header file
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

// Third-party libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// local includes
#include "row_major_types.h"
#include "equations_of_motion/equations_of_motion.h"

namespace py = pybind11;
using namespace rhbi;
// forward declarations

void bind_real_hamiltonian_diabatic_wavefunction(py::module& m);

void bind_real_hamiltonian_adiabatic_wavefunction(py::module& m);

void bind_complex_hamiltonian_diabatic_wavefunction(py::module& m);

void bind_complex_hamiltonian_adiabatic_wavefunction(py::module& m);

void bind_real_hamiltonian_diabatic_density_matrix(py::module& m);

void bind_real_hamiltonian_adiabatic_density_matrix(py::module& m);

void bind_complex_hamiltonian_diabatic_density_matrix(py::module& m);

void bind_complex_hamiltonian_adiabatic_density_matrix(py::module& m);

void bind_equations_of_motion(py::module& m);

// The implementation of the functions
void bind_real_hamiltonian_diabatic_wavefunction(py::module& m) {
  m.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXd> H, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return diabatic_equations_of_motion(H, psi);
  });
}

void bind_real_hamiltonian_adiabatic_wavefunction(py::module& m) {
  m.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXd> v_dot_d, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return adiabatic_equations_of_motion(E, v_dot_d, psi);
  });
}

void bind_complex_hamiltonian_diabatic_wavefunction(py::module& m) {
  m.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXcd> H, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return diabatic_equations_of_motion(H, psi);
  });
}

void bind_complex_hamiltonian_adiabatic_wavefunction(py::module& m) {
  m.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXcd> v_dot_d, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return adiabatic_equations_of_motion(E, v_dot_d, psi);
  });
}

void bind_real_hamiltonian_diabatic_density_matrix(py::module& m) {
  m.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXd> H, Eigen::Ref<const RowMatrixXcd> rho) {
    return diabatic_equations_of_motion(H, rho);
  });
}

void bind_real_hamiltonian_adiabatic_density_matrix(py::module& m) {
  m.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXd> v_dot_d, Eigen::Ref<const RowMatrixXcd> rho) {
    return adiabatic_equations_of_motion(E, v_dot_d, rho);
  });
}

void bind_complex_hamiltonian_diabatic_density_matrix(py::module& m) {
  m.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXcd> H, Eigen::Ref<const RowMatrixXcd> rho) {
    return diabatic_equations_of_motion(H, rho);
  });
}

void bind_complex_hamiltonian_adiabatic_density_matrix(py::module& m) {
  m.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXcd> v_dot_d, Eigen::Ref<const RowMatrixXcd> rho) {
    return adiabatic_equations_of_motion(E, v_dot_d, rho);
  });
}

void bind_equations_of_motion(py::module& m) {
  bind_real_hamiltonian_diabatic_wavefunction(m);
  bind_real_hamiltonian_adiabatic_wavefunction(m);
  bind_complex_hamiltonian_diabatic_wavefunction(m);
  bind_complex_hamiltonian_adiabatic_wavefunction(m);
  bind_real_hamiltonian_diabatic_density_matrix(m);
  bind_real_hamiltonian_adiabatic_density_matrix(m);
  bind_complex_hamiltonian_diabatic_density_matrix(m);
  bind_complex_hamiltonian_adiabatic_density_matrix(m);
}