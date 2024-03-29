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
#include "ehrenfest/ehrenfest.h"

namespace py = pybind11;
using namespace rhbi;

void bind_meanF_diabatic(py::module& m);

void bind_meanF_adiabatic(py::module& m);

void diabatic_real_wavefunction(py::module& m);

void adiabatic_real_wavefunction(py::module& m);

void diabatic_complex_wavefunction(py::module& m);

void adiabatic_complex_wavefunction(py::module& m);

void diabatic_real_density_matrix(py::module& m);

void adiabatic_real_density_matrix(py::module& m);

void diabatic_complex_density_matrix(py::module& m);

void adiabatic_complex_density_matrix(py::module& m);

void bind_ehrenfest(py::module& m);

// The implementation of the binding code
void diabatic_real_wavefunction(py::module& m) {
    m.def("ehrenfest_meanF_diabatic", [](const Tensor3d& dHdR, Eigen::Ref<const Eigen::VectorXcd> psi) {
        return ehrenfest_meanF_diabatic(dHdR, psi);
        });
}

void diabatic_complex_wavefunction(py::module& m) {
    m.def("ehrenfest_meanF_diabatic", [](const Tensor3cd& dHdR, Eigen::Ref<const Eigen::VectorXcd> psi) {
        return ehrenfest_meanF_diabatic(dHdR, psi);
        });
}

void diabatic_real_density_matrix(py::module& m) {
    m.def("ehrenfest_meanF_diabatic", [](const Tensor3d& dHdR, Eigen::Ref<const RowMatrixXcd> rho) {
        return ehrenfest_meanF_diabatic(dHdR, rho);
        });
}

void diabatic_complex_density_matrix(py::module& m) {
    m.def("ehrenfest_meanF_diabatic", [](const Tensor3cd& dHdR, Eigen::Ref<const RowMatrixXcd> rho) {
        return ehrenfest_meanF_diabatic(dHdR, rho);
        });
}

void adiabatic_real_wavefunction(py::module& m) {
    m.def("ehrenfest_meanF_adiabatic", [](Eigen::Ref<const RowMatrixXd> F, Eigen::Ref<const Eigen::VectorXd> eig_vals, const Tensor3d& d, Eigen::Ref<const Eigen::VectorXcd> psi) {
        return ehrenfest_meanF_adiabatic(F, eig_vals, d, psi);
        });
}

void adiabatic_complex_wavefunction(py::module& m) {
    m.def("ehrenfest_meanF_adiabatic", [](Eigen::Ref<const RowMatrixXcd> F, Eigen::Ref<const Eigen::VectorXd> eig_vals, const Tensor3cd& d, Eigen::Ref<const Eigen::VectorXcd> psi) {
        return ehrenfest_meanF_adiabatic(F, eig_vals, d, psi);
        });
}

void adiabatic_real_density_matrix(py::module& m) {
    m.def("ehrenfest_meanF_adiabatic", [](Eigen::Ref<const RowMatrixXd> F, Eigen::Ref<const Eigen::VectorXd> eig_vals, const Tensor3d& d, Eigen::Ref<const RowMatrixXcd> rho) {
        return ehrenfest_meanF_adiabatic(F, eig_vals, d, rho);
        });
}

void adiabatic_complex_density_matrix(py::module& m) {
    m.def("ehrenfest_meanF_adiabatic", [](Eigen::Ref<const RowMatrixXcd> F, Eigen::Ref<const Eigen::VectorXd> eig_vals, const Tensor3cd& d, Eigen::Ref<const RowMatrixXcd> rho) {
        return ehrenfest_meanF_adiabatic(F, eig_vals, d, rho);
        });
}


void bind_meanF_diabatic(py::module& m) {
    diabatic_real_wavefunction(m);
    diabatic_complex_wavefunction(m);
    diabatic_real_density_matrix(m);
    diabatic_complex_density_matrix(m);
}

void bind_meanF_adiabatic(py::module& m) {
    adiabatic_real_wavefunction(m);
    adiabatic_complex_wavefunction(m);
    adiabatic_real_density_matrix(m);
    adiabatic_complex_density_matrix(m);
}

void bind_ehrenfest(py::module& m) {
    bind_meanF_diabatic(m);
    bind_meanF_adiabatic(m);
}
