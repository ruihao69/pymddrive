// pybind11 header file
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>

// local includes
#include "row_major_types.h"

#include "surface_hopping/surface_hopping.h"

namespace py = pybind11;
using namespace rhbi;

// forward declarations
void bind_scalar_mass_real_derivative_coupling_density_matrix(py::module& m);
void bind_vector_mass_real_derivative_coupling_density_matrix(py::module& m);

void bind_surface_hopping(py::module& m);

// The implementation of the binding code

void bind_scalar_mass_real_derivative_coupling_density_matrix(py::module& m) {
    m.def("fssh_surface_hopping", [](double dt, int active_surface, Eigen::Ref<const Eigen::RowVectorXd> P_current, Eigen::Ref<const RowMatrixXcd> rho, Eigen::Ref<const Eigen::RowVectorXd> eig_vals, Eigen::Ref<const RowMatrixXd> v_dot_d, const Tensor3d& dc, double mass) {
        return fssh_surface_hopping(dt, active_surface, P_current, rho, eig_vals, v_dot_d, dc, mass);
        });
}

void bind_vector_mass_real_derivative_coupling_density_matrix(py::module& m) {
    m.def("fssh_surface_hopping", [](double dt, int active_surface, Eigen::Ref<const Eigen::RowVectorXd> P_current, Eigen::Ref<const RowMatrixXcd> rho, Eigen::Ref<const Eigen::RowVectorXd> eig_vals, Eigen::Ref<const RowMatrixXd> v_dot_d, const Tensor3d& dc, Eigen::Ref<const Eigen::RowVectorXd> mass) {
        return fssh_surface_hopping(dt, active_surface, P_current, rho, eig_vals, v_dot_d, dc, mass);
        });
}

void bind_surface_hopping(py::module& m) {
    bind_scalar_mass_real_derivative_coupling_density_matrix(m);
    bind_vector_mass_real_derivative_coupling_density_matrix(m);
}
