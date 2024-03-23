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
#include "density_matrix_state.h"

// C++ standard library
#include <complex>

namespace py = pybind11;
using namespace rhbi;

// create a overall module called low_level
PYBIND11_MODULE(_low_level, m) {
  m.doc() = "low-level module for the pymddrive package";

  /****
  The states submodule
   ****/
  py::module m_states = m.def_submodule("states", "low-level version of the states module");
  m_states.doc() = "low-level version of the states module";

  py::class_<DensityMatrixState>(m_states, "DensityMatrixState")
      .def(py::init<const Eigen::Tensor<double, 1>&, const Eigen::Tensor<double, 1>&, const Eigen::Tensor<std::complex<double>, 2>&>())
      .def_property(
          "R", [](const DensityMatrixState& state) { return Eigen::TensorMap<const Eigen::Tensor<double, 1>>(state.R.data(), state.R.dimensions()); }, [](DensityMatrixState& state, const Eigen::Tensor<double, 1>& R) { Eigen::TensorMap<Eigen::Tensor<double, 1>>(state.R.data(), state.R.dimensions()) = R; })
      .def_property(
          "P", [](const DensityMatrixState& state) { return Eigen::TensorMap<const Eigen::Tensor<double, 1>>(state.P.data(), state.P.dimensions()); }, [](DensityMatrixState& state, const Eigen::Tensor<double, 1>& P) { Eigen::TensorMap<Eigen::Tensor<double, 1>>(state.P.data(), state.P.dimensions()) = P; })
      .def_property(
          "rho", [](const DensityMatrixState& state) { return Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 2>>(state.rho.data(), state.rho.dimensions()); }, [](DensityMatrixState& state, const Eigen::Tensor<std::complex<double>, 2>& rho) { Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>>(state.rho.data(), state.rho.dimensions()) = rho; })
      .def("flatten", &DensityMatrixState::flatten)
      .def("from_unstructured", &DensityMatrixState::from_unstructured)
      .def(py::self + py::self)
      .def(py::self * double())
      .def(py::self += py::self)
      .def(py::self *= double());

  m_states.def("axpy", &axpy);
  m_states.def("get_state_from_unstructured", &get_state_from_unstructured);
  m_states.def("rk4_step_inplace", &rk4_step_inplace);
  m_states.def("rk4_step", &rk4_step);
}