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
#include "states/state.h"

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
  m_states.doc() = "Low-level state variables for the quantum / quantum classical dynamics simulation.";

  // The binding code for State class
  py::class_<State>(m_states, "State")
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double>())
      .def(py::init<const Eigen::VectorXcd&>())
      .def(py::init<const Eigen::MatrixXcd&>())
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double, const Eigen::VectorXcd&>())
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double, const Eigen::MatrixXcd&>())
      .def("flatten", &State::flatten)
      .def("get_R", &State::get_R)
      .def("set_R", &State::set_R)
      .def("get_P", &State::get_P)
      .def("set_P", &State::set_P)
      .def("get_psi", &State::get_psi)
      .def("set_psi", &State::set_psi)
      .def("get_rho", &State::get_rho)
      .def("set_rho", &State::set_rho)
      .def("get_mass", &State::get_mass)
      .def("from_unstructured", &State::from_unstructured)
      .def("rk4_step_inplace", &State::rk4_step_inplace)
      .def("zeros_like", &State::zeros_like)
      .def("get_state_type", &State::get_state_type)
      .def("get_quantum_representation", &State::get_representation)
      .def("get_variables", [](const State& state){
        if (state.get_representation() == QuantumStateRepresentation::NONE) {
          return py::make_tuple(state.get_R(), state.get_P(), py::none());
        } else if (state.get_representation() == QuantumStateRepresentation::WAVE_FUNCTION) {
          if (state.get_state_type() == StateType::MQC) {
            return py::make_tuple(state.get_R(), state.get_P(), state.get_psi());
          } else {
            return py::make_tuple(py::none(), py::none(), state.get_psi());
          }
        } else {
          if (state.get_state_type() == StateType::MQC) {
            return py::make_tuple(state.get_R(), state.get_P(), state.get_rho());
          } else {
            return py::make_tuple(py::none(), py::none(), state.get_rho());
        }
      }})
      // static methods
      .def_static("axpy", &State::axpy)
      .def_static("rk4_step", &State::rk4_step)
      // __repr__ method
      .def("__repr__", &State::__repr__);

    // The binding code for QuantumStateRepresentation enum
    py::enum_<QuantumStateRepresentation>(m_states, "QuantumStateRepresentation")
        .value("NONE", QuantumStateRepresentation::NONE)
        .value("WAVE_FUNCTION", QuantumStateRepresentation::WAVE_FUNCTION)
        .value("DENSITY_MATRIX", QuantumStateRepresentation::DENSITY_MATRIX);

    // The binding code for StateType enum
    py::enum_<StateType>(m_states, "StateType")
        .value("CLASSICAL", StateType::CLASSICAL)
        .value("QUANTUM", StateType::QUANTUM)
        .value("MQC", StateType::MQC);
}