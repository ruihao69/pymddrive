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
#include "equations_of_motion/equations_of_motion.h"
#include "floquet/floquet.h"
#include "row_major_types.h"
#include "states/state.h"
#include "states/expected_values.h"

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
      .def(py::pickle(
          [](const State& state) {  // __getstate__
            // Return a tuple that contains the necessary data to reconstruct the State object
            return py::make_tuple(state.get_R(), state.get_P(), state.get_psi(), state.get_rho(), state.get_mass(), state.get_state_type(), state.get_representation());
          },
          [](py::tuple tuple) {  // __setstate__
            // Reconstruct the State object from the tuple
            State state(tuple[0].cast<Eigen::VectorXd>(), tuple[1].cast<Eigen::VectorXd>(), tuple[4].cast<double>(), tuple[2].cast<Eigen::VectorXcd>(), tuple[3].cast<Eigen::MatrixXcd>());
            state.set_state_type(tuple[5].cast<StateType>());
            state.set_representation(tuple[6].cast<QuantumStateRepresentation>());
            return state;
          }))
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
      .def("get_v", &State::get_v)
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
      .def("get_variables", [](const State& state) {
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
      } })
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

  // real or complex quantum operator and wave_function quantum state
  m_states.def("get_expected_value", [](Eigen::Ref<const RowMatrixXd> O, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return get_expected_value(O, psi);
  });

  m_states.def("get_expected_value", [](Eigen::Ref<const RowMatrixXcd> O, Eigen::Ref<const Eigen::VectorXcd>psi) {
    return get_expected_value(O, psi);
  });

  // real or complex quantum operator and density_matrix quantum state
  m_states.def("get_expected_value", [](Eigen::Ref<const RowMatrixXd> O, Eigen::Ref<const RowMatrixXcd> rho) {
    return get_expected_value(O, rho);
  });

  m_states.def("get_expected_value", [](Eigen::Ref<const RowMatrixXcd> O, Eigen::Ref<const RowMatrixXcd> rho) {
    return get_expected_value(O, rho);
  });

  // real or complex quantum-classical operator and wave_function quantum state
  m_states.def("get_expected_value", [](const Tensor3d& O, const Eigen::Ref<const Eigen::VectorXcd>& psi) {
    return get_expected_value(O, psi);
  });

  m_states.def("get_expected_value", [](const Tensor3cd& O, const Eigen::Ref<const Eigen::VectorXcd>& psi) {
    return get_expected_value(O, psi);
  });

  // real or complex quantum-classical operator and density_matrix quantum state
  m_states.def("get_expected_value", [](const Tensor3d& O, const Eigen::Ref<const RowMatrixXcd>& rho) {
    return get_expected_value(O, rho);
  });

  m_states.def("get_expected_value", [](const Tensor3cd& O, const Eigen::Ref<const RowMatrixXcd>& rho) {
    return get_expected_value(O, rho);
  });




  /****
   * The equations_of_motion submodule
   ****/
  py::module m_eom = m.def_submodule("equations_of_motion");
  m_eom.doc() = "Low-level version of the equations_of_motion module, including the electornic equations of motion for the quantum / quantum classical dynamics simulation.";

  // real hamiltonian (non-adiabatic coupling) and wave_function quantum state
  m_eom.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXd> H, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return diabatic_equations_of_motion(H, psi);
  });

  m_eom.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXd> v_dot_d, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return adiabatic_equations_of_motion(E, v_dot_d, psi);
  });

  // real hamiltonian (non-adiabatic coupling) and density matrix quantum state
  m_eom.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXd> H, Eigen::Ref<const RowMatrixXcd> rho) {
    return diabatic_equations_of_motion(H, rho);
  });

  m_eom.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXd> v_dot_d, Eigen::Ref<const RowMatrixXcd> rho) {
    return adiabatic_equations_of_motion(E, v_dot_d, rho);
  });

  // complex hamiltonian (non-adiabatic coupling) and wave_function quantum state
  m_eom.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXcd> H, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return diabatic_equations_of_motion(H, psi);
  });

  m_eom.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXcd> v_dot_d, Eigen::Ref<const Eigen::VectorXcd> psi) {
    return adiabatic_equations_of_motion(E, v_dot_d, psi);
  });

  // complex hamiltonian (non-adiabatic coupling) and density matrix quantum state
  m_eom.def("diabatic_equations_of_motion", [](Eigen::Ref<const RowMatrixXcd> H, Eigen::Ref<const RowMatrixXcd> rho) {
    return diabatic_equations_of_motion(H, rho);
  });
  m_eom.def("adiabatic_equations_of_motion", [](Eigen::Ref<const Eigen::VectorXd> E, Eigen::Ref<const RowMatrixXcd> v_dot_d, Eigen::Ref<const RowMatrixXcd> rho) {
    return adiabatic_equations_of_motion(E, v_dot_d, rho);
  });

  /****
   * The floquet submodule
   ****/

  py::module m_floquet = m.def_submodule("floquet", "Low-level version of the floquet module");

  // instantiate for get_HF_cos: complex H0, complex V
  m_floquet.def("get_HF_cos", &get_HF_cos<RowMatrixXcd, RowMatrixXcd>); 

  // instantiate for get_HF_cos: complex H0, real V
  m_floquet.def("get_HF_cos", &get_HF_cos<RowMatrixXcd, RowMatrixXd>);

  // instantiate for get_HF_cos: real H0, complex V
  m_floquet.def("get_HF_cos", &get_HF_cos<RowMatrixXd, RowMatrixXcd>);

  // instantiate for get_HF_cos: real H0, real V
  m_floquet.def("get_HF_cos", &get_HF_cos<RowMatrixXd, RowMatrixXd>);

  // instantiate for get_dHF_dR_cos: complex dH0_dR, complex dV_dR
  m_floquet.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3cd, Tensor3cd>);

  // instantiate for get_dHF_dR_cos: complex dH0_dR, real dV_dR
  m_floquet.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3cd, Tensor3d>);

  // instantiate for get_dHF_dR_cos: real dH0_dR, complex dV_dR
  m_floquet.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3d, Tensor3cd>);

  // instantiate for get_dHF_dR_cos: real dH0_dR, real dV_dR
  m_floquet.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3d, Tensor3d>);
}