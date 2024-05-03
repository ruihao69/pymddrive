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
#include "states/expected_values.h"
#include "states/state.h"

namespace py = pybind11;
using namespace rhbi;
// forward declarations
void bind_state(py::module& m);

void bind_expected_values(py::module& m);

void bind_states(py::module& m);

// The implementation of the functions is in the following files

void bind_state(py::module& m) {
    // The binding code for State class
    py::class_<State>(m, "State")
        .def(py::pickle(
            [](const State& state) {  // __getstate__
                // Return a tuple that contains the necessary data to reconstruct the State object
                return py::make_tuple(
                    state.get_mass(),
                    state.get_R(),
                    state.get_P(),
                    state.get_psi(),
                    state.get_rho(),
                    state.get_representation(),
                    state.get_state_type(),
                    state.get_flatten_view()
                );
            },
            [](py::tuple tuple) {  // __setstate__
                // Reconstruct the State object from the tuple
                StateData state_data;
                state_data.mass = tuple[0].cast<double>();
                state_data.R = tuple[1].cast<Eigen::VectorXd>();
                state_data.P = tuple[2].cast<Eigen::VectorXd>();
                state_data.psi = tuple[3].cast<Eigen::VectorXcd>();
                state_data.rho = tuple[4].cast<RowMatrixXcd>();
                return State(
                    state_data,
                    tuple[5].cast<QuantumStateRepresentation>(),
                    tuple[6].cast<StateType>(),
                    tuple[7].cast<Eigen::VectorXcd>()
                );
                // State state(tuple[0].cast<StateData>(), tuple[1].cast<Eigen::VectorXd>(), tuple[4].cast<double>(), tuple[2].cast<Eigen::VectorXcd>(), tuple[3].cast<RowMatrixXcd>());
                // state.set_state_type(tuple[5].cast<StateType>());
                // state.set_representation(tuple[6].cast<QuantumStateRepresentation>());
                // return state;
            }))
            .def(py::init<Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>, double>())
                .def(py::init<Eigen::Ref<const Eigen::VectorXcd>>())
                .def(py::init<Eigen::Ref<const RowMatrixXcd>>())
                .def(py::init<Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>, double, Eigen::Ref<const Eigen::VectorXcd>>())
                .def(py::init<Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>, double, Eigen::Ref<const RowMatrixXcd>>())
                // .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double>())
                // .def(py::init<const Eigen::VectorXcd&>())
                // .def(py::init<const RowMatrixXcd&>())
                // .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double, const Eigen::VectorXcd&>())
                // .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double, const RowMatrixXcd&>())
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
                }
                else if (state.get_representation() == QuantumStateRepresentation::WAVE_FUNCTION) {
                    if (state.get_state_type() == StateType::MQC) {
                        return py::make_tuple(state.get_R(), state.get_P(), state.get_psi());
                    }
                    else {
                        return py::make_tuple(py::none(), py::none(), state.get_psi());
                    }
                }
                else {
                    if (state.get_state_type() == StateType::MQC) {
                        return py::make_tuple(state.get_R(), state.get_P(), state.get_rho());
                    }
                    else {
                        return py::make_tuple(py::none(), py::none(), state.get_rho());
                    }
                } })
                // static methods
                    .def_static("axpy", &State::axpy)
                    .def_static("rk4_step", &State::rk4_step)
                    // __repr__ method
                    .def("__repr__", &State::__repr__);

                // The binding code for QuantumStateRepresentation enum
                py::enum_<QuantumStateRepresentation>(m, "QuantumStateRepresentation")
                    .value("NONE", QuantumStateRepresentation::NONE)
                    .value("WAVE_FUNCTION", QuantumStateRepresentation::WAVE_FUNCTION)
                    .value("DENSITY_MATRIX", QuantumStateRepresentation::DENSITY_MATRIX);

                // The binding code for StateType enum
                py::enum_<StateType>(m, "StateType")
                    .value("CLASSICAL", StateType::CLASSICAL)
                    .value("QUANTUM", StateType::QUANTUM)
                    .value("MQC", StateType::MQC);
}

void bind_expected_values(py::module& m) {
    // real or complex quantum operator and wave_function quantum state
    m.def("get_expected_value", [](Eigen::Ref<const RowMatrixXd> O, Eigen::Ref<const Eigen::VectorXcd> psi) {
        return get_expected_value(O, psi);
        });

    m.def("get_expected_value", [](Eigen::Ref<const RowMatrixXcd> O, Eigen::Ref<const Eigen::VectorXcd> psi) {
        return get_expected_value(O, psi);
        });

    // real or complex quantum operator and density_matrix quantum state
    m.def("get_expected_value", [](Eigen::Ref<const RowMatrixXd> O, Eigen::Ref<const RowMatrixXcd> rho) {
        return get_expected_value(O, rho);
        });

    m.def("get_expected_value", [](Eigen::Ref<const RowMatrixXcd> O, Eigen::Ref<const RowMatrixXcd> rho) {
        return get_expected_value(O, rho);
        });

    // real or complex quantum-classical operator and wave_function quantum state
    m.def("get_expected_value", [](const Tensor3d& O, const Eigen::Ref<const Eigen::VectorXcd>& psi) {
        return get_expected_value(O, psi);
        });

    m.def("get_expected_value", [](const Tensor3cd& O, const Eigen::Ref<const Eigen::VectorXcd>& psi) {
        return get_expected_value(O, psi);
        });

    // real or complex quantum-classical operator and density_matrix quantum state
    m.def("get_expected_value", [](const Tensor3d& O, const Eigen::Ref<const RowMatrixXcd>& rho) {
        return get_expected_value(O, rho);
        });

    m.def("get_expected_value", [](const Tensor3cd& O, const Eigen::Ref<const RowMatrixXcd>& rho) {
        return get_expected_value(O, rho);
        });
}

void bind_states(py::module& m) {
    bind_state(m);
    bind_expected_values(m);
}






