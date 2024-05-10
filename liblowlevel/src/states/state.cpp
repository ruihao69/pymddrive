// third-party includes
#include <Eigen/Dense>

// local includes
#include "states/state.h"
#include "states/state_utils.h"

// c++ includes
#include <complex>
#include <exception>
#include <string>

namespace rhbi {
StateData StateData::zeros_like() const {
  StateData state_data;
  state_data.mass = 0.0;
  state_data.R = Eigen::VectorXd::Zero(R.size());
  state_data.P = Eigen::VectorXd::Zero(P.size());
  state_data.psi = Eigen::VectorXcd::Zero(psi.size());
  state_data.rho = RowMatrixXcd::Zero(rho.rows(), rho.cols());
  Tensor3cd delta_R_zero(delta_R.dimension(0), delta_R.dimension(1), delta_R.dimension(2));
  delta_R_zero.setZero();
  Tensor3cd delta_P_zero(delta_P.dimension(0), delta_P.dimension(1), delta_P.dimension(2));
  delta_P_zero.setZero();
  state_data.delta_R = delta_R_zero;
  state_data.delta_P = delta_P_zero;
  return state_data;
}

State::State(
  Eigen::Ref<const Eigen::VectorXd> R,  // R are the coordinates of the vibrational mode
  Eigen::Ref<const Eigen::VectorXd> P,  // P are the momenta of the vibrational mode
  double mass                //
    ) : flatten_view(Eigen::VectorXcd::Zero(R.size() + P.size())) {
  state_data.R = R;
  state_data.P = P;
  state_data.mass = mass;
  state_type = StateType::CLASSICAL;
  representation = QuantumStateRepresentation::NONE;
}

State::State(
    Eigen::Ref<const Eigen::VectorXcd> psi  //
    ) : flatten_view(Eigen::VectorXcd::Zero(psi.size())) {
  state_data.psi = psi;
  state_type = StateType::QUANTUM;
  representation = QuantumStateRepresentation::WAVE_FUNCTION;
}

State::State(
    Eigen::Ref<const RowMatrixXcd> rho  //
    ) : flatten_view(Eigen::VectorXcd::Zero(rho.size())) {
  state_data.rho = rho;
  state_type = StateType::QUANTUM;
  representation = QuantumStateRepresentation::DENSITY_MATRIX;
}

State::State(
      Eigen::Ref<const Eigen::VectorXd> R,    // R are the coordinates of the vibrational mode
      Eigen::Ref<const Eigen::VectorXd> P,    // P are the momenta of the vibrational mode
      double mass,                 //
      Eigen::Ref<const Eigen::VectorXcd> psi  //
    ) : flatten_view(Eigen::VectorXcd::Zero(R.size() + P.size() + psi.size())) {
  state_data.R = R;
  state_data.P = P;
  state_data.mass = mass;
  state_data.psi = psi;
  state_type = StateType::MQC;
  representation = QuantumStateRepresentation::WAVE_FUNCTION;
}

State::State(
      Eigen::Ref<const Eigen::VectorXd> R,    // R are the coordinates of the vibrational mode
      Eigen::Ref<const Eigen::VectorXd> P,    // P are the momenta of the vibrational mode
      double mass,                 //
      Eigen::Ref<const RowMatrixXcd> rho  //
    ) : flatten_view(Eigen::VectorXcd::Zero(R.size() + P.size() + rho.size())) {
  state_data.R = R;
  state_data.P = P;
  state_data.mass = mass;
  state_data.rho = rho;
  state_type = StateType::MQC;
  representation = QuantumStateRepresentation::DENSITY_MATRIX;
}

State::State(
  Eigen::Ref<const Eigen::VectorXd> R,  // R are the coordinates of the vibrational mode
  Eigen::Ref<const Eigen::VectorXd> P,  // P are the momenta of the vibrational mode
  double mass,                //
  Eigen::Ref<const RowMatrixXcd> rho,  // density matrix
  const Tensor3cd& delta_R,  // delta_R for A-FSSH
  const Tensor3cd& delta_P   // delta_P for A-FSSH
) : flatten_view(Eigen::VectorXcd::Zero(R.size() + P.size() + rho.size() + delta_R.size() + delta_P.size())) {
  state_data.R = R;
  state_data.P = P;
  state_data.mass = mass;
  state_data.rho = rho;
  state_data.delta_R = delta_R;
  state_data.delta_P = delta_P;
  state_type = StateType::AFSSH;
  representation = QuantumStateRepresentation::DENSITY_MATRIX;
}


State::State(StateData state_data, QuantumStateRepresentation representation, StateType state_type, const Eigen::VectorXcd& flatten_view)
    : state_data(state_data), representation(representation), state_type(state_type), flatten_view(flatten_view) {}

// State::State(
//     Eigen::Ref<const Eigen::VectorXd> R,    // R are the
//     Eigen::Ref<const Eigen::VectorXd> P,    // P are the momenta of the vibrational mode
//     double mass,                 //
//     Eigen::Ref<const Eigen::VectorXcd> psi, // wave function
//     Eigen::Ref<const RowMatrixXcd> rho // density matrix
//   ) :  flatten_view(R.size() + P.size() + psi.size() + rho.size()) {
//   state_data.R = R;
//   state_data.P = P;
//   state_data.mass = mass;
//   state_data.psi = psi;
//   state_data.rho = rho;
//   }


const Eigen::VectorXcd& State::flatten() {
  using namespace Eigen;

  switch (state_type) {
    case StateType::CLASSICAL:
      concatenate_RP(state_data.R, state_data.P, flatten_view);
      break;

    case StateType::QUANTUM:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        flatten_view = state_data.psi;
      } else if (representation == QuantumStateRepresentation::DENSITY_MATRIX) {
        flatten_view = Map<VectorXcd>(state_data.rho.data(), state_data.rho.size());
      }
      break;

    case StateType::MQC:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        concatenate_RPPsi(state_data.R, state_data.P, state_data.psi, flatten_view);
      } else if (representation == QuantumStateRepresentation::DENSITY_MATRIX) {
        concatenate_RPRho(state_data.R, state_data.P, state_data.rho, flatten_view);
      }
      break;

    case StateType::AFSSH:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        concatenate_RPPsiDeltaRP(state_data.R, state_data.P, state_data.psi, state_data.delta_R, state_data.delta_P, flatten_view);
      } else if (representation == QuantumStateRepresentation::DENSITY_MATRIX) {
        concatenate_RPRhoDeltaRP(state_data.R, state_data.P, state_data.rho, state_data.delta_R, state_data.delta_P, flatten_view);
      }
      break;
  }

  return flatten_view;
}

// non-static factory methods #1: from an flattened array
State State::from_unstructured(Eigen::Ref<const Eigen::VectorXcd> unstructured_array) const {
  // copy construct the state data
  StateData new_state_data = state_data;

  switch(state_type) {
    case StateType::CLASSICAL:
      flat_to_RP(unstructured_array, new_state_data.R, new_state_data.P);
      break;

    case StateType::QUANTUM:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        new_state_data.psi = unstructured_array;
      } else {
        new_state_data.rho = Eigen::Map<const RowMatrixXcd>(unstructured_array.data(), new_state_data.rho.rows(), new_state_data.rho.cols());
      }
      break;

    case StateType::MQC:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        flat_to_RPPsi(unstructured_array, new_state_data.R, new_state_data.P, new_state_data.psi);
      } else {
        flat_to_RPRho(unstructured_array, new_state_data.R, new_state_data.P, new_state_data.rho);
      }
      break;

    case StateType::AFSSH:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        flat_to_RPPsiDeltaRP(unstructured_array, new_state_data.R, new_state_data.P, new_state_data.psi, new_state_data.delta_R, new_state_data.delta_P);
      } else {
        flat_to_RPRhoDeltaRP(unstructured_array, new_state_data.R, new_state_data.P, new_state_data.rho, new_state_data.delta_R, new_state_data.delta_P);
      }
      break;
  }
  return State(new_state_data, representation, state_type, flatten_view);
}

// inplace rk4_step
void State::rk4_step_inplace(double dt, const State& k1, const State& k2, const State& k3, const State& k4) {
  switch (state_type) {
    case StateType::CLASSICAL:
      state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
      state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
      break;

    case StateType::QUANTUM:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        state_data.psi += dt / 6.0 * (k1.get_psi() + 2.0 * k2.get_psi() + 2.0 * k3.get_psi() + k4.get_psi());
      } else {
        state_data.rho += dt / 6.0 * (k1.get_rho() + 2.0 * k2.get_rho() + 2.0 * k3.get_rho() + k4.get_rho());
      }
      break;

    case StateType::MQC:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
        state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
        state_data.psi += dt / 6.0 * (k1.get_psi() + 2.0 * k2.get_psi() + 2.0 * k3.get_psi() + k4.get_psi());
      } else {
        state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
        state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
        state_data.rho += dt / 6.0 * (k1.get_rho() + 2.0 * k2.get_rho() + 2.0 * k3.get_rho() + k4.get_rho());
      }
      break;

    case StateType::AFSSH:
      if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
        state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
        state_data.psi += dt / 6.0 * (k1.get_psi() + 2.0 * k2.get_psi() + 2.0 * k3.get_psi() + k4.get_psi());
        state_data.delta_R += dt / 6.0 * (k1.get_delta_R() + 2.0 * k2.get_delta_R() + 2.0 * k3.get_delta_R() + k4.get_delta_R());
        state_data.delta_P += dt / 6.0 * (k1.get_delta_P() + 2.0 * k2.get_delta_P() + 2.0 * k3.get_delta_P() + k4.get_delta_P());
      } else {
        state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
        state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
        state_data.rho += dt / 6.0 * (k1.get_rho() + 2.0 * k2.get_rho() + 2.0 * k3.get_rho() + k4.get_rho());
        state_data.delta_R += dt / 6.0 * (k1.get_delta_R() + 2.0 * k2.get_delta_R() + 2.0 * k3.get_delta_R() + k4.get_delta_R());
        state_data.delta_P += dt / 6.0 * (k1.get_delta_P() + 2.0 * k2.get_delta_P() + 2.0 * k3.get_delta_P() + k4.get_delta_P());
      }
      break; 
  }
}

// zeros_like method
State State::zeros_like() const{
  StateData new_state_data = state_data.zeros_like();
  return State(new_state_data, representation, state_type, flatten_view);
}

// static factory methods #1: axpy
State State::axpy(double a, const State& x, const State& y) {
  // assertion: x and y have the same state type. otherwise, throw an exception
  const auto x_state_type = x.get_state_type();
  const auto y_state_type = y.get_state_type();
  const auto x_representation = x.get_representation();
  const auto y_representation = y.get_representation();

  if ((x_state_type != y_state_type) or (x_representation != y_representation)) {
    throw std::invalid_argument("x and y must have the same state type and representation");
  }

  const StateData x_state_data = x.get_state_data();
  StateData new_state_data = y.get_state_data();

  switch (x_state_type) {
    case StateType::CLASSICAL:
      new_state_data.R += a * x_state_data.R;
      new_state_data.P += a * x_state_data.P;
      break;

    case StateType::QUANTUM:
      if (x_representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        new_state_data.psi += a * x_state_data.psi;
      } else {
        new_state_data.rho += a * x_state_data.rho;
      }
      break;

    case StateType::MQC:
      if (x_representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        new_state_data.R += a * x_state_data.R;
        new_state_data.P += a * x_state_data.P;
        new_state_data.psi += a * x_state_data.psi;
      } else {
        new_state_data.R += a * x_state_data.R;
        new_state_data.P += a * x_state_data.P;
        new_state_data.rho += a * x_state_data.rho;
      }
      break;

    case StateType::AFSSH:
      if (x_representation == QuantumStateRepresentation::WAVE_FUNCTION) {
        new_state_data.R += a * x_state_data.R;
        new_state_data.P += a * x_state_data.P;
        new_state_data.psi += a * x_state_data.psi;
        new_state_data.delta_R += a * x_state_data.delta_R;
        new_state_data.delta_P += a * x_state_data.delta_P;
      } else {
        new_state_data.R += a * x_state_data.R;
        new_state_data.P += a * x_state_data.P;
        new_state_data.rho += a * x_state_data.rho;
        new_state_data.delta_R += a * x_state_data.delta_R;
        new_state_data.delta_P += a * x_state_data.delta_P;
      }
      break;
  }

  return State(new_state_data, x_representation, x_state_type, y.get_flatten_view());
}

// static factory methods #2: runge-kutta 4th order
State State::rk4_step(double dt, const State& x, const State& k1, const State& k2, const State& k3, const State& k4) {
  State new_state = x;
  new_state.rk4_step_inplace(dt, k1, k2, k3, k4);
  return new_state;
}

// __repr__ method
std::string State::__repr__() const {
  std::string state_str = "";
  int dim;
  std::string quantum_str;

  switch(state_type) {
    case StateType::CLASSICAL:
      state_str += "CLASSICAL_State(";
      state_str += "R.shape=" + std::to_string(state_data.R.size()) + ", ";
      state_str += "P.shape= " + std::to_string(state_data.P.size()) + ", ";
      state_str += "mass=" + std::to_string(state_data.mass) + ")\n";
      break;

    case StateType::QUANTUM:
      dim = representation == QuantumStateRepresentation::WAVE_FUNCTION ? state_data.psi.size() : state_data.rho.cols();
      quantum_str = representation == QuantumStateRepresentation::WAVE_FUNCTION ? "WAVE_FUNCTION" : "DENSITY_MATRIX";
      state_str += "QUANTUM_State("; 
      state_str += "state_representation=" + quantum_str + ", ";
      state_str += "dim=" + std::to_string(dim) + ")\n";
      break;

    case StateType::MQC:
      dim = representation == QuantumStateRepresentation::WAVE_FUNCTION ? state_data.psi.size() : state_data.rho.cols();
      quantum_str = representation == QuantumStateRepresentation::WAVE_FUNCTION ? "WAVE_FUNCTION" : "DENSITY_MATRIX";
      state_str += "MQC_State("; ;
      state_str += "state_representation=" + quantum_str + ", ";
      state_str += "dim=" + std::to_string(dim) + ", ";
      state_str += "R.shape=" + std::to_string(state_data.R.size()) + ", ";
      state_str += "P.shape=" + std::to_string(state_data.P.size()) + ")\n";
      break;

    case StateType::AFSSH:
      dim = representation == QuantumStateRepresentation::WAVE_FUNCTION ? state_data.psi.size() : state_data.rho.cols();
      quantum_str = representation == QuantumStateRepresentation::WAVE_FUNCTION ? "WAVE_FUNCTION" : "DENSITY_MATRIX";
      state_str += "AFSSH_State("; ;
      state_str += "state_representation=" + quantum_str + ", ";
      state_str += "dim=" + std::to_string(dim) + ", ";
      state_str += "R.shape=" + std::to_string(state_data.R.size()) + ", ";
      state_str += "P.shape=" + std::to_string(state_data.P.size()) + ")\n";
      break;
  }
  return state_str;
}

}  // namespace rhbi
