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
    ) : flatten_view(psi.size()) {
  state_data.psi = psi;
  state_type = StateType::QUANTUM;
  representation = QuantumStateRepresentation::WAVE_FUNCTION;
}

State::State(
    Eigen::Ref<const RowMatrixXcd> rho  //
    ) : flatten_view(rho.size()) {
  state_data.rho = rho;
  state_type = StateType::QUANTUM;
  representation = QuantumStateRepresentation::DENSITY_MATRIX;
}

State::State(
      Eigen::Ref<const Eigen::VectorXd> R,    // R are the coordinates of the vibrational mode
      Eigen::Ref<const Eigen::VectorXd> P,    // P are the momenta of the vibrational mode
      double mass,                 //
      Eigen::Ref<const Eigen::VectorXcd> psi  //
    ) : flatten_view(R.size() + P.size() + psi.size()) {
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
    ) : flatten_view(R.size() + P.size() + rho.size()) {
  state_data.R = R;
  state_data.P = P;
  state_data.mass = mass;
  state_data.rho = rho;
  state_type = StateType::MQC;
  representation = QuantumStateRepresentation::DENSITY_MATRIX;
}

State::State(StateData state_data, QuantumStateRepresentation representation, StateType state_type, const Eigen::VectorXcd& flatten_view)
    : state_data(state_data), representation(representation), state_type(state_type), flatten_view(flatten_view) {}

State::State(
    Eigen::Ref<const Eigen::VectorXd> R,    // R are the
    Eigen::Ref<const Eigen::VectorXd> P,    // P are the momenta of the vibrational mode
    double mass,                 //
    Eigen::Ref<const Eigen::VectorXcd> psi, // wave function
    Eigen::Ref<const RowMatrixXcd> rho // density matrix
  ) :  flatten_view(R.size() + P.size() + psi.size() + rho.size()) {
  state_data.R = R;
  state_data.P = P;
  state_data.mass = mass;
  state_data.psi = psi;
  state_data.rho = rho;
  }


const Eigen::VectorXcd& State::flatten() {
  if (state_type == StateType::CLASSICAL) {
    concatenate_RP(state_data.R, state_data.P, flatten_view);
  } else if (state_type == StateType::QUANTUM) {
    if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      flatten_view = state_data.psi;
    } else if (representation == QuantumStateRepresentation::DENSITY_MATRIX) {
      flatten_view = Eigen::Map<Eigen::VectorXcd>(state_data.rho.data(), state_data.rho.size());
    }
  } else if (state_type == StateType::MQC) {
    if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      concatenate_RPPsi(state_data.R, state_data.P, state_data.psi, flatten_view);
    } else if (representation == QuantumStateRepresentation::DENSITY_MATRIX) {
      concatenate_RPRho(state_data.R, state_data.P, state_data.rho, flatten_view);
    }
  }
  return flatten_view;
}

// non-static factory methods #1: from an flattened array
State State::from_unstructured(Eigen::Ref<const Eigen::VectorXcd> ) const {
  // copy construct the state data
  StateData new_state_data = state_data;

  if (state_type == StateType::CLASSICAL) {
    flat_to_RP(flatten_view, new_state_data.R, new_state_data.P);
  } else if (state_type == StateType::QUANTUM) {
    if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      new_state_data.psi = flatten_view;
    } else {
      new_state_data.rho = Eigen::Map<const RowMatrixXcd>(flatten_view.data(), new_state_data.rho.rows(), new_state_data.rho.cols());
    }
  } else {
    if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      flat_to_RPPsi(flatten_view, new_state_data.R, new_state_data.P, new_state_data.psi);
    } else {
      flat_to_RPRho(flatten_view, new_state_data.R, new_state_data.P, new_state_data.rho);
    }
  }
  return State(new_state_data, representation, state_type, flatten_view);
}

// inplace rk4_step
void State::rk4_step_inplace(double dt, const State& k1, const State& k2, const State& k3, const State& k4) {
  if (state_type == StateType::CLASSICAL) {
    state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
    state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
  } else if (state_type == StateType::QUANTUM) {
    if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      state_data.psi += dt / 6.0 * (k1.get_psi() + 2.0 * k2.get_psi() + 2.0 * k3.get_psi() + k4.get_psi());
    } else {
      state_data.rho += dt / 6.0 * (k1.get_rho() + 2.0 * k2.get_rho() + 2.0 * k3.get_rho() + k4.get_rho());
    }
  } else {
    if (representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
      state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
      state_data.psi += dt / 6.0 * (k1.get_psi() + 2.0 * k2.get_psi() + 2.0 * k3.get_psi() + k4.get_psi());
    } else {
      state_data.R += dt / 6.0 * (k1.get_R() + 2.0 * k2.get_R() + 2.0 * k3.get_R() + k4.get_R());
      state_data.P += dt / 6.0 * (k1.get_P() + 2.0 * k2.get_P() + 2.0 * k3.get_P() + k4.get_P());
      state_data.rho += dt / 6.0 * (k1.get_rho() + 2.0 * k2.get_rho() + 2.0 * k3.get_rho() + k4.get_rho());
    }
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

  if (x_state_type == StateType::CLASSICAL) {
    new_state_data.R += a * x_state_data.R;
    new_state_data.P += a * x_state_data.P;
  } else if (x_state_type == StateType::QUANTUM) {
    if (x_representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      new_state_data.psi += a * x_state_data.psi;
    } else {
      new_state_data.rho += a * x_state_data.rho;
    }
  } else {
    if (x_representation == QuantumStateRepresentation::WAVE_FUNCTION) {
      new_state_data.R += a * x_state_data.R;
      new_state_data.P += a * x_state_data.P;
      new_state_data.psi += a * x_state_data.psi;
    } else {
      new_state_data.R += a * x_state_data.R;
      new_state_data.P += a * x_state_data.P;
      new_state_data.rho += a * x_state_data.rho;
    }
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
  if (state_type == StateType::CLASSICAL) {
    state_str += "CLASSICAL_State(";
    state_str += "R.shape=" + std::to_string(state_data.R.size()) + ", ";
    state_str += "P.shape= " + std::to_string(state_data.P.size()) + ", ";
    state_str += "mass=" + std::to_string(state_data.mass) + ")\n";
  } else {
    int dim = representation == QuantumStateRepresentation::WAVE_FUNCTION ? state_data.psi.size() : state_data.rho.size();
    std::string quantum_str = representation == QuantumStateRepresentation::WAVE_FUNCTION ? "WAVE_FUNCTION" : "DENSITY_MATRIX";
    if (state_type == StateType::QUANTUM) {
      state_str += "QUANTUM_State("; 
      state_str += "state_representation=" + quantum_str + ", ";
      state_str += "dim=" + std::to_string(dim) + ")\n";
    } else {
      state_str += "MQC_State("; ;
      state_str += "state_representation=" + quantum_str + ", ";
      state_str += "dim=" + std::to_string(dim) + ", ";
      state_str += "R.shape=" + std::to_string(state_data.R.size()) + ", ";
      state_str += "P.shape=" + std::to_string(state_data.P.size()) + ")\n";
    }
  }
  return state_str;
}

}  // namespace rhbi
