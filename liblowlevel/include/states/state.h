#pragma once
// third-party includes
#include <Eigen/Dense>
#include <complex>
#include <string>

// local includes

namespace rhbi {
enum QuantumStateRepresentation { NONE = 0,
                                  WAVE_FUNCTION = 1,
                                  DENSITY_MATRIX = 2 };
enum StateType { CLASSICAL = 1,
                 QUANTUM = 2,
                 MQC = 3 };

struct StateData {
  double mass;           // mass of the particle
  Eigen::VectorXd R;     // R are the
  Eigen::VectorXd P;     // P are the momenta of the vibrational mode
  Eigen::VectorXcd psi;  // wave function
  Eigen::MatrixXcd rho;  // density matrix
  // StateData() : mass(-1.0), R(Eigen::VectorXd::Zero(0)), P(Eigen::VectorXd::Zero(0)), psi(Eigen::VectorXcd::Zero(0)), rho(Eigen::MatrixXcd::Zero(0, 0)) {}
  StateData zeros_like() const;
};

class State {
 public:
  // Classical constructor
  State(
      const Eigen::VectorXd& R,  // R are the coordinates of the vibrational mode
      const Eigen::VectorXd& P,  // P are the momenta of the vibrational mode
      double mass                //
  );
  // Qunatum constructor #1: wave function
  State(
      const Eigen::VectorXcd& psi  //
  );
  // Quantum constructor #2: density matrix
  State(
      const Eigen::MatrixXcd& rho  //
  );
  // MQC constructor #1: wave function
  State(
      const Eigen::VectorXd& R,    // R are the coordinates of the vibrational mode
      const Eigen::VectorXd& P,    // P are the momenta of the vibrational mode
      double mass,                 //
      const Eigen::VectorXcd& psi  //
  );
  // MQC constructor #2: density matrix
  State(
      const Eigen::VectorXd& R,    // R are the coordinates of the vibrational mode
      const Eigen::VectorXd& P,    // P are the momenta of the vibrational mode
      double mass,                 //
      const Eigen::MatrixXcd& rho  //
  );

  // constructor for pickle
  State(
    const Eigen::VectorXd& R,    // R are the
    const Eigen::VectorXd& P,    // P are the momenta of the vibrational mode
    double mass,                 //
    const Eigen::VectorXcd& psi, // wave function
    const Eigen::MatrixXcd& rho // density matrix
  );

  // straight forward constructor (directly construction from private data)
  State(StateData state_data, QuantumStateRepresentation representation, StateType state_type, const Eigen::VectorXcd& flatten_view);

  // small getters and setters
  double get_mass() const { return state_data.mass; }
  const Eigen::VectorXd& get_R() const { return state_data.R; }
  void set_R(const Eigen::VectorXd& R) { state_data.R = R; }
  const Eigen::VectorXd& get_P() const { return state_data.P; }
  void set_P(const Eigen::VectorXd& P) { state_data.P = P; }
  const Eigen::VectorXcd& get_psi() const { return state_data.psi; }
  Eigen::VectorXd get_v() const { return state_data.P / state_data.mass; }
  void set_psi(const Eigen::VectorXcd& psi) { state_data.psi = psi; }
  const Eigen::MatrixXcd& get_rho() const { return state_data.rho; }
  void set_rho(const Eigen::MatrixXcd& rho) { state_data.rho = rho; }
  const Eigen::VectorXcd& get_flatten_view() const { return flatten_view; }
  StateData get_state_data() const { return state_data; }
  QuantumStateRepresentation get_representation() const { return representation; }
  void set_representation(QuantumStateRepresentation representation) { this->representation = representation; }
  StateType get_state_type() const { return state_type; }
  void set_state_type(StateType state_type) { this->state_type = state_type; }


  // access the state data as an 1D array
  const Eigen::VectorXcd& flatten();

  // non-static factory methods #1: from an flattened array
  State from_unstructured(const Eigen::VectorXcd& flatten_view) const;

  // non-static factory methods #2: rk4_step inplace
  void rk4_step_inplace(double dt, const State& k1, const State& k2, const State& k3, const State& k4);

  // non-static factory methods #3: zeros_like
  State zeros_like() const;

  // static factory methods #1: axpy
  static State axpy(double a, const State& x, const State& y);

  // static factory methods #2: runge-kutta 4th order
  static State rk4_step(double dt, const State& x, const State& k1, const State& k2, const State& k3, const State& k4);

  // __repr__ method
  std::string __repr__() const;

 private:
  StateData state_data;
  QuantumStateRepresentation representation;
  StateType state_type;
  Eigen::VectorXcd flatten_view;
};

}  // namespace rhbi