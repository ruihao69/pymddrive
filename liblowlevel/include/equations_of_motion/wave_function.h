#pragma once
// Third party includes
#include <Eigen/Dense>

// local includes
#include "constants.h"
#include "equations_of_motion/math_utils.h"
#include "row_major_types.h"

// c++ includes
#include <complex>

namespace rhbi {
template <typename H_t>
void _eom_wave_function_diabatic(
    Eigen::Ref<const H_t> H,                 // Hamiltonian
    Eigen::Ref<const Eigen::VectorXcd> psi,  // Wave function
    Eigen::Ref<Eigen::VectorXcd> d_psi_dt    // Time derivative of wave function
) {
  d_psi_dt.noalias() = -constants::IM * (H * psi);
}

template <typename H_t>
Eigen::VectorXcd eom_wave_function_diabatic(
    Eigen::Ref<const H_t> H,                // Hamiltonian
    Eigen::Ref<const Eigen::VectorXcd> psi  // Wave function
) {
  Eigen::VectorXcd d_psi_dt(H.cols());
  _eom_wave_function_diabatic(H, psi, d_psi_dt);
  return d_psi_dt;
}

template <typename dc_t>
void _eom_wave_function_adiabatic(
    Eigen::Ref<const Eigen::VectorXd> E,           // Energy levels
    Eigen::Ref<const dc_t> v_dot_d,          // velocity dot nonadiabatic coupling
    Eigen::Ref<const Eigen::VectorXcd> psi,  // Wave function
    Eigen::Ref<Eigen::VectorXcd> d_psi_dt    // Time derivative of wave function
) {
  // d_psi_dt.array() = -constants::IM * E.array() * psi.array();
  d_psi_dt.noalias() += -constants::IM * psi.cwiseProduct(E);
  d_psi_dt.noalias() += -v_dot_d * psi;
}

template <typename dc_t>
Eigen::VectorXcd eom_wave_function_adiabatic(
    Eigen::Ref<const Eigen::VectorXd> E,          // Energy levels
    Eigen::Ref<const dc_t> v_dot_d,         // velocity dot nonadiabatic coupling
    Eigen::Ref<const Eigen::VectorXcd> psi  // Wave function
) {
  Eigen::VectorXcd d_psi_dt = Eigen::VectorXcd::Zero(E.size());
  _eom_wave_function_adiabatic(E, v_dot_d, psi, d_psi_dt);
  return d_psi_dt;
}
}  // namespace rhbi