#pragma once
// Third-party includes
#include <Eigen/Dense>

// local includes
#include "equations_of_motion/density_matrix.h"
#include "equations_of_motion/wave_function.h"
#include "row_major_types.h"

// C++ includes
#include <complex>

namespace rhbi {
template <typename qm_t, typename H_t>
typename std::enable_if<std::is_same<qm_t, Eigen::VectorXcd>::value, qm_t>::type
diabatic_equations_of_motion(
  Eigen::Ref<const H_t> H,           // Hamiltonian
  Eigen::Ref<const qm_t> psi_or_rho  // Wave function or density matrix
) {
  return eom_wave_function_diabatic(H, psi_or_rho);
}

template <typename qm_t, typename H_t>
typename std::enable_if<std::is_same<qm_t, RowMatrixXcd>::value, qm_t>::type
diabatic_equations_of_motion(
  Eigen::Ref<const H_t> H,           // Hamiltonian
  Eigen::Ref<const qm_t> psi_or_rho  // Wave function or density matrix
) {
  return eom_density_matrix_diabatic(H, psi_or_rho);
}

template <typename qm_t, typename dc_t>
typename std::enable_if<std::is_same<qm_t, Eigen::VectorXcd>::value, qm_t>::type
adiabatic_equations_of_motion(
  Eigen::Ref<const Eigen::VectorXd> E,     // Energy levels
  Eigen::Ref<const dc_t> v_dot_d,      // velocity dot nonadiabatic coupling
  Eigen::Ref<const qm_t> psi_or_rho    // Wave function or density matrix
) {
  return eom_wave_function_adiabatic(E, v_dot_d, psi_or_rho);
}

template <typename qm_t, typename dc_t>
typename std::enable_if<std::is_same<qm_t, RowMatrixXcd>::value, qm_t>::type
adiabatic_equations_of_motion(
  Eigen::Ref<const Eigen::VectorXd> E,     // Energy levels
  Eigen::Ref<const dc_t> v_dot_d,      // velocity dot nonadiabatic coupling
  Eigen::Ref<const qm_t> psi_or_rho    // Wave function or density matrix
) {
  return eom_density_matrix_adiabatic(E, v_dot_d, psi_or_rho);
}

}  // namespace rhbi
