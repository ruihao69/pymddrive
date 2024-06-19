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
void _eom_density_matrix_diabatic(
    Eigen::Ref<const H_t>& H,                // Hamiltonian
    Eigen::Ref<const RowMatrixXcd> rho,      // Density matrix
    Eigen::Ref<RowMatrixXcd> drho_dt         // Time derivative of density matrix 
) {
  drho_dt.setZero();
  commutator(H, rho, drho_dt, -constants::IM);
}

template <typename H_t>
RowMatrixXcd eom_density_matrix_diabatic(
  Eigen::Ref<const H_t>& H,                // Hamiltonian
  Eigen::Ref<const RowMatrixXcd> rho  // Density matrix
) {
  RowMatrixXcd drho_dt(H.rows(), H.cols());
  _eom_density_matrix_diabatic(H, rho, drho_dt);
  return drho_dt;
}

template <typename dc_t>
void _eom_density_matrix_adiabatic(
  Eigen::Ref<const Eigen::VectorXd> E,     // Energy levels
  Eigen::Ref<const dc_t> v_dot_d,      // velocity dot nonadiabatic coupling
  Eigen::Ref<const RowMatrixXcd> rho,  // Density matrix
  Eigen::Ref<RowMatrixXcd> drho_dt     // Time derivative of density matrix
) {
  drho_dt.setZero();  
  for (int ii = 0; ii < E.size(); ii++) {
    for (int jj = 0; jj < E.size(); jj++) {
      drho_dt(ii, jj) += (ii == jj) ? 0.0 : -constants::IM * (E(ii) - E(jj)) * rho(ii, jj);
    }
  }
  commutator(v_dot_d, rho, drho_dt, -constants::IM);
}

template <typename dc_t>
RowMatrixXcd eom_density_matrix_adiabatic(
  Eigen::Ref<const Eigen::VectorXd> E,     // Energy levels
  Eigen::Ref<const dc_t> v_dot_d,      // velocity dot nonadiabatic coupling
  Eigen::Ref<const RowMatrixXcd> rho  // Density matrix
) {
  RowMatrixXcd drho_dt(E.size(), E.size());
  _eom_density_matrix_adiabatic(E, v_dot_d, rho, drho_dt);
  return drho_dt;
}

}  // namespace rhbi