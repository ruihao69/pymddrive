#include <complex>

#include "states/expected_values.h"

namespace rhbi {
template <typename ComplexScalar>
ComplexScalar rho_ij(const ComplexScalar& psi_i, const ComplexScalar& psi_j) {
  return psi_i * std::conj(psi_j);
}

template <typename ComplexScalar>
double rho_ij_real(const ComplexScalar& psi_i, const ComplexScalar& psi_j) {
  return psi_i.real() * psi_j.real() + psi_i.imag() * psi_j.imag();
}

template <typename ComplexScalar>
double rho_ij_imag(const ComplexScalar& psi_i, const ComplexScalar& psi_j) {
  return psi_i.imag() * psi_j.real() - psi_i.real() * psi_j.imag();
}

template <typename TENSOR_OP_t, typename State_t>
Eigen::VectorXd ehrenfest_meanF_diabatic(
    const TENSOR_OP_t& dHdR,              // nuclear gradient of the Hamiltonian
    Eigen::Ref<const State_t> rho_or_psi  // density matrix or wavefunction
) {
  return -get_expected_value(dHdR, rho_or_psi);
}

template <typename TENSOR_OP_t>
Eigen::VectorXd ehrenfest_meanF_adiabatic(
    Eigen::Ref<const RowMatrixXd> F,             // forces on PESs
    Eigen::Ref<const Eigen::VectorXd> eig_vals,  // eigenvalues of the Hamiltonian
    const TENSOR_OP_t& dc,                       // non-adiabatic coupling
    Eigen::Ref<const RowMatrixXcd> rho           // density matrix
) {
  Eigen::VectorXd meanF = Eigen::VectorXd::Zero(F.cols());
  for (int kk = 0; kk < F.cols(); ++kk) {
    for (int ii = 0; ii < F.rows(); ++ii) {
      meanF(kk) += rho(ii, ii).real() * F(ii, kk);
      for (int jj = ii + 1; jj < F.rows(); ++jj) {
        const double dE = eig_vals(jj) - eig_vals(ii);
        const std::complex<double> rho_ij_dc_ji = rho(ii, jj) * dc(jj, ii, kk);
        meanF(kk) += 2.0 * dE * rho_ij_dc_ji.real();
      }
    }
  }
  return meanF;
}

template <typename TENSOR_OP_t>
Eigen::VectorXd ehrenfest_meanF_adiabatic(
    Eigen::Ref<const RowMatrixXd> F,             // forces on PESs
    Eigen::Ref<const Eigen::VectorXd> eig_vals,  // eigenvalues of the Hamiltonian
    const TENSOR_OP_t& dc,                       // non-adiabatic coupling
    Eigen::Ref<const Eigen::VectorXcd> psi       // wavefunction
) {
  Eigen::VectorXd meanF = Eigen::VectorXd::Zero(F.cols());
  for (int kk = 0; kk < F.cols(); ++kk) {
    for (int ii = 0; ii < F.rows(); ++ii) {
      meanF(kk) += rho_ij_real(psi(ii), psi(ii)) * F(ii, kk);
      for (int jj = ii + 1; jj < F.rows(); ++jj) {
        const double dE = eig_vals(jj) - eig_vals(ii);
        const std::complex<double> rho_ij_dc_ji = rho_ij(psi(jj), psi(ii)) * dc(jj, ii, kk);
        meanF(kk) += 2.0 * dE * rho_ij_dc_ji.real();
      }
    }
  }

  return meanF;
}

}  // namespace rhbi