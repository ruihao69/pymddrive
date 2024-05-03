#pragma once
// Third party libraries
#include <Eigen/Dense>

// local libraries
#include "row_major_types.h"

// Standard libraries
#include <vector>

namespace rhbi {
template <typename ComplexScalar>
ComplexScalar rho_ij(const ComplexScalar& psi_i, const ComplexScalar& psi_j);

template <typename ComplexScalar>
double rho_ij_real(const ComplexScalar& psi_i, const ComplexScalar& psi_j);

template <typename ComplexScalar>
double rho_ij_imag(const ComplexScalar& psi_i, const ComplexScalar& psi_j);

template <typename TENSOR_OP_t, typename State_t>
Eigen::VectorXd ehrenfest_meanF_diabatic(
    const TENSOR_OP_t& dHdR,              // nuclear gradient of the Hamiltonian
    Eigen::Ref<const State_t> rho_or_psi  // density matrix or wavefunction
);

template <typename TENSOR_OP_t, typename F_t>
Eigen::VectorXd ehrenfest_meanF_adiabatic(
    Eigen::Ref<const F_t> F,            // forces on PESs (electronic, nuclear)
    Eigen::Ref<const Eigen::VectorXd> eig_vals,  // eigenvalues of the Hamiltonian
    const TENSOR_OP_t& dc,                       // non-adiabatic coupling
    Eigen::Ref<const Eigen::VectorXcd> psi       // wavefunction
);

template <typename TENSOR_OP_t, typename F_t>
Eigen::VectorXd ehrenfest_meanF_adiabatic(
    Eigen::Ref<const F_t> F,            // forces on PESs (electronic, nuclear)
    Eigen::Ref<const Eigen::VectorXd> eig_vals,  // eigenvalues of the Hamiltonian
    const TENSOR_OP_t& dc,                       // non-adiabatic coupling
    Eigen::Ref<const RowMatrixXcd> rho           // density matrix
);

}  // namespace rhbi

#include "ehrenfest_impl.h"
