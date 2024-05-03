// Third-party includes
#include <Eigen/Dense>
#include "constants.h"

// local includes

namespace rhbi {
// function implementations
size_t dim_to_dimF(size_t dim, size_t NF) {
  return (2 * NF + 1) * dim;
}

size_t map_floquet_index_to_block_index(size_t n, size_t NF) {
  return n + NF;
}

// template function implementations
template <typename HF_t, typename H_t>
void fill_HF_diagonal(
    Eigen::Ref<HF_t> HF,       // the Floquet Hamiltonian
    Eigen::Ref<const H_t> H0,  // the Hamiltonian
    double Omega,              // the driving frequency
    size_t NF                  // the Floquet level cutoff
) {
  const size_t dim = H0.rows();
  const int iNF = static_cast<int>(NF);
  for (int n = -iNF; n <= iNF; ++n) {
    const size_t nF = map_floquet_index_to_block_index(n, NF);
    HF.block(nF * dim, nF * dim, dim, dim) = H0;
    HF.block(nF * dim, nF * dim, dim, dim) += n * Omega * Eigen::MatrixXd::Identity(dim, dim);
  }
}

template <typename HF_t, typename V_t>
void fill_HF_offdiagonal_cosine(
    Eigen::Ref<HF_t> HF,      // the Floquet Hamiltonian
    Eigen::Ref<const V_t> V,  // the Floquet perturbation (upper triangular part)
    size_t NF                 // the Floquet level cutoff
) {
  const size_t dim = V.rows();
  for (size_t block_index = 0; block_index < 2 * NF; ++block_index) {
    // fill the upper triangular part of the block
    HF.block(block_index * dim, (block_index + 1) * dim, dim, dim) = V;
    // fill the lower triangular part of the block
    HF.block((block_index + 1) * dim, block_index * dim, dim, dim) = V.adjoint();
  }
}

template <typename V_t>
void fill_HF_offdiagonal_sine(
  Eigen::Ref<RowMatrixXcd> HF,  // the Floquet Hamiltonian
  Eigen::Ref<const V_t> V,      // the Floquet perturbation (upper triangular part)
  size_t NF                     // the Floquet level cutoff
) {
  const size_t dim = V.rows();
  for (size_t block_index = 0; block_index < 2 * NF; ++block_index) {
    // fill the upper triangular part of the block
    HF.block(block_index * dim, (block_index + 1) * dim, dim, dim) = V / constants::IM;
    // fill the lower triangular part of the block
    HF.block((block_index + 1) * dim, block_index * dim, dim, dim) = V.adjoint() / (-constants::IM);
  }
}

template <typename H_t, typename V_t>
HFReturnType<H_t, V_t> get_HF_cos(Eigen::Ref<const H_t> H0, Eigen::Ref<const V_t> V, double Omega, size_t NF) {
  const size_t dim = H0.rows();
  const size_t dimF = dim_to_dimF(dim, NF);

  // Allocate the return matrix based on the ReturnType alias
  HFReturnType<H_t, V_t> HF = HFReturnType<H_t, V_t>::Zero(dimF, dimF);

  // fill the diagonal blocks
  fill_HF_diagonal<HFReturnType<H_t, V_t>, H_t>(HF, H0, Omega, NF);

  // fill the off-diagonal blocks
  fill_HF_offdiagonal_cosine<HFReturnType<H_t, V_t>, V_t>(HF, V, NF);

  return HF;
}

template <typename H_t, typename V_t>
RowMatrixXcd get_HF_sin(Eigen::Ref<const H_t> H0, Eigen::Ref<const V_t> V, double Omega, size_t NF) {
  const size_t dim = H0.rows();
  const size_t dimF = dim_to_dimF(dim, NF);

  // Allocate the return matrix based on the ReturnType alias
  RowMatrixXcd HF = RowMatrixXcd::Zero(dimF, dimF);

  // fill the diagonal blocks
  fill_HF_diagonal<RowMatrixXcd, H_t>(HF, H0, Omega, NF);

  // fill the off-diagonal blocks
  fill_HF_offdiagonal_sine<V_t>(HF, V, NF);

  return HF;
}


template <typename dHF_dR_t, typename dH0_dR_t>
void fill_dHF_dR_diagonal(
    // Eigen::Ref<dHF_dR_t> dHF_dR,        // the Floquet forces tensor
    // Eigen::Ref<const dH0_dR_t> dH0_dR,  // the forces tensor
    dHF_dR_t &dHF_dR,
    const dH0_dR_t &dH0_dR,
    size_t NF  // the Floquet level cutoff
) {
  // the dimension of the tensor is (dim_electronic, dim_electronic, dim_nuclear)
  const int dim_electronic = dH0_dR.dimension(1);
  const int dime_nuclear = dH0_dR.dimension(2);
  const int iNF = static_cast<int>(NF);

  // fill the diagonal blocks (naive implementation)
  for (int n = -iNF; n <= iNF; ++n) {
    for (int i = 0; i < dim_electronic; ++i) {
      for (int j = 0; j < dim_electronic; ++j) {
        for (int k = 0; k < dime_nuclear; ++k) {
          const size_t nF = map_floquet_index_to_block_index(n, NF);
          dHF_dR(nF * dim_electronic + i, nF * dim_electronic + j, k) = dH0_dR(i, j, k);
        }
      }
    }
  }
}

template <typename dHF_dR_t, typename dV_dR_t>
void fill_dHF_dR_offdiagonal_cosine(
    // Eigen::Ref<dHF_dR_t> dHF_dR,      // the Floquet forces tensor
    // Eigen::Ref<const dV_dR_t> dV_dR,  // the Floquet perturbation forces tensor (upper triangular part)
    dHF_dR_t &dHF_dR,
    const dV_dR_t &dV_dR,
    size_t NF  // the Floquet level cutoff
) {
  // the dimension of the tensor is (dim_electronic, dim_electronic, dim_nuclear)
  const int dim_electronic = dV_dR.dimension(1);
  const int dime_nuclear = dV_dR.dimension(2);

  // temporary conjugate of dV_dR
  dV_dR_t dV_dR_conj = dV_dR.conjugate();

  // fill the off-diagonal blocks (naive implementation)
  for (size_t block_i = 0; block_i < 2 * NF; ++block_i) {
    const size_t block_j = block_i + 1;
    for (int i = 0; i < dim_electronic; ++i) {
      for (int j = 0; j < dim_electronic; ++j) {
        for (int k = 0; k < dime_nuclear; ++k) {
          dHF_dR(block_i * dim_electronic + i, block_j * dim_electronic + j, k) = dV_dR(i, j, k);
          dHF_dR(block_j * dim_electronic + j, block_i * dim_electronic + i, k) = dV_dR_conj(i, j, k);
        }
      }
    }
  }
}

template <typename dV_dR_t>
void fill_dHF_dR_offdiagonal_sine(
  // Eigen::Ref<dHF_dR_t> dHF_dR,      // the Floquet forces tensor
  // Eigen::Ref<const dV_dR_t> dV_dR,  // the Floquet perturbation forces tensor (upper triangular part)
  Tensor3cd& dHF_dR,
  const dV_dR_t& dV_dR,
  size_t NF  // the Floquet level cutoff
) {
  // the dimension of the tensor is (dim_electronic, dim_electronic, dim_nuclear)
  const int dim_electronic = dV_dR.dimension(1);
  const int dime_nuclear = dV_dR.dimension(2);

  // temporary conjugate of dV_dR
  dV_dR_t dV_dR_conj = dV_dR.conjugate();

  // fill the off-diagonal blocks (naive implementation)
  for (size_t block_i = 0; block_i < 2 * NF; ++block_i) {
    const size_t block_j = block_i + 1;
    for (int i = 0; i < dim_electronic; ++i) {
      for (int j = 0; j < dim_electronic; ++j) {
        for (int k = 0; k < dime_nuclear; ++k) {
          dHF_dR(block_i * dim_electronic + i, block_j * dim_electronic + j, k) = dV_dR(i, j, k) / constants::IM;
          dHF_dR(block_j * dim_electronic + j, block_i * dim_electronic + i, k) = dV_dR_conj(i, j, k) / (-constants::IM);
        }
      }
    }
  }
}

template <typename dH0_dR_t, typename dV_dR_t>
dHF_dRReturnType<dH0_dR_t, dV_dR_t> get_dHF_dR_cos(
    // Eigen::Ref<const dH0_dR_t> dH0_dR,
    // Eigen::Ref<const dV_dR_t> dV_dR,
    const dH0_dR_t &dH0_dR,
    const dV_dR_t &dV_dR,
    size_t NF) {
  // the dimension of the tensor is (dim_electronic, dim_electronic, dim_nuclear)
  const size_t dim_electronic = dH0_dR.dimension(1);
  const size_t dim_nuclear = dH0_dR.dimension(2);
  const size_t dimF = dim_to_dimF(dim_electronic, NF);

  // Allocate the return tensor based on the ReturnType alias, and set zero
  // dHF_dRReturnType<dH0_dR_t, dV_dR_t> dHF_dR = dHF_dRReturnType<dH0_dR_t, dV_dR_t>::Zero(dimF, dimF, dim_nuclear);
  dHF_dRReturnType<dH0_dR_t, dV_dR_t> dHF_dR(dimF, dimF, dim_nuclear);
  dHF_dR.setZero();

  // fill the diagonal blocks
  fill_dHF_dR_diagonal(dHF_dR, dH0_dR, NF);

  // fill the off-diagonal blocks
  fill_dHF_dR_offdiagonal_cosine(dHF_dR, dV_dR, NF);

  return dHF_dR;
}

template <typename dH0_dR_t, typename dV_dR_t>
Tensor3cd get_dHF_dR_sin(
  const dH0_dR_t& dH0_dR,
  const dV_dR_t& dV_dR,
  size_t NF
){
  // the dimension of the tensor is (dim_electronic, dim_electronic, dim_nuclear)
  const size_t dim_electronic = dH0_dR.dimension(1);
  const size_t dim_nuclear = dH0_dR.dimension(2);
  const size_t dimF = dim_to_dimF(dim_electronic, NF);

  // Allocate the return tensor based on the ReturnType alias, and set zero
  Tensor3cd dHF_dR(dimF, dimF, dim_nuclear);
  dHF_dR.setZero();

  // fill the diagonal blocks
  fill_dHF_dR_diagonal(dHF_dR, dH0_dR, NF);

  // fill the off-diagonal blocks
  fill_dHF_dR_offdiagonal_sine(dHF_dR, dV_dR, NF);

  return dHF_dR;
}

}  // namespace rhbi