// Third-party includes
#include <Eigen/Dense>

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

}  // namespace rhbi