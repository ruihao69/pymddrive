#pragma once

// Third party includes
#include <Eigen/Dense>

// c++ includes
#include <complex>

namespace rhbi {

template <typename OP1_t, typename OP2_t>
auto commutator(const OP1_t& A, const OP2_t& B) -> decltype(A * B - B * A) {
  return A * B - B * A;
}

template <typename OP1_t, typename OP2_t, typename OP3_t, typename scalar_t>
void commutator(const OP1_t& A, const OP2_t& B, OP3_t& C, scalar_t scalar = 1.0) {
  C.noalias() += scalar * (A * B - B * A);
}
}  // namespace rhbi