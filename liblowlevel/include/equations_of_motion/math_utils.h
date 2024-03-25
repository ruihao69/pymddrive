#pragma once

// Third party includes
#include <Eigen/Dense>

// c++ includes
#include <complex>

namespace rhbi {

template <typename OP1_t, typename OP2_t>
auto commutator(const OP1_t& A, const OP2_t& B) {
  return A * B - B * A;
}

}  // namespace rhbi