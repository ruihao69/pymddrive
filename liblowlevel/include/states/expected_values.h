#pragma once
// third-party includes
#include <Eigen/Dense>

// local includes
#include "row_major_types.h"
#include "tensor_utils.h"

namespace rhbi {
template <typename OP_t>
double expected_value_density_matrix(
    Eigen::Ref<const OP_t> op,          // quantum operator
    Eigen::Ref<const RowMatrixXcd> rho  // quantum state as a density matrix
);

template <typename OP_t>
double expected_value_wave_function(
    Eigen::Ref<const OP_t> op,              // quantum operator
    Eigen::Ref<const Eigen::VectorXcd> psi  // quantum state as a wave function
);

template <typename TENSOR_OP_t>
Eigen::VectorXd tensor_expected_value_density_matrix(
    const TENSOR_OP_t& op,              // quantum-classical operator as a rank-3 tensor
    Eigen::Ref<const RowMatrixXcd> rho  // quantum state as a density matrix
);

template <typename TENSOR_OP_t>
Eigen::VectorXd tensor_expected_value_wave_function(
    const TENSOR_OP_t& op,                  // quantum-classical operator as a rank-3 tensor
    Eigen::Ref<const Eigen::VectorXcd> psi  // quantum state as a wave function
);

template <typename MAT_OP_t, typename State_t>
double get_expected_value(
    Eigen::Ref<const MAT_OP_t> op,   // quantum operator
    Eigen::Ref<const State_t> state  // quantum state
);

template <typename TENSOR_OP_t, typename State_t>
Eigen::VectorXd get_expected_value(
    const TENSOR_OP_t& op,           // quantum-classical operator as a rank-3 tensor
    Eigen::Ref<const State_t> state  // quantum state
);

}  // namespace rhbi

#include "expected_values_impl.h"
