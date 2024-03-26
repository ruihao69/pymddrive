namespace rhbi {
template <typename ChippedOp_t>
double psi_O_psi(
    const ChippedOp_t& op,
    const Tensor2cd& psi_bra,
    const Tensor2cd& psi_ket) {
  constexpr Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
  const Tensor2cd result = (psi_bra.contract(op.contract(psi_ket, product_dims), product_dims));
  return result(0, 0).real();
}

template <typename OP_t>
double expected_value_density_matrix(
    Eigen::Ref<const OP_t> op,
    Eigen::Ref<const RowMatrixXcd> rho) {
  return (rho * op).trace().real();
}

template <typename OP_t>
double expected_value_wave_function(
    Eigen::Ref<const OP_t> op,
    Eigen::Ref<const Eigen::VectorXcd> psi) {
  return (psi.adjoint() * op * psi).real()(0, 0);
}

template <typename TENSOR_OP_t>
Tensor1d tensor_expected_value_density_matrix(
    const TENSOR_OP_t& op,
    Eigen::Ref<const RowMatrixXcd> rho) {
  // Suppose the quantum classical operator is a rank-3 tensor,
  // shaped (n, n, m), where n is the number of electronic dofs
  // and m is the number of classical dofs.

  Tensor1d result(op.dimension(2));
  result.setZero();

  for (int i = 0; i < op.dimension(2); i++) {
    for (int j = 0; j < rho.rows(); j++) {
      // for (int k = 0; k < rho.cols(); k++) {  
      //   result(i) += (rho(j, k) * op(k, j, i)).real();
      // }
      for (int k = j; k < rho.cols(); k++) {
        result(i) += (rho(j, k) * op(k, j, i)).real();
        if (j != k) {
          result(i) += (rho(k, j) * op(j, k, i)).real();
        }
      }
    }
  }
  return result;
}

template <typename TENSOR_OP_t>
Tensor1d tensor_expected_value_wave_function(
    const TENSOR_OP_t& op,
    Eigen::Ref<const Eigen::VectorXcd> psi) {
  // Suppose the quantum classical operator is a rank-3 tensor,
  // shaped (n, n, m), where n is the number of electronic dofs
  // and m is the number of classical dofs.

  Tensor1d result(op.dimension(2));

  // map psi as a rank-2 tensor
  const Tensor1cd psi_as_tensor = TensorCast(psi);

  const Eigen::array<Eigen::Index, 2> shape_bra = {1, psi.size()};
  const Eigen::array<Eigen::Index, 2> shape_ket = {psi.size(), 1};
  const Tensor2cd psi_bra = psi_as_tensor.conjugate().reshape(shape_bra);
  const Tensor2cd psi_ket = psi_as_tensor.reshape(shape_ket);

  for (int i = 0; i < op.dimension(2); ++i) {
    result(i) = psi_O_psi(op.chip(i, 2), psi_bra, psi_ket);
  }
  return result;
}

// use enable_if to restrict the return type of expected_value
template <typename MAT_OP_t, typename State_t>
double get_expected_value(
    Eigen::Ref<const MAT_OP_t> op,
    Eigen::Ref<const State_t> state) {
  if constexpr (State_t::RowsAtCompileTime == 1 ||
                State_t::ColsAtCompileTime == 1) {
    return expected_value_wave_function(op, state);
  } else {
    return expected_value_density_matrix(op, state);
  }
}

template <typename TENSOR_OP_t, typename State_t>
Tensor1d get_expected_value(
    const TENSOR_OP_t& op,
    Eigen::Ref<const State_t> state) {
  if constexpr (State_t::RowsAtCompileTime == 1 ||
                State_t::ColsAtCompileTime == 1) {
    return tensor_expected_value_wave_function(op, state);
  } else {
    return tensor_expected_value_density_matrix(op, state);
  }
}
}  // namespace rhbi