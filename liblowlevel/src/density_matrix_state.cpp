// Third-party libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// local includes
#include "density_matrix_state.h"

// C++ standard library
#include <complex>

namespace rhbi {
DensityMatrixState::DensityMatrixState(const Eigen::Tensor<double, 1>& R, const Eigen::Tensor<double, 1>& P, const Eigen::Tensor<std::complex<double>, 2>& rho)
    : R(R), P(P), rho(rho) {}

DensityMatrixState DensityMatrixState::operator+(const DensityMatrixState& other) const {
  return DensityMatrixState(R + other.R, P + other.P, rho + other.rho);
}

DensityMatrixState& DensityMatrixState::operator+=(const DensityMatrixState& other) {
  R += other.R;
  P += other.P;
  rho += other.rho;
  return *this;
}

DensityMatrixState DensityMatrixState::operator*(double scalar) const {
  return DensityMatrixState(scalar * R, scalar * P, scalar * rho);
}

DensityMatrixState& DensityMatrixState::operator*=(double scalar) {
  R = scalar * R;
  P = scalar * P;
  rho = scalar * rho;
  return *this;
}

DensityMatrixState DensityMatrixState::from_unstructured(const Eigen::VectorXcd& flattened) const {
  const int total_size = R.size() + P.size() + rho.size();
  if (flattened.size() != total_size) {
    throw std::invalid_argument("Flattened state has incorrect size");
  }

  Eigen::Tensor<std::complex<double>, 1> R_out_cplx(R.size());
  Eigen::Tensor<std::complex<double>, 1> P_out_cplx(P.size());
  Eigen::Tensor<std::complex<double>, 2> rho_out(rho.dimensions());

  int offset = 0;

  R_out_cplx = Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 1>>(flattened.segment(offset, R.size()).data(), R.size());
  offset += R.size();

  P_out_cplx = Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 1>>(flattened.segment(offset, P.size()).data(), P.size());
  offset += P.size();

  rho_out = Eigen::TensorMap<const Eigen::Tensor<std::complex<double>, 2>>(flattened.segment(offset, rho.size()).data(), rho.dimensions());

  return DensityMatrixState(R_out_cplx.real(), P_out_cplx.real(), rho_out);
}

Eigen::VectorXcd DensityMatrixState::flatten() const {
  const int total_size = R.size() + P.size() + rho.size();
  Eigen::VectorXcd flattened(total_size);

  int offset = 0;
  const Eigen::Tensor<std::complex<double>, 1> R_cmplx_cast = R.cast<std::complex<double>>();
  flattened.segment(offset, R.size()) = Eigen::Map<const Eigen::VectorXcd>(R_cmplx_cast.data(), R.size());

  offset += R.size();
  const Eigen::Tensor<std::complex<double>, 1> P_cmplx_cast = P.cast<std::complex<double>>();
  flattened.segment(offset, P.size()) = Eigen::Map<const Eigen::VectorXcd>(P_cmplx_cast.data(), P.size());

  offset += P.size();
  flattened.segment(offset, rho.size()) = Eigen::Map<const Eigen::VectorXcd>(rho.data(), rho.size());

  return flattened;
}

DensityMatrixState axpy(double a, const DensityMatrixState& x, const DensityMatrixState& y) {
  return DensityMatrixState(a * x.R + y.R, a * x.P + y.P, a * x.rho + y.rho);
}

void rk4_step_inplace(double dt, DensityMatrixState& state, const DensityMatrixState& k1, const DensityMatrixState& k2, const DensityMatrixState& k3, const DensityMatrixState& k4) {
  state.R += dt / 6.0 * (k1.R + 2.0 * k2.R + 2.0 * k3.R + k4.R);
  state.P += dt / 6.0 * (k1.P + 2.0 * k2.P + 2.0 * k3.P + k4.P);
  state.rho += dt / 6.0 * (k1.rho + 2.0 * k2.rho + 2.0 * k3.rho + k4.rho);
}

DensityMatrixState rk4_step(double dt, const DensityMatrixState& state, const DensityMatrixState& k1, const DensityMatrixState& k2, const DensityMatrixState& k3, const DensityMatrixState& k4) {
  DensityMatrixState state_out = state;
  rk4_step_inplace(dt, state_out, k1, k2, k3, k4);
  return state_out;
}

DensityMatrixState get_state_from_unstructured(const Eigen::VectorXcd& flattened, const DensityMatrixState& state) {
  return state.from_unstructured(flattened);
}
}  // namespace rhbi
