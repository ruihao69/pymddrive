#pragma once

// Third-party libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
// C++ standard library
#include <complex>

namespace rhbi {

struct DensityMatrixState {
  /// @brief The nuclear positions.
  Eigen::Tensor<double, 1> R;

  /// @brief The nuclear momenta.
  Eigen::Tensor<double, 1> P;

  /// @brief The electronic density matrix.
  Eigen::Tensor<std::complex<double>, 2> rho;

  /**
   * @brief Constructs a new DensityMatrixState object.
   *
   * @param R The nuclear positions.
   * @param P The nuclear momenta.
   * @param rho The electronic density matrix.
   */
  DensityMatrixState(const Eigen::Tensor<double, 1>& R, const Eigen::Tensor<double, 1>& P, const Eigen::Tensor<std::complex<double>, 2>& rho);

  // Arithmetic operators
  DensityMatrixState operator+(const DensityMatrixState& other) const;
  DensityMatrixState operator*(double scalar) const;
  DensityMatrixState& operator+=(const DensityMatrixState& other);
  DensityMatrixState& operator*=(double scalar);

  /**
   * @brief Creates a DensityMatrixState object from an unstructured vector.
   *
   * This method takes a flattened vector and constructs a DensityMatrixState object from it.
   *
   * @param flattened The flattened vector representing the density matrix.
   * @return The constructed DensityMatrixState object.
   */
  DensityMatrixState from_unstructured(const Eigen::VectorXcd& flattened) const;

  /**
   * @brief Flattens the density matrix into a vector.
   *
   * This method flattens the density matrix into a vector representation.
   *
   * @return The flattened vector representing the density matrix.
   */
  Eigen::VectorXcd flatten() const;
};

/**
 * @brief Performs the AXPY operation on two DensityMatrixState objects.
 *
 * This function computes the AXPY operation, which stands for "a times x plus y",
 * on two DensityMatrixState objects and returns the result.
 *
 * @param a The scalar value.
 * @param x The first DensityMatrixState object.
 * @param y The second DensityMatrixState object.
 * @return The result of the AXPY operation.
 */
DensityMatrixState axpy(double a, const DensityMatrixState& x, const DensityMatrixState& y);

/**
 * @brief Performs a single step of the fourth-order Runge-Kutta method in-place.
 *
 * This function performs a single step of the fourth-order Runge-Kutta method in-place.
 * It updates the given DensityMatrixState object `state` using the provided intermediate states `k1`, `k2`, `k3`, and `k4`.
 *
 * @param dt The time step.
 * @param state The current state of the system.
 * @param k1 The intermediate state k1.
 * @param k2 The intermediate state k2.
 * @param k3 The intermediate state k3.
 * @param k4 The intermediate state k4.
 */
void rk4_step_inplace(double dt, DensityMatrixState& state, const DensityMatrixState& k1, const DensityMatrixState& k2, const DensityMatrixState& k3, const DensityMatrixState& k4);

/**
 * @brief Performs a single step of the fourth-order Runge-Kutta method.
 *
 * This function performs a single step of the fourth-order Runge-Kutta method.
 * It returns a new DensityMatrixState object that represents the updated state of the system.
 *
 * @param dt The time step.
 * @param state The current state of the system.
 * @param k1 The intermediate state k1.
 * @param k2 The intermediate state k2.
 * @param k3 The intermediate state k3.
 * @param k4 The intermediate state k4.
 * @return The updated state of the system.
 */
DensityMatrixState rk4_step(double dt, const DensityMatrixState& state, const DensityMatrixState& k1, const DensityMatrixState& k2, const DensityMatrixState& k3, const DensityMatrixState& k4);

/**
 * @brief Computes the norm of a DensityMatrixState object.
 *
 * This function computes the norm of a DensityMatrixState object.
 * 
 * @param flattened The flattened vector representing the density matrix.
 * @param state The DensityMatrixState object.
 * @return The norm of the DensityMatrixState object.
 */
DensityMatrixState get_state_from_unstructured(const Eigen::VectorXcd& flattened, const DensityMatrixState& state); 

}  // namespace rhbi