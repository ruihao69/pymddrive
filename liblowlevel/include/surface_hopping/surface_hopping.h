#pragma once

// Third-party includes
#include <Eigen/Dense>

// local includes
#include "row_major_types.h"

// c++ includes
#include <random>
#include <tuple>
#include <utility>

namespace rhbi {
inline double b2_4ac(double a, double b, double c) {
  return b * b - 4.0 * a * c;
}

// --- Density matrix implementation of the surface hopping algorithm ---

template <typename evecs_t>
double evaluate_diabatic_state_population(
    int active_surface,
    int target_surface,
    Eigen::Ref<const evecs_t> evecs,
    Eigen::Ref<const RowMatrixXcd> rho);

template <typename evecs_t>
Eigen::RowVectorXd get_diabatic_populations(
    int active_surface,
    Eigen::Ref<const evecs_t> evecs,
    Eigen::Ref<const RowMatrixXcd> rho);

template <typename v_dot_d_t>
double evaluate_hopping_probability_dm(
    int active_surface,
    int target_surface,
    double dt,
    Eigen::Ref<const v_dot_d_t> v_dot_d,
    Eigen::Ref<const RowMatrixXcd> rho);

template <typename v_dot_d_t>
Eigen::RowVectorXd get_hopping_probabilities_dm(
    int active_surface,
    double dt,
    Eigen::Ref<const v_dot_d_t> v_dot_d,
    Eigen::Ref<const RowMatrixXcd> rho);

inline int hop(
    Eigen::Ref<const Eigen::RowVectorXd> hopping_probabilities,
    int active_surface
) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dis(0.0, 1.0);

  double random_number = dis(gen);
  double cumulative_probability = 0.0;

  for (int surface_index=0; surface_index < hopping_probabilities.size(); surface_index++) {
    if (surface_index != active_surface) { // avoid self-hopping    
      cumulative_probability += hopping_probabilities(surface_index);
      if (random_number < cumulative_probability) {
          return surface_index;
      }
    }
  }
  return active_surface;
}

template <typename d_component_t, typename mass_t>
std::pair<bool, Eigen::RowVectorXd> momentum_rescale(
    double dE,                                  // energy difference
    // Eigen::Ref<const d_component_t> direction,  // direction of the hop
    const d_component_t& direction,  // direction of the hop
    const Eigen::RowVectorXd& P_current,        // current momentum
    const mass_t& mass                          // mass matrix
);

template <typename dc_tensor_t, typename mass_t, typename v_dot_d_t>
    std::tuple<bool, int, Eigen::RowVectorXd> fssh_surface_hopping_dm(
        double dt,
        int active_surface,
        Eigen::Ref<const Eigen::RowVectorXd> P_current,
        Eigen::Ref<const RowMatrixXcd> rho,
        Eigen::Ref<const Eigen::RowVectorXd> eig_vals,
        Eigen::Ref<const v_dot_d_t> v_dot_d,
        const dc_tensor_t& dc,
        const mass_t& mass);

template <typename dc_tensor_t, typename mass_t, typename v_dot_d_t>
    std::tuple<bool, int, Eigen::RowVectorXd> fssh_surface_hopping_wf(
        double dt,
        int active_surface,
        Eigen::Ref<const Eigen::RowVectorXd> P_current,
        Eigen::Ref<const Eigen::RowVectorXcd> psi,
        Eigen::Ref<const Eigen::RowVectorXd> eig_vals,
        Eigen::Ref<const v_dot_d_t> v_dot_d,
        const dc_tensor_t& dc,
        const mass_t& mass);

template <typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> get_dc_component_at_ij(
    int ii,
    int jj,
    const Eigen::Tensor<T, 3>& dc);


// --- Wavefunction implementation of the surface hopping algorithm ---
template <typename v_dot_d_t>
double evaluate_hopping_probability_wf(
    int active_surface,
    int target_surface,
    double dt,
    Eigen::Ref<const v_dot_d_t> v_dot_d,
    Eigen::Ref<const Eigen::RowVectorXcd> psi);

template <typename v_dot_d_t>
Eigen::RowVectorXd get_hopping_probabilities_wf(
    int active_surface,
    double dt,
    Eigen::Ref<const v_dot_d_t> v_dot_d,
    Eigen::Ref<const Eigen::RowVectorXcd> psi);


}  // namespace rhbi

#include "surface_hopping_impl.h"