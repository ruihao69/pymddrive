#include "tensor_utils.h"
#include <iostream> 

namespace rhbi {
    template <typename T>
    Eigen::Matrix<T, 1, Eigen::Dynamic> get_dc_component_at_ij(
        int ii,
        int jj,
        const Eigen::Tensor<T, 3>& dc) {
        const Eigen::Tensor<T, 1> dc_ij = dc.chip(ii, 0).chip(jj, 0);
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(dc_ij.data(), dc_ij.size());
    }

    template <typename evecs_t>
    double evaluate_diabatic_state_population(
        int active_surface,
        int target_surface,
        Eigen::Ref<const evecs_t> evecs,
        Eigen::Ref<const RowMatrixXcd> rho) {
        double abs_U_ik = std::abs(evecs(target_surface, active_surface));
        double population = abs_U_ik * abs_U_ik;
        for (int ii = 0; ii < rho.rows(); ++ii) {
            for (int jj = ii + 1; jj < rho.cols(); ++jj) {
                population += 2.0 * std::real(evecs(target_surface, ii) * std::conj(evecs(target_surface, jj)) * rho(ii, jj));
            }
        }
        return population;
    }

    template <typename evecs_t>
    Eigen::RowVectorXd get_diabatic_populations(
        int active_surface,
        Eigen::Ref<const evecs_t> evecs,
        Eigen::Ref<const RowMatrixXcd> rho) {
        Eigen::RowVectorXd populations(evecs.rows());
        for (int ii = 0; ii < evecs.rows(); ++ii) {
            populations(ii) = evaluate_diabatic_state_population(active_surface, ii, evecs, rho);
        }
        return populations;
    }

    template <typename v_dot_d_t>
    double evaluate_hopping_probability(
        int active_surface,
        int target_surface,
        double dt,
        Eigen::Ref<const v_dot_d_t> v_dot_d,
        Eigen::Ref<const RowMatrixXcd> rho) {
        double probability = 2.0 * dt * std::real(v_dot_d(active_surface, target_surface) * rho(target_surface, active_surface) / rho(active_surface, active_surface));
        return probability > 0.0 ? probability : 0.0;
    }

    template <typename v_dot_d_t>
    Eigen::RowVectorXd get_hopping_probabilities(
        int active_surface,
        double dt,
        Eigen::Ref<const v_dot_d_t> v_dot_d,
        Eigen::Ref<const RowMatrixXcd> rho) {
        Eigen::RowVectorXd probabilities = Eigen::RowVectorXd::Zero(rho.rows());
        probabilities(active_surface) = 1.0;
        for (int ii = 0; ii < rho.rows(); ++ii) {
            if (ii == active_surface) {
                continue;
            }
            probabilities(ii) = evaluate_hopping_probability(active_surface, ii, dt, v_dot_d, rho);
            probabilities(active_surface) -= probabilities(ii);
        }
        return probabilities;
    }

    template <typename d_component_t, typename mass_t>
    std::pair<bool, Eigen::RowVectorXd> momentum_rescale(
        double dE,                                  // energy difference
        // Eigen::Ref<const d_component_t> direction,  // direction of the hop
        const d_component_t& direction,  // direction of the hop
        const Eigen::RowVectorXd& P_current,        // current momentum
        const mass_t& mass                          // mass matrix
    ) {
        double a, b;
        if constexpr (std::is_same_v<mass_t, double>){
            a = 0.5 * (direction.dot(direction / mass));
            b = direction.dot(P_current / mass);
        } else{
            a = 0.5 * (direction.dot(direction.cwiseQuotient(mass)));
            b = direction.dot(P_current.cwiseQuotient(mass));
        }
        double c = dE;

        double discriminant = b2_4ac(a, b, c);
        if (discriminant < 0.0) {
            return std::make_pair(false, P_current);
        }
        else {
            double gamma = (b < 0.0) ? (b + std::sqrt(discriminant)) / a * 0.5 : (b - std::sqrt(discriminant)) / a * 0.5;
            // return std::make_pair(true, P_current - gamma * mass * direction);
            return std::make_pair(true, P_current - gamma * direction);
        }
    }

    template <typename dc_tensor_t, typename mass_t, typename v_dot_d_t>
    std::tuple<bool, int, Eigen::RowVectorXd> fssh_surface_hopping(
        double dt,
        int active_surface,
        Eigen::Ref<const Eigen::RowVectorXd> P_current,
        Eigen::Ref<const RowMatrixXcd> rho,
        Eigen::Ref<const Eigen::RowVectorXd> eig_vals,
        Eigen::Ref<const v_dot_d_t> v_dot_d,
        const dc_tensor_t& dc,
        const mass_t& mass) {
        Eigen::RowVectorXd hopping_probabilities = get_hopping_probabilities(active_surface, dt, v_dot_d, rho);
        int target_surface = hop(hopping_probabilities);
        if (target_surface == active_surface) {
            return std::make_tuple(false, active_surface, P_current);
        }
        else {
            double dE = eig_vals(target_surface) - eig_vals(active_surface);
            // Eigen::RowVectorXd direction = get_dc_component_at_ij(target_surface, active_surface, dc);
            Eigen::RowVectorXd direction = get_dc_component_at_ij(active_surface, target_surface, dc);
            auto [success, P_new] = momentum_rescale(dE, direction, P_current, mass);
            if (success) {
                return std::make_tuple(true, target_surface, P_new);
            }
            else {
                return std::make_tuple(false, active_surface, P_current);
            }
        }
    }
}  // namespace rhbi