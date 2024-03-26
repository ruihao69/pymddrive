#pragma once

// third-party includes
#include <Eigen/Dense>

namespace rhbi {
void concatenate_RP(
    const Eigen::VectorXd& R,  //
    const Eigen::VectorXd& P,  //
    Eigen::VectorXcd& flatten_view //
);

void concatenate_RPPsi(
    const Eigen::VectorXd& R,  //
    const Eigen::VectorXd& P,  //
    const Eigen::VectorXcd& psi,  //
    Eigen::VectorXcd& flatten_view //
);

void concatenate_RPRho(
    const Eigen::VectorXd& R,  //
    const Eigen::VectorXd& P,  //
    const Eigen::MatrixXcd& rho,  //
    Eigen::VectorXcd& flatten_view //
);

void flat_to_RP(
    const Eigen::VectorXcd& flatten_view,  //
    Eigen::VectorXd& R,  //
    Eigen::VectorXd& P //
);

void flat_to_RPPsi(
    const Eigen::VectorXcd& flatten_view,  //
    Eigen::VectorXd& R,  //
    Eigen::VectorXd& P,  //
    Eigen::VectorXcd& psi //
);

void flat_to_RPRho(
    const Eigen::VectorXcd& flatten_view,  //
    Eigen::VectorXd& R,  //
    Eigen::VectorXd& P,  //
    Eigen::MatrixXcd& rho //
);

} // namespace rhbi