#pragma once

// third-party includes
#include <Eigen/Dense>
#include "row_major_types.h"

namespace rhbi {
    void concatenate_RP(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view //
    );

    void concatenate_RPPsi(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<const Eigen::VectorXcd> psi,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view //
    );

    void concatenate_RPRho(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<const RowMatrixXcd> rho,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view//
    );

    void concatenate_RPPsiDeltaRP(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<const Eigen::VectorXcd> psi,  //
        const Tensor3cd &delta_R,  //
        const Tensor3cd &delta_P,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view //
    );

    void concatenate_RPRhoDeltaRP(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<const RowMatrixXcd> rho,  //
        const Tensor3cd &delta_R,  //
        const Tensor3cd &delta_P,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view //
    );

    void flat_to_RP(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P  //
        );

    void flat_to_RPPsi(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P,  //
        Eigen::Ref<Eigen::VectorXcd> psi //
    );

    void flat_to_RPRho(
        Eigen::Ref<const Eigen::VectorXcd>,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P,  //
        Eigen::Ref<RowMatrixXcd> rho //
    ); 

    void flat_to_RPPsiDeltaRP(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P,  //
        Eigen::Ref<Eigen::VectorXcd> psi,  //
        Tensor3cd& delta_R,  //
        Tensor3cd& delta_P  //
    );

    void flat_to_RPRhoDeltaRP(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P,  //
        Eigen::Ref<RowMatrixXcd> rho,  //
        Tensor3cd& delta_R,  //
        Tensor3cd& delta_P  //
    );

} // namespace rhbi