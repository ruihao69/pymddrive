// third-party libraries
#include <Eigen/Dense>

// local includes
#include "states/state_utils.h"
#include "row_major_types.h"

// c++ includes
#include <complex>
#include <iostream>

namespace rhbi {

// Eigen::Map<const Eigen::VectorXcd> complex_map(const Eigen::VectorXd& real_vector) {
//     return Eigen::Map<const Eigen::VectorXcd>(real_vector.data(), real_vector.size());
// }
Eigen::VectorXcd complex_cast(Eigen::Ref<const Eigen::VectorXd> real_vector) {
    return real_vector.cast<std::complex<double>>();
}


void concatenate_RP(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view //
) {
    // concatenate R and P into flatten_view
    flatten_view.head(R.size()) = complex_cast(R);
    flatten_view.tail(P.size()) = complex_cast(P);
}

void concatenate_RPPsi(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<const Eigen::VectorXcd> psi,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view //
){
    // concatenate R, P, and psi into flatten_view
    flatten_view.head(R.size()) = complex_cast(R);
    flatten_view.segment(R.size(), P.size()) = complex_cast(P);
    flatten_view.tail(psi.size()) = psi;
    // std::cout << "debug psi= " << psi.conjugate() << std::endl;
}

void concatenate_RPRho(
        Eigen::Ref<const Eigen::VectorXd> R,  //
        Eigen::Ref<const Eigen::VectorXd> P,  //
        Eigen::Ref<const RowMatrixXcd> rho,  //
        Eigen::Ref<Eigen::VectorXcd> flatten_view//
){
    // concatenate R, P, and rho into flatten_view
    flatten_view.head(R.size()) = complex_cast(R);
    flatten_view.segment(R.size(), P.size()) = complex_cast(P);
    flatten_view.tail(rho.size()) = Eigen::Map<const Eigen::VectorXcd>(rho.data(), rho.size());
}


void flat_to_RP(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P  //
){
    R = flatten_view.head(R.size()).real();
    P = flatten_view.segment(R.size(), P.size()).real();
    // R = Eigen::VectorXd(flatten_view.head(R.size()).real());
    // P = Eigen::VectorXd(flatten_view.segment(R.size(), P.size()).real());
}

void flat_to_RPPsi(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P,  //
        Eigen::Ref<Eigen::VectorXcd> psi //
){
    R = flatten_view.head(R.size()).real();
    P = flatten_view.segment(R.size(), P.size()).real();
    // R = Eigen::VectorXd(flatten_view.head(R.size()).real());
    // P = Eigen::VectorXd(flatten_view.segment(R.size(), P.size()).real());
    psi = flatten_view.tail(psi.size());
}

void flat_to_RPRho(
        Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
        Eigen::Ref<Eigen::VectorXd> R,  //
        Eigen::Ref<Eigen::VectorXd> P,  //
        Eigen::Ref<RowMatrixXcd> rho //
){
    R = flatten_view.head(R.size()).real();
    P = flatten_view.segment(R.size(), P.size()).real();
    // R = Eigen::VectorXd(flatten_view.head(R.size()).real());
    // P = Eigen::VectorXd(flatten_view.segment(R.size(), P.size()).real());
    rho = Eigen::Map<const RowMatrixXcd>(flatten_view.tail(rho.size()).data(), rho.rows(), rho.cols());
}

}  // namespace rhbi