// third-party libraries
#include <Eigen/Dense>

// local includes
#include "states/state_utils.h"

// c++ includes
#include <complex>
#include <iostream>

namespace rhbi {

// Eigen::Map<const Eigen::VectorXcd> complex_map(const Eigen::VectorXd& real_vector) {
//     return Eigen::Map<const Eigen::VectorXcd>(real_vector.data(), real_vector.size());
// }
Eigen::VectorXcd complex_cast(const Eigen::VectorXd& real_vector) {
    return real_vector.cast<std::complex<double>>();
}


void concatenate_RP(
    const Eigen::VectorXd& R,       //
    const Eigen::VectorXd& P,       //
    Eigen::VectorXcd& flatten_view  //
) {
    // concatenate R and P into flatten_view
    flatten_view.head(R.size()) = complex_cast(R);
    flatten_view.tail(P.size()) = complex_cast(P);
}

void concatenate_RPPsi(
    const Eigen::VectorXd& R,       //
    const Eigen::VectorXd& P,       //
    const Eigen::VectorXcd& psi,    //
    Eigen::VectorXcd& flatten_view  //
){
    // concatenate R, P, and psi into flatten_view
    flatten_view.head(R.size()) = complex_cast(R);
    flatten_view.segment(R.size(), P.size()) = complex_cast(P);
    flatten_view.tail(psi.size()) = psi;
    // std::cout << "debug psi= " << psi.conjugate() << std::endl;
}

void concatenate_RPRho(
    const Eigen::VectorXd& R,       //
    const Eigen::VectorXd& P,       //
    const Eigen::MatrixXcd& rho,    //
    Eigen::VectorXcd& flatten_view  //
){
    // concatenate R, P, and rho into flatten_view
    flatten_view.head(R.size()) = complex_cast(R);
    flatten_view.segment(R.size(), P.size()) = complex_cast(P);
    flatten_view.tail(rho.size()) = Eigen::Map<const Eigen::VectorXcd>(rho.data(), rho.size());
}


void flat_to_RP(
    const Eigen::VectorXcd& flatten_view,  //
    Eigen::VectorXd& R,  //
    Eigen::VectorXd& P //
){
    R = flatten_view.head(R.size()).real();
    P = flatten_view.segment(R.size(), P.size()).real();
    // R = Eigen::VectorXd(flatten_view.head(R.size()).real());
    // P = Eigen::VectorXd(flatten_view.segment(R.size(), P.size()).real());
}

void flat_to_RPPsi(
    const Eigen::VectorXcd& flatten_view,  //
    Eigen::VectorXd& R,  //
    Eigen::VectorXd& P,  //
    Eigen::VectorXcd& psi //
){
    R = flatten_view.head(R.size()).real();
    P = flatten_view.segment(R.size(), P.size()).real();
    // R = Eigen::VectorXd(flatten_view.head(R.size()).real());
    // P = Eigen::VectorXd(flatten_view.segment(R.size(), P.size()).real());
    psi = flatten_view.tail(psi.size());
}

void flat_to_RPRho(
    const Eigen::VectorXcd& flatten_view,  //
    Eigen::VectorXd& R,  //
    Eigen::VectorXd& P,  //
    Eigen::MatrixXcd& rho //
){
    R = flatten_view.head(R.size()).real();
    P = flatten_view.segment(R.size(), P.size()).real();
    // R = Eigen::VectorXd(flatten_view.head(R.size()).real());
    // P = Eigen::VectorXd(flatten_view.segment(R.size(), P.size()).real());
    rho = Eigen::Map<const Eigen::MatrixXcd>(flatten_view.tail(rho.size()).data(), rho.rows(), rho.cols());
}

}  // namespace rhbi