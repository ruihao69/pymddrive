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

void concatenate_RPPsiDeltaRP(
    Eigen::Ref<const Eigen::VectorXd> R,  //
    Eigen::Ref<const Eigen::VectorXd> P,  //
    Eigen::Ref<const Eigen::VectorXcd> psi,  //
    const Tensor3cd &delta_R,  //
    const Tensor3cd &delta_P,  //
    Eigen::Ref<Eigen::VectorXcd> flatten_view //
){
    int start, size;
    start = 0;

    // fill in R
    size = R.size();
    flatten_view.head(size) = complex_cast(R);

    // fill in P
    start += size;
    size = P.size();
    flatten_view.segment(start, size) = complex_cast(P);

    // fill in psi  
    start += size;
    size = psi.size();
    flatten_view.segment(start, size) = psi;

    // fill in delta_R, use TensorMap from Tensor3cd to Eigen::VectorXcd
    start += size;
    size = delta_R.size();
    flatten_view.segment(start, size) = Eigen::Map<const Eigen::VectorXcd>(delta_R.data(), delta_R.size());

    // fill in delta_P, use TensorMap from Tensor3cd to Eigen::VectorXcd
    start += size;
    size = delta_P.size();
    flatten_view.segment(start, size) = Eigen::Map<const Eigen::VectorXcd>(delta_P.data(), delta_P.size());
}

void concatenate_RPRhoDeltaRP(
    Eigen::Ref<const Eigen::VectorXd> R,  //
    Eigen::Ref<const Eigen::VectorXd> P,  //
    Eigen::Ref<const RowMatrixXcd> rho,  //
    const Tensor3cd &delta_R,  //
    const Tensor3cd &delta_P,  //
    Eigen::Ref<Eigen::VectorXcd> flatten_view //
){
    int start, size;
    start = 0;

    // fill in R
    size = R.size();
    flatten_view.head(size) = complex_cast(R);

    // fill in P
    start += size;
    size = P.size();
    flatten_view.segment(start, size) = complex_cast(P);

    // fill in rho
    start += size;
    size = rho.size();
    flatten_view.segment(start, size) = Eigen::Map<const Eigen::VectorXcd>(rho.data(), rho.size());

    // fill in delta_R, use TensorMap from Tensor3cd to Eigen::VectorXcd
    start += size;
    size = delta_R.size();
    flatten_view.segment(start, size) = Eigen::Map<const Eigen::VectorXcd>(delta_R.data(), delta_R.size());

    // fill in delta_P, use TensorMap from Tensor3cd to Eigen::VectorXcd
    start += size;
    size = delta_P.size();
    flatten_view.segment(start, size) = Eigen::Map<const Eigen::VectorXcd>(delta_P.data(), delta_P.size());
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

void flat_to_RPPsiDeltaRP(
    Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
    Eigen::Ref<Eigen::VectorXd> R,  //
    Eigen::Ref<Eigen::VectorXd> P,  //
    Eigen::Ref<Eigen::VectorXcd> psi,  //
    Tensor3cd& delta_R,  //
    Tensor3cd& delta_P  //
){
    int start, size;
    start = 0;

    // fill in R
    size = R.size();
    R = flatten_view.head(size).real();

    // fill in P
    start += size;
    size = P.size();
    P = flatten_view.segment(start, size).real();

    // fill in psi
    start += size;
    size = psi.size();
    psi = flatten_view.segment(start, size);

    // fill in delta_R
    start += size;
    size = delta_R.size();
    delta_R = Eigen::TensorMap<const Tensor3cd>(flatten_view.segment(start, size).data(), delta_R.dimension(0), delta_R.dimension(1), delta_R.dimension(2));

    // fill in delta_P
    start += size;
    size = delta_P.size();
    delta_P = Eigen::TensorMap<const Tensor3cd>(flatten_view.segment(start, size).data(), delta_P.dimension(0), delta_P.dimension(1), delta_P.dimension(2));
}

void flat_to_RPRhoDeltaRP(
    Eigen::Ref<const Eigen::VectorXcd> flatten_view,  //
    Eigen::Ref<Eigen::VectorXd> R,  //
    Eigen::Ref<Eigen::VectorXd> P,  //
    Eigen::Ref<RowMatrixXcd> rho,  //
    Tensor3cd& delta_R,  //
    Tensor3cd& delta_P  //
){
    int start, size;
    start = 0;

    // fill in R
    size = R.size();
    R = flatten_view.head(size).real();

    // fill in P
    start += size;
    size = P.size();
    P = flatten_view.segment(start, size).real();

    // fill in rho
    start += size;
    size = rho.size();
    rho = Eigen::Map<const RowMatrixXcd>(flatten_view.segment(start, size).data(), rho.rows(), rho.cols());

    // fill in delta_R
    start += size;
    size = delta_R.size();
    delta_R = Eigen::TensorMap<const Tensor3cd>(flatten_view.segment(start, size).data(), delta_R.dimension(0), delta_R.dimension(1), delta_R.dimension(2));

    // fill in delta_P
    start += size;
    size = delta_P.size();
    delta_P = Eigen::TensorMap<const Tensor3cd>(flatten_view.segment(start, size).data(), delta_P.dimension(0), delta_P.dimension(1), delta_P.dimension(2));
}

}  // namespace rhbi