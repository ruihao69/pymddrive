#pragma once

// Third-party includes
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// local includes
#include "row_major_types.h"

namespace rhbi {
size_t dim_to_dimF(size_t dim, size_t NF);

size_t map_floquet_index_to_block_index(size_t n, size_t NF);

/*
 * Floquet Hamiltonian construction
 */

template <typename HF_t, typename H_t>
void fill_HF_diagonal(
    Eigen::Ref<HF_t> HF,       // the Floquet Hamiltonian
    Eigen::Ref<const H_t> H0,  // the Hamiltonian
    double Omega,              // the driving frequency
    size_t NF                  // the Floquet level cutoff
);

template <typename HF_t, typename V_t>
void fill_HF_offdiagonal_cosine(
    Eigen::Ref<HF_t> HF,      // the Floquet Hamiltonian
    Eigen::Ref<const V_t> V,  // the Floquet perturbation (upper triangular part)
    size_t NF                 // the Floquet level cutoff
);

template <typename V_t>
void fill_HF_offdiagonal_sine(
    Eigen::Ref<RowMatrixXcd> HF,  // the Floquet Hamiltonian
    Eigen::Ref<const V_t> V,      // the Floquet perturbation (upper triangular part)
    size_t NF                     // the Floquet level cutoff
);

// define the return type for get_HF_cos using conditional compilation
// based on the types of H0 and V. if any complex, RowMatrixXcd is used
template <typename H_t, typename V_t>
using HFReturnType = std::conditional_t<
    std::is_same_v<H_t, RowMatrixXcd> || std::is_same_v<V_t, RowMatrixXcd>,
    RowMatrixXcd,
    RowMatrixXd>;

template <typename H_t, typename V_t>
HFReturnType<H_t, V_t> get_HF_cos(Eigen::Ref<const H_t> H0, Eigen::Ref<const V_t> V, double Omega, size_t NF);

template <typename H_t, typename V_t>
RowMatrixXcd get_HF_sin(Eigen::Ref<const H_t> H0, Eigen::Ref<const V_t> V, double Omega, size_t NF);

/*
 * Floquet forces tensor construction
 */
template <typename dHF_dR_t, typename dH0_dR_t>
void fill_dHF_dR_diagonal(
    dHF_dR_t& dHF_dR,
    const dH0_dR_t& dH0_dR,
    size_t NF);

template <typename dHF_dR_t, typename dV_dR_t>
void fill_dHF_dR_offdiagonal_cosine(
    dHF_dR_t& dHF_dR,
    const dV_dR_t& dV_dR,
    size_t NF);

template <typename dV_dR_t>
void fill_dHF_dR_offdiagonal_sine(
    Tensor3cd& dHF_dR,
    const dV_dR_t& dV_dR,
    size_t NF);

template <typename dH0_dR_t, typename dV_dR_t>
using dHF_dRReturnType = std::conditional_t<
    std::is_same_v<dH0_dR_t, Tensor3cd> || std::is_same_v<dV_dR_t, Tensor3cd>,
    Tensor3cd,
    Tensor3d>;

template <typename dH0_dR_t, typename dV_dR_t>
dHF_dRReturnType<dH0_dR_t, dV_dR_t> get_dHF_dR_cos(
    // Eigen::Ref<const dH0_dR_t> dH0_dR, 
    // Eigen::Ref<const dV_dR_t> dV_dR, 
    const dH0_dR_t &dH0_dR,
    const dV_dR_t &dV_dR,
    size_t NF
);


template <typename dH0_dR_t, typename dV_dR_t>
Tensor3cd get_dHF_dR_sin(
    const dH0_dR_t& dH0_dR, 
    const dV_dR_t& dV_dR, 
    size_t NF
);

}  // namespace rhbi

// template function implementations
#include "floquet/floquet_impl.h"