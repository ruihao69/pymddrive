#pragma once
// Third-party includes
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>  

// To ease the mapping between Eigen types and numpy arrays
// Defile RowMajor as the default storage order
namespace rhbi{
/// @brief row-major real matrix type
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief row-major complex matrix type
using RowMatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief row-major rank-3 complex tensor type
 * 
 * This is generally used to store the forces tensor and non-adiabatic coupling tensor
 * In our convention, the tensor is shaped (n_electornic, n_electronic, n_nuclear)
 */
// using RowTensor3cd = Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>;
using Tensor3cd = Eigen::Tensor<std::complex<double>, 3>;

/// @brief row-major rank-3 real tensor type shaped (n_electornic, n_electronic, n_nuclear)
// using RowTensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>; 
using Tensor3d = Eigen::Tensor<double, 3>; 

/// @brief row-major rank-1 real tensor type shaped (n_nuclear, )
// using RowTensor1d = Eigen::Tensor<double, 1, Eigen::RowMajor>;
using Tensor1d = Eigen::Tensor<double, 1>;

/// @brief row-major rank-2 real tensor type shaped (n_electronic, n_eletronic)
// using RowTensor2cd = Eigen::Tensor<std::complex<double>, 2, Eigen::RowMajor>;
using Tensor2cd = Eigen::Tensor<std::complex<double>, 2>;

/// @brief row-major rank-1 complex tensor type shaped (n_electronic, )
// using RowTensor1cd = Eigen::Tensor<std::complex<double>, 1, Eigen::RowMajor>;
using Tensor1cd = Eigen::Tensor<std::complex<double>, 1>;


}

