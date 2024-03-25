#pragma once
// Third-party includes
#include <Eigen/Dense>

// To ease the mapping between Eigen types and numpy arrays
// Defile RowMajor as the default storage order
namespace rhbi{
/// @brief row-major real matrix type
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief row-major complex matrix type
using RowMatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief row-major real vector type
// using RowVectorXd = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::RowMajor>;

/// @brief row-major complex vector type
// using RowVectorXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::RowMajor>;
}

