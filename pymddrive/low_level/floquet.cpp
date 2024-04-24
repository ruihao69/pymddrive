// pybind11 header file
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

// Third-party libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// local includes
#include "row_major_types.h"
#include "floquet/floquet.h"

namespace py = pybind11;
using namespace rhbi;

// forward declarations

void bind_get_HF_cos(py::module& m);

void bind_get_dHF_dR_cos(py::module& m);

void bind_floquet(py::module& m);

// The implementation of the functions

void bind_get_HF_cos(py::module& m) {
    // instantiate for get_HF_cos: complex H0, complex V
    m.def("get_HF_cos", &get_HF_cos<RowMatrixXcd, RowMatrixXcd>);

    // instantiate for get_HF_cos: complex H0, real V
    m.def("get_HF_cos", &get_HF_cos<RowMatrixXcd, RowMatrixXd>);

    // instantiate for get_HF_cos: real H0, complex V
    m.def("get_HF_cos", &get_HF_cos<RowMatrixXd, RowMatrixXcd>);

    // instantiate for get_HF_cos: real H0, real V
    m.def("get_HF_cos", &get_HF_cos<RowMatrixXd, RowMatrixXd>);
}

void bind_get_HF_sin(py::module& m) {
    // instantiate for get_HF_cos: complex H0, complex V
    m.def("get_HF_sin", &get_HF_sin<RowMatrixXcd, RowMatrixXcd>);

    // instantiate for get_HF_cos: complex H0, real V
    m.def("get_HF_sin", &get_HF_sin<RowMatrixXcd, RowMatrixXd>);

    // instantiate for get_HF_cos: real H0, complex V
    m.def("get_HF_sin", &get_HF_sin<RowMatrixXd, RowMatrixXcd>);

    // instantiate for get_HF_cos: real H0, real V
    m.def("get_HF_sin", &get_HF_sin<RowMatrixXd, RowMatrixXd>);
}

void bind_get_dHF_dR_cos(py::module& m) {
    // instantiate for get_dHF_dR_cos: complex dH0_dR, complex dV_dR
    m.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3cd, Tensor3cd>);

    // instantiate for get_dHF_dR_cos: complex dH0_dR, real dV_dR
    m.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3cd, Tensor3d>);

    // instantiate for get_dHF_dR_cos: real dH0_dR, complex dV_dR
    m.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3d, Tensor3cd>);

    // instantiate for get_dHF_dR_cos: real dH0_dR, real dV_dR
    m.def("get_dHF_dR_cos", &get_dHF_dR_cos<Tensor3d, Tensor3d>);
}

void bind_get_dHF_dR_sin(py::module& m) {
    // instantiate for get_dHF_dR_cos: complex dH0_dR, complex dV_dR
    m.def("get_dHF_dR_sin", &get_dHF_dR_sin<Tensor3cd, Tensor3cd>);

    // instantiate for get_dHF_dR_cos: complex dH0_dR, real dV_dR
    m.def("get_dHF_dR_sin", &get_dHF_dR_sin<Tensor3cd, Tensor3d>);

    // instantiate for get_dHF_dR_cos: real dH0_dR, complex dV_dR
    m.def("get_dHF_dR_sin", &get_dHF_dR_sin<Tensor3d, Tensor3cd>);

    // instantiate for get_dHF_dR_cos: real dH0_dR, real dV_dR
    m.def("get_dHF_dR_sin", &get_dHF_dR_sin<Tensor3d, Tensor3d>);
}

void bind_floquet(py::module& m) {
    bind_get_HF_cos(m);
    bind_get_dHF_dR_cos(m);
    bind_get_HF_sin(m);
    bind_get_dHF_dR_sin(m);
}

