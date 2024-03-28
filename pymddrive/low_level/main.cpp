// pybind11 header file
#include <pybind11/pybind11.h>

// impl files
#include "states.cpp"
#include "equations_of_motion.cpp"
#include "floquet.cpp"
#include "ehrenfest.cpp"
#include "surface_hopping.cpp"


namespace py = pybind11;
using namespace rhbi;

// create a overall module called low_level
PYBIND11_MODULE(_low_level, m) {
  m.doc() = "low-level module for the pymddrive package";

  /****
  The states submodule
   ****/
  py::module m_states = m.def_submodule("states", "low-level version of the states module");
  m_states.doc() = "Low-level state variables for the quantum / quantum classical dynamics simulation.";

  // The binding code for the states submodule
  bind_states(m_states);

  /****
   * The equations_of_motion submodule
   ****/
  py::module m_eom = m.def_submodule("equations_of_motion");
  m_eom.doc() = "Low-level version of the equations_of_motion module, including the electornic equations of motion for the quantum / quantum classical dynamics simulation.";

  // The binding code for the equations_of_motion submodule
  bind_equations_of_motion(m_eom);

  /****
   * The floquet submodule
   ****/

  py::module m_floquet = m.def_submodule("floquet", "Low-level version of the floquet module");
  m_floquet.doc() = "Low-level version of the floquet module, including the Floquet Hamiltonian for the quantum / quantum classical dynamics simulation.";

  // The binding code for the floquet submodule
  bind_floquet(m_floquet);

  /****
   * The ehrenfest submodule
   ****/
  py::module m_ehrenfest = m.def_submodule("ehrenfest", "Low-level version of the ehrenfest module");
  m_ehrenfest.doc() = "Low-level version of the ehrenfest module, including the Ehrenfest theorems for the quantum / quantum classical dynamics simulation.";

  // The binding code for the ehrenfest submodule
  bind_ehrenfest(m_ehrenfest);


  /***
   * The low-level module for surface hopping
   ***/

  py::module m_surface_hopping = m.def_submodule("surface_hopping", "Low-level version of the surface_hopping module");
  m_surface_hopping.doc() = "Low-level version of the surface_hopping module, including the surface hopping dynamics for the quantum / quantum classical dynamics simulation.";

  // the binding code for the surface_hopping submodule 
  bind_surface_hopping(m_surface_hopping);
}