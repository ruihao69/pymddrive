# %%
import numpy as np

from pymddrive.low_level._low_level.surface_hopping import fssh_surface_hopping
from tests.test_utils import get_random_O, get_random_vO, get_random_rho, get_random_psi, compute_dc

import unittest

# %%

class TestSurfaceHopping(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = 2
        self.n_nuclear = 3

        self.rho = get_random_rho(self.dim)
        self.psi = get_random_psi(self.dim)

        self.H_real = get_random_O(self.dim, is_complex=False)
        self.H_complex = get_random_O(self.dim, is_complex=True)

        self.dHdR_real = get_random_vO(self.dim, self.n_nuclear, is_complex=False)
        self.dHdR_complex = get_random_vO(self.dim, self.n_nuclear, is_complex=True)

        self.dc_real, self.F_real, self.evals_real = compute_dc(self.H_real, self.dHdR_real)
        self.dc_complex, self.F_complex, self.evals_complex = compute_dc(self.H_complex, self.dHdR_complex)



    def _get_inputs(self, is_complex: bool):
        if is_complex:
            H = self.H_complex
            dHdR = self.dHdR_complex
            dc = self.dc_complex
            F = self.F_complex
            evals = self.evals_complex
        else:
            H = self.H_real
            dHdR = self.dHdR_real
            dc = self.dc_real
            F = self.F_real
            evals = self.evals_real
            return H, dHdR, dc, F, evals

    def test_energy_conservation_real(self):
        nums_of_hop_to_test = 10
        while nums_of_hop_to_test > 0:
            # dt, rho, H, dHdR, dc, F, evals, P_current, mass, active_surface = self.get_cpp_inputs(is_complex=False)
            dt, active_surface, P_current, rho, evals, v_dot_d, dc, mass = self.get_cpp_inputs(is_complex=False)
            success, active_surface_new, P_new = fssh_surface_hopping(dt, active_surface, P_current, rho, evals, v_dot_d, dc, mass)
            if success:
                DeltaE = evals[active_surface_new] - evals[active_surface]
                DeltaKE = np.sum((P_new**2 - P_current**2) / (2 * mass))
                np.testing.assert_allclose(DeltaE + DeltaKE, 0, atol=1e-10)
                nums_of_hop_to_test -= 1

    def get_cpp_inputs(self, is_complex: bool):
        H, dHdR, dc, F, evals = self._get_inputs(is_complex)
        dt = 0.3
        rho = self.rho
        P_current = np.random.normal(0, 1, self.n_nuclear)
        mass = 1
        active_surface = np.random.randint(0, self.dim)
        v_dot_d = np.tensordot(P_current, dc, axes=[[0], [2]])
        return dt, active_surface, P_current, rho, evals, v_dot_d, dc, mass

# %%
if __name__ == "__main__":
    unittest.main()

