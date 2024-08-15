"""Utility function for computing scaling factors for non-dimensionalization."""

import numpy as np
from scipy.constants import elementary_charge, mu_0

from desc.geometry import Curve


def compute_scaling_factors(thing):
    """Compute dimensional quantities for normalizations."""
    # local import to avoid circular import
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface

    scales = {}

    def get_lowest_mode(basis, coeffs):
        """Return lowest order coefficient (excluding m=0 modes)."""
        # lowest order modes: [0, +1, -1, +2, -2, ...]
        m_modes = np.arange(1, thing.M + 1)
        m_modes = np.vstack((m_modes, -m_modes)).flatten(order="F")
        n_modes = np.arange(thing.N + 1)
        n_modes = np.vstack((n_modes, -n_modes)).flatten(order="F")
        for n in n_modes:
            for m in m_modes:
                try:
                    x = coeffs[basis.get_idx(M=m, N=n)]
                    if not np.isclose(x, 0):  # mode exists and coefficient is non-zero
                        return x
                except ValueError:
                    pass
        raise ValueError("No modes found, geometry is unphysical.")

    if isinstance(thing, Equilibrium):
        R00 = thing.Rb_lmn[thing.surface.R_basis.get_idx(M=0, N=0)]
        R10 = get_lowest_mode(thing.surface.R_basis, thing.Rb_lmn)
        Z10 = get_lowest_mode(thing.surface.Z_basis, thing.Zb_lmn)

        scales["R0"] = R00
        scales["a"] = np.sqrt(np.abs(R10 * Z10))
        scales["Psi"] = abs(thing.Psi)
        scales["n"] = 1e19
        scales["A"] = np.pi * scales["a"] ** 2
        scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]
        scales["B"] = scales["Psi"] / scales["A"] * np.sqrt(2)
        scales["I"] = scales["B"] * 2 * np.pi / mu_0
        scales["p"] = scales["B"] ** 2 / (2 * mu_0)
        scales["W"] = scales["p"] * scales["V"]
        scales["J"] = scales["B"] / scales["a"] / mu_0
        scales["F"] = scales["p"] / scales["a"]
        scales["f"] = scales["F"] * scales["V"]
        scales["T"] = scales["p"] / (scales["n"] * elementary_charge)

    elif isinstance(thing, FourierRZToroidalSurface):
        R00 = thing.R_lmn[thing.R_basis.get_idx(M=0, N=0)]
        R10 = get_lowest_mode(thing.R_basis, thing.R_lmn)
        Z10 = get_lowest_mode(thing.Z_basis, thing.Z_lmn)

        scales["R0"] = R00
        scales["a"] = np.sqrt(np.abs(R10 * Z10))
        scales["A"] = np.pi * scales["a"] ** 2
        scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]

    elif isinstance(thing, Curve):
        scales["a"] = thing.compute("length")["length"] / (2 * np.pi)

    # replace 0 scales to avoid normalizing by zero
    for scale in scales.keys():
        if np.isclose(scales[scale], 0):
            scales[scale] = 1

    return scales
