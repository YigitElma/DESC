import numpy as np
import copy
from termcolor import colored

from desc.backend import put
from desc.io import IOAble
from desc.utils import unpack_state, copy_coeffs
from desc.grid import Grid, LinearGrid, ConcentricGrid
from desc.transform import Transform
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
)

from desc.compute_funs import (
    compute_profiles,
    compute_toroidal_coords,
    compute_cartesian_coords,
    compute_covariant_basis,
    compute_jacobian,
    compute_contravariant_basis,
    compute_magnetic_field_magnitude_axis,
    compute_current_density,
    compute_force_error_magnitude,
)


class Configuration(IOAble):
    """Configuration contains information about a plasma state, including the
    shapes of flux surfaces and profile inputs. It can compute additional
    information, such as the magnetic field and plasma currents.
    """

    _io_attrs_ = [
        "_sym",
        "_Psi",
        "_NFP",
        "_L",
        "_M",
        "_N",
        "_x",
        "_R_lmn",
        "_Z_lmn",
        "_L_lmn",
        "_Rb_mn",
        "_Zb_mn",
        "_p_l",
        "_i_l",
        "_R_basis",
        "_Z_basis",
        "_L_basis",
        "_Rb_basis",
        "_Zb_basis",
        "_p_basis",
        "_i_basis",
        "_index",
        "_bdry_mode",
        "_zeta_ratio",
    ]

    _object_lib_ = {
        "PowerSeries": PowerSeries,
        "FourierSeries": FourierSeries,
        "DoubleFourierSeries": DoubleFourierSeries,
        "FourierZernikeBasis": FourierZernikeBasis,
        "LinearGrid": LinearGrid,
        "ConcentricGrid": ConcentricGrid,
    }

    def __init__(
        self,
        inputs: dict = None,
        load_from=None,
        file_format: str = "hdf5",
        obj_lib=None,
    ) -> None:
        """Initializes a Configuration

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs with the following required keys:
                Psi : float, total toroidal flux (in Webers) within LCFS
                NFP : int, number of field periods
                L : int, radial resolution
                M : int, poloidal resolution
                N : int, toroidal resolution
                profiles : ndarray, array of profile coeffs [l, p_l, i_l]
                boundary : ndarray, array of boundary coeffs [m, n, Rb_mn, Zb_mn]
            And the following optional keys:
                sym : bool, is the problem stellarator symmetric or not, default is False
                index : str, type of Zernike indexing scheme to use, default is 'ansi'
                bdry_mode : str, how to calculate error at bdry, default is 'spectral'
                zeta_ratio : float, Multiplier on the toroidal derivatives. Default = 1.0.
                axis : ndarray, array of magnetic axis coeffs [n, R0_n, Z0_n]
                x : ndarray, state vector [R_lmn, Z_lmn, L_lmn]
                R_lmn : ndarray, spectral coefficients of R
                Z_lmn : ndarray, spectral coefficients of Z
                L_lmn : ndarray, spectral coefficients of lambda
        load_from : str file path OR file instance
            file to initialize from
        file_format : str
            file format of file initializing from. Default is 'hdf5'

        Returns
        -------
        None

        """
        self._file_format_ = file_format
        if load_from is None:
            self.inputs = inputs
            self._init_from_inputs_(inputs=inputs)
        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )
        self._make_labels()

    def _init_from_inputs_(self, inputs: dict = None) -> None:
        # required inputs
        try:
            self._Psi = inputs["Psi"]
            self._NFP = inputs["NFP"]
            self._L = inputs["L"]
            self._M = inputs["M"]
            self._N = inputs["N"]
            profiles = inputs["profiles"]
            boundary = inputs["boundary"]
        except:
            raise ValueError(colored("input dict does not contain proper keys", "red"))

        # optional inputs
        self._sym = inputs.get("sym", False)
        self._index = inputs.get("index", "ansi")
        self._bdry_mode = inputs.get("bdry_mode", "spectral")
        self._zeta_ratio = inputs.get("zeta_ratio", 1.0)

        # keep track of where it came from
        self._parent = None
        self._children = []

        # stellarator symmetry for bases
        if self._sym:
            self._R_sym = "cos"
            self._Z_sym = "sin"
        else:
            self._R_sym = None
            self._Z_sym = None

        # create bases
        self._R_basis = FourierZernikeBasis(
            L=self._L,
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._R_sym,
            index=self._index,
        )
        self._Z_basis = FourierZernikeBasis(
            L=self._L,
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._Z_sym,
            index=self._index,
        )
        self._L_basis = FourierZernikeBasis(
            L=self._L,
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._Z_sym,
            index=self._index,
        )
        self._Rb_basis = DoubleFourierSeries(
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._R_sym,
        )
        self._Zb_basis = DoubleFourierSeries(
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._Z_sym,
        )
        self._p_basis = PowerSeries(L=int(np.max(profiles[:, 0])))
        self._i_basis = PowerSeries(L=int(np.max(profiles[:, 0])))

        # format profiles
        self._p_l, self._i_l = format_profiles(profiles, self._p_basis, self._i_basis)

        # format boundary
        self._Rb_mn, self._Zb_mn = format_boundary(
            boundary, self._Rb_basis, self._Zb_basis, self._bdry_mode
        )

        # check if state vector is provided
        try:
            self._x = inputs["x"]
            self._R_lmn, self._Z_lmn, self._L_lmn = unpack_state(
                self._x,
                self._R_basis.num_modes,
                self._Z_basis.num_modes,
            )
        # default initial guess
        except:
            axis = inputs.get("axis", boundary[np.where(boundary[:, 0] == 0)[0], 1:])
            # check if R is provided
            try:
                self._R_lmn = inputs["R_lmn"]
            except:
                self._R_lmn = initial_guess(
                    self._R_basis, self._Rb_mn, self._Rb_basis, axis
                )
            # check if Z is provided
            try:
                self._Z_lmn = inputs["Z_lmn"]
            except:
                self._Z_lmn = initial_guess(
                    self._Z_basis, self._Zb_mn, self._Zb_basis, axis
                )
            # check if lambda is provided
            try:
                self._L_lmn = inputs["L_lmn"]
            except:
                self._L_lmn = np.zeros((self._L_basis.num_modes,))
            self._x = np.concatenate([self._R_lmn, self._Z_lmn, self._L_lmn])

    @property
    def parent(self):
        """Pointer to the configuration this was derived from"""
        return self._parent

    @property
    def children(self):
        """List of configurations that were derived from this one"""
        return self._children

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this equilibrium"""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        new._parent = self
        self._children.append(new)
        return new

    def change_resolution(
        self, L: int = None, M: int = None, N: int = None, *args, **kwargs
    ) -> None:

        L_change = M_change = N_change = False
        if L is not None and L != self._L:
            L_change = True
            self._L = L
        if M is not None and M != self._M:
            M_change = True
            self._M = M
        if N is not None and N != self._N:
            N_change = True
            self._N = N

        if not np.any([L_change, M_change, N_change]):
            return

        old_modes_R = self._R_basis.modes
        old_modes_Z = self._Z_basis.modes
        old_modes_L = self._L_basis.modes

        # create bases
        self._R_basis = FourierZernikeBasis(
            L=self._L,
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._R_sym,
            index=self._index,
        )
        self._Z_basis = FourierZernikeBasis(
            L=self._L,
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._Z_sym,
            index=self._index,
        )
        self._L_basis = FourierZernikeBasis(
            L=self._L,
            M=self._M,
            N=self._N,
            NFP=self._NFP,
            sym=self._Z_sym,
            index=self._index,
        )

        self._R_lmn = copy_coeffs(self._R_lmn, old_modes_R, self._R_basis.modes)
        self._Z_lmn = copy_coeffs(self._Z_lmn, old_modes_Z, self._Z_basis.modes)
        self._L_lmn = copy_coeffs(self._L_lmn, old_modes_L, self._L_basis.modes)

        # state vector
        self._x = np.concatenate([self._R_lmn, self._Z_lmn, self._L_lmn])
        self._make_labels()

    @property
    def sym(self) -> bool:
        return self._sym

    @property
    def Psi(self) -> float:
        """ float, total toroidal flux (in Webers) within LCFS"""
        return self._Psi

    @Psi.setter
    def Psi(self, Psi) -> None:
        self._Psi = Psi

    @property
    def NFP(self) -> int:
        """ int, number of field periods"""
        return self._NFP

    @NFP.setter
    def NFP(self, NFP) -> None:
        self._NFP = NFP

    @property
    def L(self) -> int:
        return self._L

    @property
    def M(self) -> int:
        return self._M

    @property
    def N(self) -> int:
        return self._N

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x) -> None:
        self._x = x
        self._R_lmn, self._Z_lmn, self._L_lmn = unpack_state(
            self._x,
            self._R_basis.num_modes,
            self._Z_basis.num_modes,
        )

    @property
    def R_lmn(self):
        """ spectral coefficients of R """
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, R_lmn) -> None:
        self._R_lmn = R_lmn
        self._x = np.concatenate([self._R_lmn, self._Z_lmn, self._L_lmn])

    @property
    def Z_lmn(self):
        """ spectral coefficients of Z """
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, Z_lmn) -> None:
        self._Z_lmn = Z_lmn
        self._x = np.concatenate([self._R_lmn, self._Z_lmn, self._L_lmn])

    @property
    def L_lmn(self):
        """ spectral coefficients of lambda """
        return self._L_lmn

    @L_lmn.setter
    def L_lmn(self, L_lmn) -> None:
        self._L_lmn = L_lmn
        self._x = np.concatenate([self._R_lmn, self._Z_lmn, self._L_lmn])

    @property
    def Rb_mn(self):
        """ spectral coefficients of R at the boundary"""
        return self._Rb_mn

    @Rb_mn.setter
    def Rb_mn(self, Rb_mn) -> None:
        self._Rb_mn = Rb_mn

    @property
    def Zb_mn(self):
        """ spectral coefficients of Z at the boundary"""
        return self._Zb_mn

    @Zb_mn.setter
    def Zb_mn(self, Zb_mn) -> None:
        self._Zb_mn = Zb_mn

    @property
    def p_l(self):
        """ spectral coefficients of pressure """
        return self._p_l

    @p_l.setter
    def p_l(self, p_l) -> None:
        self._p_l = p_l

    @property
    def i_l(self):
        """ spectral coefficients of iota """
        return self._i_l

    @i_l.setter
    def i_l(self, i_l) -> None:
        self._i_l = i_l

    @property
    def R_basis(self) -> FourierZernikeBasis:
        """
        Spectral basis for R

        Returns
        -------
        FourierZernikeBasis

        """
        return self._R_basis

    @R_basis.setter
    def R_basis(self, R_basis: FourierZernikeBasis) -> None:
        self._R_basis = R_basis
        self._make_labels()

    @property
    def Z_basis(self) -> FourierZernikeBasis:
        """
        Spectral basis for Z

        Returns
        -------
        FourierZernikeBasis

        """
        return self._Z_basis

    @Z_basis.setter
    def Z_basis(self, Z_basis: FourierZernikeBasis) -> None:
        self._Z_basis = Z_basis
        self._make_labels()

    @property
    def L_basis(self) -> FourierZernikeBasis:
        """
        Spectral basis for lambda

        Returns
        -------
        FourierZernikeBasis

        """
        return self._L_basis

    @L_basis.setter
    def L_basis(self, L_basis: FourierZernikeBasis) -> None:
        self._L_basis = L_basis
        self._make_labels()

    @property
    def Rb_basis(self) -> DoubleFourierSeries:
        """
        Spectral basis for R at the boundary

        Returns
        -------
        DoubleFourierSeries

        """
        return self._Rb_basis

    @Rb_basis.setter
    def Rb_basis(self, Rb_basis: DoubleFourierSeries) -> None:
        self._Rb_basis = Rb_basis

    @property
    def Zb_basis(self) -> DoubleFourierSeries:
        """
        Spectral basis for Z at the boundary

        Returns
        -------
        DoubleFourierSeries

        """
        return self._Zb_basis

    @Zb_basis.setter
    def Zb_basis(self, Zb_basis: DoubleFourierSeries) -> None:
        self._Zb_basis = Zb_basis

    @property
    def p_basis(self) -> PowerSeries:
        """
        Spectral basis for pressure

        Returns
        -------
        PowerSeries

        """
        return self._p_basis

    @p_basis.setter
    def p_basis(self, p_basis: PowerSeries) -> None:
        self._p_basis = p_basis

    @property
    def i_basis(self) -> PowerSeries:
        """
        Spectral basis for rotational transform

        Returns
        -------
        PowerSeries

        """
        return self._i_basis

    @i_basis.setter
    def i_basis(self, i_basis: PowerSeries) -> None:
        self._i_basis = i_basis

    def _make_labels(self):
        R_label = ["R_{},{},{}".format(l, m, n) for l, m, n in self._R_basis.modes]
        Z_label = ["Z_{},{},{}".format(l, m, n) for l, m, n in self._Z_basis.modes]
        L_label = ["L_{},{},{}".format(l, m, n) for l, m, n in self._L_basis.modes]

        x_label = R_label + Z_label + L_label

        self.xlabel = {i: val for i, val in enumerate(x_label)}
        self.rev_xlabel = {val: i for i, val in self.xlabel.items()}

    def get_xlabel_by_idx(self, idx):
        """Find which mode corresponds to a given entry in x"""

        idx = np.atleast_1d(idx)
        labels = [self.xlabel.get(i, None) for i in idx]
        return labels

    def get_idx_by_xlabel(self, labels):
        """Find which index of x corresponds to a given mode"""

        if not isinstance(labels, (list, tuple)):
            labels = list(labels)
        idx = [self.rev_xlabel.get(label, None) for label in labels]
        return np.array(idx)

    def compute_profiles(self, grid: Grid) -> dict:
        """Computes magnetic flux, pressure, and rotational transform profiles.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        profiles : dict
            dictionary of ndarray, shape(num_nodes,) of profiles.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=0)
        Z_transform = Transform(grid, self._Z_basis, derivs=0)
        L_transform = Transform(grid, self._L_basis, derivs=0)
        p_transform = Transform(grid, self._p_basis, derivs=1)
        i_transform = Transform(grid, self._i_basis, derivs=1)

        profiles = compute_profiles(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._Z_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return profiles

    def compute_toroidal_coords(self, grid: Grid) -> dict:
        """Computes toroidal coordinates from polar coordinates.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        toroidal_coords : dict
            dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=0)
        Z_transform = Transform(grid, self._Z_basis, derivs=0)
        L_transform = Transform(grid, self._L_basis, derivs=0)
        p_transform = Transform(grid, self._p_basis, derivs=0)
        i_transform = Transform(grid, self._i_basis, derivs=0)

        toroidal_coords = compute_toroidal_coords(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return toroidal_coords

    def compute_cartesian_coords(self, grid: Grid) -> dict:
        """Computes cartesian coordinates from toroidal coordinates.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        toroidal_coords : dict
            dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=0)
        Z_transform = Transform(grid, self._Z_basis, derivs=0)
        L_transform = Transform(grid, self._L_basis, derivs=0)
        p_transform = Transform(grid, self._p_basis, derivs=0)
        i_transform = Transform(grid, self._i_basis, derivs=0)

        (cartesian_coords, toroidal_coords) = compute_cartesian_coords(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return cartesian_coords

    def compute_covariant_basis(self, grid: Grid) -> dict:
        """Computes covariant basis vectors.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        cov_basis : dict
            dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
            Keys are of the form 'e_x_y', meaning the covariant basis vector in
            the x direction, differentiated wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=1)
        Z_transform = Transform(grid, self._Z_basis, derivs=1)
        L_transform = Transform(grid, self._L_basis, derivs=0)
        p_transform = Transform(grid, self._p_basis, derivs=0)
        i_transform = Transform(grid, self._i_basis, derivs=0)

        (cov_basis, toroidal_coords) = compute_covariant_basis(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return cov_basis

    def compute_jacobian(self, grid: Grid) -> dict:
        """Computes coordinate system jacobian.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        jacobian : dict
            dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
            Keys are of the form 'g_x' meaning the x derivative of the coordinate
            system jacobian g.

        """
        R_transform = Transform(grid, self._R_basis, derivs=1)
        Z_transform = Transform(grid, self._Z_basis, derivs=1)
        L_transform = Transform(grid, self._L_basis, derivs=0)
        p_transform = Transform(grid, self._p_basis, derivs=0)
        i_transform = Transform(grid, self._i_basis, derivs=0)

        (jacobian, cov_basis, toroidal_coords) = compute_jacobian(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return jacobian

    def compute_contravariant_basis(self, grid: Grid) -> dict:
        """Computes contravariant basis vectors.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        con_basis : dict
            dictionary of ndarray, shape(3,num_nodes), of contravariant basis vectors.
            Keys are of the form 'e^x_y', meaning the contravariant basis vector
            in the x direction, differentiated wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=1)
        Z_transform = Transform(grid, self._Z_basis, derivs=1)
        L_transform = Transform(grid, self._L_basis, derivs=0)
        p_transform = Transform(grid, self._p_basis, derivs=0)
        i_transform = Transform(grid, self._i_basis, derivs=0)

        (con_basis, jacobian, cov_basis, toroidal_coords) = compute_contravariant_basis(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return con_basis

    def compute_magnetic_field(self, grid: Grid) -> dict:
        """Computes magnetic field components.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        magnetic_field: dict
            dictionary of ndarray, shape(num_nodes,) of magnetic field components.
            Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
            or contravariant (B^x) component of the magnetic field, with the
            derivative wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=2)
        Z_transform = Transform(grid, self._Z_basis, derivs=2)
        L_transform = Transform(grid, self._L_basis, derivs=1)
        p_transform = Transform(grid, self._p_basis, derivs=1)
        i_transform = Transform(grid, self._i_basis, derivs=1)

        (
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_magnetic_field_magnitude_axis(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return magnetic_field

    def compute_current_density(self, grid: Grid) -> dict:
        """Computes current density field components.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        current_density : dict
            dictionary of ndarray, shape(num_nodes,), of current density components.
            Keys are of the form 'J^x_y' meaning the contravariant (J^x)
            component of the current, with the derivative wrt to y.

        """
        R_transform = Transform(grid, self._R_basis, derivs=2)
        Z_transform = Transform(grid, self._Z_basis, derivs=2)
        L_transform = Transform(grid, self._L_basis, derivs=2)
        p_transform = Transform(grid, self._p_basis, derivs=1)
        i_transform = Transform(grid, self._i_basis, derivs=1)

        (
            current_density,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_current_density(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return current_density

    def compute_force_error(self, grid: Grid) -> dict:
        """Computes force errors and magnitude.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        force_error : dict
            dictionary of ndarray, shape(num_nodes,), of force error components.
            Keys are of the form 'F_x' meaning the covariant (F_x) component of the
            force error.

        """
        R_transform = Transform(grid, self._R_basis, derivs=2)
        Z_transform = Transform(grid, self._Z_basis, derivs=2)
        L_transform = Transform(grid, self._L_basis, derivs=2)
        p_transform = Transform(grid, self._p_basis, derivs=1)
        i_transform = Transform(grid, self._i_basis, derivs=1)

        (
            force_error,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_force_error_magnitude(
            self._Psi,
            self._R_lmn,
            self._Z_lmn,
            self._L_lmn,
            self._p_l,
            self._i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self._zeta_ratio,
        )

        return force_error


# these functions are needed to format the input arrays


def format_profiles(profiles, p_basis: PowerSeries, i_basis: PowerSeries):
    """Formats profile input arrays

    Parameters
    ----------
    profiles : ndarray, shape(Nbdry,3)
        array of fourier coeffs [l, p, i]
    p_basis : PowerSeries
        spectral basis for p_l coefficients
    i_basis : PowerSeries
        spectral basis for i_l coefficients

    Returns
    -------
    p_l : ndarray
        spectral coefficients for pressure profile
    i_l : ndarray
        spectral coefficients for rotational transform profile

    """
    p_l = np.zeros((p_basis.num_modes,))
    i_l = np.zeros((i_basis.num_modes,))

    for l, p, i in profiles:
        idx_p = np.where(p_basis.modes[:, 0] == int(l))[0]
        idx_i = np.where(i_basis.modes[:, 0] == int(l))[0]
        p_l = put(p_l, idx_p, p)
        i_l = put(i_l, idx_i, i)

    return p_l, i_l


def format_boundary(
    boundary,
    Rb_basis: DoubleFourierSeries,
    Zb_basis: DoubleFourierSeries,
    mode: str = "spectral",
):
    """Formats boundary arrays and converts between real and fourier representations

    Parameters
    ----------
    boundary : ndarray, shape(Nbdry,4)
        array of fourier coeffs [m, n, R1, Z1]
        or array of real space coordinates, [theta, phi, R, Z]
    Rb_basis : DoubleFourierSeries
        spectral basis for Rb_mn coefficients
    Zb_basis : DoubleFourierSeries
        spectral basis for Zb_mn coefficients
    mode : str
        one of 'real', 'spectral'. Whether bdry is specified in real or spectral space.

    Returns
    -------
    Rb_mn : ndarray
        spectral coefficients for R boundary
    Zb_mn : ndarray
        spectral coefficients for Z boundary

    """
    if mode == "real":
        theta = boundary[:, 0]
        phi = boundary[:, 1]
        rho = np.ones_like(theta)

        nodes = np.array([rho, theta, phi]).T
        grid = Grid(nodes)
        R1_tform = Transform(grid, Rb_basis)
        Z1_tform = Transform(grid, Zb_basis)

        # fit real data to spectral coefficients
        Rb_mn = R1_tform.fit(boundary[:, 2])
        Zb_mn = Z1_tform.fit(boundary[:, 3])

    else:
        Rb_mn = np.zeros((Rb_basis.num_modes,))
        Zb_mn = np.zeros((Zb_basis.num_modes,))

        for m, n, R1, Z1 in boundary:
            idx_R = np.where(
                np.logical_and.reduce(
                    (Rb_basis.modes[:, 1] == int(m), Rb_basis.modes[:, 2] == int(n))
                )
            )[0]
            idx_Z = np.where(
                np.logical_and.reduce(
                    (Zb_basis.modes[:, 1] == int(m), Zb_basis.modes[:, 2] == int(n))
                )
            )[0]
            Rb_mn = put(Rb_mn, idx_R, R1)
            Zb_mn = put(Zb_mn, idx_Z, Z1)

    return Rb_mn, Zb_mn


def initial_guess(
    x_basis: FourierZernikeBasis, b_mn, b_basis: DoubleFourierSeries, axis
):
    """creates the coefficients x_lmn based on the boundary coefficients b_mn"""

    x_lmn = np.zeros((x_basis.num_modes,))
    for k in range(b_basis.num_modes):
        m = b_basis.modes[k, 1]
        n = b_basis.modes[k, 2]
        idx = np.where(
            np.logical_and.reduce(
                (
                    x_basis.modes[:, 0] == np.abs(m),
                    x_basis.modes[:, 1] == m,
                    x_basis.modes[:, 2] == n,
                )
            ),
        )[0]
        if m == 0:
            idx2 = np.where(
                np.logical_and.reduce(
                    (
                        x_basis.modes[:, 0] == np.abs(m) + 2,
                        x_basis.modes[:, 1] == m,
                        x_basis.modes[:, 2] == n,
                    )
                )
            )[0]
            x0 = np.where(axis[:, 0] == n, axis[:, 1], b_mn[k])[0]
            x_lmn = put(x_lmn, idx, x0)
            x_lmn = put(x_lmn, idx2, b_mn[k] - x0)
        else:
            x_lmn = put(x_lmn, idx, b_mn[k])

    return x_lmn
