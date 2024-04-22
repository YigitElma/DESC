"""Test bounce integral methods."""

import inspect
from functools import partial

import numpy as np
import pytest
from scipy import integrate

# TODO: can use the one from interpax once .solve() is implemented
from scipy.interpolate import CubicHermiteSpline
from scipy.special import ellipkm1

from desc.backend import complex_sqrt, flatnonzero
from desc.compute.bounce_integral import (
    _affine_bijection_forward,
    _bounce_quadrature,
    _filter_not_nan,
    _poly_der,
    _poly_root,
    _poly_val,
    affine_bijection_reverse,
    automorphism_arcsin,
    automorphism_sin,
    bounce_integral,
    bounce_points,
    composite_linspace,
    grad_affine_bijection_reverse,
    grad_automorphism_arcsin,
    grad_automorphism_sin,
    pitch_of_extrema,
    plot_field_line,
    take_mask,
    tanh_sinh_quad,
)
from desc.compute.utils import dot, safediv
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import desc_grid_from_field_line_coords
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.objectives import (
    ObjectiveFromUser,
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile
from desc.utils import only1


@partial(np.vectorize, signature="(m)->()")
def _last_value(a):
    """Return the last non-nan value in ``a``."""
    a = np.ravel(a)[::-1]
    idx = np.squeeze(flatnonzero(~np.isnan(a), size=1, fill_value=0))
    return a[idx]


def _sqrt(x):
    """Reproduces jnp.sqrt with np.sqrt."""
    x = complex_sqrt(x)
    x = np.where(np.isclose(np.imag(x), 0), np.real(x), np.nan)
    return x


@pytest.mark.unit
def test_mask_operations():
    """Test custom masked array operation."""
    rows = 5
    cols = 7
    a = np.random.rand(rows, cols)
    nan_idx = np.random.choice(rows * cols, size=(rows * cols) // 2, replace=False)
    a.ravel()[nan_idx] = np.nan
    taken = take_mask(a, ~np.isnan(a))
    last = _last_value(taken)
    for i in range(rows):
        desired = a[i, ~np.isnan(a[i])]
        assert np.array_equal(
            taken[i],
            np.pad(desired, (0, cols - desired.size), constant_values=np.nan),
            equal_nan=True,
        ), "take_mask has bugs."
        assert np.array_equal(
            last[i],
            desired[-1] if desired.size else np.nan,
            equal_nan=True,
        ), "flatnonzero has bugs."


@pytest.mark.unit
def test_reshape_convention():
    """Test the reshaping convention separates data across field lines."""
    rho = np.linspace(0, 1, 3)
    alpha = np.linspace(0, 2 * np.pi, 4)
    zeta = np.linspace(0, 6 * np.pi, 5)
    r, a, z = map(np.ravel, np.meshgrid(rho, alpha, zeta, indexing="ij"))
    # functions of zeta should separate along first two axes
    # since those are contiguous, this should work
    f = z.reshape(-1, zeta.size)
    for i in range(1, f.shape[0]):
        np.testing.assert_allclose(f[i - 1], f[i])
    # likewise for rho
    f = r.reshape(rho.size, -1)
    for i in range(1, f.shape[-1]):
        np.testing.assert_allclose(f[:, i - 1], f[:, i])
    # test final reshape of bounce integral result won't mix data
    f = (a**2 + z).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[0]):
        np.testing.assert_allclose(f[i - 1], f[i])
    f = (r**2 + z).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[1]):
        np.testing.assert_allclose(f[:, i - 1], f[:, i])
    f = (r**2 + a).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[-1]):
        np.testing.assert_allclose(f[..., i - 1], f[..., i])

    err_msg = "The ordering conventions are required for correctness."
    assert "P, S, N" in inspect.getsource(bounce_points), err_msg
    src = inspect.getsource(bounce_integral)
    assert "S, knots.size" in src, err_msg
    assert "pitch.shape[0], rho.size, alpha.size" in src, err_msg
    src = inspect.getsource(desc_grid_from_field_line_coords)
    assert 'indexing="ij"' in src, err_msg
    assert 'meshgrid(rho, alpha, zeta, indexing="ij")' in src, err_msg


@pytest.mark.unit
def test_poly_root():
    """Test vectorized computation of cubic polynomial exact roots."""
    cubic = 4
    c = np.arange(-24, 24).reshape(cubic, 6, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    constant = np.broadcast_to(np.arange(c.shape[-1]), c.shape[1:])
    constant = np.stack([constant, constant])
    root = _poly_root(c, constant, sort=True)

    for i in range(constant.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                d = c[-1, j, k] - constant[i, j, k]
                np.testing.assert_allclose(
                    actual=root[i, j, k],
                    desired=np.sort(np.roots([*c[:-1, j, k], d])),
                )

    c = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, -8, 12],
            [1, -6, 11, -6],
            [0, -6, 11, -2],
        ]
    )
    root = _poly_root(c.T, sort=True, distinct=True)
    for j in range(c.shape[0]):
        unique_roots = np.unique(np.roots(c[j]))
        if j == 4:
            # There are only two distinct roots.
            unique_roots = unique_roots[[0, 1]]
        np.testing.assert_allclose(
            actual=_filter_not_nan(root[j]),
            desired=unique_roots,
            err_msg=str(j),
        )
    c = np.array([0, 1, -1, -8, 12])
    np.testing.assert_allclose(
        actual=_filter_not_nan(_poly_root(c, sort=True, distinct=True)),
        desired=np.unique(np.roots(c)),
    )


@pytest.mark.unit
def test_poly_der():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    c = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    derivative = _poly_der(c)
    for j in range(c.shape[1]):
        for k in range(c.shape[2]):
            np.testing.assert_allclose(
                actual=derivative[:, j, k], desired=np.polyder(c[:, j, k])
            )


@pytest.mark.unit
def test_poly_val():
    """Test vectorized computation of polynomial evaluation."""

    def test(x, c):
        val = _poly_val(x=x, c=c)
        if val.ndim != max(x.ndim, c.ndim - 1):
            raise ValueError(f"Incompatible shapes {x.shape} and {c.shape}.")
        for index in np.ndindex(c.shape[1:]):
            idx = (..., *index)
            np.testing.assert_allclose(
                actual=val[idx],
                desired=np.poly1d(c[idx])(x[idx]),
                err_msg=f"Failed with shapes {x.shape} and {c.shape}.",
            )

    quartic = 5
    c = np.arange(-60, 60).reshape(quartic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    x = np.linspace(0, 20, c.shape[1] * c.shape[2]).reshape(c.shape[1], c.shape[2])
    test(x, c)

    x = np.stack([x, x * 2], axis=0)
    x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
    # make sure broadcasting won't hide error in implementation
    assert np.unique(x.shape).size == x.ndim
    assert c.shape[1:] == x.shape[x.ndim - (c.ndim - 1) :]
    assert np.unique((c.shape[0],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
    test(x, c)


@pytest.mark.unit
def test_pitch_of_extrema():
    """Test that these pitch intersect extrema of |B|."""
    start = -np.pi
    end = -2 * start
    k = np.linspace(start, end, 5)
    B = CubicHermiteSpline(
        k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
    )
    B_z_ra = B.derivative()
    pitch_scipy = 1 / B(B_z_ra.roots(extrapolate=False))
    pitch = pitch_of_extrema(k, B.c, B_z_ra.c)
    np.testing.assert_allclose(_filter_not_nan(pitch), pitch_scipy)


@pytest.mark.unit
def test_composite_linspace():
    """Test this utility function useful for Newton-Cotes integration over pitch."""
    B_min_tz = np.array([0.1, 0.2])
    B_max_tz = np.array([1, 3])
    pitch_knot = np.linspace(1 / B_min_tz, 1 / B_max_tz, num=5)
    b_knot = 1 / pitch_knot
    print()
    print(b_knot)
    b = composite_linspace(b_knot, resolution=3)
    print()
    print(b)
    np.testing.assert_allclose(b, np.sort(b, axis=0), atol=0, rtol=0)
    for i in range(pitch_knot.shape[0]):
        for j in range(pitch_knot.shape[1]):
            assert only1(np.isclose(b_knot[i, j], b[:, j]).tolist())


@pytest.mark.unit
def test_bounce_points():
    """Test that bounce points are computed correctly."""

    def test_bp1_first(plot):
        start = np.pi / 3
        end = 6 * np.pi
        knots = np.linspace(start, end, 5)
        B = CubicHermiteSpline(knots, np.cos(knots), -np.sin(knots))
        pitch = 2
        bp1, bp2 = bounce_points(pitch, knots, B.c, B.derivative().c, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        if plot:
            plot_field_line(B, pitch, bp1, bp2)
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    def test_bp2_first(plot):
        start = -3 * np.pi
        end = -start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch = 2
        bp1, bp2 = bounce_points(pitch, k, B.c, B.derivative().c, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        if plot:
            plot_field_line(B, pitch, bp1, bp2)
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[1::2])
        np.testing.assert_allclose(bp2, intersect[0::2][1:])

    def test_bp1_before_extrema(plot):
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[3]
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        if plot:
            plot_field_line(B, pitch, bp1, bp2)
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[1], 1.9827671337414938)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[1]), bp1[1])
        np.testing.assert_allclose(bp1, intersect[[1, 2]])
        np.testing.assert_allclose(bp2, intersect[[2, 3]])

    def test_bp2_before_extrema(plot):
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 4,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 4,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2]
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        if plot:
            plot_field_line(B, pitch, bp1, bp2)
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[[0, -2]])
        np.testing.assert_allclose(bp2, intersect[[1, -1]])

    def test_extrema_first_and_before_bp1(plot):
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 20,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 20,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2]
        bp1, bp2 = bounce_points(pitch, k[2:], B.c[:, 2:], B_z_ra.c[:, 2:], check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        if plot:
            plot_field_line(B, pitch, bp1, bp2, start=k[2])
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], 0.8353192766102349)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[0]), bp1[0])
        intersect = intersect[intersect >= k[2]]
        np.testing.assert_allclose(bp1, intersect[[0, 1, 3]])
        np.testing.assert_allclose(bp2, intersect[[0, 2, 4]])

    def test_extrema_first_and_before_bp2(plot):
        start = -1.2 * np.pi
        end = -2 * start + 1
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 10,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 10,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[1]
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        if plot:
            plot_field_line(B, pitch, bp1, bp2)
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], -0.6719044147510538)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[0]), bp1[0])
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    # These are all the unique cases, if all tests pass then the bounce_points
    # should work correctly for all inputs.
    test_bp1_first(True)
    test_bp2_first(True)
    test_bp1_before_extrema(True)
    test_bp2_before_extrema(True)
    # In theory, this test should only pass if distinct=True when computing the
    # intersections in bounce points. However, we can get lucky due to floating
    # point errors, and it may also pass when distinct=False.
    test_extrema_first_and_before_bp1(True)
    test_extrema_first_and_before_bp2(True)


@pytest.mark.unit
def test_automorphism():
    """Test automorphisms."""
    a, b = -312, 786
    x = np.linspace(a, b, 10)
    y = _affine_bijection_forward(x, a, b)
    x_1 = affine_bijection_reverse(y, a, b)
    np.testing.assert_allclose(x_1, x)
    np.testing.assert_allclose(_affine_bijection_forward(x_1, a, b), y)
    np.testing.assert_allclose(automorphism_arcsin(automorphism_sin(y)), y)
    np.testing.assert_allclose(automorphism_sin(automorphism_arcsin(y)), y)

    np.testing.assert_allclose(
        grad_affine_bijection_reverse(a, b),
        1 / (2 / (b - a)),
    )
    np.testing.assert_allclose(
        grad_automorphism_sin(y),
        1 / grad_automorphism_arcsin(automorphism_sin(y)),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        1 / grad_automorphism_arcsin(y),
        grad_automorphism_sin(automorphism_arcsin(y)),
        atol=1e-14,
    )


@pytest.mark.unit
def test_bounce_quadrature():
    """Test bounce integral matches elliptic integral."""
    p = 1e-4
    m = 1 - p
    # Some prime number that doesn't appear anywhere in calculation.
    # Ensures no lucky cancellation occurs from this test case since otherwise
    # (bp2 - bp1) / pi = pi / (bp2 - bp1) which could mask errors since pi
    # appears often in transformations.
    v = 7
    truth = v * 2 * ellipkm1(p)
    rtol = 1e-3

    bp1 = -np.pi / 2 * v
    bp2 = -bp1
    knots = np.linspace(bp1, bp2, 15)
    bp1 = np.atleast_3d(bp1)
    bp2 = np.atleast_3d(bp2)
    B_sup_z = np.ones((1, knots.size))
    B = (np.sin(knots / v) ** 2).reshape(1, -1)
    B_z_ra = (np.sin(2 * knots / v) / v).reshape(1, -1)
    pitch = np.ones((1, 1))

    def integrand(B, pitch, Z):
        return 1 / np.sqrt(1 - pitch * m * B)

    # augment the singularity
    x_t, w_t = tanh_sinh_quad(18, grad_automorphism_arcsin)
    x_t = automorphism_arcsin(x_t)
    tanh_sinh_arcsin = _bounce_quadrature(
        bp1,
        bp2,
        x_t,
        w_t,
        integrand,
        [],
        B_sup_z,
        B,
        B_z_ra,
        pitch,
        knots,
        check=True,
    )
    np.testing.assert_allclose(tanh_sinh_arcsin, truth, rtol=rtol)
    x_g, w_g = np.polynomial.legendre.leggauss(16)
    # suppress the singularity
    w_g = w_g * grad_automorphism_sin(x_g)
    x_g = automorphism_sin(x_g)
    leg_gauss_sin = _bounce_quadrature(
        bp1,
        bp2,
        x_g,
        w_g,
        integrand,
        [],
        B_sup_z,
        B,
        B_z_ra,
        pitch,
        knots,
        check=True,
    )
    np.testing.assert_allclose(leg_gauss_sin, truth, rtol=rtol)


@pytest.mark.unit
def test_example_code():
    """Test example code in bounce_integral docstring."""

    def integrand_num(g_zz, B, pitch, Z):
        """Integrand in integral in numerator of bounce average."""
        f = (1 - pitch * B) * g_zz
        return safediv(f, _sqrt(1 - pitch * B))

    def integrand_den(B, pitch, Z):
        """Integrand in integral in denominator of bounce average."""
        return safediv(1, _sqrt(1 - pitch * B))

    eq = get("HELIOTRON")
    rho = np.linspace(1e-12, 1, 6)
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 5)

    bounce_integrate, items = bounce_integral(eq, rho, alpha, check=True)
    g_zz = eq.compute("g_zz", grid=items["grid_desc"])["g_zz"]
    pitch = pitch_of_extrema(items["knots"], items["B.c"], items["B_z_ra.c"])
    num = bounce_integrate(integrand_num, g_zz, pitch)
    den = bounce_integrate(integrand_den, [], pitch)
    average = num / den
    assert np.isfinite(average).any()

    # Now we can group the data by field line.
    average = average.reshape(pitch.shape[0], rho.size, alpha.size, -1)
    # The bounce averages stored at index i, j
    i, j = 0, 0
    print(average[:, i, j])
    # are the bounce averages along the field line with nodes
    # given in Clebsch-Type field-line coordinates ρ, α, ζ
    nodes = items["grid_fl"].nodes.reshape(rho.size, alpha.size, -1, 3)
    print(nodes[i, j])
    # for the pitch values stored in
    pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
    print(pitch[:, i, j])
    # Some of these bounce averages will evaluate as nan.
    # You should filter out these nan values when computing stuff.
    print(np.nansum(average, axis=-1))


# @pytest.mark.unit
def test_elliptic_integral_limit():
    """Test bounce integral matches elliptic integrals.

    In the limit of a low beta, large aspect ratio tokamak the bounce integral
    should converge to the elliptic integrals of the first kind.
    todo: would be nice to understand physics for why these are supposed
        to be proportional to bounce integral. Is this discussed in any book?
        Also, looking at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html
        Are we saying that in this limit, we expect that |B| ~ sin(t)^2, with m as the
        pitch angle? I assume that we want to add g_zz to the integrand in the
        definition of the function in the scipy documentation above,
        and after a change of variables the bounce points will be the endpoints of
        the integration.
        So this test will test whether the quadrature is accurate
        (and not whether the bounce points were accurate).

    """
    assert False, "Test not finished yet."
    L, M, N, NFP, sym = 6, 6, 6, 1, True
    surface = FourierRZToroidalSurface(
        R_lmn=[1.0, 0.1],
        Z_lmn=[0.0, -0.1],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
        sym=sym,
        NFP=NFP,
    )
    eq = Equilibrium(
        L=L,
        M=M,
        N=N,
        NFP=NFP,
        surface=surface,
        pressure=PowerSeriesProfile([1e2, 0, -1e2]),
        iota=PowerSeriesProfile([1, 0, 2]),
        Psi=1.0,
    )
    eq = solve_continuation_automatic(eq)[-1]

    def beta(grid, data):
        return data["<beta>_vol"]

    low_beta = 0.01
    # todo: error that objective function has no linear attribute?
    objective = ObjectiveFunction(
        (ObjectiveFromUser(fun=beta, eq=eq, target=low_beta),)
    )

    constraints = (*get_fixed_boundary_constraints(eq), get_equilibrium_objective(eq))
    opt = Optimizer("proximal-lsq-exact")
    eq, result = eq.optimize(
        objective=objective, constraints=constraints, optimizer=opt
    )
    print(result)

    rho = np.array([0.5])
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 10)
    knots = np.linspace(0, 6 * np.pi, 20)
    # TODO now compare result to elliptic integral
    bounce_integrate, items = bounce_integral(eq, rho, alpha, knots, check=True)
    pitch = pitch_of_extrema(knots, items["B.c"], items["B_z_ra.c"])
    bp1, bp2 = bounce_points(pitch, knots, items["B.c"], items["B_z_ra.c"], check=True)


@pytest.mark.unit
def test_integral_0(k=0.9, resolution=10):
    """4 / k * ellipkinc(np.arcsin(k), 1 / k**2)."""
    k = np.atleast_1d(k)
    bp1 = np.zeros_like(k)
    bp2 = np.arcsin(k)
    x, w = tanh_sinh_quad(resolution, grad_automorphism_arcsin)
    Z = affine_bijection_reverse(
        automorphism_arcsin(x), bp1[..., np.newaxis], bp2[..., np.newaxis]
    )
    k = k[..., np.newaxis]

    def integrand(Z, k):
        return safediv(4 / k, np.sqrt(1 - 1 / k**2 * np.sin(Z) ** 2))

    quad = np.dot(integrand(Z, k), w) * grad_affine_bijection_reverse(bp1, bp2)
    if k.size == 1:
        q = integrate.quad(integrand, bp1.item(), bp2.item(), args=(k.item(),))[0]
        np.testing.assert_allclose(quad, q, rtol=1e-5)
    return quad


@pytest.mark.unit
def test_integral_1(k=0.9, resolution=10):
    """4 * k * ellipeinc(np.arcsin(k), 1 / k**2)."""
    k = np.atleast_1d(k)
    bp1 = np.zeros_like(k)
    bp2 = np.arcsin(k)
    x, w = tanh_sinh_quad(resolution, grad_automorphism_arcsin)
    Z = affine_bijection_reverse(
        automorphism_arcsin(x), bp1[..., np.newaxis], bp2[..., np.newaxis]
    )
    k = k[..., np.newaxis]

    def integrand(Z, k):
        return 4 * k * np.sqrt(1 - 1 / k**2 * np.sin(Z) ** 2)

    quad = np.dot(integrand(Z, k), w) * grad_affine_bijection_reverse(bp1, bp2)
    if k.size == 1:
        q = integrate.quad(integrand, bp1.item(), bp2.item(), args=(k.item(),))[0]
        np.testing.assert_allclose(quad, q, rtol=1e-4)
    return quad


@pytest.mark.unit
def test_bounce_averaged_drifts():
    """Test bounce-averaged drift with analytical expressions.

    Calculate bounce-averaged drifts using the bounce-average routine and
    compare it with the analytical expression
    # Note 1: This test can be merged with the elliptic integral test as
    we do calculate elliptic integrals here
    # Note 2: Remove tests/test_equilibrium :: test_shifted_circle_geometry
    # once all the epsilons and Gammas have been implemented and tested
    """
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
    psi = 0.25  # normalized psi
    rho = np.sqrt(psi)
    data = eq.compute(["iota", "iota_r", "a", "rho", "psi"])

    # normalization
    Lref = data["a"]
    epsilon = Lref * rho
    psi_boundary = np.max(
        np.abs(data["psi"])
    )  # data["psi"][np.argmax(np.abs(data["psi"]))]
    Bref = 2 * np.abs(psi_boundary) / Lref**2

    # Creating a grid along a field line
    iota = np.interp(rho, data["rho"], data["iota"])
    shear = np.interp(rho, data["rho"], data["iota_r"])
    N = (2 * eq.M_grid) * 4 + 1
    zeta = np.linspace(-np.pi / iota, np.pi / iota, N)
    alpha = 0
    theta_PEST = alpha + iota * zeta
    coords1 = np.zeros((N, 3))
    coords1[:, 0] = np.broadcast_to(rho, N)
    coords1[:, 1] = theta_PEST
    coords1[:, 2] = zeta
    # c1 = eq.compute_theta_coords(coords1)  # noqa: E800
    # grid = Grid(c1, sort=False)  # noqa: E800
    # TODO: Request: The bounce integral operator should be able to take a grid.
    #       Response: Currently the API is such that the method does all the
    #                 above preprocessing for you. Let's test it for correctness
    #                 first then do this later.
    bounce_integrate, items = bounce_integral(
        # FIXME: Question
        #  add normalize to compute matching bounce points for the test
        #  below, but should everything related to B be normalized?
        #  or just things relevant for computing bounce points?
        #  e.g. should I normalize B dot e^zeta = B^zeta by Bref as well?
        #  Response (R.G.): Yes, it would be better to normalize everything
        #  All the quantities can be normalized using combinations of Lref
        #  and Bref. To see what normalizations I use see below.
        #  For B^zeta the normalization should be Lref/Bref. Since we only
        #  use b dot grad zeta, we need B^zeta/|B| * Lref
        eq,
        rho,
        alpha,
        knots=zeta,
        check=True,
        normalize=Bref,
    )
    data_keys = [
        "|grad(psi)|^2",
        "grad(psi)",
        "B",
        "iota",
        "|B|",
        "B^zeta",
        "cvdrift0",
        "cvdrift",
        "gbdrift",
    ]
    # FIXME (outside scope of the bounce branch):
    #  override_grid should not be required for the test to pass.
    #  and anytime override_grid is true we should print a blue warning.
    data_bounce = eq.compute(data_keys, grid=items["grid_desc"], override_grid=False)

    # normalizations
    bmag = data_bounce["|B|"] / Bref
    B0 = np.mean(bmag)
    bmag_an = B0 * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(bmag, bmag_an, atol=5e-3, rtol=5e-3)

    x = Lref * rho
    s_hat = -x / iota * shear / Lref
    gradpar = Lref * data_bounce["B^zeta"] / data_bounce["|B|"]
    gradpar_an = 2 * Lref * data_bounce["iota"] * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(gradpar, gradpar_an, atol=9e-3, rtol=5e-3)

    # Comparing coefficient calculation here with coefficients from compute/_metric
    cvdrift = -2 * np.sign(psi_boundary) * Bref * Lref**2 * rho * data_bounce["cvdrift"]
    gbdrift = -2 * np.sign(psi_boundary) * Bref * Lref**2 * rho * data_bounce["gbdrift"]
    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * data_bounce["|B|"] ** 2)
    alpha_MHD = -np.mean(dPdrho * 1 / data_bounce["iota"] ** 2 * 0.5)

    gds21 = (
        -np.sign(iota)
        * dot(data_bounce["grad(psi)"], data_bounce["grad(alpha)"])
        * s_hat
        / Bref
    )

    gds21_an = (
        -1 * s_hat * (s_hat * theta_PEST - alpha_MHD / bmag**4 * np.sin(theta_PEST))
    )
    np.testing.assert_allclose(gds21, gds21_an, atol=1.7e-2, rtol=5e-4)

    fudge_factor2 = 0.19
    gbdrift_an = fudge_factor2 * (
        -s_hat + (np.cos(theta_PEST) - gds21_an / s_hat * np.sin(theta_PEST))
    )

    fudge_factor3 = 0.07
    cvdrift_an = gbdrift_an + fudge_factor3 * alpha_MHD / bmag**2
    # Comparing coefficients with their analytical expressions
    np.testing.assert_allclose(gbdrift, gbdrift_an, atol=1.2e-2, rtol=5e-3)
    np.testing.assert_allclose(cvdrift, cvdrift_an, atol=1.8e-2, rtol=5e-3)

    # Values of pitch angle lambda for which to evaluate the bounce averages.
    pitch = np.linspace(1 / np.max(bmag), 1 / np.min(bmag), 11)
    pitch = pitch.reshape(pitch.shape[0], -1)

    k2 = 0.5 * ((1 - pitch * B0) / (pitch * B0 * epsilon) + 1)
    k = np.sqrt(k2)
    # Here are the notes that explain these integrals.
    # https://github.com/PlasmaControl/DESC/files/15010927/bavg.pdf.
    I_0 = test_integral_0(k)
    I_1 = test_integral_1(k)
    I_2 = 16 * k * I_0
    I_3 = 4 / 9 * (8 * k * (-1 + 2 * k2) * I_1 - 4 * k * (-1 + k2) * I_0)
    I_4 = (
        2
        * np.sqrt(2)
        / 3
        * (4 * np.sqrt(2) * k * (-1 + 2 * k2) * I_0 - 2 * (-1 + k2) * I_1)
    )
    I_5 = (
        2
        / 30
        * (32 * k * (1 - k2 + k2**2) * I_0 - 16 * k * (1 - 3 * k2 + 2 * k2**2) * I_1)
    )
    I_6 = 2 / 3 * (k * (-2 + 4 * k2) * I_0 - 4 * (-1 + k2) * I_1)
    I_7 = 4 / k * (2 * k2 * I_0 + (1 - 2 * k2) * I_1)

    bavg_drift_an = fudge_factor3 * dPdrho / B0**2 * I_1 - 0.5 * fudge_factor2 * (
        s_hat * (I_0 + I_1 + I_2 + I_3) + alpha_MHD / B0**4 * (I_4 + I_5) - (I_6 + I_7)
    )

    def integrand(cvdrift, gbdrift, B, pitch, Z):
        # The arguments to this function will be interpolated
        # onto the quadrature points before these quantities are evaluated.
        g = _sqrt(1 - pitch * B)
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    bavg_drift_num = bounce_integrate(
        integrand=integrand,
        # additional things to interpolate onto quadrature points besides B and pitch
        f=[cvdrift, gbdrift],
        pitch=pitch,
    )
    assert np.isfinite(bavg_drift_num).any(), "Quadrature failed."
    # there's only one field line on the grid, so squeeze out that axis
    bavg_drift_num = np.squeeze(bavg_drift_num, axis=1)
    for i in range(pitch.shape[0]):
        np.testing.assert_allclose(
            # this will have size equal to the number of bounce integrals
            # found along the field line (there's only one field line in the grid)
            _filter_not_nan(bavg_drift_num[i]),
            # this will have size equal to the number of nodes used to discretize
            # that field line, so this test will always fail.
            bavg_drift_an[i],
            atol=2e-2,
            rtol=1e-2,
            err_msg=f"Failed on index {i} for pitch {pitch[i]}",
        )
