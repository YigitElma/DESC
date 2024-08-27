"""Benchmarks for timing comparison on cpu (that are small enough to run on CI)."""

import jax
import pytest

import desc

desc.set_device("cpu")
import desc.examples
from desc.basis import FourierZernikeBasis
from desc.grid import ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.optimize import LinearConstraintProjection
from desc.transform import Transform


@pytest.mark.benchmark()
def test_build_transform_fft_lowres(benchmark):
    """Test time to build a transform (after compilation) for low resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 5
        M = 5
        N = 5
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_dshape_current(benchmark):
    """Benchmark compiling objective."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("DSHAPE_current")
        objective = LinearConstraintProjection(
            get_equilibrium_objective(eq),
            ObjectiveFunction(
                maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
            ),
        )
        objective.build(eq)
        args = (
            objective,
            eq,
        )
        kwargs = {}
        return args, kwargs

    def run(objective, eq):
        objective.compile()

    benchmark.pedantic(run, setup=setup, rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_atf(benchmark):
    """Benchmark computing objective."""
    jax.clear_caches()
    eq = desc.examples.get("ATF")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x, objective):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=50, iterations=1)
