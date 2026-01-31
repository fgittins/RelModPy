"""Test relativistic stellar structure class `Star`."""

from unittest import TestCase

import numpy as np

from relmodpy import Star
from relmodpy.eos import EnergyPolytrope


class TestEnergyPolytrope(TestCase):
    """Test case for highly relativistic stellar model.

    Implements model in Ref. [1]. Star has `epsilonc = 1e16 g cm^-3`,
    `R = 6.465 km` and `M = 1.3 Msol` (`2 M / R = 0.594`).

    References
    ----------
    [1] Andersson, Kokkotas and Schutz (1995), "A new numerical approach to the
        oscillation modes of relativistic stars," Mon. Not. R. Astron. Soc. 274
        (4), 1039.
    """

    n = 1
    K = 100
    poly = EnergyPolytrope(n, K)
    pc = 5.515e-3
    polystar = Star(poly, pc)

    def test_interpolation(self):
        """Compare interpolation with integration results."""
        m = self.polystar.m
        p = self.polystar.p
        nu = self.polystar.nu

        np.testing.assert_allclose(m(self.polystar.rsol), self.polystar.msol)
        np.testing.assert_allclose(p(self.polystar.rsol), self.polystar.psol)
        np.testing.assert_allclose(nu(self.polystar.rsol), self.polystar.nusol)

    def test_surface(self):
        """Test that pressure vanishes at surface."""
        np.testing.assert_allclose(self.polystar.psol[-1], 0, rtol=0, atol=1e-15)

    def test_stellar_mass(self):
        R = 6.465
        M = R / 2 * 0.594
        np.testing.assert_allclose(self.polystar.M, M, rtol=0, atol=1e-3)

    def test_stellar_radius(self):
        R = 6.465
        np.testing.assert_allclose(self.polystar.R, R, rtol=0, atol=1e-3)

    def test_monotonicity(self):
        """Test that integration results are monotonic."""
        np.testing.assert_equal(np.diff(self.polystar.psol) <= 0, True)
        np.testing.assert_equal(np.diff(self.polystar.msol) >= 0, True)
        np.testing.assert_equal(np.diff(self.polystar.nusol) >= 0, True)


class TestRange(TestCase):
    """Test case for range of `n = 1` polytropic equations of state with
    different proportionality constants `K` and central pressures `pc`.
    """

    n = 1
    Krange = (80, 90, 100, 110, 120)
    pcrange = (1e-4, 5e-4, 1e-3, 5e-3)

    def test_interpolation(self):
        """Compare interpolation with integration points."""
        for K in self.Krange:
            for pc in self.pcrange:
                with self.subTest(K=K, pc=pc):
                    poly = EnergyPolytrope(self.n, K)
                    polystar = Star(poly, pc)

                    m = polystar.m
                    p = polystar.p
                    nu = polystar.nu

                    np.testing.assert_allclose(m(polystar.rsol), polystar.msol)
                    np.testing.assert_allclose(p(polystar.rsol), polystar.psol)
                    np.testing.assert_allclose(nu(polystar.rsol), polystar.nusol)

    def test_surface(self):
        """Test that pressure vanishes at surface."""
        for K in self.Krange:
            for pc in self.pcrange:
                with self.subTest(K=K, pc=pc):
                    poly = EnergyPolytrope(self.n, K)
                    polystar = Star(poly, pc)

                    np.testing.assert_allclose(polystar.psol[-1], 0, rtol=0, atol=1e-15)

    def test_monotonicity(self):
        """Test that integration results are monotonic."""
        for K in self.Krange:
            for pc in self.pcrange:
                with self.subTest(K=K, pc=pc):
                    poly = EnergyPolytrope(self.n, K)
                    polystar = Star(poly, pc)

                    np.testing.assert_equal(np.diff(polystar.psol) <= 0, True)
                    np.testing.assert_equal(np.diff(polystar.msol) >= 0, True)
                    np.testing.assert_equal(np.diff(polystar.nusol) >= 0, True)
