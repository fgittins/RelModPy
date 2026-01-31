"""Test equation-of-state module `eos`."""

from unittest import TestCase

import numpy as np

from relmodpy.eos import Polytrope, EnergyPolytrope


class TestPolytrope(TestCase):
    """Test case for `Polytrope` equation of state."""

    rng = np.random.default_rng(11102023)

    n = 1
    K = 100
    poly = Polytrope(n, K)

    point = 10 ** rng.uniform(-13, -3)
    prange = np.logspace(-13, -3, 11)

    def test_trivial(self):
        """Test at zero."""
        self.assertEqual(self.poly.epsilon(0), 0)

    def test_point(self):
        """Test at arbitrary point."""
        epsilontrue = (self.point / self.K) ** (
            self.n / (self.n + 1)
        ) + self.n * self.point
        self.assertAlmostEqual(self.poly.epsilon(self.point), epsilontrue)

    def test_Gamma(self):
        """Test constancy of adiabatic index."""
        for p in self.prange:
            with self.subTest(p=p):
                self.assertEqual(self.poly.Gamma(p), 1 + 1 / self.n)

    def test_barotropic(self):
        """Test for barotropic matter."""
        for p in self.prange:
            with self.subTest(p=p):
                self.assertEqual(self.poly.Gamma(p), self.poly.Gamma1(p))


class TestEnergyPolytrope(TestCase):
    """Test case for `EnergyPolytrope` equation of state."""

    rng = np.random.default_rng(11102023)

    n = 1
    K = 100
    poly = EnergyPolytrope(n, K)

    point = 10 ** rng.uniform(-13, -3)
    prange = np.logspace(-13, -3, 11)

    def test_trivial(self):
        """Test at zero."""
        self.assertEqual(self.poly.epsilon(0), 0)
        self.assertEqual(self.poly.Gamma(0), 1 + 1 / self.n)

    def test_point(self):
        """Test at arbitrary point."""
        epsilontrue = (self.point / self.K) ** (self.n / (self.n + 1))
        Gammatrue = (1 + 1 / self.n) * (
            1 + self.K * (self.point / self.K) ** (1 / (self.n + 1))
        )
        self.assertAlmostEqual(self.poly.epsilon(self.point), epsilontrue)
        self.assertAlmostEqual(self.poly.Gamma(self.point), Gammatrue)

    def test_barotropic(self):
        """Test for barotropic matter."""
        for p in self.prange:
            with self.subTest(p=p):
                self.assertEqual(self.poly.Gamma(p), self.poly.Gamma1(p))
