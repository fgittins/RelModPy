"""Test oscillation mode class `Mode`.

Implements highly relativistic stellar model in Ref. [1]. Background has
`epsilonc = 1e16 g cm^-3`, `R = 6.465 km` and `M = 1.3 Msol`
(`2 M / R = 0.594`).

Compares with eigenfrequencies of Table 3.3 in Ref. [2].

References
----------
[1] Andersson, Kokkotas and Schutz (1995), "A new numerical approach to the
    oscillation modes of relativistic stars," Mon. Not. R. Astron. Soc. 274
    (4), 1039.
[2] Krüger (2015), "Seismology of adolescent general relativistic neutron
    stars," PhD thesis, University of Southampton.
"""

import pytest
from unittest import TestCase

from relmodpy import Mode, Star
from relmodpy.eos import EnergyPolytrope


class TestMuller(TestCase):
    """Test case for polar modes obtained using Muller's method."""

    n = 1
    K = 100
    poly = EnergyPolytrope(n, K)
    pc = 5.52e-3
    polystar = Star(poly, pc)
    M = polystar.M
    ell = 2

    def test_f(self):
        """Test f-mode."""
        Reomegatrue = 0.171 / self.M
        Imomegatrue = 6.19e-5 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-7
        )

    def test_w1(self):
        """Test w-mode with one overtone."""
        Reomegatrue = 0.471 / self.M
        Imomegatrue = 0.056 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    def test_w2(self):
        """Test w-mode with two overtones."""
        Reomegatrue = 0.653 / self.M
        Imomegatrue = 0.164 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    def test_w3(self):
        """Test w-mode with three overtones."""
        Reomegatrue = 0.891 / self.M
        Imomegatrue = 0.227 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=2e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    def test_w4(self):
        """Test w-mode with four overtones."""
        Reomegatrue = 1.127 / self.M
        Imomegatrue = 0.261 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=2e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=2e-3
        )

    def test_w5(self):
        """Test w-mode with five overtones."""
        Reomegatrue = 1.362 / self.M
        Imomegatrue = 0.287 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=7e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    def test_w6(self):
        """Test w-mode with six overtones."""
        Reomegatrue = 1.598 / self.M
        Imomegatrue = 0.307 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=2e-2
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=3e-3
        )

    @pytest.mark.xfail(reason="Test is broken.")
    def test_w7(self):
        """Test w-mode with seven overtones."""
        Reomegatrue = 1.835 / self.M
        Imomegatrue = 0.324 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=4e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=2e-2
        )

    def test_w8(self):
        """Test w-mode with eight overtones."""
        Reomegatrue = 2.072 / self.M
        Imomegatrue = 0.339 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=4e-2
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    @pytest.mark.xfail(reason="Test is broken.")
    def test_w9(self):
        """Test w-mode with nine overtones."""
        Reomegatrue = 2.309 / self.M
        Imomegatrue = 0.351 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=4e-2
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=5e-3
        )

    @pytest.mark.xfail(reason="Test is broken.")
    def test_w10(self):
        """Test w-mode with ten overtones."""
        Reomegatrue = 2.546 / self.M
        Imomegatrue = 0.363 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=2e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=3e-2
        )

    def test_w11(self):
        """Test w-mode with eleven overtones."""
        Reomegatrue = 2.782 / self.M
        Imomegatrue = 0.374 / self.M

        polymode = Mode(self.polystar)

        omegaguess1 = Reomegatrue + 1j * Imomegatrue
        omegaguess2 = omegaguess1.real + 1.0001j * omegaguess1.imag
        omegaguess3 = 1.0001 * omegaguess1.real + 1j * omegaguess1.imag

        polymode.solve(
            self.ell, (omegaguess1, omegaguess2, omegaguess3), method="Muller"
        )
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=4e-2
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=3e-2
        )


class TestSimplex(TestCase):
    """Test case for polar modes obtained using Simplex method."""

    n = 1
    K = 100
    poly = EnergyPolytrope(n, K)
    pc = 5.52e-3
    polystar = Star(poly, pc)
    M = polystar.M
    ell = 2

    def test_f(self):
        """Test f-mode."""
        Reomegatrue = 0.171 / self.M
        Imomegatrue = 6.19e-5 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-7
        )

    def test_w1(self):
        """Test w-mode with one overtone."""
        Reomegatrue = 0.471 / self.M
        Imomegatrue = 0.056 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    def test_w2(self):
        """Test w-mode with two overtones."""
        Reomegatrue = 0.653 / self.M
        Imomegatrue = 0.164 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-3
        )

    def test_p1(self):
        """Test p-mode with one overtone."""
        Reomegatrue = 0.344 / self.M
        Imomegatrue = 2.46e-6 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=2e-8
        )

    def test_p2(self):
        """Test p-mode with two overtones."""
        Reomegatrue = 0.503 / self.M
        Imomegatrue = 3.97e-5 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-7
        )

    def test_p3(self):
        """Test p-mode with three overtones."""
        Reomegatrue = 0.658 / self.M
        Imomegatrue = 3.38e-6 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-8
        )

    def test_p4(self):
        """Test p-mode with four overtones."""
        Reomegatrue = 0.810 / self.M
        Imomegatrue = 6.74e-7 / self.M

        polymode = Mode(self.polystar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-9
        )


class TestStratified(TestCase):
    """Test case for polar modes with stratification in equation of state."""

    class StratifiedEnergyPoltrope(EnergyPolytrope):
        def Gamma1(self, p):
            return 1.1 * self.Gamma(p)

    n = 1
    K = 100
    strat = StratifiedEnergyPoltrope(n, K)
    pc = 5.52e-3
    stratstar = Star(strat, pc)
    M = stratstar.M
    ell = 2

    def test_f(self):
        """Test f-mode."""
        Reomegatrue = 0.171 / self.M
        Imomegatrue = 6.20e-5 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-7
        )

    def test_p1(self):
        """Test p-mode with one overtone."""
        Reomegatrue = 0.366 / self.M
        Imomegatrue = 2.52e-6 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-8
        )

    def test_p2(self):
        """Test p-mode with two overtones."""
        Reomegatrue = 0.532 / self.M
        Imomegatrue = 2.17e-5 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-3
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-7
        )

    def test_g1(self):
        """Test g-mode with one overtones."""
        Reomegatrue = 4.54e-2 / self.M
        Imomegatrue = 1.4e-12 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-4
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-13
        )

    def test_g2(self):
        """Test g-mode with two overtones."""
        Reomegatrue = 3.07e-2 / self.M
        Imomegatrue = 1.1e-13 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-4
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=2e-14
        )

    @pytest.mark.xfail(reason="Test is broken.")
    def test_g3(self):
        """Test g-mode with three overtones."""
        Reomegatrue = 2.32e-2 / self.M
        Imomegatrue = 8e-16 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue + 1j * Imomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-4
        )
        self.assertAlmostEqual(
            polymode.omega.imag * self.M, Imomegatrue * self.M, delta=1e-16
        )

    @pytest.mark.xfail(reason="Test is broken.")
    def test_g4(self):
        """Test g-mode with four overtones."""
        Reomegatrue = 1.87e-2 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=3e-4
        )

    def test_g5(self):
        """Test g-mode with five overtones."""
        Reomegatrue = 1.57e-2 / self.M

        polymode = Mode(self.stratstar)

        omegaguess = Reomegatrue

        polymode.solve(self.ell, omegaguess, method="Simplex")
        self.assertAlmostEqual(
            polymode.omega.real * self.M, Reomegatrue * self.M, delta=1e-4
        )
