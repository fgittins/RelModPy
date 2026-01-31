"""Script demonstrates example of computing oscillation modes of star described by polytropic equation of state."""

import numpy as np
import matplotlib.pyplot as plt

from relmodpy import Mode, Star
from relmodpy.eos import EnergyPolytrope
import units


# compute stellar background with provided equation of state
class StratifiedEnergyPolytrope(EnergyPolytrope):
    def Gamma1(self, p):
        return 1.1 * self.Gamma(p)


def main():
    n, K = 1, 100
    eos = StratifiedEnergyPolytrope(n, K)
    pc = 5.52e-3
    star = Star(eos, pc)

    print(f"R = {star.R} km, M = {units.mass_geometric_to_Msol * star.M} Msol")

    # initialise mode, focusing on quadrupolar f-mode
    mode = Mode(star)

    ell = 2
    omegaguess = (0.171 + 6.2e-5j) / star.M

    # search for eigenfrequency with Muller's method, which requires three initial
    # guesses
    mode.solve(
        ell,
        (
            omegaguess,
            1.001 * omegaguess.real + 1j * omegaguess.imag,
            omegaguess.real + 1.001j * omegaguess.imag,
        ),
    )

    print(
        f"Re(omega M) = {mode.omega.real * star.M}, "
        + f"Im(omega M) = {mode.omega.imag * star.M}"
    )
    print(
        f"Re[omega / (2 pi)] = {mode.omega.real / (2 * np.pi) / units.time_geometric_to_CGS / 1e3} kHz, "
        + f"1 / Im(omega) = {1 / (mode.omega.imag) * units.time_geometric_to_CGS * 1e3} ms"
    )

    # plot eigenfunctions
    r = np.linspace(mode.r0, star.R, num=100)
    H1, K, W, X = [], [], [], []
    for r0 in r:
        H1.append(mode.H1(r0))
        K.append(mode.K(r0))
        W.append(mode.W(r0))
        X.append(mode.X(r0))
    H1, K, W, X = np.array(H1), np.array(K), np.array(W), np.array(X)

    fig, ax = plt.subplots()
    ax.plot(r, H1.real, color="C0", label=r"$\mathrm{Re}(H_1)$")
    ax.plot(r, K.real, color="C1", label=r"$\mathrm{Re}(K)$")
    ax.plot(r, W.real, color="C2", label=r"$\mathrm{Re}(W)$ / km$^2$")
    ax.plot(r, X.real, color="C3", label=r"$\mathrm{Re}(X)$ / km$^{-2}$")
    ax.set_xlabel(r"$r$ / km")
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(r, H1.imag, color="C0", label=r"$\mathrm{Im}(H_1)$")
    ax.plot(r, K.imag, color="C1", label=r"$\mathrm{Im}(K)$")
    ax.plot(r, W.imag, color="C2", label=r"$\mathrm{Im}(W)$ / km$^2$")
    ax.plot(r, X.imag, color="C3", label=r"$\mathrm{Im}(X)$ / km$^{-2}$")
    ax.set_xlabel(r"$r$ / km")
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
