"""Define functions for mode perturbations in exterior of relativistic star.

Based on method described in Ref. [1]. `transformation` relies on equations in
Refs. [2,3], noting different definitions of metric perturbations.

Functions
---------
V
dVdr
U
dUdr
modified_zerilli
jac_modified_zerilli
solve_perturbations_exterior
transformation
Ain
Aout

Notes
-----
Assumes geometric units, where G = c = 1. Due to numerical noise at low
frequencies, amplitudes `Ain` and `Aout` are renormalised.

References
----------
[1] Andersson, Kokkotas and Schutz (1995), "A new numerical approach to the
    oscillation modes of relativistic stars," Mon. Not. R. Astron. Soc. 274
    (4), 1039.
[2] Detweiler and Lindblom (1985), "On the nonradial pulsations of general
    relativistic stellar models," Astrophys. J. 292, 12.
[3] Fackerell (1971), "Solutions of Zerilli's equation for even-parity
    gravitational perturbations," Astrophys. J. 166, 197.
"""

import numpy as np
from scipy.integrate import solve_ivp


def V(r, M, n):
    """Effective potential from Zerilli equation.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].

    Returns
    -------
    V : float
        Potential [km^-2].
    """
    return (
        2
        * (1 - 2 * M / r)
        * (n**2 * (n + 1) * r**3 + 3 * n**2 * M * r**2 + 9 * n * M**2 * r + 9 * M**3)
        / (r**3 * (n * r + 3 * M) ** 2)
    )


def dVdr(r, M, n):
    """Derivative of effective potential from Zerilli equation.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].

    Returns
    -------
    dVdr : float
        Derivative of potential [km^-3].
    """
    return (
        432 * M**5
        + 54 * (10 * n - 3) * M**4 * r
        + 18 * n * (14 * n - 11) * M**3 * r**2
        + 6 * n**2 * (10 * n - 13) * M**2 * r**3
        + 6 * n**3 * (2 * n - 1) * M * r**4
        - 4 * n**3 * (n + 1) * r**5
    ) / (r**5 * (n * r + 3 * M) ** 3)


def U(r, M, n, omega2):
    """New effective potential.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    U : complex
        Potential [km^-2].
    """
    return (1 - 2 * M / r) ** (-2) * (
        omega2 - V(r, M, n) + 2 * M / r**3 - 3 * M**2 / r**4
    )


def dUdr(r, M, n, omega2):
    """Derivative of new effective potential.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    dUdr : complex
        Derivative of potential [km^-3].
    """
    return (
        2 * M * (6 * M**2 - 8 * M * r + 3 * r**2 + 2 * omega2 * r**4)
        - 4 * M * V(r, M, n) * r**4
        + (r - 2 * M) * dVdr(r, M, n) * r**5
    ) / (r * (2 * M - r)) ** 3


def modified_zerilli(rho, y, R, M, n, omega):
    """Exterior perturbation equation derived from Zerilli equation with
    introduction of new variable `q`.

    Parameters
    ----------
    rho : float
        Real distance [km].
    y : (2,) complex array_like
        Variables `q` [km^-1] and `dqdrho` [km^-2] at `rho`.
    R : float
        Stellar radius [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].
    omega : complex
        Mode frequency [km^-1].

    Returns
    -------
    dqrho : complex
        Derivative of variable `q` [km^-2] at `rho`.
    d2qdrho2 : complex
        Second derivative of variable `q` [km^-3] at `rho`.
    """
    q, dqdrho = y

    theta = -np.arctan(omega.imag / omega.real)
    r = R + rho * np.exp(1j * theta)

    d2qdrho2 = 3 * dqdrho**2 / (2 * q) - 2 * q * np.exp(2j * theta) * (
        q**2 - U(r, M, n, omega**2)
    )

    return [dqdrho, d2qdrho2]


def jac_modified_zerilli(rho, y, R, M, n, omega):
    """Jacobian of exterior perturbation equation `modified_zerilli`.

    Parameters
    ----------
    rho : float
        Real distance [km].
    y : (2,) complex array_like
        Variables `q` [km^-1] and `dqdrho` [km^-2] at `rho`.
    R : float
        Stellar radius [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].
    omega : complex
        Mode frequency [km^-1].

    Returns
    -------
    jac : (2, 2) complex array_like
        Jacobian matrix.
    """
    q, dqdrho = y

    theta = -np.arctan(omega.imag / omega.real)
    r = R + rho * np.exp(1j * theta)

    return [
        [0, 1],
        [
            -3 / 2 * (dqdrho / q) ** 2
            - 2 * np.exp(2j * theta) * (3 * q**2 - U(r, M, n, omega**2)),
            3 * dqdrho / q,
        ],
    ]


def solve_perturbations_exterior(R, M, n, omega):
    """Integrate perturbation equation along straight line in complex plane
    from far from star to its surface.

    Parameters
    ----------
    R : float
        Stellar radius [km].
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].
    omega : complex
        Mode frequency [km^-1].

    Returns
    -------
    sol : solve_ivp object
        Contains results of integration.

    Notes
    -----
    Integration result depends strongly on distance from star `rhoinf` [km].
    Follow Ref. [1] and use
    ```
    rhoinf = 50 / abs(omega)
    ```

    References
    ----------
    [1] Kokkotas and Schutz (1992) "W-modes: a new family of normal modes of
        pulsating relativistic stars," Mon. Not. R. Astron. Soc. 255, 119.
    """
    theta = -np.arctan(omega.imag / omega.real)
    rhoinf = 50 / abs(omega)
    N = 10 / abs(omega * M)
    step = rhoinf / N
    rinf = R + rhoinf * np.exp(1j * theta)
    Uinf = U(rinf, M, n, omega**2)
    dUdrinf = dUdr(rinf, M, n, omega**2)

    sol = solve_ivp(
        modified_zerilli,
        [rhoinf, 0],
        [Uinf ** (1 / 2), np.exp(1j * theta) * dUdrinf / (2 * Uinf ** (1 / 2))],
        args=(R, M, n, omega),
        method="BDF",
        max_step=step,
        rtol=1e-11,
        atol=1e-11,
        jac=jac_modified_zerilli,
    )
    return sol


def transformation(r, x, M, n):
    """Transform metric perturbations to functions in Zerilli equation.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    x : (2,) complex array_like
        Metric perturbations `H1` [dimensionless] and `K` [dimensionless] at
        `r`.
    M : float
        Stellar mass [km].
    n : int
        Quantum number [dimensionless].

    Returns
    -------
    Z : complex
        Function `Z` [km] at `r`.
    dZdrstar : complex
        Derivative of function `Z` with respect to tortoise coordinate `rstar`
        [dimensionless] at `r`.
    """
    g = (n * (n + 1) * r**2 + 3 * n * M * r + 6 * M**2) / (r**2 * (n * r + 3 * M))
    h = (n * r**2 - 3 * n * M * r - 3 * M**2) / (r * (r - 2 * M) * (n * r + 3 * M))
    k = r / (r - 2 * M)
    A = np.array([[1, -k], [-g, h]]) / (h - g * k)
    b = A @ x
    return b


def Ain(r, q, dqdr, Z, dZdrstar, M):
    """Amplitude of ingoing radiation.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    q : complex
        Variable `q` [km^-1] at `r`.
    dqdr : complex
        Derivative `dqdr` [km^-1] at `r`.
    Z : complex
        Variable `Z` [dimensionless] at `r`.
    dZdrstar : complex
        Derivative `dZdrstar` [km^-1] at `r`.
    M : float
        Stellar mass [km].

    Returns
    -------
    Ain : complex
        Amplitude of ingoing radiation [dimensionless] at `r`.
    """
    return r * ((M / r**2 + (1 - 2 * M / r) * (dqdr / (2 * q) + 1j * q)) * Z + dZdrstar)


def Aout(r, q, dqdr, Z, dZdrstar, M):
    """Amplitude of outgoing radiation.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    q : complex
        Variable `q` [km^-1] at `r`.
    dqdr : complex
        Derivative `dqdr` [km^-1] at `r`.
    Z : complex
        Variable `Z` [dimensionless] at `r`.
    dZdrstar : complex
        Derivative `dZdrstar` [km^-1] at `r`.
    M : float
        Stellar mass [km].

    Returns
    -------
    Aout : complex
        Amplitude of outgoing radiation [dimensionless] at `r`.
    """
    return -r * (
        (M / r**2 + (1 - 2 * M / r) * (dqdr / (2 * q) - 1j * q)) * Z + dZdrstar
    )
