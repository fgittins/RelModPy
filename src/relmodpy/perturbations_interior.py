"""Define functions for mode perturbations in interior of relativistic star.

Based on equations derived in Ref. [1]. Low-frequency formulation comes from
Ref. [2].

Functions
---------
perturbations_interior
taylor_coefficients
solve_perturbations_interior
perturbations_interior_low_frequency
taylor_coefficients_low_frequency
solve_perturbations_interior_low_frequency
X

Notes
-----
Assumes geometric units, where G = c = 1.

References
----------
[1] Detweiler and Lindblom (1985), "On the nonradial pulsations of general
    relativistic stellar models," Astrophys. J. 292, 12.
[2] Krüger (2015), "Seismology of adolescent general relativistic neutron
    stars," PhD thesis, University of Southampton.
"""

import numpy as np
from scipy.integrate import solve_ivp


def perturbations_interior(r, y, background, ell, omega2):
    """Interior polar perturbation equations for relativistic star.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    y : (4,) complex array_like
        Metric perturbations `H1` [dimensionless] and `K` [dimensionless],
        radial displacement function `W` [km^2] and Lagrangian pressure
        perturbation function `X` [km^-2] at `r`.
    background : Star object
        Object contains following methods as functions of radius `r` and
        pressure `p`:

        * `m(r)` for mass [km],
        * `p(r)` for pressure [km^-2],
        * `nu(r)` for metric potential [dimensionless],
        * `eos.epsilon(p)` for energy density [km^-2],
        * `eos.Gamma1(p)` for adiabatic index [dimensionless].

    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    dH1dr : complex
        Derivative of metric perturbation `H1` [km^-1] at `r`.
    dKdr : complex
        Derivative of metric perturbation `K` [km^-1] at `r`.
    dWdr : complex
        Derivative of radial displacement function `W` [km] at `r`.
    dXdr : complex
        Derivative of Lagrangian pressure perturbation function `X` [km^-3] at
        `r`.
    """
    H1, K, W, X = y

    # background
    m, p, nu = background.m(r), background.p(r), background.nu(r)
    if p < 0:
        epsilon = np.nan
        Gamma1 = np.nan
    else:
        epsilon = background.eos.epsilon(p)
        Gamma1 = background.eos.Gamma1(p)

    dmdr = 4 * np.pi * r**2 * epsilon
    dnudr = 2 * (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m))
    dpdr = -(epsilon + p) * dnudr / 2

    expnu = np.exp(nu)
    explambda = 1 / (1 - 2 * m / r)
    dlambdadr = 2 * explambda * (dmdr / r - m / r**2)

    d2nudr2 = (
        2
        / (r * (r - 2 * m))
        * (dmdr + 4 * np.pi * r**2 * (3 * p + r * dpdr) + (m + r * dmdr - r) * dnudr)
    )

    # perturbations
    H0 = (
        8 * np.pi * r**3 / expnu ** (1 / 2) * X
        - (
            ell * (ell + 1) / 2 * (m + 4 * np.pi * r**3 * p)
            - omega2 * r**3 / (explambda * expnu)
        )
        * H1
        + (
            (ell + 2) * (ell - 1) / 2 * r
            - omega2 * r**3 / expnu
            - explambda
            / r
            * (m + 4 * np.pi * r**3 * p)
            * (3 * m - r + 4 * np.pi * r**3 * p)
        )
        * K
    ) / (3 * m + (ell + 2) * (ell - 1) / 2 * r + 4 * np.pi * r**3 * p)
    V = (
        expnu ** (1 / 2)
        / omega2
        * (
            1 / (epsilon + p) * X
            - dnudr / (2 * r) * (expnu / explambda) ** (1 / 2) * W
            - expnu ** (1 / 2) / 2 * H0
        )
    )

    dH1dr = -(
        ell + 1 + 2 * m * explambda / r + 4 * np.pi * r**2 * explambda * (p - epsilon)
    ) / r * H1 + explambda / r * (H0 + K - 16 * np.pi * (epsilon + p) * V)
    dKdr = (
        1 / r * H0
        + ell * (ell + 1) / (2 * r) * H1
        - ((ell + 1) / r - dnudr / 2) * K
        - 8 * np.pi * (epsilon + p) * explambda ** (1 / 2) / r * W
    )
    dWdr = -(ell + 1) / r * W + r * explambda ** (1 / 2) * (
        1 / (Gamma1 * p * expnu ** (1 / 2)) * X
        - ell * (ell + 1) / r**2 * V
        + 1 / 2 * H0
        + K
    )
    dXdr = -ell / r * X + (epsilon + p) * expnu ** (1 / 2) * (
        (1 / r - dnudr / 2) / 2 * H0
        + (r * omega2 / expnu + ell * (ell + 1) / (2 * r)) / 2 * H1
        + (3 * dnudr / 2 - 1 / r) / 2 * K
        - ell * (ell + 1) * dnudr / (2 * r**2) * V
        - (
            4 * np.pi * (epsilon + p) * explambda ** (1 / 2)
            + omega2 * explambda ** (1 / 2) / expnu
            + (dnudr * (dlambdadr / 2 + 2 / r) - d2nudr2) / (2 * explambda ** (1 / 2))
        )
        / r
        * W
    )

    return [dH1dr, dKdr, dWdr, dXdr]


def taylor_coefficients(Kc, Wc, background, ell, omega2):
    """Calculate coefficients for Taylor expansion near centre.

    Parameters
    ----------
    Kc : complex
        Metric perturbation `K` [dimensionless] at stellar centre.
    Wc : complex
        Radial displacement function `W` [km^2] at stellar centre.
    background : Star object
        Object contains following attributes at stellar centre and methods as
        functions of pressure `p`:

        * `pc` for central pressure [km^-2],
        * `nuc` for central metric potential [dimensionless],
        * `epsilonc` for central energy density [km^-2],
        * `eos.Gamma(p)` for adiabatic index of background [dimensionless],
        * `eos.Gamma1(p)` for adiabatic index of perturbations [dimensionless].

    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    H1c : complex
        Metric perturbation `H1` [dimensionless] at stellar centre.
    d2H1dr2c : complex
        Second derivative of metric perturbation `H1` [km^-2] at stellar
        centre.
    d2Kdr2c : complex
        Second derivative of metric perturbation `K` [km^-2] at stellar centre.
    d2Wdr2c : complex
        Second derivative of radial displacement function `W` [dimensionless]
        at stellar centre.
    Xc : complex
        Lagrangian pressure perturbation function `X` [km^-2] at stellar
        centre.
    d2Xdr2c : complex
        Second derivative of Lagrangian pressure perturbation function `X`
        [km^-4] at stellar centre.
    """
    pc = background.pc
    epsilonc = background.epsilonc
    expnuc = np.exp(background.nuc)
    Gammac = background.eos.Gamma(pc)
    Gamma1c = background.eos.Gamma1(pc)

    # coefficients for background quantities
    p2 = -4 * np.pi / 3 * (epsilonc + pc) * (epsilonc + 3 * pc)
    epsilon2 = p2 * (epsilonc + pc) / (Gammac * pc)
    nu2 = 8 * np.pi / 3 * (epsilonc + 3 * pc)

    p4 = -(
        2 * np.pi / 5 * (epsilonc + pc) * (epsilon2 + 5 * p2)
        + 2 * np.pi / 3 * (epsilon2 + p2) * (epsilonc + 3 * pc)
        + 32 * np.pi**2 / 9 * epsilonc * (epsilonc + pc) * (epsilonc + 3 * pc)
    )
    nu4 = 4 * np.pi / 5 * (epsilon2 + 5 * p2) + 64 * np.pi**2 / 9 * epsilonc * (
        epsilonc + 3 * pc
    )

    # zeroth-order coefficients
    H1c = (2 * ell * Kc + 16 * np.pi * (epsilonc + pc) * Wc) / (ell * (ell + 1))
    Xc = (
        (epsilonc + pc)
        * expnuc ** (1 / 2)
        * (1 / 2 * Kc + (nu2 / 2 - omega2 / (ell * expnuc)) * Wc)
    )

    # solve for second-order coefficients
    Q0 = (
        4
        / ((ell + 2) * (ell - 1))
        * (
            8 * np.pi / expnuc ** (1 / 2) * Xc
            - (8 * np.pi / 3 * epsilonc + omega2 / expnuc) * Kc
            - (2 * np.pi / 3 * ell * (ell + 1) * (epsilonc + 3 * pc) - omega2 / expnuc)
            * H1c
        )
    )
    Q1 = (
        2
        / (ell * (ell + 1))
        * (
            1 / (Gamma1c * pc * expnuc ** (1 / 2)) * Xc
            + 3 / 2 * Kc
            + 4 * np.pi / 3 * (ell + 1) * epsilonc * Wc
        )
    )

    b = np.array(
        [
            1 / 4 * nu2 / expnuc ** (1 / 2) * Xc
            + 1 / 4 * (epsilon2 + p2) * Kc
            + 1 / 4 * (epsilonc + pc) * Q0
            + 1 / 2 * omega2 * (epsilonc + pc) / expnuc * Q1
            - (
                p4
                - 4 * np.pi / 3 * epsilonc * p2
                + omega2 / (2 * ell) * (epsilon2 + p2 - (epsilonc + pc) * nu2) / expnuc
            )
            * Wc,
            ###
            4 * np.pi / 3 * (epsilonc + 3 * pc) * Kc
            + 1 / 2 * Q0
            - 4
            * np.pi
            * (epsilon2 + p2 + 8 * np.pi / 3 * epsilonc * (epsilonc + pc))
            * Wc,
            ###
            4 * np.pi * (1 / 3 * (2 * ell + 3) * epsilonc - pc) * H1c
            + 8 * np.pi / ell * (epsilon2 + p2) * Wc
            - 8 * np.pi * (epsilonc + pc) * Q1
            + 1 / 2 * Q0,
            ###
            1
            / 2
            * (epsilon2 + p2 + 1 / 2 * (epsilonc + pc) * nu2)
            * ell
            / (epsilonc + pc)
            * Xc
            + (epsilonc + pc)
            * expnuc ** (1 / 2)
            * (
                1 / 2 * nu2 * Kc
                + 1 / 4 * Q0
                + 1 / 2 * omega2 / expnuc * H1c
                - 1 / 4 * ell * (ell + 1) * nu2 * Q1
                + (
                    1 / 2 * (ell + 1) * nu4
                    - 2 * np.pi * (epsilon2 + p2)
                    - 16 * np.pi**2 / 3 * epsilonc * (epsilonc + pc)
                    + 1 / 2 * (nu4 - 4 * np.pi / 3 * epsilonc * nu2)
                    + 1 / 2 * omega2 / expnuc * (nu2 - 8 * np.pi / 3 * epsilonc)
                )
                * Wc
            ),
        ]
    )

    A = np.array(
        [
            [
                0,
                -1 / 4 * (epsilonc + pc),
                1
                / 2
                * (
                    p2
                    + (epsilonc + pc) * omega2 * (ell + 3) / (ell * (ell + 1) * expnuc)
                ),
                1 / (2 * expnuc ** (1 / 2)),
            ],
            ###
            [
                -1 / 4 * ell * (ell + 1),
                1 / 2 * (ell + 2),
                4 * np.pi * (epsilonc + pc),
                0,
            ],
            ###
            [
                1 / 2 * (ell + 3),
                -1,
                -8 * np.pi * (epsilonc + pc) * (ell + 3) / (ell * (ell + 1)),
                0,
            ],
            ###
            [
                -1 / 8 * ell * (ell + 1) * (epsilonc + pc) * expnuc ** (1 / 2),
                0,
                -(epsilonc + pc)
                * expnuc ** (1 / 2)
                * (
                    1 / 4 * (ell + 2) * nu2
                    - 2 * np.pi * (epsilonc + pc)
                    - 1 / 2 * omega2 / expnuc
                ),
                1 / 2 * (ell + 2),
            ],
        ]
    )

    x = np.linalg.solve(A, b)
    d2H1dr2c, d2Kdr2c, d2Wdr2c, d2Xdr2c = x
    return (H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Xc, d2Xdr2c)


def solve_perturbations_interior(background, ell, omega2):
    """Integrate perturbation equations in interior.

    Parameters
    ----------
    background : Star object
        In addition to attributes and methods required for
        `perturbations_interior` and `taylor_coefficients`, object contains
        attribute `R` for stellar radius [km].
    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    x : (5,) complex array_like
        Coefficients of general solution [dimensionless].
    sol1, sol2 : solve_ivp object
        Solutions 1 and 2 from stellar centre.
    sol3, sol4, sol5 : solve_ivp object
        Solutions 3, 4 and 5 from stellar surface.

    Notes
    -----
    If omega2 is real, integration focuses on real values for efficiency.
    """
    R = background.R
    # starting point
    r0 = 1e-3
    # matching point
    rmatch = R / 2

    # from centre
    # Solution 1
    if isinstance(omega2, complex):
        Kc, Wc = 1 + 0j, 0 + 0j
    else:
        Kc, Wc = 1, 0
    H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Xc, d2Xdr2c = taylor_coefficients(
        Kc, Wc, background, ell, omega2
    )
    H10 = H1c + 1 / 2 * r0**2 * d2H1dr2c
    K0 = Kc + 1 / 2 * r0**2 * d2Kdr2c
    W0 = Wc + 1 / 2 * r0**2 * d2Wdr2c
    X0 = Xc + 1 / 2 * r0**2 * d2Xdr2c

    sol1 = solve_ivp(
        perturbations_interior,
        [r0, rmatch],
        [H10, K0, W0, X0],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 2
    if isinstance(omega2, complex):
        Kc, Wc = 0 + 0j, 1 + 0j
    else:
        Kc, Wc = 0, 1
    H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Xc, d2Xdr2c = taylor_coefficients(
        Kc, Wc, background, ell, omega2
    )
    H10 = H1c + 1 / 2 * r0**2 * d2H1dr2c
    K0 = Kc + 1 / 2 * r0**2 * d2Kdr2c
    W0 = Wc + 1 / 2 * r0**2 * d2Wdr2c
    X0 = Xc + 1 / 2 * r0**2 * d2Xdr2c

    sol2 = solve_ivp(
        perturbations_interior,
        [r0, rmatch],
        [H10, K0, W0, X0],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # from surface
    # Solution 3
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 1 + 0j, 0 + 0j, 0 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 1, 0, 0
        Xf = 0

    sol3 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 4
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 0 + 0j, 1 + 0j, 0 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 0, 1, 0
        Xf = 0

    sol4 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 5
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 0 + 0j, 0 + 0j, 1 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 0, 0, 1
        Xf = 0

    sol5 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # solve for coefficients of general solution
    A = np.zeros((5, 5), dtype=type(omega2))
    A[:4, 0] = sol1.y[:, -1]
    A[:4, 1] = sol2.y[:, -1]
    A[:4, 2] = -sol3.y[:, -1]
    A[:4, 3] = -sol4.y[:, -1]
    A[:4, 4] = -sol5.y[:, -1]
    # normalising at surface
    A[4, :] = [0, 0, sol3.y[2, 0], sol4.y[2, 0], sol5.y[2, 0]]
    b = np.array([0, 0, 0, 0, 1])
    x = np.linalg.solve(A, b)

    return (x, sol1, sol2, sol3, sol4, sol5)


def perturbations_interior_low_frequency(r, y, background, ell, omega2):
    """Interior polar perturbation equations for relativistic star with
    formulation appropriate for low-frequency oscillations.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    y : (4,) complex array_like
        Metric perturbations `H1` [dimensionless] and `K` [dimensionless],
        radial displacement function `W` [km^2] and horizontal displacement
        function `V` [km^2] at `r`.
    background : Star object
        Object contains following methods as functions of radius `r` and
        pressure `p`:

        * `m(r)` for mass [km],
        * `p(r)` for pressure [km^-2],
        * `nu(r)` for metric potential [dimensionless],
        * `eos.epsilon(p)` for energy density [km^-2],
        * `eos.Gamma(p)` for adiabatic index of background [dimensionless],
        * `eos.Gamma1(p)` for adiabatic index of perturbations [dimensionless].

    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    dH1dr : complex
        Derivative of metric perturbation `H1` [km^-1] at `r`.
    dKdr : complex
        Derivative of metric perturbation `K` [km^-1] at `r`.
    dWdr : complex
        Derivative of radial displacement function `W` [km] at `r`.
    dVdr : complex
        Derivative of horizontal displacement function `V` [km] at `r`.
    """
    H1, K, W, V = y

    # background
    m, p, nu = background.m(r), background.p(r), background.nu(r)
    if p < 0:
        epsilon = np.nan
        Gamma = np.nan
        Gamma1 = np.nan
    else:
        epsilon = background.eos.epsilon(p)
        Gamma = background.eos.Gamma(p)
        Gamma1 = background.eos.Gamma1(p)

    explambda = 1 / (1 - 2 * m / r)
    expnu = np.exp(nu)

    dlambdadr = (1 - explambda) / r + 8 * np.pi * r * explambda * epsilon
    dnudr = (explambda - 1) / r + 8 * np.pi * r * explambda * p
    dpdr = -(epsilon + p) * dnudr / 2

    n = (ell + 2) * (ell - 1) / 2
    A = (1 / Gamma - 1 / Gamma1) * dpdr / p

    # perturbations
    H0 = (
        r**2 / explambda * (omega2 * r / expnu - (n + 1) * dnudr / 2) * H1
        + (
            n * r
            - omega2 * r**3 / expnu
            - r**2 * dnudr / (4 * explambda) * (r * dnudr - 2)
        )
        * K
        + 4
        * np.pi
        * r**2
        * (epsilon + p)
        * (dnudr / explambda ** (1 / 2) * W + 2 * r * omega2 / expnu * V)
    ) / ((n + 1) * r - r / (2 * explambda) * (r * dlambdadr + 2))

    dH1dr = ((dlambdadr - dnudr) / 2 - (ell + 1) / r) * H1 + explambda / r * (
        H0 + K - 16 * np.pi * (epsilon + p) * V
    )
    dKdr = (
        1 / r * H0
        + (n + 1) / r * H1
        + (dnudr / 2 - (ell + 1) / r) * K
        - 8 * np.pi * (epsilon + p) * explambda ** (1 / 2) / r * W
    )
    dWdr = -((ell + 1) / r + dpdr / (Gamma1 * p)) * W + r * explambda ** (1 / 2) * (
        (epsilon + p) / (Gamma1 * p) * (omega2 / expnu * V + 1 / 2 * H0)
        - 2 * (n + 1) / r**2 * V
        + 1 / 2 * H0
        + K
    )
    dVdr = (
        (-A + dnudr - ell / r) * V
        - expnu / (2 * omega2) * A * (H0 + dnudr / (r * explambda ** (1 / 2)) * W)
        + r * H1
        - explambda ** (1 / 2) / r * W
    )

    return [dH1dr, dKdr, dWdr, dVdr]


def taylor_coefficients_low_frequency(Kc, Wc, background, ell, omega2):
    """Calculate coefficients for Taylor expansion near centre with formulation
    appropriate to low-frequency oscillations.

    Parameters
    ----------
    Kc : complex
        Metric perturbation `K` [dimensionless] at stellar centre.
    Wc : complex
        Radial displacement function `W` [km^2] at stellar centre.
    background : Star object
        Object contains following attributes at stellar centre and methods as
        functions of pressure `p`:

        * `pc` for central pressure [km^-2],
        * `nuc` for central metric potential [dimensionless],
        * `epsilonc` for central energy density [km^-2],
        * `eos.Gamma(p)` for adiabatic index of background [dimensionless],
        * `eos.Gamma1(p)` for adiabatic index of perturbations [dimensionless].

    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    H1c : complex
        Metric perturbation `H1` [dimensionless] at stellar centre.
    d2H1dr2c : complex
        Second derivative of metric perturbation `H1` [km^-2] at stellar
        centre.
    d2Kdr2c : complex
        Second derivative of metric perturbation `K` [km^-2] at stellar centre.
    d2Wdr2c : complex
        Second derivative of radial displacement function `W` [dimensionless]
        at stellar centre.
    Vc : complex
        Horizontal displacement function `V` [km^2] at stellar centre.
    d2Vdr2c : complex
        Second derivative of horizontal displacement function `V`
        [dimensionless] at stellar centre.
    """
    pc = background.pc
    epsilonc = background.epsilonc
    expnuc = np.exp(background.nuc)
    Gammac = background.eos.Gamma(pc)
    Gamma1c = background.eos.Gamma1(pc)

    # coefficients for background quantities
    lambda2 = 16 * np.pi / 3 * epsilonc
    nu2 = 8 * np.pi / 3 * (epsilonc + 3 * pc)
    p2 = -nu2 / 2 * (epsilonc + pc)
    epsilon2 = p2 * (epsilonc + pc) / (Gammac * pc)

    # zeroth-order coefficients
    H1c = (2 * ell * Kc + 16 * np.pi * (epsilonc + pc) * Wc) / (ell * (ell + 1))
    Vc = -1 / ell * Wc

    n = (ell + 2) * (ell - 1) / 2

    # solve for second-order coefficients
    b = np.array(
        [
            (nu2 - lambda2) / 2 * H1c
            - lambda2 * Kc
            + 8 * np.pi * (epsilon2 + p2 + lambda2 * (epsilonc + pc)) * Vc,
            ###
            -nu2 / 2 * Kc
            + 2 * np.pi * (2 * (epsilon2 + p2) + lambda2 * (epsilonc + pc)) * Wc,
            ###
            -1 / 2 * (3 * Gamma1c * pc + pc + epsilonc) * Kc
            + (
                (n + 1) / 2 * lambda2 * Gamma1c * pc
                - ell * p2
                - omega2 / expnuc * (epsilonc + pc)
            )
            * Vc,
            ###
            2 * H1c
            - expnuc / omega2 * (epsilon2 / (epsilonc + pc) - p2 / (Gamma1c * pc)) * Kc
            + (
                ell / 2 * lambda2
                + 2 * nu2
                - (2 - ell * expnuc * nu2 / omega2)
                * (epsilon2 / (epsilonc + pc) - p2 / (Gamma1c * pc))
            )
            * Vc,
            ###
            ((n + 1) / 2 * nu2 - omega2 / expnuc) * H1c
            + (omega2 / expnuc - nu2 / 2) * Kc
            - 8 * np.pi * omega2 / expnuc * (epsilonc + pc) * Vc
            + 8 * np.pi * p2 * Wc,
        ]
    )

    A = np.array(
        [
            [-(ell + 3) / 2, 1 / 2, 0, -8 * np.pi * (epsilonc + pc), 1 / 2],
            ###
            [(n + 1) / 2, -(ell + 3) / 2, -4 * np.pi * (epsilonc + pc), 0, 1 / 2],
            ###
            [0, 0, -(ell + 3) / 2 * Gamma1c * pc, -(n + 1) * Gamma1c * pc, 0],
            ###
            [0, 0, 1, ell + 2, 0],
            ###
            [0, n / 2, 0, 0, -n / 2],
        ]
    )

    x = np.linalg.solve(A, b)
    d2H1dr2c, d2Kdr2c, d2Wdr2c, d2Vdr2c, d2H0dr2c = x
    return (H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Vc, d2Vdr2c)


def solve_perturbations_interior_low_frequency(background, ell, omega2):
    """Integrate perturbation equations in interior with formulation
    appropriate for low-frequency oscillations.

    Parameters
    ----------
    background : Star object
        In addition to attributes and methods required for
        `perturbations_interior`, `perturbations_interior_low_frequency` and
        `taylor_coefficients_low_frequency`, object contains attribute `R` for
        stellar radius [km].
    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    x : (5,) complex array_like
        Coefficients of general solution [dimensionless].
    sol1, sol2 : solve_ivp object
        Solutions 1 and 2 from stellar centre.
    sol3, sol4, sol5 : solve_ivp object
        Solutions 3, 4 and 5 from stellar surface.

    Notes
    -----
    If omega2 is real, integration focuses on real values for efficiency.
    Integration with low-frequency perturbation equations only occurs from
    stellar centre.
    """
    R = background.R
    # starting point
    r0 = 1e-3
    # matching point
    rmatch = R / 2

    # from centre
    # Solution 1
    if isinstance(omega2, complex):
        Kc, Wc = 1 + 0j, 0 + 0j
    else:
        Kc, Wc = 1, 0
    (H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Vc, d2Vdr2c) = taylor_coefficients_low_frequency(
        Kc, Wc, background, ell, omega2
    )
    H10 = H1c + 1 / 2 * r0**2 * d2H1dr2c
    K0 = Kc + 1 / 2 * r0**2 * d2Kdr2c
    W0 = Wc + 1 / 2 * r0**2 * d2Wdr2c
    V0 = Vc + 1 / 2 * r0**2 * d2Vdr2c

    sol1 = solve_ivp(
        perturbations_interior_low_frequency,
        [r0, rmatch],
        [H10, K0, W0, V0],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 2
    if isinstance(omega2, complex):
        Kc, Wc = 0 + 0j, 1 + 0j
    else:
        Kc, Wc = 0, 1
    (H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Vc, d2Vdr2c) = taylor_coefficients_low_frequency(
        Kc, Wc, background, ell, omega2
    )
    H10 = H1c + 1 / 2 * r0**2 * d2H1dr2c
    K0 = Kc + 1 / 2 * r0**2 * d2Kdr2c
    W0 = Wc + 1 / 2 * r0**2 * d2Wdr2c
    V0 = Vc + 1 / 2 * r0**2 * d2Vdr2c

    sol2 = solve_ivp(
        perturbations_interior_low_frequency,
        [r0, rmatch],
        [H10, K0, W0, V0],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # from surface
    # Solution 3
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 1 + 0j, 0 + 0j, 0 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 1, 0, 0
        Xf = 0

    sol3 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 4
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 0 + 0j, 1 + 0j, 0 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 0, 1, 0
        Xf = 0

    sol4 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 5
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 0 + 0j, 0 + 0j, 1 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 0, 0, 1
        Xf = 0

    sol5 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # solve for coefficients of general solution
    A = np.zeros((5, 5), dtype=type(omega2))
    A[:3, 0] = sol1.y[:3, -1]
    A[3, 0] = X(rmatch, sol1.y[:, -1], background, ell, omega2)
    A[:3, 1] = sol2.y[:3, -1]
    A[3, 1] = X(rmatch, sol2.y[:, -1], background, ell, omega2)
    A[:4, 2] = -sol3.y[:, -1]
    A[:4, 3] = -sol4.y[:, -1]
    A[:4, 4] = -sol5.y[:, -1]
    # normalising at surface
    A[4, :] = [0, 0, sol3.y[2, 0], sol4.y[2, 0], sol5.y[2, 0]]
    b = np.array([0, 0, 0, 0, 1])
    x = np.linalg.solve(A, b)

    return (x, sol1, sol2, sol3, sol4, sol5)


def X(r, y, background, ell, omega2):
    """Return Lagrangian pressure perturbation function.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    y : (4,) complex array_like
        Metric perturbations `H1` [dimensionless] and `K` [dimensionless],
        radial displacement function `W` [km^2] and horizontal displacement
        function `V` [km^2] at `r`.
    background : Star object
        Object contains following methods as functions of radius `r` and
        pressure `p`:

        * `m(r)` for mass [km],
        * `p(r)` for pressure [km^-2],
        * `nu(r)` for metric potential [dimensionless],
        * `eos.epsilon(p)` for energy density [km^-2].

    ell : int
        Quantum number [dimensionless].
    omega2 : complex
        Square of mode frequency [km^-2].

    Returns
    -------
    X : complex
        Lagrangian pressure perturbation function `X` [km^-2] at `r`.
    """
    H1, K, W, V = y

    m, p, nu = background.m(r), background.p(r), background.nu(r)
    epsilon = background.eos.epsilon(p)

    explambda = 1 / (1 - 2 * m / r)
    expnu = np.exp(nu)

    dlambdadr = (1 - explambda) / r + 8 * np.pi * r * explambda * epsilon
    dnudr = (explambda - 1) / r + 8 * np.pi * r * explambda * p

    n = (ell + 2) * (ell - 1) / 2

    H0 = (
        r**2 / explambda * (omega2 * r / expnu - (n + 1) * dnudr / 2) * H1
        + (
            n * r
            - omega2 * r**3 / expnu
            - r**2 * dnudr / (4 * explambda) * (r * dnudr - 2)
        )
        * K
        + 4
        * np.pi
        * r**2
        * (epsilon + p)
        * (dnudr / explambda ** (1 / 2) * W + 2 * r * omega2 / expnu * V)
    ) / ((n + 1) * r - r / (2 * explambda) * (r * dlambdadr + 2))

    return (epsilon + p) * (
        omega2 / expnu ** (1 / 2) * V
        + dnudr / (2 * r) * (expnu / explambda) ** (1 / 2) * W
        + expnu ** (1 / 2) / 2 * H0
    )
