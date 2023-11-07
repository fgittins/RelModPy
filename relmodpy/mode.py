import numpy as np
from scipy.optimize import minimize

from .perturbations_interior import (
        solve_perturbations_interior,
        solve_perturbations_interior_low_frequency)
from .perturbations_exterior import (
        solve_perturbations_exterior, transformation, Ain, Aout)
from .muller import muller


class Mode:
    """Polar mode of relativistic star.

    Parameters
    ----------
    background : Star object
        Object contains following attributes and methods as functions of radius
        `r` and pressure `p`:

        * `R` for stellar radius [km],
        * `M` for stellar mass [km],
        * `m(r)` for mass [km],
        * `p(r)` for pressure [km^-2],
        * `nu(r)` for metric potential [dimensionless],
        * `eos.epsilon(p)` for energy density [km^-2],
        * `eos.Gamma1(p)` for adiabatic index [dimensionless].

    Attributes
    ----------
    background : Star object
        Records `background` parameter.
    l : int
        Quantum number [dimensionless].
    omega : complex
        Mode eigenfrequency [km^-1].
    r0 : float
        Radius [km] near stellar centre for integration. In general, this can be
        different to `background.r0`.

    Methods
    -------
    eigenproblem
    spectrum
    solve
    H1
    K
    W

    Notes
    -----
    Assumes geometric units, where G = c = 1. For low frequencies
    ```
    omegaguess.real*M < 0.01
    ```
    specialist formulation of interior perturbation equations is used.
    """
    def __init__(self, background):
        self.background = background

        self.l = None
        self.omega = None

        self.__x = None
        self.__sol1, self.__sol2 = None, None
        self.__sol3, self.__sol4, self.__sol5 = None, None, None

        self.r0 = None
        self.__rmatch = None

    def eigenproblem(self, l, omegaguess):
        """Defines function to find root of for eigenfrequency.

        Parameters
        ----------
        l : int
            Quantum number [dimensionless].
        omegaguess : complex
            Guess of mode frequency [km^-1].

        Returns
        -------
        Ain/Aout : complex
            Ratio of ingoing to outgoing radiation [dimensionless].
        """
        R = self.background.R
        M = self.background.M

        if omegaguess.real*M < 0.01:
            interior_solver = solve_perturbations_interior_low_frequency
        else:
            interior_solver = solve_perturbations_interior

        # interior
        (self.__x,
         self.__sol1, self.__sol2,
         self.__sol3, self.__sol4, self.__sol5) = interior_solver(
                self.background, l, omegaguess**2)
        H1, K = (self.__x[2]*self.__sol3.y[:2, 0]
                 + self.__x[3]*self.__sol4.y[:2, 0]
                 + self.__x[4]*self.__sol5.y[:2, 0])

        # exterior
        n = (l + 2)*(l - 1)/2
        sol = solve_perturbations_exterior(R, M, n, omegaguess)
        q, dqdrho = sol.y[:, -1]
        theta = -np.arctan(omegaguess.imag/omegaguess.real)
        dqdr = np.exp(-1j*theta)*dqdrho
        Z, dZdrstar = transformation(R, (H1, K), M, n)

        return Ain(R, q, dqdr, Z, dZdrstar, M)/Aout(R, q, dqdr, Z, dZdrstar, M)

    def spectrum(self, l, omegaguess):
        """Calculate "spectrum" using amplitude of ingoing radiation at
        surface.

        Parameters
        ----------
        l : int
            Quantum number [dimensionless].
        omegaguess : complex
            Guess of mode frequency [km^-1].

        Returns
        -------
        Ain : complex
            Amplitude of ingoing radiation [dimensionless] at `R`.
        """
        R = self.background.R
        M = self.background.M

        if omegaguess.real*M < 0.01:
            interior_solver = solve_perturbations_interior_low_frequency
        else:
            interior_solver = solve_perturbations_interior

        # interior
        (self.__x,
         self.__sol1, self.__sol2,
         self.__sol3, self.__sol4, self.__sol5) = interior_solver(
                self.background, l, omegaguess**2)
        H1, K = (self.__x[2]*self.__sol3.y[:2, 0]
                 + self.__x[3]*self.__sol4.y[:2, 0]
                 + self.__x[4]*self.__sol5.y[:2, 0])

        # exterior
        n = (l + 2)*(l - 1)/2
        sol = solve_perturbations_exterior(R, M, n, omegaguess)
        q, dqdrho = sol.y[:, -1]
        theta = -np.arctan(omegaguess.imag/omegaguess.real)
        dqdr = np.exp(-1j*theta)*dqdrho
        Z, dZdrstar = transformation(R, (H1, K), M, n)

        return Ain(R, q, dqdr, Z, dZdrstar, M)

    def solve(self, l, omegaguess, method='Muller'):
        """Solve for mode eigenfrequency and eigenfunctions.

        Parameters
        ----------
        l : int
            Quantum number [dimensionless].
        omegaguess : complex
            Guess of mode frequency [km^-1]. If `method` is 'Muller', this must
            be three values. If `method` is 'Simplex', this must be one value.
        method : str, optional
            Type of solver. Must be one of

            * 'Muller',
            * 'Simplex'.

            Default is 'Muller'.
        
        Notes
        -----
        At very low frequencies
        ```
        omegaguess.real*M < 0.02
        ```
        only real values are considered.
        """
        # solve eigenfrequency
        if method == 'Muller':
            def fmuller(omegaguess):
                return self.eigenproblem(l, omegaguess)

            res = muller(fmuller, omegaguess, xtol=1e-8, ftol=1e-5)
            omega = res.root
        elif method == 'Simplex':
            M = self.background.M
            # ignore damping for very low frequencies
            if omegaguess.real*M < 0.02:
                def fsimplex(x):
                    return abs(self.eigenproblem(l, x[0]))

                res = minimize(fsimplex, (omegaguess.real,),
                               method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-5,
                                        'disp': False})
                omega = res.x[0]
            else:
                def fsimplex(x):
                    return abs(self.eigenproblem(l, x[0] + 1j*x[1]))

                res = minimize(fsimplex, (omegaguess.real, omegaguess.imag),
                               method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-5,
                                        'disp': False})
                omega = res.x[0] + 1j*res.x[1]
        else:
            raise ValueError("method must be either 'Muller' or 'Simplex'")

        self.omega = omega
        self.l = l

        self.r0 = self.__sol1.t[0]
        self.__rmatch = self.__sol1.t[-1]

    def H1(self, r):
        """Return metric perturbation `H1`.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        H1 : complex
            Metric perturbation `H1` [dimensionless].
        """
        if self.r0 <= r <= self.__rmatch:
            H1, K, W, _ = (self.__x[0]*self.__sol1.sol(r)
                           + self.__x[1]*self.__sol2.sol(r))
        elif self.__rmatch < r <= self.background.R:
            H1, K, W, _ = (self.__x[2]*self.__sol3.sol(r)
                           + self.__x[3]*self.__sol4.sol(r)
                           + self.__x[4]*self.__sol5.sol(r))
        else:
            raise ValueError('r is outside range')
        return H1

    def K(self, r):
        """Return metric perturbation `K`.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        K : complex
            Metric perturbation `K` [dimensionless].
        """
        if self.r0 <= r <= self.__rmatch:
            H1, K, W, _ = (self.__x[0]*self.__sol1.sol(r)
                           + self.__x[1]*self.__sol2.sol(r))
        elif self.__rmatch < r <= self.background.R:
            H1, K, W, _ = (self.__x[2]*self.__sol3.sol(r)
                           + self.__x[3]*self.__sol4.sol(r)
                           + self.__x[4]*self.__sol5.sol(r))
        else:
            raise ValueError('r is outside range')
        return K

    def W(self, r):
        """Return radial displacement function `W`.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        W : complex
            Radial displacement function `W` [km^2].
        """
        if self.r0 <= r <= self.__rmatch:
            H1, K, W, _ = (self.__x[0]*self.__sol1.sol(r)
                           + self.__x[1]*self.__sol2.sol(r))
        elif self.__rmatch < r <= self.background.R:
            H1, K, W, _ = (self.__x[2]*self.__sol3.sol(r)
                           + self.__x[3]*self.__sol4.sol(r)
                           + self.__x[4]*self.__sol5.sol(r))
        else:
            raise ValueError('r is outside range')
        return W
