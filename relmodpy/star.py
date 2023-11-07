import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class Star:
    """Structure of relativistic star.

    Parameters
    ----------
    eos : EOS object
        Object contains method `epsilon(p)` for energy density [km^-2] as
        function of pressure [km^-2].
    pc : float
        Pressure [km^-2] at stellar centre.

    Attributes
    ----------
    eos : EOS object
        Records `eos` parameter.
    pc : float
        Records `pc` parameter.
    nuc : float
        Metric potential [dimensionless] at stellar centre.
    epsilonc : float
        Energy density [km^-2] at stellar centre.
    r0 : float
        Radius [km] near stellar centre for integration.
    R : float
        Stellar radius [km].
    M : float
        Stellar mass [km].
    nu : CubicSpline object
        Return metric potential `nu(r)` [dimensionless] at given radial
        coordinate.
    rsol : array_like
        Radial coordinates [km] from integration.
    msol : array_like
        Masse [km] from integration.
    psol : array_like
        Pressure [km^-2] from integration.
    nusol : array_like
        Metric potential [dimensionless] from integration.

    Methods
    -------
    structure
    surface
    m
    p

    Notes
    -----
    Assumes geometric units, where G = c = 1.
    """
    def __init__(self, eos, pc):
        self.eos = eos
        self.pc = pc
        self.r0 = 1e-5
        self.epsilonc = eos.epsilon(pc)

        # Taylor expansion near centre for accuracy
        p2 = - 4*np.pi/3*(self.epsilonc + pc)*(self.epsilonc + 3*pc)
        m3 = 4*np.pi*self.epsilonc

        m0 = 1/3*self.r0**3*m3
        p0 = pc + 1/2*self.r0**2*p2

        # integrate solution
        self.__sol = solve_ivp(self.structure, [self.r0, 20], [m0, p0, 0],
                               method='DOP853', dense_output=True,
                               events=self.surface, rtol=1e-10, atol=1e-10)

        self.rsol = self.__sol.t
        self.msol, self.psol, nusol = self.__sol.y
        self.M, self.R = self.msol[-1], self.rsol[-1]

        # adjust to match surface boundary condition
        self.nusol = nusol - nusol[-1] + np.log(1 - 2*self.M/self.R)

        # calculate central metric potential
        nu2 = 8*np.pi/3*(self.epsilonc + pc)
        self.nuc = self.nusol[0] - self.r0**2*nu2/2

        # interpolate metric potential separately
        self.nu = CubicSpline(self.rsol, self.nusol, extrapolate=False)

    def structure(self, r, y):
        """Stellar structure equations.

        Parameters
        ----------
        r : float
            Radial coordinate [km].
        y : (3,) array_like
            Mass `m` [km], pressure `p` [km^-2] and metric potential `nu`
            [dimensionless] at `r`.

        Returns
        -------
        dmdr : float
            Derivative of mass `m` [dimensionless] at `r`.
        dpdr : float
            Derivative of pressure `p` [km^-3] at `r`.
        dnudr : float
            Derivative of metric potential `nu` [km^-1] at `r`.
        """
        m, p, nu = y

        if p < 0:
            epsilon = np.nan
        else:
            epsilon = self.eos.epsilon(p)

        dmdr = 4*np.pi*r**2*epsilon
        dnudr = 2*(m + 4*np.pi*r**3*p)/(r*(r - 2*m))
        dpdr = - (epsilon + p)*dnudr/2

        return [dmdr, dpdr, dnudr]

    def surface(self, r, y):
        """Definition of stellar surface: `p = 0`."""
        m, p, nu = y
        return p
    surface.terminal = True
    surface.direction = -1

    def m(self, r):
        """Return mass.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        m : float
            Mass `m` [km] at `r`.
        """
        m, p, nu = self.__sol.sol(r)
        return m

    def p(self, r):
        """Return pressure.

        Parameters
        ----------
        r : float
            Radial coordinate [km].

        Returns
        -------
        p : float
            Pressure `p` [km^-2] at `r`.
        """
        m, p, nu = self.__sol.sol(r)
        return p
