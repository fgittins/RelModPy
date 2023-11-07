"""Polytropic equations of state for baroptropic matter.

Classes
-------
Polytrope
EnergyPolytrope

Notes
-----
Assumes geometric units, where G = c = 1.

* For `Star` class, `EOS` object must have `epsilon(p)` method.
* For `Mode` class, `EOS` object contained in `background` must also have
`Gamma(p)` and `Gamma1(p)` methods.
"""


class Polytrope:
    """Polytrope defined with rest-mass density.

    Methods are functions of pressure `p` [km^-2].

    Attributes
    ----------
    n : int
        Polytropic index [dimensionless].
    K : float
        Proportionality constant [km^2/n].

    Methods
    -------
    epsilon
    Gamma
    Gamma1
    """
    def __init__(self, n, K):
        self.n = n
        self.K = K

    def epsilon(self, p):
        """Return energy density [km^-2]."""
        return (p / self.K)**(self.n/(self.n + 1)) + self.n*p

    def Gamma(self, p):
        """Return adiabatic index [dimensionless] associated with background.
        """
        return 1 + 1/self.n

    def Gamma1(self, p):
        """Return adiabatic index [dimensionless] associated with
        perturbations.

        Equation of state is assumed to be barotropic.
        """
        return self.Gamma(p)


class EnergyPolytrope:
    """Polytrope defined with energy density.

    Methods are functions of pressure `p` [km^-2].

    Attributes
    ----------
    n : int
        Polytropic index [dimensionless].
    K : float
        Proportionality constant [km^2/n].

    Methods
    -------
    epsilon
    Gamma
    Gamma1
    """
    def __init__(self, n, K):
        self.n = n
        self.K = K

    def epsilon(self, p):
        """Return energy density [km^-2]."""
        return (p / self.K)**(self.n/(self.n + 1))

    def Gamma(self, p):
        """Return adiabatic index [dimensionless] associated with background.
        """
        return (1 + 1/self.n)*(1 + self.K*(p / self.K)**(1/(self.n + 1)))

    def Gamma1(self, p):
        """Return adiabatic index [dimensionless] associated with
        perturbations.

        Equation of state is assumed to be barotropic.
        """
        return self.Gamma(p)
