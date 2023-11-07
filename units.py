"""Record physical constants and useful conversion factors.

Constants are given in Centimetre-Gram-Second (CGS) system of units.

Geometric units assume G = c = 1 and measure all variables in powers of km.
Natural units assume hbar = c = kB = 1.
"""

# speed of light in vacuum [cm s^-1]
c = 29979245800
# gravitational constant [cm^3 g^-1 s^-2]
G = 6.67430e-8
# solar mass [g]
Msol = 1.98841e33
# electron volt [erg]
eV = 1.602176634e-12

# conversion factors from geometric to CGS units
# length [cm km^-1]
length_geometric_to_CGS = 1e5
# mass [g km^-1]
mass_geometric_to_CGS = 1e5*c**2/G
# time [s km-1]
time_geometric_to_CGS = 1e5/c

# conversion factor for pressure and energy density [erg cm^-3 km^2]
pressure_geometric_to_CGS = (mass_geometric_to_CGS
                             /length_geometric_to_CGS
                             /time_geometric_to_CGS**2)

# conversion factor for mass to Msol [Msol km^-1]
mass_geometric_to_Msol = mass_geometric_to_CGS/Msol

# conversion factor for pressure and energy density [MeV fm^-3 km^2]
pressure_geometric_to_natural = pressure_geometric_to_CGS/(eV*1e45)
