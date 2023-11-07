# RelModPy
Computing the oscillation modes of relativistic stars in Python.

For the uninitiated, a good textbook on oscillation modes in the context of Newtonian gravity is Ref. [1]. This implementation follows closely Ref. [2].

## Usage
`relmodpy` depends on `numpy` and `scipy`. It uses `numpy` for mathematical functions and linear-algebra operations. `scipy` is used for solving differential equations.

An example of its use is shown in `example.py`. Throughout its implementation, geometric units, where $G = c = 1$, are assumed.

### Equation of state
To calculate oscillation modes, the user must supply, first and foremost, an equation of state for the stellar matter. This will be described by an `EOS` object that contains the following thermodynamic relationships as methods:
* `epsilon(p)`: the energy density [km^-2] as a function of pressure [km^-2],
* `Gamma(p)`: the adiabatic index associated with the background [dimensionless] as a function of pressure [km^-2],
* `Gamma1(p)`: the adiabatic index associated with the perturbations [dimensionless] as a function of pressure [km^-2].

`relmodpy` provides some simple examples in `eos.py`, one of which is used in `example.py`. These are assumed to be barotropic, where $\Gamma_1 = \Gamma$, and are called `Polytrope` and `EnergyPolytrope`. These are not intended to be accurate descriptions of neutron-star matter.

### Stellar background
The perturbations are calculated on top of a spherically symmetric background. This must be initialised before the modes are computed. 

The background is contained in a `Star` object, which has two parameters: an `EOS` object and `pc` central pressure [km^-2]. It can be initialised as
```
from relmodpy import Star
from relmodpy.eos import EnergyPolytrope

eos = EnergyPolytrope(1, 100)
star = Star(eos, 5.52e-3)
```

### Searching for oscillations
Provided the equation of state and the computed background, one is in a position to obtain the oscillation modes. The `Mode` object is initialised as
```
from relmodpy import Mode

mode = mode(star)
```
Note: `mode` accesses the equation of state from `star`, since the background and mode will use the same equation of state.

Searching for the quadrupole *f*-mode can be done as
```
l = 2
omegaguess = (0.171 + 6.2e-5j)/star.M
mode.solve(l,
           (omegaguess,
            1.001*omegaguess.real + 1j*omegaguess.imag,
            omegaguess.real + 1.001j*omegaguess.imag))
```
From `mode`, the user can access the eigenfrequency `omega`, as well as the eigenfunctions `(H1, K, W, X)`.

## Testing

### Disclaimer
I do not claim that this code is suited to finding the infinite spectrum of oscillation modes of a star, since this is a substantially challenging, computational task. Furthermore, it should be noted that for oscillations that are damped extremely weakly (*g*-modes), the imaginary parts of their eigenfrequencies can be below machine precision. In `relmodpy`, an artifical cut-off frequency is assumed, below which only the real parts are calculated. I have, however, made an effort to test this code on established results in the literature [3,4]. These can be found in `test.py`.

### Running the tests
For example, to run the tests on `mode.py`, write
```
python -m unittest tests.test_mode
```
in the root directory. Warning: this can take quite a while...

## References
[1] Unno *et al.* (1989), "Nonradial oscillations of stars," University of
    Tokyo Press, Tokyo.
[2] Detweiler and Lindblom (1985), "On the nonradial pulsations of general
    relativistic stellar models," *Astrophys. J.* **292**, 12.
[3] Andersson, Kokkotas and Schutz (1995), "A new numerical approach to the
    oscillation modes of relativistic stars," *Mon. Not. R. Astron. Soc.*
    **274** (4), 1039.
[4] Krüger (2015), "Seismology of adolescent general relativistic neutron
    stars," PhD thesis, University of Southampton.
