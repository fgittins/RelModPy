from collections.abc import Callable


class Results:
    """Represents root-finding result.

    Attributes
    ----------
    root : complex
        Estimated root.
    iterations : int
        Number of iterations needed to find root.
    converged : bool
        True if routine converged.
    flag : str
        Description of cause of termination.
    """

    def __init__(
        self, root: complex, iterations: int, converged: bool, flag: str
    ) -> None:
        self.root = root
        self.iterations = iterations
        self.converged = converged
        self.flag = flag


def muller(
    f: Callable[[complex], complex],
    x: tuple[complex, complex, complex],
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    maxiter: int = 50,
) -> Results:
    """Muller's method for root finding of scalar function.

    Implementation is based on description of Muller's method in Sec. 9.5.2 of
    Ref. [1].

    Parameters
    ----------
    f : callable
        Function to find root of.
    x : (3,) array_like
        Three initial guesses.
    xtol : float, optional
        Absolute error in `x` between iterations that is acceptable for
        convergence.
    ftol : float, optional
        Minimum absolute value of function `f` that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum number of iterations.
    verbose : bool, optional
        Prints final number of iterations.

    Returns
    -------
    res : Results object
        Contains results of routine.

    References
    ----------
    [1] Press et al. (2007), "Numerical recipes. The Art of Scientific
        Computing, 3rd Edition" (Cambridge University Press, Cambridge, UK;
        http://numerical.recipes/book.html).
    """
    if xtol <= 0:
        raise ValueError(f"xtol is too small (xtol = {xtol} <= 0)")
    if ftol <= 0:
        raise ValueError(f"ftol is too small (ftol = {ftol} <= 0)")
    if not isinstance(maxiter, int):
        raise ValueError("maxiter must be integer")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    ximinus2, ximinus1, xi = x
    yiminus2, yiminus1, yi = f(ximinus2), f(ximinus1), f(xi)

    converged = False
    flag = "Routine did not converge"

    for i in range(maxiter):
        q = (xi - ximinus1) / (ximinus1 - ximinus2)
        A = q * yi - q * (1 + q) * yiminus1 + q**2 * yiminus2
        B = (2 * q + 1) * yi - (1 + q) ** 2 * yiminus1 + q**2 * yiminus2
        C = (1 + q) * yi

        denomplus = B + (B**2 - 4 * A * C) ** (1 / 2)
        denomminus = B - (B**2 - 4 * A * C) ** (1 / 2)

        if abs(denomplus) >= abs(denomminus):
            xiplus1 = xi - (xi - ximinus1) * 2 * C / denomplus
        else:
            xiplus1 = xi - (xi - ximinus1) * 2 * C / denomminus

        yiplus1 = f(xiplus1)

        if ftol >= abs(yiplus1):
            flag = (
                "Routine has reached desired tolerance "
                f"in function value ({abs(yiplus1)} <= {ftol})"
            )
            converged = True
            break
        if xtol >= abs(xiplus1 - xi):
            flag = (
                "Routine has reached desired tolerance "
                f"in root value ({abs(xiplus1 - xi)} <= {xtol})"
            )
            converged = True
            break

        ximinus2, ximinus1, xi = ximinus1, xi, xiplus1
        yiminus2, yiminus1, yi = yiminus1, yi, yiplus1

    return Results(xiplus1, i + 1, converged, flag)
