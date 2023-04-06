# optimisers.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 30 Mar 2023 11:22:14 BST

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Optional, Union

import numpy as np
import scipy as sp

from nmrespy.nlp._funcs import FunctionFactory
from nmrespy._misc import start_end_wrapper
from nmrespy._timing import timer


result_messages = {
    "success": "Optimiser successfully converged.",
    "maxiter": "Maximum allowed iterations reached.",
    "noimprov": "Improvement could not be predicted.",
    "negamp": "Negative amplitude(s) detected.",
}

TABLE_WIDTHS = (0, 7, 14, 14, 14, 0)


@dataclass
class NLPResult:
    x: np.ndarray
    errors: Optional[np.ndarray]
    trajectory: Optional[Iterable[np.ndarray]]
    result_message: Iterable[str]
    iterations: int
    time: float


def print_title():
    titles = (
        "",
        "Iter.",
        "Objective",
        "Grad. Norm",
        "Trust Radius",
        "",
    )
    title = "│".join(f"{x:^{w}}" for x, w in zip(titles, TABLE_WIDTHS))
    print("┌" + "┬".join(w * "─" for w in TABLE_WIDTHS[1:-1]) + "┐")
    print(title)
    print("├" + "┼".join(w * "─" for w in TABLE_WIDTHS[1:-1]) + "┤")


def print_entry(k: int, m: FunctionFactory, trust_radius: float) -> None:
    entries = (
        "",
        f" {k}",
        f" {m.objective:.6g}",
        f" {m.gradient_norm:.6g}",
        f" {trust_radius:.6g}",
        "",
    )
    msg = "│".join(f"{x:<{w}}" for x, w in zip(entries, TABLE_WIDTHS))
    print(msg)


@timer
@start_end_wrapper("TRUST NCG ALGORITHM STARTED", "TRUST NCG ALGORITHM COMPLETE")
def trust_ncg(
    x0: Union[np.ndarray, NLPResult],
    function_factory: FunctionFactory,
    args: Iterable[Any] = (),
    eta: float = 0.15,
    initial_trust_radius: Optional[float] = None,
    max_trust_radius: float = 4.0,
    epsilon: float = 1.e-8,
    output_mode: Optional[int] = 5,
    max_iterations: int = 100,
    save_trajectory: bool = False,
    monitor_negative_amps: bool = False,
    check_neg_amps_every: int = 10,
) -> NLPResult:
    r"""Newton Conjugate Gradient Trust-Region Algorithm.

    Parameters
    ----------
    x0
        The initial guess of parameters.

    function_factory
        Generator of numerous useful functions (objective, gradient, Hessian, etc.)

    eta
        Criterion for accepting an update. An update will be accepted if the ratio
        of the actual reduction and the predicted reduction is greater than ``eta``:

        ..math ::

            \rho_k = \frac{f(x_k) - f(x_k - p_k)}{m_k(0) - m_k(p_k)} > \eta

    initial_trust_radius
        The initial value of the radius of the trust region. By default (``None``):

        .. math ::

            1 / 10 \right \rVert
                \mathcal{F} \left( \boldsymbol{x}^{(0)} \right)
            \left \lVert_2

    max_trust_radius
        The largest permitted radius for the trust region. If ``None``, this will be
        ``16 * initial_trust_radius``.

    epsilon
        Sets the convergence criterion. Convergence will occur when
        :math:`\lVert \boldsymbol{g}_k \rVert_2 < \epsilon.

    output_mode
        Should be an integer greater than or equal to ``0`` or ``None``. If ``None``,
        no output will be given. If ``0``, only a message on the outcome of the
        optimisation will be printed. If an integer greater than ``0``, information
        for each iteration ``k`` which satisfies ``k % output_mode == 0`` will be
        printed.

    max_iterations
        The greaterest number of iterations allowed before the optimiser is
        terminated.

    save_trajectory
        If ``True``, a list of parameters at each iteration will be saved, and
        accessible via the ``trajectory`` attribute of the ``NLPResult`` object.

    monitor_negative_amps
        If ``True``, checks for negative amplitudes after each iteration, and
        terminates the optimiser if there are any.

    check_neg_amps_every
        For every iteration that is a multiple of this, negative amplitudes
        will be checked for and dealt with if found.

    Returns
    -------
    An object with information about the optimisation.
    """
    start = time.time()
    x = x0 if isinstance(x0, np.ndarray) else x0.x
    m = function_factory(x, *args)

    if initial_trust_radius is None:
        # See Gould 2005: Sensitivity of trust-region algorithms and
        # their parameters (p 231)
        initial_trust_radius = 0.1 * m.gradient_norm
    if max_trust_radius is None:
        max_trust_radius = 16 * initial_trust_radius
    trust_radius = min(initial_trust_radius, max_trust_radius)

    k = 0

    if monitor_negative_amps:
        oscs = m.args[2]
        active_idx = m.args[4]
        if 0 in active_idx:
            amp_slice = slice(0, oscs)

    if isinstance(output_mode, int) and output_mode > 0:
        print_title()
        print_entry(k, m, trust_radius)

    if save_trajectory:
        trajectory = [np.copy(x)]
    else:
        trajectory = None

    while True:
        # Solve the subproblem using the Steihaug CG algorithm
        # Required accuracy of the computed solution
        epsi = min(0.5, math.sqrt(m.gradient_norm)) * m.gradient_norm

        z = np.zeros_like(x)
        r = m.gradient
        d = -r

        while True:
            Bd = m.hessian @ d
            dBd = d.T @ Bd

            if dBd <= 0:
                ta, tb = get_boundaries(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if m.model(pa) < m.model(pb):
                    p = pa
                else:
                    p = pb
                hits_boundary = True
                break

            r_sq = r.T @ r
            alpha = r_sq / dBd
            z_next = z + alpha * d

            if sp.linalg.norm(z_next) >= trust_radius:
                ta, tb = get_boundaries(z, d, trust_radius)
                p = z + tb * d
                hits_boundary = True
                break

            r_next = r + alpha * Bd
            r_next_sq = r_next.T @ r_next

            if math.sqrt(r_next_sq) < epsi:
                hits_boundary = False
                p = z_next
                break

            beta_next = r_next_sq / r_sq
            d_next = -r_next + beta_next * d

            z = z_next
            r = r_next
            d = d_next

        predicted_value = m.model(p)
        x_proposed = x + p
        m_proposed = function_factory(x_proposed, *args)

        actual_reduction = m.objective - m_proposed.objective
        predicted_reduction = m.objective - predicted_value

        if predicted_reduction <= 0:
            result_message = result_messages["noimprov"]
            break

        rho = actual_reduction / predicted_reduction
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2 * trust_radius, max_trust_radius)

        if rho > eta:
            x = x_proposed
            m = m_proposed

        k += 1

        # Save iterate
        if save_trajectory:
            trajectory.append(np.copy(x))

        # Print output
        if isinstance(output_mode, int) and (k % output_mode == 0):
            print_entry(k, m, trust_radius)

        # Check negative amps
        if monitor_negative_amps and (k != 0) and (k % check_neg_amps_every == 0):
            neg_amps = np.where(x[amp_slice] <= 0)[0]
            if neg_amps.size > 0:
                result_message = result_messages["negamp"]
                break

        # Check for convergence
        if m.gradient_norm < epsilon:
            result_message = result_messages["success"]
            break

        # Check whether number of iterations exceeds max
        if k == max_iterations:
            result_message = result_messages["maxiter"]
            break

    if isinstance(output_mode, int) and output_mode > 0:
        if (k % output_mode != 0):
            print_entry(k, m, trust_radius)
        print("└" + "┴".join(w * "─" for w in TABLE_WIDTHS[1:-1]) + "┘")

    if isinstance(output_mode, int):
        print(result_message)
    time_elapsed = time.time() - start

    errors = np.sqrt(m.objective * np.abs(np.diag(np.linalg.inv(m.hessian))))

    return NLPResult(x, errors, trajectory, result_message, k, time_elapsed)


def get_boundaries(z, d, trust_radius):
    a = d.T @ d
    b = 2 * z.T @ d
    c = (z.T @ z) - (trust_radius ** 2)
    aux = b + math.copysign(
        math.sqrt(b * b - 4 * a * c),
        b,
    )
    return sorted([-aux / (2 * a), -(2 * c) / aux])
