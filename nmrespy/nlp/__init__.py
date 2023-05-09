# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 05 May 2023 11:14:39 BST

"""Nonlinear programming for generating parameter estiamtes.

MWE
---

.. literalinclude:: examples/nlp_example.py
"""

import copy
import functools
import operator
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import numpy.linalg as nlinalg

from nmrespy import ExpInfo
from nmrespy._colors import ORA, END, USE_COLORAMA
from nmrespy._misc import start_end_wrapper
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy._result_fetcher import ResultFetcher
from nmrespy._timing import timer

from . import _funcs as funcs, optimisers

if USE_COLORAMA:
    import colorama
    colorama.init()


@timer
@start_end_wrapper("OPTIMISATION STARTED", "OPTIMISATION COMPLETE")
def nonlinear_programming(
    expinfo: ExpInfo,
    data: np.ndarray,
    theta0: np.ndarray,
    start_time: Optional[Iterable[int]] = None,
    phase_variance: bool = True,
    hessian: str = "gauss-newton",
    bound: bool = False,
    max_iterations: int = 100,
    mode: str = "apfd",
    amp_thold: Optional[float] = None,
    freq_thold: Optional[float] = None,
    negative_amps: str = "remove",
    output_mode: Optional[int] = 10,
    save_trajectory: bool = False,
    epsilon: float = 1.0e-8,
    eta: float = 0.15,
    initial_trust_radius: float = 1.0,
    max_trust_radius: float = 4.0,
    check_neg_amps_every: int = 10,
):
    r"""
    Parameters
    ----------
    expinfo
        Experiment information.

    data
        Signal to be considered.

    theta0
        Initial parameter guess in the following form:

        * **1-dimensional data:**

          .. code:: python3

             theta0 = numpy.array([
                 [a_1, φ_1, f_1, η_1],
                 [a_2, φ_2, f_2, η_2],
                 ...,
                 [a_m, φ_m, f_m, η_m],
             ])

        * **2-dimensional data:**

          .. code:: python3

             theta0 = numpy.array([
                 [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                 [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                 ...,
                 [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
             ])

    start_time
        The start time in each dimension. If set to ``None``, the initial
        point in each dimension with be ``0.0``. To set non-zero start times,
        a list of floats or strings can be used. If floats are used, they
        specify the start time in each dimension in seconds. Alternatively,
        strings of the form ``r"\d+dt"``, may be used, which indicates a
        cetain multiple of the difference in time between two adjacent
        points.

    phase_variance
        Specifies whether or not to include the variance of oscillator
        phases into the NLP routine.

    hessian
        Specifies how to construct the Hessian matrix.

        * ``"exact"`` The Hessian will be exact.
        * ``"gauss-newton"`` The Hessian will be approximated as is done with
          the `Gauss-Newton method <https://en.wikipedia.org/wiki/
          Gauss%E2%80%93Newton_algorithm>`_

    bound
        .. warning::

            Not yet supported. Hard-coded to ``False``.

        Specifies whether or not to bound the parameters during
        optimisation. Bounds are given by:

        * :math:`0 \leq a_m \leq \infty`
        * :math:`-\pi < \phi_m \leq \pi`
        * :math:`-f_{\mathrm{sw}} / 2 + f_{\mathrm{off}} \leq f_m \leq\
          f_{\mathrm{sw}} / 2 + f_{\mathrm{off}}`
        * :math:`0 \leq \eta_m \leq \infty`

        :math:`(\forall m \in \{1, \cdots, M\})`

    max_iterations
        A value specifiying the number of iterations the routine may run
        through before it is terminated. If ``None``, the default number
        of maximum iterations is set (``100`` if ``method`` is
        ``"exact"`` or ``"gauss-newton"``, and ``500`` if ``"method"`` is
        ``"lbfgs"``).

    mode
        A string containing a subset of the characters ``"a"`` (amplitudes),
        ``"p"`` (phases), ``"f"`` (frequencies), and ``"d"`` (damping factors).
        Specifies which types of parameters should be considered for optimisation.

    amp_thold
        A value that imposes a threshold for deleting oscillators of
        negligible ampltiude. If ``None``, does nothing. If a float,
        oscillators with amplitudes satisfying :math:`a_m <
        a_{\mathrm{thold}} \lVert \boldsymbol{a} \rVert_2`` will be
        removed from the parameter array, where :math:`\lVert
        \boldsymbol{a} \rVert_2` is the Euclidian norm of the vector of
        all the oscillator amplitudes. It is advised to set ``amp_thold``
        at least a couple of orders of magnitude below 1.

    freq_thold
        .. warning::

            Note yet supprted. Hard-coded to be ``None``.

        If ``None``, does nothing. If a float, oscillator pairs with
        frequencies satisfying
        :math:`\lvert f_m - f_p \rvert < f_{\mathrm{thold}}` will be
        removed from the parameter array. A new oscillator will be included
        in the array, with parameters:

        * amplitude: :math:`a = a_m + a_p`
        * phase: :math:`\phi = \left(\phi_m + \phi_p\right) / 2`
        * frequency: :math:`f = \left(f_m + f_p\right) / 2`
        * damping: :math:`\eta = \left(\eta_m + \eta_p\right) / 2`

    negative_amps
        Indicates how to treat oscillators which have gained negative
        amplitudes during the optimisation.

        * ``"remove"`` will result in such oscillators being purged from
          the parameter estimate. The optimisation routine will the be
          re-run recursively until no oscillators have a negative
          amplitude.
        * ``"flip_phase"`` will retain oscillators with negative
          amplitudes, but the the amplitudes will be multiplied by -1,
          and a π radians phase shift will be applied.
        * ``"ignore"`` will do nothing (negative amplitude oscillators will remain).

    output_mode
        Should be an integer greater than or equal to ``0`` or ``None``. If
        ``None``, no output will be given. If ``0``, only a message on the
        outcome of the optimisation will be printed. If an integer greater
        than ``0``, information for each iteration ``k`` which satisfies
        ``k % output_mode == 0`` will be printed.

    save_trajectory
        If ``True``, a list of parameters at each iteration will be saved, and
        accessible via the ``trajectory`` attribute.

    epsilon
        Sets the convergence criterion. Convergence will occur when
        :math:`\lVert \boldsymbol{g}_k \rVert_2 < epsilon`.

    eta
        Criterion for accepting an update. An update will be accepted if the ratio
        of the actual reduction and the predicted reduction is greater than ``eta``:

        ..math ::

            \rho_k = \frac{f(x_k) - f(x_k - p_k)}{m_k(0) - m_k(p_k)} > \eta

    initial_trust_radius
        The initial value of the radius of the trust region.

    max_trust_radius
        The largest permitted radius for the trust region.

    check_neg_amps_every
        For every iteration that is a multiple of this, negative amplitudes
        will be checked for and dealt with if found.
    """
    sanity_check(
        ("expinfo", expinfo, sfuncs.check_expinfo),
        ("phase_variance", phase_variance, sfuncs.check_bool),
        ("hessian", hessian, sfuncs.check_one_of, ("exact", "gauss-newton")),
        ("bound", bound, sfuncs.check_bool),
        (
            "max_iterations", max_iterations, sfuncs.check_int, (),
            {"min_value": 1},
        ),
        ("mode", mode, sfuncs.check_optimiser_mode),
        (
            "amp_thold", amp_thold, sfuncs.check_float, (),
            {"greater_than_zero": True}, True,
        ),
        (
            "freq_thold", freq_thold, sfuncs.check_float, (),
            {"greater_than_zero": True}, True,
        ),
        (
            "negative_amps", negative_amps, sfuncs.check_one_of,
            ("remove", "flip_phase", "ignore"),
        ),
        ("output_mode", output_mode, sfuncs.check_int, (), {"min_value": 0}, True),
        ("save_trajectory", save_trajectory, sfuncs.check_bool),
        (
            "epsilon", epsilon, sfuncs.check_float, (),
            {"min_value": np.finfo(float).eps},
        ),
        (
            "eta", eta, sfuncs.check_float, (),
            {"greater_than_zero": True, "max_value": 1.0},
        ),
        (
            "initial_trust_radius", initial_trust_radius, sfuncs.check_float, (),
            {"greater_than_zero": True},
        ),
    )

    dim = expinfo.dim

    sanity_check(
        ("data", data, sfuncs.check_ndarray, (dim,)),
        ("theta0", theta0, sfuncs.check_parameter_array, (dim,)),
        (
            "start_time", start_time, sfuncs.check_start_time,
            (dim,), {"len_one_can_be_listless": True}, True,
        ),
        (
            "max_trust_radius", max_trust_radius, sfuncs.check_float, (),
            {"min_value": initial_trust_radius},
        ),
        (
            "check_neg_amps_every", check_neg_amps_every, sfuncs.check_int, (),
            {"min_value": 1, "max_value": max_iterations},
        ),
    )

    # Hard-code features that have not yet been implemented
    bound = False
    freq_thold = None

    # Normalise the data
    norm = np.linalg.norm(data)
    normed_data = data / norm

    # Number of oscillators and number of parameters per oscillator
    m, p = theta0.shape
    # Amplitude threshold procesing
    if amp_thold is None:
        amp_thold = 0.0
    # TODO: Frequency threshold processing

    if "p" not in mode:
        phase_variance = False

    active_idx, passive_idx = _get_active_passive_indices(mode, dim)
    # Flatten parameter array: (m, p) -> (p * m,)
    theta0_proc = theta0.flatten(order="F")
    # Normalise amplitudes
    theta0_proc[: m] /= norm
    # Center frequecnies at 0
    offset = expinfo.offset()
    theta0_proc = _shift_offset(theta0_proc, m, offset, "center")
    # Split parameter array into active and passive components
    active, passive = _split_parameters(theta0_proc, m, active_idx, passive_idx)
    # Get function factory
    function_factory = _get_function_factory(dim, hessian)

    # Extra arguments needed for the objective, grad, and Hessian, which are not
    # the parameters being optimised (`active`)
    opt_args = [
        normed_data,
        expinfo.get_timepoints(
            pts=normed_data.shape, start_time=start_time, meshgrid=False,
        ),
        m,
        passive,
        active_idx,
        phase_variance,
    ]

    # Create list to save parameters at each iteration
    trajectory = [] if save_trajectory else None
    opt_messages = []
    iterations = 0
    opt_time = 0.

    while True:
        result = optimisers.trust_ncg(
            x0=active,
            function_factory=function_factory,
            args=tuple(opt_args),
            eta=eta,
            initial_trust_radius=initial_trust_radius,
            max_trust_radius=max_trust_radius,
            epsilon=epsilon,
            output_mode=output_mode,
            max_iterations=max_iterations,
            save_trajectory=save_trajectory,
            monitor_negative_amps=negative_amps == "remove",
            check_neg_amps_every=check_neg_amps_every,
        )

        active = result.x
        if save_trajectory:
            # Need to know the passive params associated with trajectory to
            # reconstruct later on
            trajectory.append([result.trajectory, passive])
        opt_messages.append(result.result_message)
        iterations += result.iterations
        opt_time += result.time

        # --- Tackle negative and negligible amplitudes ---
        # `rerun` flag specifies whether or not to rerun the optimiser as some
        # oscillators have been purged.
        rerun = False
        if negative_amps == "ignore":
            pass

        elif 0 in active_idx:
            # Check for negative ampltiudes
            negative_idx = list(np.where(active[: m] <= 0.)[0])

            if negative_idx:
                if output_mode is not None:
                    print(f"{ORA}Negative amplitudes detected.")

                # Remove negative oscillators
                if negative_amps == "remove":
                    if output_mode is not None:
                        print("These oscillators will be removed")

                    rerun = True
                    active = np.delete(
                        active, _get_slice(m, active_idx, osc_idx=negative_idx),
                    )
                    passive = np.delete(
                        passive, _get_slice(m, passive_idx, osc_idx=negative_idx),
                    )
                    m -= len(negative_idx)

                    if output_mode is not None:
                        print(f"Updated number of oscillators: {m}{END}")

                # Make negative amplitude oscillators positive and flip
                # phase by 180°
                elif negative_amps == "flip_phase":
                    if output_mode is not None:
                        print(
                            "These oscillators will have their amplitudes "
                            "multiplied by -1, and phases shifted by 180°"
                        )
                    amp_slice = _get_slice(m, [0], osc_idx=negative_idx)
                    active[amp_slice] *= -1
                    phase_slice = _get_slice(m, [1], osc_idx=negative_idx)
                    active[phase_slice] = \
                        (active[phase_slice] + 2 * np.pi) % (2 * np.pi) - np.pi

            # Check for negligible ampltiudes
            thold = amp_thold * np.linalg.norm(active[: m])
            negligible_idx = list(np.where(active[: m] <= thold)[0])

            if negligible_idx:
                if output_mode is not None:
                    print(
                        f"{ORA}Negligible amplitudes detected (smaller than "
                        f"{amp_thold:.4g}). These oscillators will be removed{END}"
                    )

                rerun = True

                active = np.delete(
                    active, _get_slice(m, active_idx, osc_idx=negligible_idx),
                )
                passive = np.delete(
                    passive, _get_slice(m, passive_idx, osc_idx=negligible_idx),
                )
                m -= len(negligible_idx)

                if output_mode is not None:
                    print(f"{ORA}Updated number of oscillators: {m}{END}")

        if not rerun:
            break

        else:
            opt_args[2] = m
            opt_args[3] = passive

    # --- Generate the errors ---
    # Switch off phase variance for objective and Hessian computation
    opt_args[-1] = False
    if dim == 1:
        func = funcs.obj_grad_true_hess_1d
    elif dim == 2:
        func = funcs.obj_grad_true_hess_2d

    obj, _, hess = func(active, *opt_args)
    errors = np.sqrt(obj * np.abs(np.diag(np.linalg.inv(hess))) / np.prod(data.shape))

    # --- Format the result array ---
    active_slice = _get_slice(m, active_idx)
    passive_slice = _get_slice(m, passive_idx)
    if not passive_idx:
        theta = active
    else:
        theta = np.zeros((m * p,), dtype="float64")
        theta[active_slice] = active
        theta[passive_slice] = passive

    # Move frequencies back to their true values
    theta = _shift_offset(theta, m, offset, "displace")
    # Re-scale amplitudes
    theta[: m] *= norm
    # Set phases between 0 and 2π
    theta[m : 2 * m] = (theta[m : 2 * m] + np.pi) % (2 * np.pi) - np.pi
    # Reshape: (m * p,) -> (m, p)
    theta = theta.reshape((m, p), order="F")

    # --- Format the error array ---
    proc_errors = np.full((m * p,), np.nan, dtype="float64")
    proc_errors[active_slice] = errors
    if 0 in active_idx:
        proc_errors[: m] *= norm
    proc_errors = proc_errors.reshape((m, p), order="F")

    # --- Format the trajectory arrays ---
    if save_trajectory:
        proc_trajectories = []
        for i, (trajs, passive) in enumerate(trajectory):
            m = trajs[0].size // len(active_idx)
            for traj in trajs:
                if not passive_idx:
                    traj_proc = traj
                else:
                    traj_proc = np.zeros((m * p,), dtype="float64")
                    traj_proc[_get_slice(m, active_idx)] = traj
                    traj_proc[_get_slice(m, passive_idx)] = passive
                traj_proc = _shift_offset(traj_proc, m, offset, "displace")
                traj_proc[: m] *= norm
                traj_proc[m : 2 * m] = \
                    (traj_proc[m : 2 * m] + np.pi) % (2 * np.pi) - np.pi
                traj_proc = traj_proc.reshape((m, p), order="F")
                proc_trajectories.append(traj_proc)
    else:
        proc_trajectories = None

    return optimisers.NLPResult(
        theta, proc_errors, proc_trajectories, opt_messages, iterations, opt_time,
    )


def _get_active_passive_indices(
    mode: str, dim: int
) -> Tuple[Iterable[int], Iterable[int]]:
    active_idx = []
    for c in mode:
        if c == "a":
            active_idx.append(0)
        elif c == "p":
            active_idx.append(1)
        elif c == "f":
            active_idx.extend(list(range(2, 2 + dim)))
        elif c == "d":
            active_idx.extend(list(range(2 + dim, 2 + 2 * dim)))

    passive_idx = [i for i in range(2 + 2 * dim) if i not in active_idx]

    return list(sorted(active_idx)), list(sorted(passive_idx))


def _split_parameters(
    theta: np.ndarray, m: int, active_idx: Iterable[int], passive_idx: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if not passive_idx:
        return theta, np.array([], dtype="float64")

    else:
        active_slice = _get_slice(m, active_idx)
        passive_slice = _get_slice(m, passive_idx)
        return theta[active_slice], theta[passive_slice]


def _get_slice(
    m: int,
    idx: Iterable[int],
    osc_idx: Optional[Iterable[int]] = None,
) -> Iterable[int]:
    if osc_idx is None:
        osc_idx = list(range(m))

    slice_ = []
    for i in idx:
        slice_.extend([i * m + j for j in osc_idx])

    return slice_


def _shift_offset(
    params: np.ndarray, m: int, offset: Iterable[float], direction: str,
) -> np.ndarray:
    for i, off in enumerate(offset):
        slice_ = _get_slice(m, [2 + i])
        if direction == "center":
            params[slice_] -= off
        elif direction == "displace":
            params[slice_] += off

    return params


def _get_function_factory(dim: int, hessian: str) -> funcs.FunctionFactory:
    if dim == 1 and hessian == "exact":
        return funcs.FunctionFactory1DExact
    elif dim == 1 and hessian == "gauss-newton":
        return funcs.FunctionFactory1DGaussNewton
    elif dim == 2 and hessian == "exact":
        return funcs.FunctionFactory2DExact
    elif dim == 2 and hessian == "gauss-newton":
        return funcs.FunctionFactory2DGaussNewton
