# test_nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 29 Mar 2022 23:28:19 BST

from itertools import combinations
import sys

import numpy as np
import numpy.linalg as nlinalg

from nmrespy import ExpInfo
from nmrespy.nlp import NonlinearProgramming, _funcs
sys.path.insert(0, ".")
from utils import similar  # noqa: E402


EXPINFO1D = ExpInfo(dim=1, sw=20., default_pts=1024)
PARAMS1D = np.array(
    [
        [4, 0, -4.5, 1],
        [3, 0, -2.5, 2],
        [1, 0, 2.5, 1],
        [2, 0, 5.5, 2],
    ]
)

EXPINFO2D = ExpInfo(dim=2, sw=(20., 20.), default_pts=[128, 128])
PARAMS2D = np.array(
    [
        [4, 0, -4.5, 4.5, 1, 1],
        [3, 0, -2.5, 2.5, 2, 2],
        [1, 0, 2.5, 3.5, 1, 1],
        [2, 0, 5.5, -0.5, 2, 2],
    ]
)


def test_1d():
    fid = EXPINFO1D.make_fid(PARAMS1D)
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 1.1],
            [4.2, -0.1, -2.3, 1.8],
            [1.1, 0.05, 2.6, 0.9],
            [1.8, -0.1, 5.3, 2.2],
        ]
    )

    for hessian in ("exact", "gauss-newton"):
        nlp = NonlinearProgramming(
            fid, x0, EXPINFO1D, hessian=hessian, phase_variance=False, fprint=False,
        )
        assert similar(nlp.get_result(), PARAMS1D, 1e-4)

    nlp = NonlinearProgramming(
        fid, x0, EXPINFO1D, phase_variance=False, method="lbfgs", fprint=False,
    )
    assert similar(nlp.get_result(), PARAMS1D, 1e-2)

    # test with FID not starting at t=0
    nlp = NonlinearProgramming(
        fid[20:], x0, EXPINFO1D, phase_variance=False, start_time=["20dt"],
    )
    assert similar(nlp.get_result(), PARAMS1D, 1e-4)


def test_nlp_2d():
    fid = EXPINFO2D.make_fid(PARAMS2D)
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 4.7, 0.9, 1.1],
            [4.2, -0.1, -2.3, 2.4, 2.1, 1.8],
            [1.1, 0.05, 2.6, 3.7, 1.3, 0.9],
            [1.8, -0.1, 5.3, -0.3, 1.8, 2.2],
        ]
    )

    for hessian in ("exact", "gauss-newton"):
        nlp = NonlinearProgramming(
            fid, x0, EXPINFO2D, hessian=hessian, phase_variance=False, fprint=False,
        )
        assert similar(nlp.get_result(), PARAMS2D, 1e-4)

    nlp = NonlinearProgramming(
        fid, x0, EXPINFO2D, phase_variance=False, method="lbfgs", fprint=False,
    )
    assert similar(nlp.get_result(), PARAMS2D, 1e-2)

    nlp = NonlinearProgramming(
        fid[10:, 5:], x0, EXPINFO2D, phase_variance=False, start_time=["10dt", "5dt"],
        negative_amps="flip_phase",
    )
    assert similar(nlp.get_result(), PARAMS2D, 1e-4)


def test_analytic_grad_hess():
    # Compare analytic and finite difference grad and hessian and check they
    # all closely match.
    h = 0.000001

    # --- 1D ---
    params = np.array([[1.0, 0.0, 5.0, 1.0]])
    x0 = np.array([[0.9, 0.4, 6, 1.2]])
    expinfo = ExpInfo(dim=1, sw=100., default_pts=4048)

    tp = expinfo.get_timepoints()
    fid = expinfo.make_fid(params)
    norm = nlinalg.norm(fid)
    fid /= norm
    params[0, 0] /= norm
    x0[0, 0] /= norm
    x0 = x0.flatten(order='F')

    idxs = []
    for i in range(1, 5):
        idxs.extend([list(c) for c in combinations(range(4), i)])

    for idx in idxs:
        active = x0[idx]
        passive = x0[[i for i in range(4) if i not in idx]]
        args = (fid, tp, 1, passive, idx, False)

        obj_fd, grad_fd, hess_fd = _funcs.obj_finite_diff_grad_hess_1d(active, h, *args)
        obj_ex, grad_ex, hess_ex = _funcs.obj_grad_true_hess_1d(active, *args)

        assert similar(grad_fd, grad_ex, 1e-2)
        assert similar(hess_fd, hess_ex, 1e-2)

    # --- 2D ---
    params = np.array([[1.0, 0.0, 5.0, -7.0, 2.0, 1.0]])
    x0 = np.array([[0.9, 0.4, 6, -8.0, 1.8, 1.2]])
    expinfo = ExpInfo(dim=2, sw=(100., 100.), default_pts=(128, 128))

    tp = expinfo.get_timepoints(meshgrid=False)
    fid = expinfo.make_fid(params)
    norm = nlinalg.norm(fid)
    fid /= norm
    params[0, 0] /= norm
    x0[0, 0] /= norm
    x0 = x0.flatten(order='F')

    idxs = []
    for i in range(1, 7):
        idxs.extend([list(c) for c in combinations(range(6), i)])

    for idx in idxs:
        # Filter away combinations that are not feasible.
        # 2 and 3 will always exist together (f1 and f2).
        # 4 and 5 will always exist together (η1 and η2).
        if not (
            (2 in idx and 3 not in idx) or
            (3 in idx and 2 not in idx) or
            (4 in idx and 5 not in idx) or
            (5 in idx and 4 not in idx)
        ):
            active = x0[idx]
            passive = x0[[i for i in range(6) if i not in idx]]
            args = (fid, tp, 1, passive, idx, False)

            obj_fd, grad_fd, hess_fd = _funcs.obj_finite_diff_grad_hess_2d(
                active, h, *args,
            )
            obj_ex, grad_ex, hess_ex = _funcs.obj_grad_true_hess_2d(active, *args)

            assert similar(grad_fd, grad_ex, 2e-2)
            assert similar(hess_fd, hess_ex, 2e-2)
