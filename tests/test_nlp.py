# test_nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 04 Feb 2022 12:00:21 GMT

from itertools import combinations
import numpy as np
import numpy.linalg as nlinalg
from nmrespy import ExpInfo, sig
from nmrespy.nlp import NonlinearProgramming, _funcs


def test_nlp_1d():
    params = np.array(
        [
            [4, 0, -4.5, 1],
            [3, 0, -2.5, 2],
            [1, 0, 2.5, 1],
            [2, 0, 5.5, 2],
        ]
    )
    pts = [1024]
    expinfo = ExpInfo(sw=20)
    fid = sig.make_fid(params, expinfo, pts)[0]
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 1.1],
            [4.2, -0.1, -2.3, 1.8],
            [1.1, 0.05, 2.6, 0.9],
            [1.8, -0.1, 5.3, 2.2],
        ]
    )

    nlp = NonlinearProgramming(
        fid, x0, expinfo, hessian="gauss-newton", phase_variance=False
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)

    nlp = NonlinearProgramming(
        fid, x0, expinfo, hessian="exact", phase_variance=False
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)

    nlp = NonlinearProgramming(
        fid, x0, expinfo, phase_variance=False, method="lbfgs"
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-2)

    # test with FID not starting at t=0
    nlp = NonlinearProgramming(
        fid[20:], x0, expinfo, phase_variance=False, start_time=["20dt"],
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)


def test_nlp_2d():
    params = np.array(
        [
            [4, 0, -4.5, 4.5, 1, 1],
            [3, 0, -2.5, 2.5, 2, 2],
            [1, 0, 2.5, 3.5, 1, 1],
            [2, 0, 5.5, -0.5, 2, 2],
        ]
    )
    pts = [128, 128]
    expinfo = ExpInfo(sw=20, dim=2)
    fid = sig.make_fid(params, expinfo, pts)[0]
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 4.7, 0.9, 1.1],
            [4.2, -0.1, -2.3, 2.4, 2.1, 1.8],
            [1.1, 0.05, 2.6, 3.7, 1.3, 0.9],
            [1.8, -0.1, 5.3, -0.3, 1.8, 2.2],
        ]
    )
    nlp = NonlinearProgramming(
        fid, x0, expinfo, hessian="gauss-newton", phase_variance=False
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)

    nlp = NonlinearProgramming(
        fid, x0, expinfo, hessian="exact", phase_variance=False
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)

    nlp = NonlinearProgramming(
        fid, x0, expinfo, phase_variance=False, method="lbfgs"
    )

    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-2)

    nlp = NonlinearProgramming(
        fid[10:, 5:], x0, expinfo, phase_variance=False,
        start_time=["10dt", "5dt"],
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-4)


def test_analytic_grad_hess():
    # Compare analytic and finite difference grad and hessian and check they
    # all closely match.

    h = 0.000001

    # --- 1D ---
    params = np.array([[1.0, 0.0, 5.0, 1.0]])
    x0 = np.array([[0.9, 0.4, 6, 1.2]])
    pts = [4048]
    expinfo = ExpInfo(sw=100)

    fid, tp = sig.make_fid(params, expinfo, pts)
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

        assert np.allclose(grad_fd, grad_ex, rtol=0, atol=0.01)
        assert np.allclose(hess_fd, hess_ex, rtol=0, atol=0.01)

    # --- 2D ---
    params = np.array([[1.0, 0.0, 5.0, -7.0, 2.0, 1.0]])
    x0 = np.array([[0.9, 0.4, 6, -8.0, 1.8, 1.2]])
    pts = [128, 128]
    expinfo = ExpInfo(sw=100, dim=2)

    fid, tp = sig.make_fid(params, expinfo, pts)
    norm = nlinalg.norm(fid)
    fid /= norm
    params[0, 0] /= norm
    x0[0, 0] /= norm
    x0 = x0.flatten(order='F')

    idxs = []
    for i in range(1, 7):
        idxs.extend([list(c) for c in combinations(range(6), i)])

    for i, idx in enumerate(idxs):
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

            assert np.allclose(grad_fd, grad_ex, rtol=0, atol=0.02)
            assert np.allclose(hess_fd, hess_ex, rtol=0, atol=0.02)
