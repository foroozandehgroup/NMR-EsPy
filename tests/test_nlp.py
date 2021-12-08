# test_nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 15 Oct 2021 10:10:03 BST

import numpy as np
from context import nmrespy  # noqa: F401
from nmrespy import ExpInfo
from nmrespy.nlp import NonlinearProgramming, _funcs
from nmrespy.sig import make_fid, get_timepoints


def test_funcs():
    params = np.array(
        [
            [4, 0, -4.5, 1],
            [3, 0, -2.5, 2],
            [1, 0, 2.5, 1],
            [2, 0, 5.5, 2],
        ]
    )
    expinfo = ExpInfo(pts=1024, sw=20)
    fid = make_fid(params, expinfo)[0]
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 1.1],
            [4.2, -0.1, -2.3, 1.8],
            [1.1, 0.05, 2.6, 0.9],
            [1.8, -0.1, 5.3, 2.2],
        ]
    ).flatten(order="F")
    args = (
        fid,
        get_timepoints(expinfo),
        params.shape[0],
        np.array([]),
        list(range(4)),
        True,
    )
    old = (
        _funcs.f_1d(x0, *args),
        _funcs.g_1d(x0, *args),
        _funcs.h_1d(x0, *args),
    )
    new = _funcs.obj_grad_hess_1d(x0, *args)
    assert all(
        [
            np.array_equal(a, b)
            for a, b in zip(old, new)
        ]
    )


def test_nlp_1d():
    params = np.array(
        [
            [4, 0, -4.5, 1],
            [3, 0, -2.5, 2],
            [1, 0, 2.5, 1],
            [2, 0, 5.5, 2],
        ]
    )
    expinfo = ExpInfo(pts=1024, sw=20)
    fid = make_fid(params, expinfo)[0]
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 1.1],
            [4.2, -0.1, -2.3, 1.8],
            [1.1, 0.05, 2.6, 0.9],
            [1.8, -0.1, 5.3, 2.2],
        ]
    )
    nlp = NonlinearProgramming(fid, x0, expinfo, phase_variance=False)
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)

    # test with FID not starting at t=0
    nlp = NonlinearProgramming(
        fid[20:],
        x0,
        expinfo,
        phase_variance=False,
        start_point=[20],
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
    expinfo = ExpInfo(pts=128, sw=20, dim=2)
    fid = make_fid(params, expinfo)[0]
    x0 = np.array(
        [
            [3.9, 0.1, -4.3, 4.7, 0.9, 1.1],
            [4.2, -0.1, -2.3, 2.4, 2.1, 1.8],
            [1.1, 0.05, 2.6, 3.7, 1.3, 0.9],
            [1.8, -0.1, 5.3, -0.3, 1.8, 2.2],
        ]
    )
    nlp = NonlinearProgramming(fid, x0, expinfo, phase_variance=False)
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1e-4)

    # nlp = NonlinearProgramming(
    #     fid[10:, 5:], x0, sw, offset=offset, phase_variance=False,
    #     start_point=[10, 5],
    # )
    # result = nlp.get_result()
    # assert np.allclose(result, params, rtol=0, atol=1E-4)
