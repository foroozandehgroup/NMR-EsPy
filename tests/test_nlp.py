# test_nlp.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 14 Dec 2021 19:30:31 GMT

import pytest

import numpy as np
from context import nmrespy  # noqa: F401
from nmrespy import ExpInfo, sig
from nmrespy.nlp import NonlinearProgramming


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
    fid = sig.make_fid(params, expinfo)[0]
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
        fid[20:], x0, expinfo, phase_variance=False, start_point=[20],
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
    fid = sig.make_fid(params, expinfo)[0]
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
        start_point=[10, 5],
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-4)
