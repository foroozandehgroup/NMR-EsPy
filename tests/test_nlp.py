import numpy as np
from nmrespy.nlp.nlp import NonlinearProgramming
from nmrespy.sig import make_fid


def test_nlp_1d():
    params = np.array([
        [4, 0, -4.5, 1],
        [3, 0, -2.5, 2],
        [1, 0, 2.5, 1],
        [2, 0, 5.5, 2],
    ])
    sw = [20.]
    offset = [0.]
    n = [1024]

    fid = make_fid(params, n, sw, offset=offset)[0]
    x0 = np.array([
        [3.9, 0.1, -4.3, 1.1],
        [4.2, -0.1, -2.3, 1.8],
        [1.1, 0.05, 2.6, 0.9],
        [1.8, -0.1, 5.3, 2.2],
    ])
    nlp = NonlinearProgramming(
        fid, x0, sw, offset=offset, phase_variance=False,
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-4)


def test_nlp_2d():
    params = np.array([
        [4, 0, -4.5, 4.5, 1, 1],
        [3, 0, -2.5, 2.5, 2, 2],
        [1, 0, 2.5, 3.5, 1, 1],
        [2, 0, 5.5, -0.5, 2, 2],
    ])
    sw = [20., 20.]
    offset = [0., 0.]
    n = [128, 128]

    fid = make_fid(params, n, sw, offset=offset)[0]

    x0 = np.array([
        [3.9, 0.1, -4.3, 4.7, 0.9, 1.1],
        [4.2, -0.1, -2.3, 2.4, 2.1, 1.8],
        [1.1, 0.05, 2.6, 3.7, 1.3, 0.9],
        [1.8, -0.1, 5.3, -0.3, 1.8, 2.2],
    ])
    nlp = NonlinearProgramming(
        fid, x0, sw, offset=offset, phase_variance=False,
    )
    result = nlp.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-4)
