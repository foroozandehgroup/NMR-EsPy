import numpy as np
from nmrespy.mpm import MatrixPencil
from nmrespy.sig import make_fid


def test_mpm_2d():
    params = np.array([
        [1, 0, 2.5, 2.5, 1, 1],
        [2, 0, 5.5, 3.5, 1, 1],
    ])
    sw = [20., 20.]
    offset = [0., 0.]
    n = [128, 128]
    sfo = [10., 10.]

    fid = make_fid(params, n, sw, offset=offset)[0]
    mpm = MatrixPencil(fid, sw, offset=offset, sfo=sfo, M=2)
    result = mpm.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-8)
