import numpy as np
from nmrespy.mpm import MatrixPencil
from nmrespy.sig import make_fid


def test_mpm_1d():
    params = np.array([
        [4, 0, -4.5, 1],
        [3, 0, -2.5, 2],
        [1, 0, 2.5, 1],
        [2, 0, 5.5, 2],
    ])
    sw = [20.]
    offset = [0.]
    n = [1024]
    sfo = [10.]

    fid = make_fid(params, n, sw, offset=offset)[0]
    mpm = MatrixPencil(fid, sw, offset=offset, sfo=sfo, M=4)
    result = mpm.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-8)

    # test with FID not starting at t=0
    mpm = MatrixPencil(
        fid[20:], sw, offset=offset, sfo=sfo, M=4, start_point=[20],
    )
    result = mpm.get_result()
    assert np.allclose(result, params, rtol=0, atol=1E-8)


def test_mpm_2d():
    params = np.array([
        [3, 0, -2.5, 4.5, 1, 1],
        [1, 0, 2.5, 2.5, 1, 1],
        [2, 0, 5.5, 3.5, 1, 1],
        [0.5, 0, 8, 1, 1, 1],
    ])
    sw = [20., 20.]
    offset = [0., 0.]
    n = [128, 128]
    sfo = [10., 10.]

    fid = make_fid(params, n, sw, offset=offset)[0]
    mpm = MatrixPencil(fid, sw, offset=offset, sfo=sfo, M=4)
    assert np.allclose(mpm.get_result(), params, rtol=0, atol=1E-8)

    # test _remove_negative_damping
    neg_damping = np.vstack((
        mpm.result,
        np.array([
            [1, 0, 4, 3, -1, 1],
            [1, 0, 4, 3, 1, -1],
        ])
    ))
    mpm.result, mpm.M = mpm._remove_negative_damping(neg_damping)
    assert np.allclose(mpm.get_result(), params, rtol=0, atol=1E-8)


if __name__ == '__main__':
    test_mpm_1d()
    test_mpm_2d()
