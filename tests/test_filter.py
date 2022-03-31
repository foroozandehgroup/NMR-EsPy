# test_filter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 31 Mar 2022 13:51:28 BST

import matplotlib as mpl
import numpy as np

from nmrespy import ExpInfo, sig
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming

mpl.use("tkAgg")


def round_tuple(tup, x=3):
    return tuple([round(i, x) for i in tup])


def round_region(region, x=3):
    return tuple([(round(r[0], x), round(r[1], x)) for r in region])


class TestFilterParameters1D:
    #  |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
    #  0     2     4     6     8     10    12    14    16    18   idx
    #  5.5   4.5   3.5   2.5   1.5   0.5   -0.5  -1.5  -2.5  -3.5 Hz
    #                    ^     ^           ^     ^
    #                    |     |   region  |     |
    #                    |                       |
    #                    |       cut region      |

    expinfo = ExpInfo(1, sw=9.0, offset=1.0, sfo=2.0, default_pts=10)
    filt = Filter(
        expinfo.make_fid(np.array([[1, 0, 2, 0.1]])),
        expinfo,
        [1.5, -0.5],
        [4.5, 3.5],
    )

    def test_sw(self):
        _, expinfo = self.filt.get_filtered_spectrum(cut_ratio=None)
        assert expinfo.sw("hz") == (9.0,)
        _, expinfo = self.filt.get_filtered_spectrum(cut_ratio=2.0)
        assert expinfo.sw("hz") == (4.0,)

    def test_offset(self):
        _, expinfo = self.filt.get_filtered_spectrum(cut_ratio=None)
        assert expinfo.offset("hz") == (1.0,)
        _, expinfo = self.filt.get_filtered_spectrum(cut_ratio=2.0)
        assert expinfo.offset("hz") == (0.5,)

    def test_region(self):
        assert round_region(self.filt.get_region(unit="hz")) == ((1.5, -0.5),)
        assert round_region(self.filt.get_region(unit="ppm")) == ((0.75, -0.25),)
        assert round_region(self.filt.get_region(unit="idx")) == ((8, 12),)

    def test_noise_region(self):
        assert round_region(self.filt.get_noise_region(unit="hz")) == ((4.5, 3.5),)
        assert round_region(self.filt.get_noise_region(unit="ppm")) == ((2.25, 1.75),)
        assert round_region(self.filt.get_noise_region(unit="idx")) == ((2, 4),)

    def test_center(self):
        assert round_tuple(self.filt.get_center(unit="hz")) == (0.5,)
        assert round_tuple(self.filt.get_center(unit="ppm")) == (0.25,)
        assert round_tuple(self.filt.get_center(unit="idx")) == (10,)

    def test_bw(self):
        assert round_tuple(self.filt.get_bw(unit="hz")) == (2.0,)
        assert round_tuple(self.filt.get_bw(unit="ppm")) == (1.0,)
        assert round_tuple(self.filt.get_bw(unit="idx")) == (4,)

    def test_shape(self):
        # 10 * 2 - 1
        assert self.filt.shape == (19,)

    def test_sg_power(self):
        assert self.filt.sg_power == 40.0


class TestFilterPerformance:
    expinfo = ExpInfo(dim=1, sw=1000., offset=0., sfo=500., default_pts=4096)
    params = np.array(
        [
            [10, 0, 350, 10],
            [10, 0, 100, 10],
        ]
    )
    filt = Filter(
        expinfo.make_fid(params, snr=30.0),
        expinfo,
        [400., 300.],
        [-250., -225.],
    )

    def test_uncut(self):
        expected = np.array([[10, 0, 350, 10]], dtype="float64")
        fid, expinfo = self.filt.get_filtered_fid(cut_ratio=None)
        mpm = MatrixPencil(fid, expinfo)
        print(mpm.get_result() - expected)
        assert np.allclose(expected, mpm.get_result(), rtol=0, atol=1e-2)

    def test_cut(self):
        expected = np.array([[10, 0, 350, 10]], dtype="float64")
        fid_cut, expinfo_cut = self.filt.get_filtered_fid(cut_ratio=1.0)
        fid_uncut, expinfo_uncut = self.filt.get_filtered_fid(cut_ratio=None)
        mpm = MatrixPencil(fid_cut, expinfo_cut)
        nlp = NonlinearProgramming(fid_uncut, mpm.get_result(), expinfo_uncut)
        print(nlp.get_result())
        assert np.allclose(expected, mpm.get_result(), rtol=0, atol=1e-1)


# def test_linear_fit():
#     from functools import reduce
#     from numpy.random import normal

#     model = np.array([2, 3, 4])
#     shape = (16, 32)
#     dim = len(shape)
#     prod = reduce(lambda x, y: x * y, shape)

#     xs = (np.indices(shape).reshape(dim, prod)).T
#     X = np.ones((prod, dim + 1))
#     X[:, :-1] = xs

#     noise = normal(scale=2, size=shape)
#     y = (X @ model).reshape(shape) + noise
#     n = ff.superg_noise(y, [None, None], np.zeros(shape))
#     x1, x2 = np.meshgrid(*[np.arange(x) for x in shape], indexing="ij")
#     import matplotlib as mpl
#     mpl.use("tkAgg")
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     ax.plot_wireframe(x1, x2, y, lw=0.5)
#     ax.plot_wireframe(x1, x2, n, lw=0.5)

#     plt.show()

#     model = np.array([2, 4])
#     shape = (128,)
#     dim = 1
#     prod = 128

#     xs = (np.indices(shape).reshape(dim, prod)).T
#     X = np.ones((prod, dim + 1))
#     X[:, :-1] = xs

#     noise = normal(scale=2, size=shape)
#     y = (X @ model).reshape(shape) + noise
#     n = ff.superg_noise(y, [None], np.zeros(shape))

#     plt.plot(y)
#     plt.plot(n)
#     plt.show()
