# test_filter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 28 Jan 2022 18:28:40 GMT

import numpy as np
from nmrespy import ExpInfo, mpm, sig, freqfilter as ff
import matplotlib as mpl
mpl.use("tkAgg")


def round_tuple(tup, x=3):
    return tuple([round(i, x) for i in tup])


def round_region(region, x=3):
    return tuple([(round(r[0], x), round(r[1], x)) for r in region])


class TestFilterParameters:
    def make_filter(self):
        #  |----|----|----|----|----|----|----|----|----|
        #  0    1    2    3    4    5    6    7    8    9  idx
        # 5.5  4.5  3.5  2.5  1.5  0.5 -0.5 -1.5 -2.5 -3.5 Hz
        #                 ^    ^         ^    ^
        #                 |    |  region |    |
        #                 |                   |
        #                 |     cut region    |

        params = np.array([[1, 0, 2, 0.1]])
        sw = 9.0
        offset = 1.0
        sfo = 2.0
        pts = [10]
        expinfo = ExpInfo(sw=sw, offset=offset, sfo=sfo)
        fid = sig.make_fid(params, expinfo, pts)[0]
        region = ((4, 6),)
        noise_region = ((1, 2),)  # Doesn't matter
        spectrum = sig.ft(fid)

        return ff.filter_spectrum(
            spectrum,
            expinfo,
            region,
            noise_region,
            region_unit="idx",
        )

    def test_sw(self):
        finfo = self.make_filter()
        _, expinfo = finfo.get_filtered_spectrum(cut_ratio=None)
        assert expinfo.unpack("sw") == (9.0,)
        _, expinfo = finfo.get_filtered_spectrum(cut_ratio=2.0)
        assert expinfo.unpack("sw") == (4.0,)

    def test_offset(self):
        finfo = self.make_filter()
        _, expinfo = finfo.get_filtered_spectrum(cut_ratio=None)
        assert expinfo.unpack("offset") == (1.0,)
        _, expinfo = finfo.get_filtered_spectrum(cut_ratio=2.0)
        assert expinfo.unpack("offset") == (0.5,)

    def test_region(self):
        finfo = self.make_filter()
        assert round_region(finfo.get_region(unit="hz")) == ((1.5, -0.5),)
        assert round_region(finfo.get_region(unit="ppm")) == ((0.75, -0.25),)
        assert round_region(finfo.get_region(unit="idx")) == ((4, 6),)

    def test_noise_region(self):
        finfo = self.make_filter()
        assert round_region(finfo.get_noise_region(unit="hz")) == ((4.5, 3.5),)
        assert round_region(finfo.get_noise_region(unit="ppm")) == ((2.25, 1.75),)
        assert round_region(finfo.get_noise_region(unit="idx")) == ((1, 2),)

    def test_center(self):
        finfo = self.make_filter()
        assert round_tuple(finfo.get_center(unit="hz")) == (0.5,)
        assert round_tuple(finfo.get_center(unit="ppm")) == (0.25,)
        assert round_tuple(finfo.get_center(unit="idx")) == (5,)

    def test_bw(self):
        finfo = self.make_filter()
        assert round_tuple(finfo.get_bw(unit="hz")) == (2.0,)
        assert round_tuple(finfo.get_bw(unit="ppm")) == (1.0,)
        assert round_tuple(finfo.get_bw(unit="idx")) == (2.0,)

    def test_shape(self):
        finfo = self.make_filter()
        assert finfo.shape == (10,)

    def test_sg_power(self):
        finfo = self.make_filter()
        assert finfo.sg_power == 40.0


class TestFilterPerformance:
    def make_filter(self):
        # Construct a 2-oscillator signal. Filter out a single component,
        # and estimate both the cut and uncut signals.
        params = np.array(
            [
                [10, 0, 350, 10],
                [10, 0, 100, 10],
            ]
        )
        sw = 1000.0
        offset = 0.0
        sfo = 500.0
        pts = [4096]
        expinfo = ExpInfo(sw=sw, offset=offset, sfo=sfo)
        region = ((300.0, 400.0),)
        noise_region = ((-225.0, -250.0),)

        # make spectrum from virtual echo signal
        fid = sig.make_fid(params, expinfo, pts, snr=40.0)[0]
        ve = sig.make_virtual_echo([fid])
        spectrum = sig.ft(ve)
        expinfo._pts = spectrum.shape
        return ff.filter_spectrum(spectrum, expinfo, region, noise_region)

    def test_uncut(self):
        expected = np.array([[10, 0, 350, 10]])
        finfo = self.make_filter()
        fid, expinfo = finfo.get_filtered_fid(cut_ratio=None)
        mpm_object = mpm.MatrixPencil(fid, expinfo)
        mpm_result = mpm_object.get_result()
        assert np.allclose(expected, mpm_result, rtol=0, atol=1e-2)

    def test_cut(self):
        expected = np.array([[10, 0, 350, 10]])
        finfo = self.make_filter()
        fid, expinfo = finfo.get_filtered_fid(cut_ratio=1.000001, fix_baseline=False)
        mpm_object = mpm.MatrixPencil(fid, expinfo)
        mpm_result = mpm_object.get_result()
        assert np.allclose(expected, mpm_result, rtol=0, atol=1e-1)
