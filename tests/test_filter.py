#!/usr/bin/python3

import numpy as np
from scipy.optimize import minimize
from context import nmrespy
from nmrespy import ExpInfo, _misc, mpm, sig, freqfilter as ff


def round_region(region):
    return tuple([(round(r[0], 3), round(r[1], 3)) for r in region])


def test_filter_parameters():
    # Ensure derived parameters, such as regions, sw, offset are correct
    # by considering a very simply signal with only 10 points, with a cut
    # ratio of 2:
    #  |----|----|----|----|----|----|----|----|----|
    #  0    1    2    3    4    5    6    7    8    9  idx
    # 5.5  4.5  3.5  2.5  1.5  0.5 -0.5 -1.5 -2.5 -3.5 Hz
    #                 ^    ^         ^    ^
    #                 |    |  region |    |
    #                 |     cut region    |

    p = np.array([[1, 0, 2, 0.1]])
    sw = 9.
    offset = 1.
    sfo = 9.
    pts = 10
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)
    fid = sig.make_fid(p, expinfo)[0]
    region = [[4, 6]]
    noise_region = [[1, 2]]  # Doesn't matter
    spectrum = sig.ft(fid)

    filter_ = ff.filter_spectrum(
        spectrum, region, noise_region, expinfo, region_unit='idx',
        cut_ratio=2.
    )

    assert filter_.get_sw() == filter_.get_sw(unit='hz') ==  (9.,)
    assert filter_.get_sw(unit='ppm') == (1.,)
    assert filter_.get_cut_sw() == filter_.get_cut_sw(unit='hz') == (4.,)
    assert round(filter_.get_cut_sw(unit='ppm')[0], 3) == round(4. / 9., 3)

    assert filter_.get_offset() == filter_.get_offset(unit='hz') == (1.,)
    assert round(filter_.get_offset(unit='ppm')[0], 3) == round(1. / 9., 3)
    assert round(filter_.get_cut_offset()[0], 3) == \
           round(filter_.get_cut_offset(unit='hz')[0], 3) == 0.5
    assert round(filter_.get_cut_offset(unit='ppm')[0], 3) == round(0.5 / 9., 3)

    assert round_region(filter_.get_region()) == ((1.5, -0.5),)
    assert round_region(filter_.get_region(unit='ppm')) == \
           round_region(((1.5 / 9, -0.5 / 9),))
    assert filter_.get_region(unit='idx') == ((4, 6),)

    assert round_region(filter_.get_noise_region()) == ((4.5, 3.5),)
    assert round_region(filter_.get_noise_region(unit='ppm')) == \
           round_region(((4.5 / 9, 3.5 / 9),))
    assert filter_.get_noise_region(unit='idx') == ((1, 2),)

    assert round_region(filter_.get_cut_region()) == ((2.5, -1.5),)
    assert round_region(filter_.get_cut_region(unit='ppm')) == \
           round_region(((2.5 / 9, -1.5 / 9),))
    assert filter_.get_cut_region(unit='idx') == ((3, 7),)

    assert filter_.sfo == (9.,)

    assert round(filter_.get_bw()[0], 3) == 2.
    assert round(filter_.get_bw(unit='ppm')[0], 3) == round(2. / 9., 3)
    assert filter_.get_bw(unit='idx') == (2,)

    assert round(filter_.get_center()[0], 3) == 0.5
    assert round(filter_.get_center(unit='ppm')[0], 3) == round(0.5 / 9., 3)
    assert filter_.get_center(unit='idx') == (5,)

    assert filter_.shape == (10,)
    assert filter_.cut_spectrum.shape[0] == filter_.cut_shape[0] == 5


def test_filter_performance():
    # Construct a 2-oscillator signal. Filter out a single component,
    # and estimate both the cut and uncut signals.
    p = np.array([
        [10, 0, 350, 10],
        [10, 0, 100, 10],
    ])
    sw = 1000.
    offset = 0.
    sfo = 500.
    pts = 4096
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)
    region = ((300., 400.),)
    noise_region = ((-225., -250.),)

    # make spectrum from virtual echo signal
    fid = sig.make_fid(p, expinfo, snr=40.)[0]
    ve = sig.make_virtual_echo([fid])
    spectrum = sig.ft(ve)
    filter_ = ff.filter_spectrum(
        spectrum, region, noise_region, expinfo
    )
    uncut_result = mpm.MatrixPencil(
        filter_.filtered_fid, filter_.uncut_expinfo
    ).get_result()

    cut_result = mpm.MatrixPencil(
        filter_.cut_fid, filter_.cut_expinfo
    ).get_result()

    assert np.allclose(p[0], uncut_result, rtol=0, atol=1e-2)
    # give cut signal a larger tolerance due to greater uncertainty generated
    # by cutting
    assert np.allclose(p[0], cut_result, rtol=0, atol=1e-1)

def test_fix_baseline():
    p = np.array([
        [10, 0, 350, 10],
        [20, 0, 340, 10],
        [10, 0, 330, 10],
        [10, 0, -100, 10],
        [30, 0, -90, 10],
        [30, 0, -80, 10],
        [10, 0, -70, 10],
    ])
    sw = 1000.
    offset = 0.
    sfo = 500.
    pts = 4096
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)
    region = ((300., 380.),)
    noise_region = ((-225., -250.),)
    fid = sig.make_fid(p, expinfo, snr=20.)[0]
    spectrum = sig.ft(fid)
    finfo = ff.filter_spectrum(spectrum, region, noise_region, expinfo)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(spectrum)
    ax2.plot(finfo.filtered_spectrum)
    plt.show()
