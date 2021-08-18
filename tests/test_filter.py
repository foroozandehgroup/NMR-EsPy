#!/usr/bin/python3

import numpy as np
from nmrespy import mpm, sig, freqfilter as ff


def round_region(region):
    return [[round(r[0], 3), round(r[1], 3)] for r in region]


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
    sw = [9.]
    offset = [1.]
    sfo = [9.]
    n = [10]
    fid = sig.make_fid(p, n, sw, offset=offset)[0]
    region = [[4, 6]]
    noise_region = [[1, 2]]  # Doesn't matter
    spectrum = sig.ft(fid)

    filter = ff.filter_spectrum(
        spectrum, region, noise_region, sw, offset, sfo=sfo, region_unit='idx',
        cut_ratio=2.
    )

    assert filter.get_sw() == filter.get_sw(unit='hz') == [9.]
    assert filter.get_sw(unit='ppm') == [1.]
    assert filter.get_cut_sw() == filter.get_cut_sw(unit='hz') == [4.]
    assert round(filter.get_cut_sw(unit='ppm')[0], 3) == round(4. / 9., 3)

    assert filter.get_offset() == filter.get_offset(unit='hz') == [1.]
    assert round(filter.get_offset(unit='ppm')[0], 3) == round(1. / 9., 3)
    assert round(filter.get_cut_offset()[0], 3) == \
           round(filter.get_cut_offset(unit='hz')[0], 3) == 0.5
    assert round(filter.get_cut_offset(unit='ppm')[0], 3) == round(0.5 / 9., 3)

    assert round_region(filter.get_region()) == [[1.5, -0.5]]
    assert round_region(filter.get_region(unit='ppm')) == \
           round_region([[1.5 / 9, -0.5 / 9]])
    assert filter.get_region(unit='idx') == [[4, 6]]

    assert round_region(filter.get_noise_region()) == [[4.5, 3.5]]
    assert round_region(filter.get_noise_region(unit='ppm')) == \
           round_region([[4.5 / 9, 3.5 / 9]])
    assert filter.get_noise_region(unit='idx') == [[1, 2]]

    assert round_region(filter.get_cut_region()) == [[2.5, -1.5]]
    assert round_region(filter.get_cut_region(unit='ppm')) == \
           round_region([[2.5 / 9, -1.5 / 9]])
    assert filter.get_cut_region(unit='idx') == [[3, 7]]

    assert filter.sfo == [9.]

    assert round(filter.get_bw()[0], 3) == 2.
    assert round(filter.get_bw(unit='ppm')[0], 3) == round(2. / 9., 3)
    assert filter.get_bw(unit='idx') == [2]

    assert round(filter.get_center()[0], 3) == 0.5
    assert round(filter.get_center(unit='ppm')[0], 3) == round(0.5 / 9., 3)
    assert filter.get_center(unit='idx') == [5]

    assert filter.shape == [10]
    assert filter.cut_spectrum.shape[0] == filter.cut_shape[0] == 5


def test_filter_performance():
    # Construct a 2-oscillator signal. Filter out a single component,
    # and estimate both the cut and uncut signals.
    p = np.array([
        [10, 0, 350, 10],
        [10, 0, 100, 10],
    ])
    sw = [1000.]
    offset = [0.]
    sfo = [500.]
    n = [4096]
    region = [[300., 400.]]
    noise_region = [[-225., -250.]]

    # make spectrum from virtual echo signal
    fid = sig.make_fid(p, n, sw, offset=offset, snr=40.)[0]
    ve = sig.make_virtual_echo([fid])
    spectrum = sig.ft(ve)
    filter = ff.filter_spectrum(
        spectrum, region, noise_region, sw, offset, sfo=sfo,
    )
    uncut_result = mpm.MatrixPencil(
        filter.filtered_fid, filter.get_sw(), offset=filter.get_offset(),
        sfo=filter.sfo
    ).get_result()

    cut_result = mpm.MatrixPencil(
        filter.cut_fid, filter.get_cut_sw(), offset=filter.get_cut_offset(),
        sfo=filter.sfo
    ).get_result()

    assert np.allclose(p[0], uncut_result, rtol=0, atol=1e-2)
    # give cut signal a larger tolerance due to greater uncertainty generated
    # by cutting
    assert np.allclose(p[0], cut_result, rtol=0, atol=1e-1)
