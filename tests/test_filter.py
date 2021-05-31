#!/usr/bin/python3

import numpy as np
from nmrespy.sig import make_fid
from nmrespy.freqfilter import FrequencyFilter


def test_filter():

    # Ensure correct sw and offset derived after cutting
    # are correct

    # Singal like this:
    #  |----|----|----|----|----|----|----|----|----|
    #  0    1    2    3    4    5    6    7    8    9  idx
    # 4.5  3.5  2.5  1.5  0.5 -0.5 -1.5 -2.5 -3.5 -4.5 Hz
    #                      ^         ^
    #                      |  region |
    #                      |         |
    #
    # Using a cut ratio of 2.
    # sw should be 9Hz for uncut
    # sw should be (8 / 18) x 9 = 4Hz for cut
    # offset should be -0.5 for uncut
    # offset should be 4.5 - (10 * 9 / 18) = -0.5Hz for cut

    p = np.array([[1, 0, 2, 0.1]])
    sw = [9.]
    offset = [0.]
    sfo = [100.]
    n = [10]
    fid = make_fid(p, n, sw, offset=offset)[0]
    region = [[4, 6]]
    noise_region = [[1, 2]]  # Doesn't matter

    filter = FrequencyFilter(fid, region, noise_region, sw, offset, sfo,
                             region_unit='idx', cut_ratio=2.)

    assert filter.get_sw(cut=False)[0] == 9.
    assert round(filter.get_sw(cut=True)[0], 3) == 4.
    assert filter.get_offset(cut=False)[0] == 0.
    assert round(filter.get_offset(cut=True)[0], 3) == -0.5

    region = [[7, 9]]
    filter = FrequencyFilter(fid, region, noise_region, sw, offset, sfo,
                             region_unit='idx', cut_ratio=3.)

    assert filter.get_sw(cut=False)[0] == 9.
    assert round(filter.get_sw(cut=True)[0], 3) == 4.
    assert filter.get_offset(cut=False)[0] == 0.
    assert round(filter.get_offset(cut=True)[0], 3) == -2.5
