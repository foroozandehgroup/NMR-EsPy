# test_frequency_converter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 30 Mar 2022 11:16:45 BST

import sys

import pytest

from nmrespy._freqconverter import FrequencyConverter
sys.path.insert(0, ".")
from utils import errstr  # noqa: E402


def test():
    sfo = (500.0, 250.0)
    sw = (10.0, 100.0)
    offset = (0.0, 50.0)
    pts = (101, 101)

    converter = FrequencyConverter(sfo, sw, offset, pts)

    dims = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
    idxs = (0, 25, 50, 75, 100, 0, 25, 50, 75, 100)
    hzs = (5.0, 2.5, 0.0, -2.5, -5.0, 100.0, 75.0, 50.0, 25.0, 0.0)
    ppms = (0.01, 0.005, 0.0, -0.005, -0.01, 0.4, 0.3, 0.2, 0.1, 0.0)
    for dim, idx, hz, ppm in zip(dims, idxs, hzs, ppms):
        assert converter._convert_value(idx, dim, "idx->hz") == hz
        assert converter._convert_value(hz, dim, "hz->idx") == idx
        assert converter._convert_value(idx, dim, "idx->ppm") == ppm
        assert converter._convert_value(ppm, dim, "ppm->idx") == idx
        assert converter._convert_value(hz, dim, "hz->ppm") == ppm
        assert converter._convert_value(ppm, dim, "ppm->hz") == hz

    sfo = (None, 250.0)
    converter = FrequencyConverter(sfo, sw, offset, pts)

    idxs_list = [idxs[:5], idxs[5:]]
    hzs_list = [hzs[:5], hzs[5:]]
    ppms_list = [hzs[:5], ppms[5:]]
    assert converter.convert(hzs_list, "hz->ppm") == ppms_list
    assert converter.convert(hzs_list, "hz->idx") == idxs_list
    with pytest.raises(TypeError) as exc_info:
        converter.convert(ppms_list, "ppm->idx")
    assert (
        "Conversion between ppm and array indices is not possible." in
        errstr(exc_info)
    )
