import pytest
from pathlib import Path
import pickle

import numpy as np

from nmrespy._misc import PathManager, FrequencyConverter, ArgumentChecker

def test_argchecker(self):
    with pytest.raises(TypeError):
        ArgumentChecker(
            [
                # should be a valid parameter array
                (np.array([[1, 2, 3, 4]]), 'theta0', 'parameter'),
                # should be an invalid sw
                ([10], 'sw', 'float_list'),
                # should be an invalid boolean
                ('thisisastring', 'fprint', 'bool'),
            ],
            # dimension
            1
        )

    ArgumentChecker(
        [
            (np.arange(12).reshape(2,6), 'parameter', 'parameter'),
            ([10, 43], 'int_list', 'int_list'),
            ([10.21, 43.74], 'float_list', 'float_list'),
            (True, 'bool', 'bool'),
            (-10, 'int', 'int'),
            (-10.563, 'float', 'float'),
            (10, 'positive_int', 'positive_int'),
            (10.563, 'positive_float', 'positive_float'),
            ('afd', 'optimiser_mode', 'optimiser_mode'),
            (0.53425, 'zero_to_one', 'zero_to_one'),
            ('remove', 'negative_amplitude', 'negative_amplitude')
        ],
        # dimension
        2
    )


def test_converter():
    n = [101, 101]
    sw = [10.,100.]
    offset = [0., 50.]
    sfo = [500., 500.]

    # test failures
    # non-list arguments
    with pytest.raises(TypeError):
        FrequencyConverter(n, 'not_a_list', offset, sfo)
    # lists not same length
    with pytest.raises(ValueError):
        FrequencyConverter(n, [sw[0]], offset, sfo)
    # some elements are not numerical values
    with pytest.raises(TypeError):
        FrequencyConverter(n, [sw[0], 'not_a_number'], offset, sfo)

    converter = FrequencyConverter(n, sw, offset, sfo)

    # index -> Hz
    assert converter._convert_value(0, 0, 'idx->hz') == 5.
    assert converter._convert_value(25, 0, 'idx->hz') == 2.5
    assert converter._convert_value(50, 0, 'idx->hz') == 0.
    assert converter._convert_value(75, 0, 'idx->hz') == -2.5
    assert converter._convert_value(100, 0, 'idx->hz') == -5
    assert converter._convert_value(0, 1, 'idx->hz') == 100.
    assert converter._convert_value(50, 1, 'idx->hz') == 50.
    assert converter._convert_value(100, 1, 'idx->hz') == 0.

    # Hz -> index
    assert converter._convert_value(-5, 0, 'hz->idx') == 100
    assert converter._convert_value(-2.5, 0, 'hz->idx') == 75
    assert converter._convert_value(0, 0, 'hz->idx') == 50
    assert converter._convert_value(2.5, 0, 'hz->idx') == 25
    assert converter._convert_value(5, 0, 'hz->idx') == 0

    # index -> ppm
    assert converter._convert_value(0, 0, 'idx->ppm') == 0.01
    assert converter._convert_value(25, 0, 'idx->ppm') == 0.005
    assert converter._convert_value(50, 0, 'idx->ppm') == 0.
    assert converter._convert_value(75, 0, 'idx->ppm') == -0.005
    assert converter._convert_value(100, 0, 'idx->ppm') == -0.01

    # ppm -> index
    assert converter._convert_value(0.01, 0, 'ppm->idx') == 0
    assert converter._convert_value(0.005, 0, 'ppm->idx') == 25
    assert converter._convert_value(0., 0, 'ppm->idx') == 50
    assert converter._convert_value(-0.005, 0, 'ppm->idx') == 75
    assert converter._convert_value(-0.01, 0, 'ppm->idx') == 100

    # ppm -> Hz
    assert converter._convert_value(0.01, 0, 'ppm->hz') == 5.
    assert converter._convert_value(0.005, 0, 'ppm->hz') == 2.5
    assert converter._convert_value(0., 0, 'ppm->hz') == 0.
    assert converter._convert_value(-0.005, 0, 'ppm->hz') == -2.5
    assert converter._convert_value(-0.01, 0, 'ppm->hz') == -5.

    # Hz -> ppm
    assert converter._convert_value(5., 0, 'hz->ppm') == 0.01
    assert converter._convert_value(2.5, 0, 'hz->ppm') == 0.005
    assert converter._convert_value(0., 0, 'hz->ppm') == 0.
    assert converter._convert_value(-2.5, 0, 'hz->ppm') == -0.005
    assert converter._convert_value(-5., 0, 'hz->ppm') == -0.01

    assert converter.convert([50, 50], 'idx->hz') == [0., 50.]
    assert converter.convert([[25, 50], [75, 100]], 'idx->hz') == [[2.5, 0.], [25., 0.]]
