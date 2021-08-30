import pytest

import numpy as np

from nmrespy import *
from nmrespy._misc import FrequencyConverter, ArgumentChecker


def test_argchecker():
    with pytest.raises(TypeError) as exc_info:
        checker = ArgumentChecker(dim=1)
        checker.stage(
            # should be a valid parameter array
            (np.array([[1, 2, 3, 4]]), 'theta0', 'parameter'),
            # should be an invalid sw
            ([10], 'sw', 'float_list'),
            # should be an invalid boolean
            ('thisisastring', 'fprint', 'bool'),
        )
        checker.check()

    assert str(exc_info.value) == \
        (f'{RED}The following arguments are invalid:\n'
         '--> sw\n--> fprint\n'
         f'Have a look at the documentation for more info.{END}')

    checker = ArgumentChecker(dim=2)
    checker.stage(
        (np.arange(12).reshape(2, 6), 'a', 'parameter'),
        (None, 'b', 'int_list', True),
        ([10.21, 43.74], 'c', 'float_list'),
        (True, 'd', 'bool'),
        (-10, 'e', 'int'),
        (-10.563, 'f', 'float'),
        (None, 'g', 'positive_int', True),
        (10.563, 'h', 'positive_float'),
        ('afd', 'i', 'optimiser_mode'),
        (0.53425, 'j', 'zero_to_one'),
        ('remove', 'k', 'negative_amplitude')
    )
    checker.check()


def test_converter():
    n = [101, 101]
    sw = [10., 100.]
    offset = [0., 50.]
    sfo = [500., 500.]

    # test failures
    # non-list arguments
    with pytest.raises(TypeError):
        FrequencyConverter(n, 'not_a_list', offset, sfo)
    # lists not same length
    with pytest.raises(TypeError):
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
    assert (converter.convert([[25, 50], [75, 100]], 'idx->hz') ==
            [[2.5, 0.], [25., 0.]])
