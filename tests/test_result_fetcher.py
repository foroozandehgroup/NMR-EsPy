# test_result_fetcher.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 29 Mar 2022 16:16:54 BST

import copy
import sys
from typing import Iterable, Optional

import numpy as np
import pytest

from nmrespy._result_fetcher import ResultFetcher
sys.path.append(".")
from utils import errstr  # noqa: E402


RESULT = np.array(
    [
        [1, 0, -10, 1990, 2, 2],
        [2, 0, 0, 2000, 2, 2],
        [1, 0, 10, 2010, 2, 2],
    ],
    dtype="float64",
)

RESULT_PPM = copy.deepcopy(RESULT)
RESULT_PPM[:, 3] = np.array([3.98, 4, 4.02])

ERRORS = np.array([
    [0.01, 1e-4, 0.1, 1, 0.2, 0.2],
    [0.02, 1e-4, 0.1, 0.5, 0.2, 0.2],
    [0.01, 1e-4, 0.1, 1, 0.2, 0.2],
])

ERRORS_PPM = copy.deepcopy(ERRORS)
ERRORS_PPM[:, 3] = np.array([0.002, 0.001, 0.002])


# Imitates classes like MatrixPencil, NonlinearProgramming, and Estimator
class Imitator(ResultFetcher):

    result = RESULT

    def __init__(
        self,
        sfo: Optional[Iterable[Optional[float]]] = None,
        errors: bool = True
    ):
        if errors:
            self.errors = ERRORS

        super().__init__(sfo)


def test_sfo_none_errors_none():
    cls = Imitator(None, errors=False)

    assert np.array_equal(cls.get_result(), RESULT)

    with pytest.raises(TypeError) as exc_info:
        cls.get_result("ppm")
    assert "Cannot generate ppm values without sfo specification." in errstr(exc_info)

    with pytest.raises(AttributeError) as exc_info:
        cls.get_errors()
    assert "This class does not possess the attribute `errors`." in errstr(exc_info)


def test_sfo_none():
    cls = Imitator(None)

    assert np.array_equal(cls.get_result(), RESULT)

    with pytest.raises(TypeError) as exc_info:
        cls.get_result("ppm")
    assert "Cannot generate ppm values without sfo specification." in errstr(exc_info)

    assert np.array_equal(cls.get_errors(), ERRORS)


def test_errors_none():
    cls = Imitator((None, 500.), errors=False)

    assert np.array_equal(cls.get_result(), RESULT)
    assert np.array_equal(cls.get_result("ppm"), RESULT_PPM)

    with pytest.raises(AttributeError) as exc_info:
        cls.get_errors()
    assert "This class does not possess the attribute `errors`." in errstr(exc_info)


def test_neither_none():
    cls = Imitator((None, 500.))

    assert np.array_equal(cls.get_result(), RESULT)
    assert np.array_equal(cls.get_result("ppm"), RESULT_PPM)
    assert np.array_equal(cls.get_errors(), ERRORS)
    assert np.array_equal(cls.get_errors("ppm"), ERRORS_PPM)
