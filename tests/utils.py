# utils.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 29 Mar 2022 17:44:55 BST

import numpy as np


def errstr(error: Exception) -> str:
    return str(error.value)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, atol=1e-8, rtol=0)


def similar(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    return np.allclose(a, b, atol=tolerance, rtol=0)
