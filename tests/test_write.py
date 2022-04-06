# test_write.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 06 Apr 2022 10:13:14 BST

import copy

import numpy as np

from nmrespy import ExpInfo, write as nwrite

USER_INPUT = True

params = np.array(
    [
        [1, 0, 20, 3020, 5, 5],
        [3, 0, 10, 3010, 5, 5],
        [6, 0, 0, 3000, 5, 5],
        [3, 0, -10, 2990, 5, 5],
        [1, 0, -20, 2980, 5, 5],
    ],
    dtype="float64",
)
params += np.random.uniform(-0.01, 0.01, size=params.shape)
errors = copy.deepcopy(params)
errors /= 100.

expinfo = ExpInfo(
    dim=2,
    sw=[50.0, 5000.0],
    offset=[0.0, 2000.0],
    sfo=[None, 400.0],
    nuclei=[None, "1H"],
    default_pts=[64, 4096],
)
WRITER = nwrite.ResultWriter(expinfo, params, errors, description="Test example")


def test_write():
    WRITER.write("hello_there", "pdf", force_overwrite=True)
    WRITER.write("hello_there", "txt", force_overwrite=True)


def test_format_string():
    sig_figs = 4
    sci_lims = (-2, 3)
    tests = {
        123456789012: "1.235e+11",
        1435678.349: "1.436e+6",
        0.0000143: "1.43e-5",
        -0.000004241: "-4.241e-6",
        1000.0: "1e+3",
        -999.9: "-999.9",
        999.99: "1e+3",
        0.0: "0",
        0.01: "1e-2",
        0.09999: "9.999e-2",
        0.1: "0.1",
    }

    for value, result in tests.items():
        assert WRITER._fmtstr(value, sig_figs, sci_lims) == result
