# test_expinfo.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 30 Mar 2022 16:57:21 BST

"""Test :py:mod:`nmrespy.__init__`."""

import sys

import numpy as np

from nmrespy import ExpInfo, sig
sys.path.insert(0, ".")
from utils import same  # noqa: E402


EXPINFO2D = ExpInfo(
    2,
    [50.0, 2000.0],
    [0.0, 2000.0],
    [None, 500.0],
    [None, "1H"],
    [64, 1024],
    "QF",
)


def check_expinfo_correct(
    expinfo, dim, sw, offset, sfo, nuclei, default_pts, fn_mode, kwargs=None,
):
    checks = [
        expinfo.dim == dim,
        expinfo._sw == sw,
        expinfo._offset == offset,
        expinfo._sfo == sfo,
        expinfo._nuclei == nuclei,
        expinfo._default_pts == default_pts,
        expinfo._fn_mode == fn_mode,
    ]

    if kwargs is not None:
        for key, value in kwargs.items():
            checks.append(expinfo.__dict__[key] == value)

    return all(checks)


def matching_array_iter(a, b):
    return all([same(a_, b_) for a_, b_ in zip(a, b)])


def test_1d():
    expinfo = ExpInfo(
        dim=1,
        sw=5000.0,
        offset=[2000.0],
        sfo=500.0,
        nuclei="13C",
    )
    assert check_expinfo_correct(
        expinfo,
        1,
        (5000.0,),
        (2000.0,),
        (500.0,),
        ("13C",),
        None,
        None,
    )

    assert expinfo.sw() == (5000.0,)
    assert expinfo.sw("ppm") == (10.0,)

    assert expinfo.offset() == (2000.0,)
    assert expinfo.offset("ppm") == (4.0,)


def test_2d():

    expinfo = ExpInfo(
        2,
        sw=[50.0, 5000.0],
        offset=[0.0, 2000.0],
        sfo=[None, 500.0],
        nuclei=[None, "1H"],
        default_pts=[32, 8192],
        array=[1, 2, 3, 4],
        dic={"a": 10, "b": 20},
    )

    assert check_expinfo_correct(
        expinfo,
        2,
        (50.0, 5000.0),
        (0.0, 2000.0),
        (None, 500.0),
        (None, "1H"),
        (32, 8192),
        "QF",
        {"array": [1, 2, 3, 4], "dic": {"a": 10, "b": 20}},
    )

    assert expinfo.sw() == (50.0, 5000.0)
    assert expinfo.sw("ppm") == (50.0, 10.0)
    assert expinfo.offset() == (0.0, 2000.0)
    assert expinfo.offset("ppm") == (0.0, 4.0)


def test_timepoints():
    assert matching_array_iter(
        EXPINFO2D.get_timepoints(meshgrid=False),
        (
            np.linspace(0, 63 / 50, 64),
            np.linspace(0, 1023 / 2000, 1024),
        ),
    )


def test_shifts():
    assert matching_array_iter(
        EXPINFO2D.get_shifts(meshgrid=False),
        (
            np.linspace(25, -25, 64),
            np.linspace(3000, 1000, 1024),
        ),
    )

    assert matching_array_iter(
        EXPINFO2D.get_shifts(meshgrid=False, unit="ppm"),
        (
            np.linspace(25, -25, 64),
            np.linspace(6, 2, 1024),
        ),
    )


def test_make_fid():
    def expected_idx(pts, fractions):
        return tuple(
            [
                int(np.round(frac * p, 0)) - 1
                for p, frac in zip(pts, fractions)
            ]
        )

    def check_peak(fid, expected_fractions):
        return (
            np.unravel_index(np.argmax(np.abs(sig.ft(fid))), shape=fid.shape) ==
            expected_idx(fid.shape, expected_fractions)
        )

    # Single oscillator
    params = np.array([[1.0, 0.0, 15.0, 1500.0, 10.0, 10.0]])
    fid = EXPINFO2D.make_fid(params)
    assert fid[0, 0] == 1.0 + 0.0 * 1j
    assert check_peak(fid, (0.2, 0.75))

    # Three oscillators
    params = np.array(
        [
            [1.0, 0.0, 15.0, 2515.0, 10.0, 10.0],
            [2.0, 0.0, 0.0, 2500.0, 10.0, 10.0],
            [1.0, 0.0, -15.0, 2485.0, 10.0, 10.0],
        ]
    )
    fid = EXPINFO2D.make_fid(params)
    assert fid[0, 0] == 4.0 + 0.0 * 1j
    assert check_peak(fid, (0.5, 0.25))
