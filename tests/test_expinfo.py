# test_expinfo.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 17:50:33 GMT

"""Test :py:mod:`nmrespy.__init__`."""

import pytest
from nmrespy import ExpInfo


def check_expinfo_correct(expinfo, dim, sw, offset, sfo, nuclei, kwargs=None):
    """Ensure expinfo attributes match the function args."""
    checks = [
        expinfo.dim == dim,
        expinfo.sw == sw,
        expinfo.offset == offset,
        expinfo.sfo == sfo,
        expinfo.nuclei == nuclei,
    ]

    if kwargs is not None:
        for key, value in kwargs.items():
            checks.append(expinfo.__dict__[key] == value)
    print(checks)

    return all(checks)


def test():
    sw = 5000.0
    offset = [2000.0]
    sfo = 500.0
    nuclei = "13C"
    expinfo = ExpInfo(1, sw, offset, sfo=sfo, nuclei=nuclei)
    assert check_expinfo_correct(
        expinfo,
        1,
        (5000.0,),
        (2000.0,),
        (500.0,),
        ("13C",),
    )

    sw = [1000.0, 5000.0]
    offset = [0.0, 2000.0]
    sfo = [None, 500.0]
    nuclei = [None, "13C"]
    expinfo = ExpInfo(2, sw, offset, sfo=sfo, nuclei=nuclei)
    assert check_expinfo_correct(
        expinfo,
        2,
        (1000.0, 5000.0),
        (0.0, 2000.0),
        (None, 500.0),
        (None, "13C"),
    )

    expinfo = ExpInfo(
        2,
        sw,
        offset,
        sfo,
        nuclei,
        array=[1, 2, 3, 4],
        dic={"a": 10, "b": 20},
    )
    assert check_expinfo_correct(
        expinfo,
        2,
        (1000.0, 5000.0),
        (0.0, 2000.0),
        (None, 500.0),
        (None, "13C"),
        {"array": [1, 2, 3, 4], "dic": {"a": 10, "b": 20}},
    )

    assert expinfo.unpack("sw") == (1000.0, 5000.0)
    assert expinfo.unpack("sw", "offset", "sfo") == (
        (1000.0, 5000.0),
        (0.0, 2000.0),
        (None, 500.0),
    )

    expinfo.sw = [8000.0, 8000.0]
    assert expinfo.sw == (8000.0, 8000.0)
