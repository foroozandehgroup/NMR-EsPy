"""Test :py:mod:`nmrespy.__init__`."""

import pytest
from nmrespy import ExpInfo, RED, END


def check_expinfo_correct(
    expinfo, pts, sw, offset, sfo, nuclei, dim, kwargs=None
):
    """Ensure expinfo attributes match the function args."""
    checks = [
        expinfo.pts == pts,
        expinfo.sw == sw,
        expinfo.offset == offset,
        expinfo.sfo == sfo,
        expinfo.nuclei == nuclei,
        expinfo.dim == dim,
    ]

    if kwargs is not None:
        for key, value in kwargs.items():
            checks.append(expinfo.__dict__[key] == value)

    return all(checks)


def test_expinfo():
    """Test :py:class:`nmrespy.ExpInfo`."""
    pts = 2048
    sw = 5000
    offset = [2000.]
    sfo = 500
    nuclei = '13C'
    expinfo = ExpInfo(pts, sw, offset, sfo=sfo, nuclei=nuclei)
    assert check_expinfo_correct(
        expinfo, (2048.,), (5000.,), (2000.,), (500.,), ('13C',), 1,
    )

    expinfo = ExpInfo(pts, sw, offset, sfo=sfo, nuclei=nuclei, dim=2)
    assert check_expinfo_correct(
        expinfo, (2048, 2048), (5000., 5000.), (2000., 2000.),
        (500., 500.), ('13C', '13C'), 2,
    )

    expinfo = ExpInfo(
        pts, sw, offset, sfo=sfo, nuclei=nuclei, dim=2, array=[1, 2, 3, 4],
        dic={'a': 10, 'b': 20}
    )
    assert check_expinfo_correct(
        expinfo, (2048, 2048), (5000., 5000.), (2000., 2000.),
        (500., 500.), ('13C', '13C'), 2,
        {'array': [1, 2, 3, 4], 'dic': {'a': 10, 'b': 20}},
    )

    assert expinfo.unpack('sw') == ((5000., 5000.),)
    assert expinfo.unpack('sw', 'offset', 'sfo') == \
        ((5000., 5000.), (2000., 2000.), (500., 500.))

    for input_ in ['fail', 1024, [1024., 1024]]:
        with pytest.raises(ValueError) as exc_info:
            expinfo.pts = input_
        assert str(exc_info.value) == \
            f'{RED}Invalid value supplied to `pts`: {repr(input_)}{END}'
    expinfo.pts = [1024, 1024]
    assert expinfo.pts == (1024, 1024)

    for input_ in ['fail', 1024, ['fail', 1024]]:
        with pytest.raises(ValueError) as exc_info:
            expinfo.sw = input_
        assert str(exc_info.value) == \
            f'{RED}Invalid value supplied to `sw`: {repr(input_)}{END}'
    expinfo.sw = [8000, 8000.]
    assert expinfo.sw == (8000., 8000.)

    for input_ in ['13C', 1024, ['13C', 1024]]:
        with pytest.raises(ValueError) as exc_info:
            expinfo.nuclei = input_
        assert str(exc_info.value) == \
            f'{RED}Invalid value supplied to `nuclei`: {repr(input_)}{END}'
    expinfo.nuclei = ['205Pb', '19F']
    assert expinfo.nuclei == ('205Pb', '19F')
