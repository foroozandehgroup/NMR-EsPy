from pathlib import Path

import numpy as np
import pytest

import nmrespy._errors as err
from nmrespy.load import bruker as bload
from nmrespy.load import *
from nmrespy.sig import make_fid


FIDPATHS = [Path(f'data/{i}').resolve() for i in range(1, 3)]
PDATAPATHS = [Path(f'data/{i}/pdata/1').resolve() for i in range(1, 3)]
ALLPATHS = FIDPATHS + PDATAPATHS


def test_parse_jcampdx():
    path = FIDPATHS[0] / 'acqus'
    params = parse_jcampdx(path)
    assert params['BYTORDA'] == '0'
    assert params['CPDPRG'] == 4 * ['<>'] + 5 * ['<mlev>']
    assert params['SFO1'] == '500.132249206'


def test_determine_data_type():
    for i, (fidp, pdatap) in enumerate(zip(FIDPATHS, PDATAPATHS), start=1):
        fid_dset = bload._determine_data_type(fidp)
        pdata_dset = bload._determine_data_type(pdatap)

        assert fid_dset.dim == pdata_dset.dim == i

        assert fid_dset.dtype == 'fid'
        assert pdata_dset.dtype == 'pdata'

        assert fid_dset.datafile == fidp / ('fid' if i == 1 else 'ser')
        assert pdata_dset.datafile == pdatap / f"{i}{i * 'r'}"

        for s in ['', '2', '3'][:i]:
            assert fid_dset.paramfiles[f'acqu{s}s'] == fidp / f'acqu{s}s'
            assert pdata_dset.paramfiles[f'acqu{s}s'] == \
                pdatap.parents[1] / f'acqu{s}s'
            assert pdata_dset.paramfiles[f'proc{s}s'] == pdatap / f'proc{s}s'


def test_fetch_parameters():
    directory = Path(FIDPATHS[0])
    info = bload._determine_bruker_data_type(directory)
    params = bload._fetch_parameters(info['param'])
    print(params)


def test_get_expinfo():
    directory = FIDPATHS[0]
    parampaths = bload._determine_bruker_data_type(directory)['param']
    params = bload._fetch_parameters(parampaths)
    expinfo = bload._get_expinfo(params)
    print(expinfo)


def _funky_mod(x, r):
    """Same as normal % operator, but if the result is 0, ``r`` is returned
    instead. i.e. 6 % 3 -> 3 rather than the usual 0."""
    m = x % r
    return m if m != 0 else r


def test_determine_bruker_data_type():
    for i, path in enumerate(ALLPATHS, start=1):
        info = bload.determine_bruker_data_type(path)
        assert info['dim'] == _funky_mod(i, 2)
        assert info['dtype'] == 'fid' if i < 3 else 'pdata'
        assert list(info['param'].keys()) == (
            [f'acqu{j}s' for j in ['', '2', '3'][:i]]
            if i < 3
            else [f'{t}{j}s' for t in ['acqu', 'proc']
                  for j in ['', '2', '3'][:i - 2]]
        )
        if i == 1:
            assert list(info['bin'].values())[0].name == 'fid'
        elif i == 2:
            assert list(info['bin'].values())[0].name == 'ser'
        else:
            assert list(info['bin'].values())[0].name == \
                f"{info['dim']}{info['dim'] * 'r'}"


def test_remove_zeros():
    params = np.array([[1, 0, 2, 3, 0.1, 0.1]])
    n = [4, 16]
    sw = [10., 10.]
    fid = make_fid(params, n, sw)[0].flatten()
    for zeropad in range(8):
        shape = [n[0], n[1] - zeropad]
        for t in range(n[0]):
            fid[t * n[1] - zeropad: t * n[1]] = 0
        assert np.any(fid == 0) if zeropad != 0 else True
        assert fid.size == n[0] * n[1]
        slyce = bload.remove_zeros([fid], shape)[0]
        assert slyce.size == n[0] * (n[1] - zeropad)
        assert np.all(slice != 0)


def test_load_bruker():
    for i, path in enumerate(ALLPATHS):
        load_bruker(path, ask_convdta=False)
