from pathlib import Path

import numpy as np
import pytest

import nmrespy._errors as err
from nmrespy.load import bruker as bload
from nmrespy.load import load_bruker
from nmrespy.sig import make_fid


FIDPATHS = [Path(f'data/{i}').resolve() for i in range(1, 3)]
PDATAPATHS = [Path(f'data/{i}/pdata/1').resolve() for i in range(1, 3)]
ALLPATHS = FIDPATHS + PDATAPATHS


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


def test_get_params_from_jcampdx():
    path = FIDPATHS[0] / 'acqus'
    params = {
        'BF1': '500.13',
        'FnMODE': '0',
        'NUC1': '<1H>',
        'O1': '2249.20599998768',
        'SFO1': '500.132249206',
        'SW_h': '5494.50549450549',
        # Example of a multiline arrayed parameter
        'GPNAM': ' '.join(32 * ['<sine.100>'])
    }

    assert bload.get_params_from_jcampdx(list(params.keys()), path) == \
        list(params.values())

    with pytest.raises(err.ParameterNotFoundError):
        bload.get_params_from_jcampdx(['IDONTEXIST'], path)
    with pytest.raises(FileNotFoundError):
        bload.get_params_from_jcampdx(['BF1'], FIDPATHS[0] / 'idontexist')


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
