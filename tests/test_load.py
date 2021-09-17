from pathlib import Path
import pytest
from bruker_utils import parse_jcampdx
from nmrespy import load


FIDPATHS = [Path(f'data/{i}').resolve() for i in range(1, 3)]
PDATAPATHS = [Path(f'data/{i}/pdata/1').resolve() for i in range(1, 3)]
ALLPATHS = FIDPATHS + PDATAPATHS


def test_bruker_dataset_for_nmrespy():
    dataset = load._BrukerDatasetForNmrespy(FIDPATHS[0])
    expinfo = dataset.expinfo
    params = parse_jcampdx(FIDPATHS[0] / 'acqus')
    assert expinfo.sw == (float(params['SW_h']),)
    assert expinfo.offset == (float(params['O1']),)
    assert expinfo.sfo == (float(params['SFO1']),)
    # Strip '<' and '>' characters from ends of string.
    assert expinfo.nuclei == (params['NUC1'][1:-1],)


def test_load_bruker():
    with pytest.raises(OSError):
        data, expinfo = load.load_bruker('blah')

    with pytest.raises(ValueError):
        data, expinfo = load.load_bruker('.')

    data, expinfo = load.load_bruker(FIDPATHS[0], ask_convdta=False)
