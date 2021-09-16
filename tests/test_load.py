from pathlib import Path

import numpy as np
import pytest
from bruker_utils import parse_jcampdx

from context import nmrespy
import nmrespy._errors as err
from nmrespy.load import bruker
from nmrespy.sig import make_fid


FIDPATHS = [Path(f'data/{i}').resolve() for i in range(1, 3)]
PDATAPATHS = [Path(f'data/{i}/pdata/1').resolve() for i in range(1, 3)]
ALLPATHS = FIDPATHS + PDATAPATHS


def test_bruker_dataset_for_nmrespy():
    dataset = bruker._BrukerDatasetForNmrespy(FIDPATHS[0])
    expinfo = dataset.expinfo
    params = parse_jcampdx(FIDPATHS[0] / 'acqus')
    assert expinfo.sw == (float(params['SW_h']),)
    assert expinfo.offset == (float(params['O1']),)
    assert expinfo.sfo == (float(params['SFO1']),)
    # Strip '<' and '>' characters from ends of string.
    assert expinfo.nuclei == (params['NUC1'][1:-1],)
