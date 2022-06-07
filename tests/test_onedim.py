# test_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 07 Jun 2022 10:59:34 BST

import nmrespy as ne
import numpy as np


def test_new_synthetic_from_parameters():
    estimator = ne.Estimator1D.new_synthetic_from_parameters(
        np.array(
            [
                [1, 0, 700, 2],
                [3, 0, 710, 2],
                [3, 0, 720, 2],
                [1, 0, 730, 2],
                [1, 0, 730, 2],
                [15, 0, 340, 2],
                [15, 0, 350, 2],
            ]
        ),
        pts=4096,
        sw=1000.,
        offset=500.,
        sfo=500.,
    )

    assert estimator.default_pts == (4096,)
    assert estimator.dim == 1
    assert estimator.fn_mode is None
