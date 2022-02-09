# test_estimator.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 07 Feb 2022 13:39:49 GMT

import numpy as np
import pytest

from nmrespy import ExpInfo
from nmrespy.core import Estimator


class TestSyntheticEstimator:
    pts_1d = [4048]
    params_1d = np.array(
        [
            [1, 0, 3000, 10],
            [3, 0, 3050, 10],
            [3, 0, 3100, 10],
            [1, 0, 3150, 10],
            [2, 0, 150, 10],
            [4, 0, 100, 10],
            [2, 0, 50, 10],
        ]
    )
    expinfo_1d = ExpInfo(sw=5000, offset=2000, sfo=500, nuclei="1H")
    expinfo_1d_no_nuc = ExpInfo(sw=5000, offset=2000, sfo=500)
    expinfo_1d_no_sfo = ExpInfo(sw=5000, offset=2000, nuclei="1H")
    expinfo_1d_no_sfo_no_nuc = ExpInfo(sw=5000, offset=2000)

    pts_2d = [512, 512]
    params_2d = np.array(
        [
            [1, 0, 3000, 3000, 10, 10],
            [3, 0, 3050, 3050, 10, 10],
            [3, 0, 3100, 3100, 10, 10],
            [1, 0, 3150, 3150, 10, 10],
            [2, 0, 150, 150, 10, 10],
            [4, 0, 100, 100, 10, 10],
            [2, 0, 50, 50, 10, 10],
        ]
    )
    expinfo_2d = ExpInfo(sw=5000, offset=2000, sfo=500, nuclei="1H", dim=2)
    expinfo_2d_no_nuc = ExpInfo(sw=5000, offset=2000, sfo=500, dim=2)
    expinfo_2d_no_sfo = ExpInfo(sw=5000, offset=2000, nuclei="1H", dim=2)
    expinfo_2d_no_sfo_no_nuc = ExpInfo(sw=5000, offset=2000, dim=2)

    def test_init(self):
        # Create with valid parameters
        Estimator.new_synthetic_from_parameters(
            self.params_1d, self.expinfo_1d, self.pts_1d,
        )

        # Try to create with list instead of np array of parameters.
        with pytest.raises(TypeError) as exc_info:
            Estimator.new_synthetic_from_parameters(
                self.params_1d.tolist(), self.expinfo_1d, self.pts_1d,
            )
        assert (
            "`parameters` is invalid:\nShould be a numpy array." in
            str(exc_info.value)
        )

        # Try to create with 1D parameter array but 2D expinfo and pts.
        with pytest.raises(TypeError) as exc_info:
            Estimator.new_synthetic_from_parameters(
                self.params_1d, self.expinfo_2d, self.pts_2d,
            )
        assert (
            "`parameters` is invalid:\nShould be a 2-dimensional array with "
            "shape (M, 6)." in str(exc_info.value)
        )
