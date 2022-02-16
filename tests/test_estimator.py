# test_estimator.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Feb 2022 10:10:05 GMT

import os
from pathlib import Path
import pickle
import random

import numpy as np
import pytest

from nmrespy import ExpInfo
from nmrespy.core import Estimator


def basic_estimator():
    return Estimator.new_synthetic_from_parameters(
        np.array([[random.uniform(1, 10), 0, 5, 1]]), ExpInfo(50), [4048],
    )


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
        estimator = Estimator.new_synthetic_from_parameters(
            self.params_1d, self.expinfo_1d, self.pts_1d,
        )
        print(estimator)

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

    def test_estimate(self):
        estimator = Estimator.new_synthetic_from_parameters(
            self.params_1d, self.expinfo_1d, self.pts_1d,
        )
        estimator.estimate([[3350., 2800.]], [[1200., 1000.]], region_unit="hz")


class TestPickle:
    def test_default(self):
        # Default options: check that unique pathname generation is working
        estimator = basic_estimator()
        for i in range(1, 4):
            estimator.to_pickle()
            assert Path(f"estimator_{i}.pkl").resolve().is_file()
        [os.remove(f"estimator_{x}.pkl") for x in range(1, 4)]

    def test_force_overwrite(self):
        estimator = basic_estimator()
        path_name = "my_estimator"
        path = Path(path_name).with_suffix(".pkl").resolve()
        estimator.to_pickle(path=path_name)
        assert path.is_file()
        creation_time = os.path.getmtime(path)

        # Try to save different estimator to same file with force_overwrite = False
        # Should be no change in modified time, as overwrite not permitted
        estimator2 = basic_estimator()
        estimator2.to_pickle(path=path_name)
        assert os.path.getmtime(path) == creation_time
        # Now allow overwrite
        estimator2.to_pickle(path=path_name, force_overwrite=True)
        assert os.path.getmtime(path) != creation_time
        os.remove("my_estimator.pkl")

    def test_invalid_directory(self):
        estimator = basic_estimator()
        with pytest.raises(TypeError) as exc_info:
            estimator.to_pickle("not_a_dir/estimator")
        assert "The parent directory" in str(exc_info.value)

    def test_from_pickle(self):
        estimator = basic_estimator()
        estimator.to_pickle("example")
        assert Path("example.pkl").resolve().is_file()
        estimator_cp = Estimator.from_pickle("example")
        assert np.array_equal(estimator._data, estimator_cp._data)
        os.remove("example.pkl")

    def test_from_pickle_not_estimator(self):
        path = "estimator.pkl"
        with open(path, "wb") as fh:
            pickle.dump(1, fh)

        with pytest.raises(TypeError) as exc_info:
            Estimator.from_pickle(path)
        assert "It is expected that the object loaded by" in str(exc_info.value)
        os.remove("estimator.pkl")
