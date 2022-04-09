# test_estimator.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 14:01:51 GMT

import os
from pathlib import Path
import pickle
import random

import numpy as np
import pytest

from nmrespy import ExpInfo, sig
# from nmrespy.core import Estimator
from nmrespy import Estimator1D


def basic_estimator(dim: int = 1, with_sfo: bool = True) -> Estimator1D:
    if dim == 1:
        pts = [4048]
        params = np.array(
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
        if with_sfo:
            expinfo = ExpInfo(sw=5000, offset=2000, sfo=500, nuclei="1H")
        else:
            expinfo = ExpInfo(sw=5000, offset=2000, nuclei="1H")
    else:
        pts = [512, 512]
        params = np.array(
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
        if with_sfo:
            expinfo = ExpInfo(sw=5000, offset=2000, sfo=500, nuclei="1H", dim=2)
        else:
            expinfo = ExpInfo(sw=5000, offset=2000, nuclei="1H", dim=2)

    return Estimator1D(sig.make_fid(params, expinfo, pts), expinfo, None)
    # return Estimator.new_synthetic_from_parameters(params, expinfo, pts, snr=25.0)


def test_simulated():
    from nmr_sims.spin_system import SpinSystem

    system = SpinSystem(
        {
            1: {
                "shift": 3.7,
                "couplings": {
                    2: -10.1,
                    3: 4.3,
                },
                "nucleus": "1H",
            },
            2: {
                "shift": 3.92,
                "couplings": {
                    3: 11.3,
                },
                "nucleus": "1H",
            },
            3: {
                "shift": 4.5,
                "nucleus": "1H",
            },
        }
    )

    e = Estimator1D.new_synthetic_from_simulation(
        system,
        2.,    # sw
        4.,    # offset
        4096,  # number of pts
        freq_unit="ppm",
    )

    e.estimate([4.59, 4.41], [4.2, 4.15], region_unit="ppm")
    e.estimate([3.97, 3.87], [4.2, 4.15], region_unit="ppm")
    e.write_result(path="helloooo", fmt="pdf", description="Hello there")
    e.view_log
    exit()
    print(e._results[0].get_result(funit="ppm"))
    e.write_results(force_overwrite=True)
    p = e.plot_results()
    p[0].fig.savefig("hello.pdf")

# class TestPickle:
#     def test_default(self):
#         # Default options: check that unique pathname generation is working
#         estimator = basic_estimator()
#         for i in range(1, 4):
#             estimator.to_pickle()
#             assert Path(f"estimator_{i}.pkl").resolve().is_file()
#         [os.remove(f"estimator_{x}.pkl") for x in range(1, 4)]

#     def test_force_overwrite(self):
#         estimator = basic_estimator()
#         path_name = "my_estimator"
#         path = Path(path_name).with_suffix(".pkl").resolve()
#         estimator.to_pickle(path=path_name)
#         assert path.is_file()
#         creation_time = os.path.getmtime(path)

#         # Try to save different estimator to same file with force_overwrite = False
#         # Should be no change in modified time, as overwrite not permitted
#         estimator2 = basic_estimator()
#         estimator2.to_pickle(path=path_name)
#         assert os.path.getmtime(path) == creation_time
#         # Now allow overwrite
#         estimator2.to_pickle(path=path_name, force_overwrite=True)
#         assert os.path.getmtime(path) != creation_time
#         os.remove("my_estimator.pkl")

#     def test_invalid_directory(self):
#         estimator = basic_estimator()
#         with pytest.raises(TypeError) as exc_info:
#             estimator.to_pickle("not_a_dir/estimator")
#         assert "The parent directory" in str(exc_info.value)

#     def test_from_pickle(self):
#         estimator = basic_estimator()
#         estimator.to_pickle("example")
#         assert Path("example.pkl").resolve().is_file()
#         estimator_cp = Estimator.from_pickle("example")
#         assert np.array_equal(estimator._data, estimator_cp._data)
#         os.remove("example.pkl")

#     def test_from_pickle_not_estimator(self):
#         path = "estimator.pkl"
#         with open(path, "wb") as fh:
#             pickle.dump(1, fh)

#         with pytest.raises(TypeError) as exc_info:
#             Estimator.from_pickle(path)
#         assert "It is expected that the object loaded by" in str(exc_info.value)
#         os.remove("estimator.pkl")


# class TestWrite:
#     def test_txt(self):
#         estimator = basic_estimator()
#         estimator.estimate([[3350., 2800.]], [[1200., 1000.]], region_unit="hz")
#         estimator.estimate([[300., -50.]], [[1200., 1000.]], region_unit="hz")
#         estimator.write_results(force_overwrite=True)
#         estimator.write_results(fmt="pdf", force_overwrite=True, sci_lims=(-5, 6))
#         estimator.view_log
#         estimator.save_log()

#         plots = estimator.plot_results()
#         plots[0].fig.savefig("figure.pdf")
