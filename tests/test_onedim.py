# test_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 16 Jun 2022 19:46:07 BST

import copy
from pathlib import Path
import platform
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import nmrespy as ne
from nmr_sims.spin_system import SpinSystem
import numpy as np
import pytest
import utils
mpl.use("tkAgg")

VIEW_CONTENT = False


def latex_exists():
    if platform.system() == "Windows":
        cmd = "where"
    else:
        cmd = "which"

    return subprocess.run(
        [cmd, "pdflatex"], stdout=subprocess.DEVNULL,
    ).returncode == 0


def test_new_bruker():
    estimator = ne.Estimator1D.new_bruker("tests/data/1/pdata/1")
    if VIEW_CONTENT:
        estimator.view_data()


def test_new_synthetic_from_simulation():
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
    estimator = ne.Estimator1D.new_synthetic_from_simulation(
        spin_system=system,
        sw=2.,
        offset=4.,
        pts=4048,
        freq_unit="ppm",
    )
    if VIEW_CONTENT:
        estimator.view_data()


@pytest.mark.usefixtures("cleanup_files")
def test_new_synthetic_from_parameters(monkeypatch):
    params = np.array(
        [
            [15, 0, 340, 7],
            [15, 0, 350, 7],
            [1, 0, 700, 7],
            [3, 0, 710, 7],
            [3, 0, 720, 7],
            [1, 0, 730, 7],
        ]
    )
    estimator = ne.Estimator1D.new_synthetic_from_parameters(
        params=params,
        pts=4096,
        sw=1000.,
        offset=500.,
        sfo=500.,
    )

    if not VIEW_CONTENT:
        monkeypatch.setattr(plt, 'show', lambda: None)

    estimator.view_data()
    estimator.view_data(freq_unit="ppm")
    estimator.view_data(domain="time", components="both")

    assert estimator.dim == 1
    assert estimator.default_pts == (4096,)
    assert estimator.sw() == estimator.sw(unit="hz") == (1000.,)
    assert estimator.sw(unit="ppm") == (2.,)
    assert estimator.offset() == estimator.offset(unit="hz") == (500.,)
    assert estimator.offset(unit="ppm") == (1.,)
    assert estimator.sfo == (500.,)
    assert estimator.nuclei is None
    assert estimator.latex_nuclei is None
    assert estimator.unicode_nuclei is None

    tp = estimator.get_timepoints()
    assert isinstance(tp, tuple)
    assert len(tp) == 1
    assert utils.equal(tp[0], np.linspace(0, 4095 / 1000, 4096))

    tp = estimator.get_timepoints(pts=8192, start_time="20dt")
    assert utils.equal(tp[0], np.linspace(20 / 1000, 8211 / 1000, 8192))

    shifts = estimator.get_shifts()
    assert isinstance(shifts, tuple)
    assert len(shifts) == 1
    assert utils.equal(shifts[0], np.linspace(1000, 0, 4096))

    shifts = estimator.get_shifts(pts=8192, unit="ppm", flip=False)
    assert utils.equal(shifts[0], np.linspace(0, 2, 8192))

    assert estimator.fn_mode is None
    assert estimator.get_results() is None
    assert estimator.get_params() is None
    assert estimator.get_errors() is None

    with pytest.raises(ValueError):
        estimator.write_result()

    with pytest.raises(ValueError):
        estimator.plot_result()

    # --- Phasing ---
    og_data = copy.deepcopy(estimator.data)
    estimator.phase_data(p0=np.pi / 2)
    assert utils.close(estimator.data[0].real, 0, tol=0.2)
    estimator.phase_data(p0=-np.pi / 2)
    assert utils.equal(estimator.data, og_data)

    # --- Estimations ---
    regions = (None, (750., 680.), (375., 315.))
    noise_regions = (None, (550., 525.), (550., 525.))

    # Not going to actually test the accuracy of estimation.
    # Quite a trivial example.
    for i, (region, noise_region) in enumerate(zip(regions, noise_regions)):
        if i == 0:
            methods = ["gauss-newton", "exact", "lbfgs"]
        else:
            methods = ["gauss-newton"]

        for method in methods:
            estimator.estimate(
                region=region,
                noise_region=noise_region,
                method=method,
                fprint=False,
            )
    del estimator._results[1]
    del estimator._results[1]

    # --- Accessing parameters/errors ---
    all_params_hz = estimator.get_params([0])
    all_errors_hz = estimator.get_errors([0])
    assert all_params_hz.shape == all_errors_hz.shape == (6, 4)

    for i, sort in enumerate(("a", "p", "f", "d")):
        params = estimator.get_params([0], sort_by=sort)
        errors = estimator.get_errors([0], sort_by=sort)
        assert list(np.argsort(params[:, i])) == list(range(params.shape[0]))
        # Check errors correctly map to their parameters
        assert utils.equal(errors[np.argsort(params[:, 2])], all_errors_hz)

    all_params_ppm = estimator.get_params([0], funit="ppm")
    all_errors_ppm = estimator.get_errors([0], funit="ppm")
    assert utils.equal(all_params_hz[:, (0, 1, 3)], all_params_ppm[:, (0, 1, 3)])
    assert utils.equal(all_params_hz[:, 2] / estimator.sfo, all_params_ppm[:, 2])
    assert utils.equal(all_errors_hz[:, (0, 1, 3)], all_errors_ppm[:, (0, 1, 3)])
    assert utils.equal(all_errors_hz[:, 2] / estimator.sfo, all_errors_ppm[:, 2])

    region_params = estimator.get_params([2, 1], merge=False)
    assert len(region_params) == 2
    assert utils.equal(np.vstack(region_params), estimator.get_params([1, 2]))

    # --- Make FID ---
    fid1 = estimator.make_fid([0])
    fid2 = estimator.make_fid([1, 2])
    assert utils.close(fid1, fid2, tol=0.1)
    assert fid1.shape == fid2.shape == estimator.data.shape
    fid3 = estimator.make_fid(indices=[0], pts=8192)
    assert utils.equal(fid1, fid3[:estimator.default_pts[0]])

    # --- Result editing (merge, split, add, remove) ---
    true_params = estimator.get_params([1])
    # Remove an oscillator and re-add
    estimator._results[1].params = true_params[(0, 1, 3), ...]
    estimator.add_oscillators(np.array([[2.5, 0, 717, 6.5]]), index=1)
    assert utils.close(true_params, estimator.get_params([1]), tol=0.1)
    # Add an extra oscillator and remove
    estimator._results[1].params = np.vstack(
        (
            np.array([[0.2, 0, 7.15, 6]]),
            true_params,
        ),
    )
    estimator.remove_oscillators([0], index=1)
    assert utils.close(true_params, estimator.get_params([1]), tol=0.1)
    # Split an oscillator and re-merge
    estimator.split_oscillator(1, index=1)
    # Determine two closest oscs
    freqs = estimator._results[1].params[:, 2]
    min_diff = np.argmin([np.absolute(b - a) for a, b in zip(freqs, freqs[1:])])
    to_merge = [min_diff, min_diff + 1]
    estimator.merge_oscillators(to_merge, index=1)
    assert utils.close(true_params, estimator.get_params([1]), tol=0.1)

    to_view = []
    # --- Writing results ---
    write_result_options = [
        {},
        {
            "indices": [0],
            "path": "custom_path",
            "fmt": "pdf",
            "description": "Testing",
            "sig_figs": 3,
            "sci_lims": None,
            "integral_mode": "absolute",
        }
    ]
    for options in write_result_options:
        if not latex_exists() and options.get("fmt", "txt") == "pdf":
            pass
        else:
            estimator.write_result(**options)
            to_view.append(
                Path(
                    ".".join([
                        options.get("path", "nmrespy_result"),
                        options.get("fmt", "txt"),
                    ])
                )
            )

    # --- Result plotting ---
    plot_result_options = [
        {},
        {
            "indices": [0],
            "plot_residual": False,
            "plot_model": True,
            "shifts_unit": "hz",
            "data_color": "#b0b0b0",
            "model_color": "#505050",
            "oscillator_colors": "inferno",
            "show_labels": False,
            "stylesheet": "ggplot"
        }
    ]

    counter = 1
    for options in plot_result_options:
        plots = estimator.plot_result(**options)
        for plot in plots:
            plot.save(f"plot{counter}", fmt="pdf")
            to_view.append(Path(f"plot{counter}.pdf"))
            counter += 1

    # --- Save logfile ---
    save_log_options = [
        {},
        {"path": "test_log"},
        {"path": "test_log", "force_overwrite": True},
    ]
    for options in save_log_options:
        estimator.save_log(**options)
        to_view.append(Path(options.get("path", "espy_logfile")).with_suffix(".log"))

    # --- Pickling ---
    pickling_options = [
        {},
        {"path": "test_pickle"},
        {"path": "test_pickle", "force_overwrite": True}
    ]
    for options in pickling_options:
        estimator.to_pickle(**options)
    # TODO: Simply checks that `from_pickle` runs successfully.
    # Should include an `__eq__` method into `Estimator` to check for equality.
    ne.Estimator1D.from_pickle("test_pickle")

    # --- Viewing outputs ---
    if VIEW_CONTENT:
        for path in to_view:
            if path.suffix in [".txt", ".log"]:
                prog = "vi"
            elif path.suffix == ".pdf":
                prog = "evince"
            subprocess.run([prog, str(path)])
