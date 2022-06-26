# test_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 24 Jun 2022 13:47:20 BST

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


def view_files(to_view):
    if VIEW_CONTENT:
        for path in to_view:
            if path.suffix in [".txt", ".log"]:
                prog = "vi"
            elif path.suffix == ".pdf":
                prog = "evince"
            subprocess.run([prog, str(path)])


class DefaultEstimator:

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

    _before_estimation = ne.Estimator1D.new_synthetic_from_parameters(
        params=params,
        pts=4096,
        sw=1000.,
        offset=500.,
        sfo=500.,
    )

    @classmethod
    def before_estimation(cls):
        return copy.deepcopy(cls._before_estimation)

    @classmethod
    def after_estimation(cls):
        if hasattr(cls, "_after_estimation"):
            return copy.deepcopy(cls._after_estimation)
        else:
            cls._after_estimation = cls.before_estimation()
            regions = ((750., 680.), (375., 315.))
            noise_regions = ((550., 525.), (550., 525.))

            for region, noise_region in zip(regions, noise_regions):
                cls._after_estimation.estimate(
                    region=region,
                    noise_region=noise_region,
                    fprint=False,
                )

            return cls.after_estimation()


def latex_exists():
    if platform.system() == "Windows":
        cmd = "where"
    else:
        cmd = "which"

    return subprocess.run(
        [cmd, "pdflatex"], stdout=subprocess.DEVNULL,
    ).returncode == 0


def test_new_bruker(monkeypatch):
    if not VIEW_CONTENT:
        monkeypatch.setattr(plt, 'show', lambda: None)

    estimator = ne.Estimator1D.new_bruker("tests/data/1/pdata/1")
    estimator.view_data()
    estimator.view_data(freq_unit="ppm")
    estimator.view_data(domain="time", components="both")


def test_new_synthetic_from_simulation(monkeypatch):
    if not VIEW_CONTENT:
        monkeypatch.setattr(plt, 'show', lambda: None)

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
        pts=4096,
        freq_unit="ppm",
    )

    estimator.view_data()
    estimator.view_data(freq_unit="ppm")
    estimator.view_data(domain="time", components="both")


def test_new_synthetic_from_parameters(monkeypatch):
    if not VIEW_CONTENT:
        monkeypatch.setattr(plt, 'show', lambda: None)

    estimator = DefaultEstimator.before_estimation()

    estimator.view_data()
    estimator.view_data(freq_unit="ppm")
    estimator.view_data(domain="time", components="both")


def test_basic_parameters():
    estimator = DefaultEstimator.before_estimation()
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
    assert estimator.fn_mode is None


def test_timepoints():
    estimator = DefaultEstimator.before_estimation()
    tp = estimator.get_timepoints()
    assert isinstance(tp, tuple)
    assert len(tp) == 1
    assert utils.equal(tp[0], np.linspace(0, 4095 / 1000, 4096))

    tp = estimator.get_timepoints(pts=8192, start_time="20dt")
    assert utils.equal(tp[0], np.linspace(20 / 1000, 8211 / 1000, 8192))


def test_shifts():
    estimator = DefaultEstimator.before_estimation()
    shifts = estimator.get_shifts()
    assert isinstance(shifts, tuple)
    assert len(shifts) == 1
    assert utils.equal(shifts[0], np.linspace(1000, 0, 4096))

    shifts = estimator.get_shifts(pts=8192, unit="ppm", flip=False)
    assert utils.equal(shifts[0], np.linspace(0, 2, 8192))

    assert estimator.get_results() is None
    assert estimator.get_params() is None
    assert estimator.get_errors() is None


def test_phase_data():
    estimator = DefaultEstimator.before_estimation()
    og_data = copy.deepcopy(estimator.data)
    estimator.phase_data(p0=np.pi / 2)
    assert utils.close(estimator.data[0].real, 0, tol=0.2)
    estimator.phase_data(p0=-np.pi / 2)
    assert utils.equal(estimator.data, og_data)

    with pytest.raises(ValueError):
        estimator.write_result()

    with pytest.raises(ValueError):
        estimator.plot_result()


def test_estimate():
    # `after_estimation` will run estimate on two different regions.
    # Afterwards, estimate is tested with `region` set to `None`, and with different
    # optimisation methods invoked.
    estimator = DefaultEstimator.after_estimation()
    methods = ["gauss-newton", "exact", "lbfgs"]
    for method in methods:
        estimator.estimate(
            region=None,
            noise_region=None,
            method=method,
            fprint=False,
        )


def test_get_params_and_errors():
    estimator = DefaultEstimator.after_estimation()

    params_hz = estimator.get_params([0])
    errors_hz = estimator.get_errors([0])
    for i, sort in enumerate(("a", "p", "f", "d")):
        params = estimator.get_params([0], sort_by=sort)
        errors = estimator.get_errors([0], sort_by=sort)
        assert list(np.argsort(params[:, i])) == list(range(params.shape[0]))
        # Check errors correctly map to their parameters
        assert utils.equal(errors[np.argsort(params[:, 2])], errors_hz)

    params_ppm = estimator.get_params([0], funit="ppm")
    errors_ppm = estimator.get_errors([0], funit="ppm")
    assert utils.equal(params_hz[:, (0, 1, 3)], params_ppm[:, (0, 1, 3)])
    assert utils.equal(params_hz[:, 2] / estimator.sfo, params_ppm[:, 2])
    assert utils.equal(errors_hz[:, (0, 1, 3)], errors_ppm[:, (0, 1, 3)])
    assert utils.equal(errors_hz[:, 2] / estimator.sfo, errors_ppm[:, 2])

    # Ensure freqs in order
    all_region_params = estimator.get_params([1, 0], merge=False)
    assert len(all_region_params) == 2
    assert utils.equal(np.vstack(all_region_params), estimator.get_params())


def test_make_fid():
    estimator = DefaultEstimator.after_estimation()
    full_fid = estimator.make_fid()
    r1_fid = estimator.make_fid([0])
    r2_fid = estimator.make_fid([1])
    assert utils.close(full_fid, r1_fid + r2_fid, tol=0.1)
    assert full_fid.shape == estimator.data.shape
    long_fid = estimator.make_fid(pts=8192)
    assert utils.equal(full_fid, long_fid[:estimator.default_pts[0]])


def test_result_editing():
    estimator = DefaultEstimator.after_estimation()
    true_params = estimator.get_params([0])
    # Remove an oscillator and re-add
    estimator._results[0].params = true_params[(0, 1, 3), ...]
    estimator.add_oscillators(np.array([[2.5, 0, 717, 6.5]]), index=0)
    assert utils.close(true_params, estimator.get_params([0]), tol=0.1)
    # Add an extra oscillator and remove
    estimator._results[0].params = np.vstack(
        (
            np.array([[0.2, 0, 7.15, 6]]),
            true_params,
        ),
    )
    estimator.remove_oscillators([0], index=0)
    assert utils.close(true_params, estimator.get_params([0]), tol=0.1)
    # Split an oscillator and re-merge
    estimator.split_oscillator(1, index=0)
    # Determine two closest oscs
    freqs = estimator._results[0].params[:, 2]
    min_diff = int(np.argmin([np.absolute(b - a) for a, b in zip(freqs, freqs[1:])]))
    to_merge = [min_diff, min_diff + 1]
    estimator.merge_oscillators(to_merge, index=0)


@pytest.mark.usefixtures("cleanup_files")
def test_write_result():
    estimator = DefaultEstimator.after_estimation()
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

    view_files(to_view)


@pytest.mark.usefixtures("cleanup_files")
def test_plot_result():
    estimator = DefaultEstimator.after_estimation()
    to_view = []
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

    view_files(to_view)


@pytest.mark.usefixtures("cleanup_files")
def test_save_log():
    estimator = DefaultEstimator.after_estimation()
    to_view = []
    save_log_options = [
        {},
        {"path": "test_log"},
        {"path": "test_log", "force_overwrite": True},
    ]
    for options in save_log_options:
        estimator.save_log(**options)
        to_view.append(Path(options.get("path", "espy_logfile")).with_suffix(".log"))

    view_files(to_view)


@pytest.mark.usefixtures("cleanup_files")
def test_pickle():
    estimator = DefaultEstimator.after_estimation()
    pickling_options = [
        {},
        {"path": "test_pickle"},
        {"path": "test_pickle", "force_overwrite": True}
    ]
    for options in pickling_options:
        estimator.to_pickle(**options)
    # TODO: Simply checks that `from_pickle` runs successfully.
    # Should include an `__eq__` method into `Estimator` to check for equality.
    assert isinstance(ne.Estimator1D.from_pickle("test_pickle"), ne.Estimator1D)


def test_subband_estimate():
    estimator = DefaultEstimator.before_estimation()
    estimator.subband_estimate((550., 525.))
    assert utils.close(estimator.get_params(), DefaultEstimator.params, tol=0.2)


def test_write_to_bruker():
    estimator = DefaultEstimator.after_estimation()
    estimator.write_to_topspin("hello", 3)
