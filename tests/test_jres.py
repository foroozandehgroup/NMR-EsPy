# test_jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 03 Oct 2022 18:39:00 BST

import copy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nmrespy as ne
import numpy as np
import pytest
import utils
mpl.use("tkAgg")
mpl.rcParams["text.usetex"] = False


VIEW_CONTENT = True


class DefaultEstimator:
    params = np.array(
        [
            [2, 0, 10, 180, 5, 5],
            [4, 0, 0, 170, 5, 5],
            [2, 0, -10, 160, 5, 5],
            [1, 0, 15, 110, 5, 5],
            [3, 0, 5, 100, 5, 5],
            [3, 0, -5, 90, 5, 5],
            [1, 0, -15, 80, 5, 5],
            [4, 0, 5, 40, 5, 5],
            [4, 0, -5, 30, 5, 5],
        ],
        dtype="float64",
    )
    expinfo = ne.ExpInfo(
        dim=2,
        sw=(40., 200.),
        offset=(0., 100.),
        default_pts=(64, 1024),
        nuclei=(None, "1H"),
        sfo=(None, 100.),
    )
    fid = expinfo.make_fid(params, snr=30.)

    _before_estimation = ne.Estimator2DJ(fid, expinfo, None)

    @classmethod
    def before_estimation(cls):
        return copy.deepcopy(cls._before_estimation)

    @classmethod
    def after_estimation(cls):
        if hasattr(cls, "_after_estimation"):
            return copy.deepcopy(cls._after_estimation)
        else:
            cls._after_estimation = cls.before_estimation()
            regions = ((200., 140.), (130., 60.), (60., 10.))
            noise_region = (138., 132.)

            for region in regions:
                cls._after_estimation.estimate(
                    region=region,
                    noise_region=noise_region,
                    fprint=False,
                    region_unit="hz",
                    max_iterations=40,
                )

            return cls.after_estimation()


@pytest.mark.xfail
def test_new_bruker(monkeypatch):
    if not VIEW_CONTENT:
        monkeypatch.setattr(plt, 'show', lambda: None)

    estimator = ne.Estimator2DJ.new_bruker("tests/data/1/pdata/1")
    estimator.view_data()


def test_new_spinach(monkeypatch):
    kwargs = {
        "shifts": [1., 2., 3.],
        "couplings": [(1, 2, 10.), (1., 3., 5.), (2., 3., 4.)],
        "pts": (64, 4096),
        "sw": (30., 2000.),
        "offset": 1000.,
        "sfo": 500.,
        "snr": 30.,
    }

    if not ne.MATLAB_AVAILABLE:
        with pytest.raises(NotImplementedError):
            ne.Estimator2DJ.new_spinach(**kwargs)

    else:
        if not VIEW_CONTENT:
            monkeypatch.setattr(plt, 'show', lambda: None)
        estimator = ne.Estimator2DJ.new_spinach(**kwargs)
        estimator.view_data()


def test_basic_parameters():
    estimator = DefaultEstimator.before_estimation()
    assert estimator.dim == 2
    assert estimator.default_pts == (64, 1024)
    assert utils.equal(estimator.sw(unit="hz"), (40., 200.))
    assert utils.equal(estimator.sw(unit="ppm"), (40., 2.))
    assert utils.equal(estimator.offset(unit="hz"), (0., 100.))
    assert utils.equal(estimator.offset(unit="ppm"), (0., 1.,))
    assert utils.equal(estimator.sfo, (None, 100.))
    assert estimator.nuclei == (None, "1H")
    assert estimator.latex_nuclei == (None, "\\textsuperscript{1}H")
    assert estimator.unicode_nuclei == (None, "¹H")
    assert estimator.fn_mode == "QF"


def test_timepoints():
    estimator = DefaultEstimator.before_estimation()
    tp = estimator.get_timepoints(meshgrid=False)
    assert isinstance(tp, tuple)
    assert len(tp) == 2
    assert utils.aequal(tp[0], np.linspace(0, 63 / 40, 64))
    assert utils.aequal(tp[1], np.linspace(0, 1023 / 200, 1024))

    tp = estimator.get_timepoints(
        pts=(256, 2048), start_time=[0., "10dt"], meshgrid=False,
    )
    assert utils.aequal(tp[0], np.linspace(0, 255 / 40, 256))
    assert utils.aequal(tp[1], np.linspace(10 / 200, 2057 / 200, 2048))

    tp_mesh = estimator.get_timepoints()
    assert tp_mesh[0].shape == (64, 1024)


def test_shifts():
    estimator = DefaultEstimator.before_estimation()
    shifts = estimator.get_shifts(meshgrid=False)
    assert isinstance(shifts, tuple)
    assert len(shifts) == 2
    assert utils.aequal(shifts[0], np.linspace(20, -20, 65)[1:])
    assert utils.aequal(shifts[1], np.linspace(200, 0, 1025)[1:])

    shifts = estimator.get_shifts(
        pts=(256, 2048), unit="ppm", flip=False, meshgrid=False,
    )
    assert utils.aequal(shifts[0], np.linspace(-20, 20, 257)[:-1])
    assert utils.aequal(shifts[1], np.linspace(0., 2., 2049)[:-1])


def test_estimate():
    # `after_estimation` will run estimate on three different regions.
    # Afterwards, estimate is tested with `region` set to `None`, and with different
    # optimisation methods invoked.
    DefaultEstimator.after_estimation()


def test_negative_45_signal(monkeypatch):
    if not VIEW_CONTENT:
        monkeypatch.setattr(plt, 'show', lambda: None)

    estimator = DefaultEstimator.after_estimation()
    neg45 = estimator.diagonal_signal(pts=8192)
    neg45[0] *= 0.5
    spectrum = ne.sig.ft(neg45)
    _, shifts = estimator.get_shifts(pts=(1, 8192), unit="hz", meshgrid=False)
    import matplotlib as mpl
    mpl.use("tkAgg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(shifts, spectrum.real)
    ax.set_xlim(shifts[0], shifts[-1])
    plt.show()


# def test_get_params_and_errors():
#     estimator = DefaultEstimator.after_estimation()

#     params_hz = estimator.get_params([0])
#     errors_hz = estimator.get_errors([0])
#     for i, sort in enumerate(("a", "p", "f", "d")):
#         params = estimator.get_params([0], sort_by=sort)
#         errors = estimator.get_errors([0], sort_by=sort)
#         assert list(np.argsort(params[:, i])) == list(range(params.shape[0]))
#         # Check errors correctly map to their parameters
#         assert utils.equal(errors[np.argsort(params[:, 2])], errors_hz)

#     params_ppm = estimator.get_params([0], funit="ppm")
#     errors_ppm = estimator.get_errors([0], funit="ppm")
#     assert utils.equal(params_hz[:, (0, 1, 3)], params_ppm[:, (0, 1, 3)])
#     assert utils.equal(params_hz[:, 2] / estimator.sfo, params_ppm[:, 2])
#     assert utils.equal(errors_hz[:, (0, 1, 3)], errors_ppm[:, (0, 1, 3)])
#     assert utils.equal(errors_hz[:, 2] / estimator.sfo, errors_ppm[:, 2])

#     # Ensure freqs in order
#     all_region_params = estimator.get_params([1, 0], merge=False)
#     assert len(all_region_params) == 2
#     assert utils.equal(np.vstack(all_region_params), estimator.get_params())


def test_make_fid():
    estimator = DefaultEstimator.after_estimation()
    full_fid = estimator.make_fid_from_result()
    sub_fids = [
        estimator.make_fid_from_result(indices=[i])
        for i in range(len(estimator._results))
    ]
    assert np.allclose(
        full_fid,
        sum(sub_fids),
    )
    long_fid = estimator.make_fid_from_result(pts=(128, 8192))
    assert np.array_equal(
        full_fid,
        long_fid[:estimator.default_pts[0], :estimator.default_pts[1]],
    )


# # def test_result_editing():
# #     estimator = DefaultEstimator.after_estimation()
# #     true_params = estimator.get_params([0])
# #     # Remove an oscillator and re-add
# #     estimator._results[0].params = true_params[(0, 1, 3), ...]
# #     estimator.add_oscillators(np.array([[2.5, 0, 717, 6.5]]), index=0)
# #     assert utils.close(true_params, estimator.get_params([0]), tol=0.1)
# #     # Add an extra oscillator and remove
# #     estimator._results[0].params = np.vstack(
# #         (
# #             np.array([[0.2, 0, 7.15, 6]]),
# #             true_params,
# #         ),
# #     )
# #     estimator.remove_oscillators([0], index=0)
# #     assert utils.close(true_params, estimator.get_params([0]), tol=0.1)
# #     # Split an oscillator and re-merge
# #     estimator.split_oscillator(1, index=0)
# #     # Determine two closest oscs
# #     freqs = estimator._results[0].params[:, 2]
# #     min_diff = int(np.argmin([np.absolute(b - a) for a, b in zip(freqs, freqs[1:])]))
# #     to_merge = [min_diff, min_diff + 1]
# #     estimator.merge_oscillators(to_merge, index=0)


# # @pytest.mark.usefixtures("cleanup_files")
# # def test_write_result():
# #     estimator = DefaultEstimator.after_estimation()
# #     to_view = []
# #     # --- Writing results ---
# #     write_result_options = [
# #         {},
# #         {
# #             "indices": [0],
# #             "path": "custom_path",
# #             "fmt": "pdf",
# #             "description": "Testing",
# #             "sig_figs": 3,
# #             "sci_lims": None,
# #             "integral_mode": "absolute",
# #         }
# #     ]
# #     for options in write_result_options:
# #         if not latex_exists() and options.get("fmt", "txt") == "pdf":
# #             pass
# #         else:
# #             estimator.write_result(**options)
# #             to_view.append(
# #                 Path(
# #                     ".".join([
# #                         options.get("path", "nmrespy_result"),
# #                         options.get("fmt", "txt"),
# #                     ])
# #                 )
# #             )

# #     utils.view_files(to_view, VIEW_CONTENT)


# # @pytest.mark.usefixtures("cleanup_files")
# # def test_plot_result():
# #     estimator = DefaultEstimator.after_estimation()
# #     to_view = []
# #     plot_result_options = [
# #         {},
# #         {
# #             "indices": [0],
# #             "plot_residual": False,
# #             "plot_model": True,
# #             "shifts_unit": "hz",
# #             "data_color": "#b0b0b0",
# #             "model_color": "#505050",
# #             "oscillator_colors": "inferno",
# #             "show_labels": False,
# #             "stylesheet": "ggplot"
# #         }
# #     ]

# #     counter = 1
# #     for options in plot_result_options:
# #         plots = estimator.plot_result(**options)
# #         for plot in plots:
# #             plot.save(f"plot{counter}", fmt="pdf")
# #             to_view.append(Path(f"plot{counter}.pdf"))
# #             counter += 1

# #     utils.hview_files(to_view)


# # @pytest.mark.usefixtures("cleanup_files")
# # def test_save_log():
# #     estimator = DefaultEstimator.after_estimation()
# #     to_view = []
# #     save_log_options = [
# #         {},
# #         {"path": "test_log"},
# #         {"path": "test_log", "force_overwrite": True},
# #     ]
# #     for options in save_log_options:
# #         estimator.save_log(**options)
# #         to_view.append(Path(options.get("path", "espy_logfile")).with_suffix(".log"))

# #     view_files(to_view)


# # @pytest.mark.usefixtures("cleanup_files")
# # def test_pickle():
# #     estimator = DefaultEstimator.after_estimation()
# #     pickling_options = [
# #         {},
# #         {"path": "test_pickle"},
# #         {"path": "test_pickle", "force_overwrite": True}
# #     ]
# #     for options in pickling_options:
# #         estimator.to_pickle(**options)
# #     # TODO: Simply checks that `from_pickle` runs successfully.
# #     # Should include an `__eq__` method into `Estimator` to check for equality.
# #     assert isinstance(ne.Estimator1D.from_pickle("test_pickle"), ne.Estimator1D)


# # def test_subband_estimate():
# #     estimator = DefaultEstimator.before_estimation()
# #     estimator.subband_estimate((550., 525.))
# #     assert utils.close(estimator.get_params(), DefaultEstimator.params, tol=0.2)


# # def test_write_to_bruker():
# #     estimator = DefaultEstimator.after_estimation()
# #     estimator.write_to_topspin("hello", 3)

