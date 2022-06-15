# test_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 15 Jun 2022 10:40:42 BST

from pathlib import Path
import platform
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import nmrespy as ne
import numpy as np
import pytest
import utils
mpl.use("tkAgg")

VIEW_CONTENT = False


def check_latex_exists():
    if platform.system() == "Windows":
        cmd = "where"
    else:
        cmd = "which"

    return subprocess.run(
        [cmd, "pdflatex"], stdout=subprocess.DEVNULL,
    ).returncode == 0


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

    monkeypatch.setattr(plt, 'show', lambda: None)
    estimator.view_data()

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

    regions = ((750., 680.), (375., 315.))
    noise_region = (550., 525.)

    # Not going to actually test the accuracy of estimation.
    # Quite a trivial example.
    for region in regions:
        estimator.estimate(region=region, noise_region=noise_region, fprint=False)

    to_view = []
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
        estimator.write_result(**options)
        to_view.append(
            Path(
                f"{options.get('path', 'nmrespy_result')}.{options.get('fmt', 'txt')}"
            )
        )

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

    if VIEW_CONTENT:
        for path in to_view:
            if path.suffix == ".txt":
                prog = "vi"
            elif path.suffix == ".pdf":
                prog = "evince"
            subprocess.run([prog, str(path)])
