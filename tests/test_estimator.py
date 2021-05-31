import os
from pathlib import Path
import pickle
import subprocess

import pytest

import numpy as np

import nmrespy._errors as errors
from nmrespy.core import Estimator
from nmrespy import sig

# Set this to True if you want to check interactive and visual things.
MANUAL_TEST = False
RUN_PDFLATEX = False


def test_estimator():
    # --- Create Estimator instance from Bruker path -----------------
    path = Path().cwd() / 'data/1/pdata/1'
    estimator = Estimator.new_bruker(path)
    assert repr(estimator)
    assert str(estimator)

    sw_h = 5494.50549450549
    sw_p = 10.9861051816364
    off_h = 2249.20599998768
    off_p = 2249.20599998768 / 500.132249206
    sfo = 500.132249206
    bf = 500.13

    # --- Data path --------------------------------------------------
    assert estimator.get_datapath() == path
    assert estimator.get_datapath(type_='str') == str(path)

    estimator.path = None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_datapath()

    assert estimator.get_datapath(kill=False) is None

    estimator.path = path

    # --- Data dimension ---------------------------------------------
    assert estimator.get_dim() == 1

    # --- Sweep width ------------------------------------------------
    assert round(estimator.get_sw()[0], 4) == round(sw_h, 4)
    assert round(estimator.get_sw(unit='hz')[0], 4) == round(sw_h, 4)
    assert round(estimator.get_sw(unit='ppm')[0], 4) == round(sw_p, 4)

    with pytest.raises(errors.InvalidUnitError):
        estimator.get_sw(unit='invalid')

    _sfo = estimator.get_sfo()
    estimator.sfo = None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_sw(unit='ppm')

    assert estimator.get_sw(unit='ppm', kill=False) is None

    estimator.sfo = _sfo

    # --- Offset -----------------------------------------------------
    assert round(estimator.get_offset()[0], 4) == round(off_h, 4)
    assert round(estimator.get_offset(unit='hz')[0], 4) == round(off_h, 4)
    assert round(estimator.get_offset(unit='ppm')[0], 4) == round(off_p, 4)

    with pytest.raises(errors.InvalidUnitError):
        estimator.get_offset(unit='invalid')

    estimator.sfo = None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_offset(unit='ppm')

    assert estimator.get_offset(unit='ppm', kill=False) is None
    estimator.sfo = _sfo

    # --- Transmitter and basic frequency ----------------------------
    assert round(estimator.get_sfo()[0], 4) == round(sfo, 4)
    assert round(estimator.get_bf()[0], 2) == round(bf, 4)

    estimator.sfo = None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_sfo()
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_bf()

    assert estimator.get_sfo(kill=False) is None
    assert estimator.get_bf(kill=False) is None

    estimator.sfo = _sfo

    # --- Nucleus ----------------------------------------------------
    assert estimator.get_nucleus()[0] == '1H'

    estimator.nuc = None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_nucleus()

    assert estimator.get_nucleus(kill=False) is None
    estimator.nuc = ['1H']

    # --- Chemical shifts --------------------------------------------
    pts = estimator.get_n()[0]
    shifts = np.linspace(
        (sw_h / 2) + off_h, (-sw_h / 2) + off_h, pts,
    )

    # Seem to get different numbers of sig figs, so had to revert to
    # all close rather than array_equal
    assert np.allclose(
        np.round(estimator.get_shifts()[0], decimals=4),
        np.round(shifts, decimals=4),
    )
    assert np.allclose(
        np.round(estimator.get_shifts(unit='ppm')[0], decimals=4),
        np.round(shifts / sfo, decimals=4),
    )

    estimator.sfo = None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_shifts(unit='ppm')

    assert estimator.get_shifts(unit='ppm', kill=False) is None
    estimator.sfo = _sfo

    # --- Time-points ------------------------------------------------
    tp = np.round(
        np.linspace(0., (pts - 1) / sw_h, pts),
        decimals=4,
    )

    assert np.allclose(
        np.round(estimator.get_timepoints()[0], decimals=4),
        tp,
    )

    # --- View data ------------------------------------------------------
    if MANUAL_TEST:
        # Spectrum, real, ppm
        estimator.view_data(domain='frequency')
        # Spectrum, real, Hz
        estimator.view_data(domain='frequency', freq_xunit='hz')
        # Spectrum, imaginary, ppm
        estimator.view_data(domain='frequency', component='imag')
        # Spectrum, imaginary and real, ppm
        estimator.view_data(domain='frequency', component='both')
        # FID, real
        estimator.view_data(domain='time')
        # FID, imaginary
        estimator.view_data(domain='time', component='imag')
        # FID, real and imaginary
        estimator.view_data(domain='time', component='both')

    # --- Frequency filter -------------------------------------------
    # Apply same filter to same region, using both hz and ppm for regions.
    # Ensure all attributes are matching
    assert estimator.get_filter_info(kill=False) is None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_filter_info(kill=True)

    # ppm
    estimator.frequency_filter([[4.85, 5.05]], [[6.6, 6.5]])
    ppm_filter = estimator.get_filter_info()

    # Hz
    estimator.frequency_filter(
        [[2425.6414, 2525.6679]], [[3300.8728, 3250.8596]], region_unit='hz',
    )
    hz_filter = estimator.get_filter_info()

    assert hz_filter.get_sw() == ppm_filter.get_sw()
    assert hz_filter.get_offset() == ppm_filter.get_offset()
    assert hz_filter.get_region() == ppm_filter.get_region()
    assert hz_filter.get_noise_region() == ppm_filter.get_noise_region()

    # Ensure that two signals are identical, given some margin of error
    # for noise. Have found that it is incredibly rare for the difference
    # between two points to exceed 200
    assert np.allclose(
        hz_filter.get_fid(), ppm_filter.get_fid(), rtol=0, atol=200,
    )

    # --- Phase data -----------------------------------------------------
    # Apply phasing twice, in different directions, and assert that the net
    # effect is no change
    before = estimator.get_data()
    estimator.phase_data(p0=[0.8], p1=[1.2])
    phased = estimator.get_data()
    assert not np.array_equal(before, phased)
    estimator.phase_data(p0=[-0.8], p1=[-1.2])
    after = estimator.get_data()
    assert np.array_equal(np.round(before, 4), np.round(after, 4))

    if MANUAL_TEST:
        estimator.manual_phase_data()

    f_n = list(estimator.get_filter_info().get_fid().shape)
    f_sw = estimator.get_filter_info().get_sw()
    f_off = estimator.get_filter_info().get_offset()

    params = np.array([
        [1, 0, f_off[0], 1],
        [2, 0, f_off[0] + (f_sw[0] / 4), 1],
        [2, 0, f_off[0] - (f_sw[0] / 4), 1],
        [1, 0, f_off[0] + (f_sw[0] / 8), 1],
    ])

    params = params[np.argsort(params[:, 2])]
    fid = sig.make_fid(params, f_n, f_sw, offset=f_off)[0]
    estimator.filter_info.fid['cut'] = fid

    # --- Matrix Pencil ----------------------------------------------
    # With specified number of params
    estimator.matrix_pencil(M=4, fprint=False)
    res = estimator.get_result()
    assert np.allclose(res, params)

    with pytest.raises(ValueError):
        estimator.plot_result()

    with pytest.raises(ValueError):
        estimator.write_result()

    # --- NonlinearProgramming ---------------------------------------
    estimator.result = np.array([
        [1 + 0.1, 0.01, f_off[0] + 2, 1 + 0.2],
        [2 - 0.1, 0.02, f_off[0] + (f_sw[0] / 4) - 4, 1 - 0.2],
        [2 + 0.2, -0.05, f_off[0] - (f_sw[0] / 4) + 3, 1 + 0.2],
        [1 - 0.05, -0.03, f_off[0] + (f_sw[0] / 8) - 1, 1 - 0.1],
    ])

    estimator.nonlinear_programming(phase_variance=False, max_iterations=1000,
                                    fprint=False)
    res = estimator.get_result()
    assert np.allclose(res, params, rtol=0, atol=1E-6)

    # --- Writing result files ---------------------------------------
    for fmt in ['txt', 'pdf', 'csv']:
        if (fmt in ['txt', 'csv']) or (fmt == 'pdf' and RUN_PDFLATEX):
            estimator.write_result(
                path='./test', description='Testing', fmt=fmt,
                force_overwrite=True, sig_figs=7, sci_lims=(-3, 4),
                fprint=False
            )
        # View output files
        if MANUAL_TEST:
            if fmt == 'txt':
                subprocess.run(['vi', 'test.txt'])
            elif fmt == 'pdf' and RUN_PDFLATEX:
                subprocess.run(['evince', 'test.pdf'])
            elif fmt == 'csv':
                subprocess.run(['libreoffice', 'test.csv'])

        try:
            os.remove(f'test.{fmt}')
            if fmt == 'pdf':
                os.remove('test.tex')
        except Exception:
            pass

    # --- Check result array-amending methods-------------------------
    # After manually changing the result, it should not be possible to
    # save the result.
    estimator.result = np.array([[1., 0., -4., 3.], [2., 1., 2., 3.]])
    estimator.add_oscillators(np.array([[1., 0., 0., 1.]]))
    assert np.array_equal(
        estimator.result,
        np.array(
            [
                [1., 0., -4., 3.],
                [1., 0., 0., 1.],
                [2., 1., 2., 3.],
            ]
        ),
    )

    with pytest.raises(ValueError):
        estimator.write_result()

    estimator.remove_oscillators([0])
    assert np.array_equal(
        estimator.result,
        np.array(
            [
                [1., 0., 0., 1.],
                [2., 1., 2., 3.],
            ]
        ),
    )

    estimator.merge_oscillators([0, 1])
    assert np.array_equal(estimator.result,
                          np.array([[3., 0.5, 1., 2.]]))

    estimator.split_oscillator(0, separation_frequency=5, split_number=3)
    assert np.array_equal(
        estimator.result,
        np.array(
            [
                [1., 0.5, -4., 2.],
                [1., 0.5, 1., 2.],
                [1., 0.5, 6., 2.],
            ]
        ),
    )

    # --- Saving logfile ---------------------------------------------
    estimator.save_logfile('test', force_overwrite=True)
    if MANUAL_TEST:
        subprocess.run(['vi', 'test.log'])
    os.remove('test.log')

    # --- Pickling ---------------------------------------------------
    estimator.to_pickle('info', force_overwrite=True)
    Estimator.from_pickle('info')
    os.remove('info.pkl')

    # Check that opening a file which doesn't contain an Estimator
    # instance rasies an error
    with open('fail.pkl', 'wb') as fh:
        pickle.dump(1, fh)
    with pytest.raises(TypeError):
        Estimator.from_pickle('fail')
    os.remove('fail.pkl')
