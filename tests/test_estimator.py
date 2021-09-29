import os
from pathlib import Path
import pickle
import subprocess

import pytest

import numpy as np
from numpy import random as nrandom

from context import nmrespy
from nmrespy import RED, END, ExpInfo, _errors, sig
from nmrespy.core import Estimator


# Set this to True if you want to check interactive and visual things.
INTERACT = True


class TestSyntheticEstimator:
    params_1d = np.array([
        [1, 0, 3000, 10],
        [3, 0, 3050, 10],
        [3, 0, 3100, 10],
        [1, 0, 3150, 10],
        [2, 0, 150, 10],
        [4, 0, 100, 10],
        [2, 0, 50, 10],
    ])
    params_2d = np.array([
        [1, 0, 3000, 3000, 10, 10],
        [3, 0, 3050, 3050, 10, 10],
        [3, 0, 3100, 3100, 10, 10],
        [1, 0, 3150, 3150, 10, 10],
        [2, 0, 150, 150, 10, 10],
        [4, 0, 100, 100, 10, 10],
        [2, 0, 50, 50, 10, 10],
    ])
    expinfo_1d = ExpInfo(pts=4096, sw=5000, offset=2000, sfo=500, nuclei='1H')
    expinfo_2d = ExpInfo(pts=512, sw=5000, offset=2000, sfo=500, nuclei='1H',
                         dim=2)
    expinfo_1d_no_nuc = ExpInfo(pts=4096, sw=5000, offset=2000, sfo=500)
    expinfo_2d_no_nuc = ExpInfo(pts=512, sw=5000, offset=2000, sfo=500, dim=2)
    expinfo_1d_no_sfo = ExpInfo(pts=4096, sw=5000, offset=2000, nuclei='1H')
    expinfo_2d_no_sfo = ExpInfo(pts=512, sw=5000, offset=2000, nuclei='1H',
                                dim=2)
    expinfo_1d_no_sfo_no_nuc = ExpInfo(pts=4096, sw=5000, offset=2000)
    expinfo_2d_no_sfo_no_nuc = ExpInfo(pts=512, sw=5000, offset=2000, dim=2)

    def estimator_1d(self, sfo=True, nuclei=True):
        if sfo and nuclei:
            expinfo = self.expinfo_1d
        elif sfo:
            expinfo = self.expinfo_1d_no_nuc
        elif nuclei:
            expinfo = self.expinfo_1d_no_sfo
        else:
            expinfo = self.expinfo_1d_no_sfo_no_nuc
        return Estimator.new_synthetic_from_parameters(
            self.params_1d, expinfo, snr=40.)

    def estimator_2d(self, sfo=True, nuclei=True):
        if sfo and nuclei:
            expinfo = self.expinfo_2d
        elif sfo:
            expinfo = self.expinfo_2d_no_nuc
        elif nuclei:
            expinfo = self.expinfo_2d_no_sfo
        else:
            expinfo = self.expinfo_2d_no_sfo_no_nuc
        return Estimator.new_synthetic_from_parameters(
            self.params_2d, expinfo, snr=40.)

    def test_init(self):
        self.estimator_1d()
        self.estimator_2d()

        with pytest.raises(TypeError) as exc_info:
            Estimator.new_synthetic_from_parameters(
                self.params_1d.tolist(), self.expinfo_1d)
        assert str(exc_info.value) == \
            f'{RED}`params` should be a numpy array.{END}'

        with pytest.raises(ValueError) as exc_info:
            Estimator.new_synthetic_from_parameters(
                self.params_1d[:, :-1], self.expinfo_1d)
        assert str(exc_info.value) == \
            f'{RED}`params` should have a size of 4 or 6 in axis 1.{END}'

        with pytest.raises(TypeError) as exc_info:
            Estimator.new_synthetic_from_parameters(self.params_1d, 'blah')
        assert str(exc_info.value) == \
            f'{RED}`expinfo` should be an instance of nmrespy.ExpInfo{END}'

        with pytest.raises(ValueError) as exc_info:
            Estimator.new_synthetic_from_parameters(
                self.params_1d, self.expinfo_2d)
        assert str(exc_info.value) == \
            (f'{RED}The dimension implied by `params` and `expinfo` do not '
             f'match!{END}')

        with pytest.raises(Exception) as exc_info:
            Estimator.new_synthetic_from_parameters(
                self.params_1d, self.expinfo_1d, snr='blah')
        assert str(exc_info.value) == \
            (f'{RED}The following arguments are invalid:\n'
             f'--> snr\n'
             f'Have a look at the documentation for more info.{END}')

    def test_datapath(self):
        estimator = self.estimator_1d()
        assert estimator.datapath is None
        with pytest.raises(ValueError) as exc_info:
            estimator.datapath = Path('.').resolve()
        assert str(exc_info.value) == f'{RED}`datapath` is not mutable!{END}'

    def test_data(self):
        estimator = self.estimator_1d()
        assert isinstance(estimator.data, np.ndarray)
        with pytest.raises(ValueError) as exc_info:
            estimator.data = np.array([1, 2, 3, 4, 5, 6])
        assert str(exc_info.value) == f'{RED}`data` is not mutable!{END}'

    def test_dim(self):
        estimator = self.estimator_1d()
        assert estimator.dim == 1
        with pytest.raises(ValueError) as exc_info:
            estimator.dim = 2
        assert str(exc_info.value) == f'{RED}`dim` is not mutable!{END}'

    def test_expinfo(self):
        estimator = self.estimator_1d()
        assert isinstance(estimator.expinfo, ExpInfo)
        with pytest.raises(ValueError) as exc_info:
            estimator.expinfo = ExpInfo(pts=2048, sw=1000, offset=1000)
        assert str(exc_info.value) == f'{RED}`expinfo` is not mutable!{END}'

    def test_sw(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            assert estimator.get_sw() == tuple([5000. for _ in range(i)])
            assert estimator.get_sw(unit='ppm') == \
                tuple([10. for _ in range(i)])

            with pytest.raises(_errors.InvalidUnitError) as exc_info:
                estimator.get_sw(unit='blah')
            assert str(exc_info.value) == \
                (f'{RED}unit should be one of the following: \'hz\', '
                 f'\'ppm\'{END}')

            estimator = getattr(self, f'estimator_{i}d')(sfo=False)
            with pytest.raises(_errors.InvalidUnitError) as exc_info:
                estimator.get_sw(unit='ppm')
            assert str(exc_info.value) == \
                f'{RED}unit should be one of the following: \'hz\'{END}'

    def test_offset(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            assert estimator.get_offset() == tuple([2000. for _ in range(i)])
            assert estimator.get_offset(unit='ppm') == \
                tuple([4. for _ in range(i)])

            with pytest.raises(_errors.InvalidUnitError) as exc_info:
                estimator.get_offset(unit='blah')
            assert str(exc_info.value) == \
                (f'{RED}unit should be one of the following: \'hz\', '
                 f'\'ppm\'{END}')

            estimator = getattr(self, f'estimator_{i}d')(sfo=False)
            with pytest.raises(_errors.InvalidUnitError) as exc_info:
                estimator.get_offset(unit='ppm')
            assert str(exc_info.value) == \
                f'{RED}unit should be one of the following: \'hz\'{END}'

    def test_sfo(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            assert estimator.sfo == tuple([500. for _ in range(i)])
            estimator = getattr(self, f'estimator_{i}d')(sfo=False)
            assert estimator.sfo is None

    def test_bf(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            assert estimator.bf == tuple([499.998 for _ in range(i)])
            estimator = getattr(self, f'estimator_{i}d')(sfo=False)
            assert estimator.sfo is None

    def test_nuclei(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            assert estimator.nuclei == tuple(['1H' for _ in range(i)])
            estimator = getattr(self, f'estimator_{i}d')(nuclei=False)
            assert estimator.nuclei is None

    def test_shifts(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            pts = estimator.data.shape
            sw = estimator.get_sw()
            offset = estimator.get_offset()
            sfo = estimator.sfo

            assert np.allclose(
                np.array(estimator.get_shifts()),
                np.array(
                    [np.linspace(sw_ / 2 + offset_, -sw_ / 2 + offset_, pts_)
                     for sw_, offset_, pts_ in zip(sw, offset, pts)]
                ),
                atol=1E-8, rtol=0.
            )

            assert np.allclose(
                np.array(estimator.get_shifts(unit='ppm')),
                np.array(
                    [np.linspace(sw_ / 2 + offset_, -sw_ / 2 + offset_, pts_)
                     for sw_, offset_, pts_ in zip(sw, offset, pts)]
                ) / np.expand_dims(np.array(sfo), axis=-1),
                atol=1E-8, rtol=0.
            )

    def test_timepoints(self):
        for i in range(1, 3):
            estimator = getattr(self, f'estimator_{i}d')()
            pts = estimator.data.shape
            sw = estimator.get_sw()

            assert np.allclose(
                np.array(estimator.timepoints),
                np.array(
                    [np.linspace(0, (pts_ - 1) / sw_, pts_)
                     for sw_, pts_ in zip(sw, pts)]
                ),
                atol=1E-8, rtol=0.
            )

    def test_view_data(self):
        if INTERACT:
            estimator = self.estimator_1d()
            estimator.view_data(domain='frequency')
            estimator.view_data(domain='frequency', freq_xunit='hz')
            estimator.view_data(domain='frequency', component='imag')
            estimator.view_data(domain='frequency', component='both')
            estimator.view_data(domain='time')
            estimator.view_data(domain='time', component='imag')
            estimator.view_data(domain='time', component='both')

    def test_phase(self):
        estimator = self.estimator_1d()
        init_data = estimator.data
        estimator.phase_data(p0=(1.2,), p1=(2.3,))
        assert not np.allclose(estimator.data, init_data)
        estimator.phase_data(p0=(-1.2,), p1=(-2.3,))
        assert np.allclose(estimator.data, init_data)

    def test_manual_phase(self):
        if INTERACT:
            estimator = self.estimator_1d()
            print('p1 should be ±10 radians')
            estimator.manual_phase_data()
            print('p1 should be ±2 radians')
            estimator.manual_phase_data(max_p1=2 * np.pi)


def test_synthetic_estimator():
    # --- Nucleus (doesn't exist for synthetic data) -----------------
    assert estimator.get_nucleus(kill=False) is None
    with pytest.raises(errors.AttributeIsNoneError):
        estimator.get_nucleus()

    # --- Chemical shifts --------------------------------------------
    shifts = np.linspace(
        (sw[0] / 2) + offset[0], (-sw[0] / 2) + offset[0], n[0],
    )

    assert np.array_equal(
        np.round(estimator.get_shifts()[0], decimals=4),
        np.round(shifts, decimals=4),
    )
    assert np.array_equal(
        np.round(estimator.get_shifts(unit='ppm')[0], decimals=4),
        np.round(shifts / sfo[0], decimals=4),
    )

    # --- Time-points ------------------------------------------------
    tp = np.linspace(0., (n[0] - 1) / sw[0], n[0])

    assert np.array_equal(
        np.round(estimator.get_timepoints()[0], decimals=4),
        np.round(tp, decimals=4)
    )

    # --- View data ------------------------------------------------------
    if VIEW_DATA:
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

    region_hz = [[3350., 2800.]]
    noise_region_hz = [[2000., 1500.]]
    region_ppm = [[3350. / sfo[0], 2800. / sfo[0]]]
    noise_region_ppm = [[2000. / sfo[0], 1500. / sfo[0]]]

    # ppm
    estimator.frequency_filter(region_ppm, noise_region_ppm)
    ppm_filter = estimator.get_filter_info()

    # Hz
    estimator.frequency_filter(
        region_hz, noise_region_hz, region_unit='hz'
    )
    hz_filter = estimator.get_filter_info()

    assert hz_filter.get_sw() == ppm_filter.get_sw()
    assert hz_filter.get_offset() == ppm_filter.get_offset()
    assert hz_filter.get_region() == ppm_filter.get_region()
    assert hz_filter.get_noise_region() == ppm_filter.get_noise_region()

    # Ensure that two signals are identical, given some margin of error
    # for noise. Have found that it is incredibly rare for the difference
    # between two points to exceed 0.1
    assert np.allclose(
        hz_filter.cut_fid, ppm_filter.cut_fid, rtol=0, atol=0.1,
    )

    # Ensure filtred signal estmation gives good result
    expected = params[:4]

    # 1. Cut signal. Should give slightly less accurate estimation
    estimator.matrix_pencil()
    estimator.nonlinear_programming(phase_variance=False)
    result = estimator.get_result()
    assert np.allclose(result, expected, atol=0.2, rtol=0)

    # 2. Uncut signal. Should give more accurate estimation
    estimator.frequency_filter(region_ppm, noise_region_ppm, cut=False)
    estimator.matrix_pencil()
    estimator.nonlinear_programming(phase_variance=False)
    result = estimator.get_result()
    assert np.allclose(result, expected, atol=0.1, rtol=0)

    # --- Writing result files ---------------------------------------
    for fmt in ['txt', 'pdf', 'csv']:
        if (fmt in ['txt', 'csv']) or (fmt == 'pdf' and RUN_PDFLATEX):
            estimator.write_result(
                path='./test', description='Testing', fmt=fmt,
                force_overwrite=True, sig_figs=7, sci_lims=(-3, 4),
                fprint=False
            )
        # View output files
        if VIEW_RESULT_FILES:
            if fmt == 'txt':
                subprocess.run(['gedit', 'test.txt'])
                os.remove('test.txt')
            elif fmt == 'pdf' and RUN_PDFLATEX:
                subprocess.run(['evince', 'test.pdf'])
                os.remove('test.pdf')
                os.remove('test.tex')
            elif fmt == 'csv':
                subprocess.run(['libreoffice', 'test.csv'])
                os.remove('test.csv')

        try:
            os.remove(f'test.{fmt}')
            if fmt == 'pdf':
                os.remove('test.tex')
        except Exception:
            pass

    # --- Result plotting --------------------------------------------
    plot = estimator.plot_result()
    plot.fig.savefig('test_default.pdf', dpi=300)
    plot = estimator.plot_result(
        shifts_unit='hz', plot_residual=False, plot_model=True,
        model_shift=100., data_color='#0000ff', oscillator_colors='inferno',
        model_color='#ff0000', show_labels=False,
    )
    plot.fig.savefig('test_custom.pdf', dpi=300)
    if VIEW_RESULT_FIGURES:
        subprocess.run(['evince', 'test_default.pdf'])
        subprocess.run(['evince', 'test_custom.pdf'])

    os.remove('test_default.pdf')
    os.remove('test_custom.pdf')


def test_bruker_estimator():
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
    if VIEW_DATA:
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
    # between two points to exceed 1000
    assert np.allclose(
        hz_filter.cut_fid, ppm_filter.cut_fid, rtol=0, atol=1000,
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

    if MANUAL_PHASE:
        estimator.manual_phase_data()

    # --- Replace data with simple synthetic signal to test estimation ---
    n = [4096]
    sw = [5000.]
    offset = [1000.]
    sfo = [500.]

    params = np.array([
        [1, 0, offset[0] - (sw[0] / 8), 1],
        [3, 0, offset[0] - (sw[0] / 16), 1],
        [6, 0, offset[0], 1],
        [3, 0, offset[0] + (sw[0] / 16), 1],
        [1, 0, offset[0] + (sw[0] / 8), 1],
    ])

    params = params[np.argsort(params[:, 2])]
    fid = sig.make_fid(params, n, sw, offset=offset, snr=30.)[0]
    estimator.data = fid
    estimator.sw = sw
    estimator.offset = offset
    estimator.sfo = sfo
    estimator.frequency_filter(
        region=[[200., 1800.]], noise_region=[[-1000., -500.]],
        region_unit='hz',
    )

    # --- Matrix Pencil ----------------------------------------------
    # With specified number of params
    estimator.matrix_pencil(M=5, fprint=False)
    res = estimator.get_result()
    assert np.allclose(res, params, rtol=0, atol=2E-2)

    for method, word in zip(
        (estimator.plot_result, estimator.write_result),
        ('plotting', 'saving')
    ):
        with pytest.raises(ValueError) as exc_info:
            method()

        assert str(exc_info.value) == \
            (f'{ORA}The last action to be applied to the estimation '
             f'result was not `nonlinear_programming`. You should ensure '
             f'this is so before {word} the result.{END}')

    # --- NonlinearProgramming ---------------------------------------
    # create initial guess
    estimator.result += 0.4 * nrandom.random_sample(size=(5, 4)) - 0.2
    estimator.nonlinear_programming(
        phase_variance=False, max_iterations=1000, fprint=False,
    )
    res = estimator.get_result()
    assert np.allclose(res, params, rtol=0, atol=2E-2)

    # --- Writing result files ---------------------------------------
    for fmt in ['txt', 'pdf', 'csv']:
        if (fmt in ['txt', 'csv']) or (fmt == 'pdf' and RUN_PDFLATEX):
            estimator.write_result(
                path='./test', description='Testing', fmt=fmt,
                force_overwrite=True, sig_figs=7, sci_lims=(-3, 4),
                fprint=False
            )
        # View output files
        if VIEW_RESULT_FILES:
            if fmt == 'txt':
                subprocess.run(['gedit', 'test.txt'])
                os.remove('test.txt')
            elif fmt == 'pdf' and RUN_PDFLATEX:
                subprocess.run(['evince', 'test.pdf'])
                os.remove('test.pdf')
                os.remove('test.tex')
            elif fmt == 'csv':
                subprocess.run(['libreoffice', 'test.csv'])
                os.remove('test.csv')

    # --- Check result array-amending methods-------------------------
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
    if VIEW_RESULT_FILES:
        subprocess.run(['gedit', 'test.log'])
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
