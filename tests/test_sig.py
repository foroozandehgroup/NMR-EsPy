# test_sig.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 28 Jan 2022 18:04:45 GMT

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy.signal import argrelextrema
from nmrespy import ExpInfo, sig


MANUAL_PHASE = True


def same(x, y):
    return np.allclose(x, y, atol=1e-10, rtol=0)


def simple_fid(dim=1):
    pts = dim * [1024]
    sw = 2000
    expinfo = ExpInfo(sw=sw, dim=dim)
    if dim == 1:
        params = np.array([[1.0, 0.0, 500.0, 10.0]])
    elif dim == 2:
        params = np.array([[1.0, 0.0, 500.0, 500.0, 10.0, 10.0]])
    return sig.make_fid(params, expinfo, pts)[0]


def test_make_fid():
    # --- 1D ---
    pts = [2048]
    sw = 2000
    expinfo = ExpInfo(sw=sw)

    # Simple FID with one oscillator
    params = np.array([[1.0, 0.0, 500.0, 10.0]])
    fid, tp = sig.make_fid(params, expinfo, pts)
    assert np.round(fid[0], 3) == 1.0 + 0.0 * 1j
    # N.B. Signal is at 500Hz with range of -1000 to 1000Hz, i.e. 3/4 along
    assert np.argmax(fftshift(fft(fid))) == int(0.75 * 2048)
    assert np.array_equal(tp[0], np.linspace(0, (pts[0] - 1) / sw, pts[0]))

    # Phase of π/2. Amplitude of first point should be 0 + i
    params[:, 1] = np.pi / 2
    fid, _ = sig.make_fid(params, expinfo, pts)
    assert np.round(fid[0], 3) == 0.0 + 1.0 * 1j

    # Three oscillators
    params = np.array(
        [[1.0, 0.0, -250.0, 10.0], [2.0, 0.0, 250.0, 10.0], [1.0, 0.0, 500.0, 10.0]]
    )
    fid, _ = sig.make_fid(params, expinfo, pts)
    assert np.round(fid[0], 3) == 4.0 + 0.0 * 1j
    # Peaks should be 3/8, 5/8 and 3/4 along the spectrum
    spectrum = fftshift(fft(fid))
    assert [x for x in argrelextrema(spectrum, np.greater)[0]] == [
        int(i / 8 * 2048) for i in (3, 5, 6)
    ]
    # Maximum should be the peak that is 5/8 along
    assert np.argmax(spectrum) == int(5 / 8 * 2048)

    # Include an offset and check signal is identical
    offset = 200.0
    expinfo = ExpInfo(sw=sw, offset=offset)
    params[:, 2] += offset
    fid_with_offset, _ = sig.make_fid(params, expinfo, pts)
    assert np.array_equal(fid, fid_with_offset)

    # Make noisy version
    snr = 30.0
    expected_std = np.std(np.abs(fid)) / (10 ** (snr / 20))
    noisy_fid, _ = sig.make_fid(params, expinfo, pts, snr=snr)
    assert np.allclose(np.std(np.abs(noisy_fid - fid)), expected_std, rtol=0, atol=0.01)

    # --- 2D ---
    pts = [2048, 2048]
    sw, offset = 2000.0, 1000.0

    # Single oscillator
    params = np.array([[1.0, 0.0, 750.0, 1250.0, 10.0, 10.0]])
    expinfo = ExpInfo(sw=sw, offset=offset, dim=2)
    fid, tp = sig.make_fid(params, expinfo, pts)
    assert fid[0, 0] == 1.0 + 0.0 * 1j
    # Take absolute value of spectrum, as phase-twisted
    assert (
        np.argmax(abs_spectrum_2d(fid)) ==
        pts[0] * int(0.375 * pts[0]) + int(0.625 * pts[0])
    )
    assert np.array_equal(tp[0], tp[1])
    assert np.array_equal(tp[0], np.linspace(0, (pts[0] - 1) / sw, pts[0]))

    # Three oscillators
    params = np.array(
        [
            [1.0, 0.0, 700.0, 1300.0, 10.0, 10.0],
            [2.0, 0.0, 1500.0, 500.0, 10.0, 10.0],
            [1.0, 0.0, 500.0, 1100.0, 10.0, 10.0],
        ]
    )
    fid, _ = sig.make_fid(params, expinfo, pts)
    assert fid[0, 0] == 4.0 + 0.0 * 1j
    assert (
        np.argmax(abs_spectrum_2d(fid)) ==
        pts[0] * int(0.75 * pts[0]) + int(0.25 * pts[0])
    )

    # TODO modulation stuff...


def abs_spectrum_2d(fid):
    return np.abs(fftshift(fft(fft(fid, axis=0), axis=1)))


def test_virtual_echo():
    # --- 1D ---
    fid = simple_fid()
    ve = sig.make_virtual_echo([fid])
    assert same(np.imag(fftshift(fft(ve))), np.zeros(ve.shape))
    assert ve.size == 2 * fid.size - 1
    assert np.array_equal(ve[1 : fid.size], fid[1:])
    assert ve[0] == np.real(fid[0]) + 0.0 * 1j
    assert np.array_equal(ve[fid.size :], fid.conj()[1:][::-1])

    # TODO 2D


def test_get_timepoints():
    # --- 1D ---
    pts = [10]
    sw = 500
    expinfo = ExpInfo(sw=500)
    test = np.arange(pts[0]) / sw
    # start at t = 0
    assert np.array_equal(sig.get_timepoints(expinfo, pts)[0], test)

    # non-zero start times, using 'dt' notation
    for pt, st in zip((-4, 6), (["-4dt"], ["6dt"])):
        assert np.array_equal(
            sig.get_timepoints(expinfo, pts, start_time=st)[0], test + (pt / sw),
        )

    # non-zero start times, using floats
    for st in ([0.03], [-0.04]):
        assert np.array_equal(
            sig.get_timepoints(expinfo, pts, start_time=st)[0], test + st
        )

    # --- 2D ---
    # For these I was getting errors if I didn't round the values, I assume
    # due to imprecision of floating point arithmetic.
    pts = [10, 15]
    sw = [500, 1000]
    expinfo = ExpInfo(sw=sw)
    test = [np.arange(pts_) / sw_ for pts_, sw_ in zip(pts, sw)]

    # start at t = 0
    assert all(same(a, b) for a, b in zip(
        sig.get_timepoints(expinfo, pts, meshgrid_2d=False),
        test,
    ))

    # non-zero start times, using 'dt' notation
    start_time = ["-4dt", "6dt"]
    test2 = [t + i / sw_ for i, t, sw_ in zip((-4, 6), test, sw)]
    assert all(
        same(a, b)
        for a, b in zip(
            sig.get_timepoints(expinfo, pts, start_time=start_time, meshgrid_2d=False),
            test2,
        )
    )

    # non-zero start times, using floats
    start_time = [0.003, -0.005]
    test3 = [t + i for i, t in zip(start_time, test)]
    assert all(
        same(a, b)
        for a, b in zip(
            sig.get_timepoints(expinfo, pts, start_time=start_time, meshgrid_2d=False),
            test3,
        )
    )


def test_get_shifts():
    # --- 1D ---
    pts = [10]
    sw, offset, sfo = 500, 100, 500
    expinfo = ExpInfo(sw=sw, offset=offset, sfo=sfo)
    test = np.linspace(sw / 2 + offset, -sw / 2 + offset, pts[0])
    assert same(sig.get_shifts(expinfo, pts)[0], test)
    assert same(sig.get_shifts(expinfo, pts, unit="ppm")[0], test / sfo)
    # Check flipping
    assert same(
        sig.get_shifts(expinfo, pts, flip=True)[0],
        sig.get_shifts(expinfo, pts, flip=False)[0][::-1],
    )

    # --- 2D ---
    pts = [10, 20]
    sw, offset, sfo = (500, 1000), (100, -100), (500, 125)
    expinfo = ExpInfo(sw=sw, offset=offset, sfo=sfo)
    test = tuple(
        [
            np.linspace(sw_ / 2 + offset_, -sw_ / 2 + offset_, pts_)
            for sw_, offset_, pts_ in zip(sw, offset, pts)
        ]
    )

    shifts_hz = sig.get_shifts(expinfo, pts, meshgrid_2d=False)
    assert all(isinstance(x, tuple) for x in (test, shifts_hz))
    for a, b in zip(test, shifts_hz):
        assert same(a, b)

    shifts_ppm = sig.get_shifts(expinfo, pts, unit="ppm", meshgrid_2d=False)
    for a, b, sfo_ in zip(test, shifts_ppm, sfo):
        assert same(a / sfo_, b)

    shifts_no_flip = sig.get_shifts(expinfo, pts, flip=False, meshgrid_2d=False)
    for a, b in zip(test, shifts_no_flip):
        assert same(a[::-1], b)


def test_ft():
    # --- 1D ---
    fid = simple_fid()
    assert same(sig.ft(fid), fftshift(fft(fid))[::-1])
    assert same(sig.ft(fid, flip=False), fftshift(fft(fid)))

    # --- 2D ---
    fid = simple_fid(dim=2)
    assert same(sig.ft(fid), fftshift(fft(fft(fid, axis=0), axis=1))[::-1, ::-1])
    assert same(sig.ft(fid, flip=False), fftshift(fft(fft(fid, axis=0), axis=1)))


def test_ift():
    # --- 1D ---
    spectrum = sig.ft(simple_fid())
    assert same(sig.ift(spectrum), ifft(ifftshift(spectrum[::-1])))
    assert same(sig.ift(spectrum, flip=False), ifft(ifftshift(spectrum)))

    # --- 2D ---
    spectrum = sig.ft(simple_fid(dim=2))
    assert same(
        sig.ift(spectrum), ifft(ifft(ifftshift(spectrum[::-1, ::-1]), axis=0), axis=1)
    )
    assert same(
        sig.ift(spectrum, flip=False), ifft(ifft(ifftshift(spectrum), axis=0), axis=1)
    )


def test_phase():
    fid = simple_fid()
    # 90 degree phase. First point should go from (1 + 0i) -> (0 + i)
    assert same(sig.phase(fid, p0=[np.pi / 2], p1=[0.0])[0], 0.0 + 1j)
    # Check two opposing phasings recover the original signal
    assert same(
        fid, sig.phase(sig.phase(fid, p0=[1.24], p1=[-0.15]), p0=[-1.24], p1=[0.15])
    )

    # TODO: 2D


# def test_maunal_phase():
#     if MANUAL_PHASE:
#         spectrum = sig.ft(simple_fid())
#         p0, p1 = sig.manual_phase_spectrum(spectrum)
#         sig.phase(spectrum, [p0], [p1])


def test_oscillator_integral():
    # --- 1D ---
    pts = [300000]
    sw = 2000
    expinfo = ExpInfo(sw=sw)
    params = np.array([1.0, 0.0, 500.0, 10.0])
    integral1 = sig.oscillator_integral(params, expinfo, pts)
    params[0] *= 2.0
    integral2 = sig.oscillator_integral(params, expinfo, pts)
    assert same(2.0 * integral1, integral2)
    params[1] = np.pi
    integral3 = sig.oscillator_integral(params, expinfo, pts, abs_=False)
    assert same(-1.0 * integral2, integral3)

    # TODO 2D


def test_random_signal():
    # --- 1D ---
    pts = [4048]
    sw, offset = 2000.0, 1000.0
    expinfo = ExpInfo(sw=sw, offset=offset)
    fid, tp, params = sig._generate_random_signal(10, expinfo, pts)
    assert params.shape == (10, 4)
    assert all(-sw / 2 + offset <= f <= sw / 2 + offset for f in params[:, 2])
    assert same(tp[0], np.linspace(0, (pts[0] - 1) / sw, pts[0]))
    assert same(sig.make_fid(params, expinfo, pts)[0], fid)

    # --- 2D ---
    pts = [128, 128]
    sw, offset = 2000.0, 1000.0
    expinfo = ExpInfo(sw=sw, offset=offset, dim=2)
    fid, tp, params = sig._generate_random_signal(10, expinfo, pts)
    assert params.shape == (10, 6)
    assert all(
        -sw / 2 + offset <= f <= sw / 2 + offset for f in params[:, 2:4].flatten()
    )
    assert same(tp[0], tp[1])
    assert same(tp[0], np.linspace(0, (pts[0] - 1) / sw, pts[0]))
    assert same(sig.make_fid(params, expinfo, pts)[0], fid)


def test_zf():
    fid = simple_fid()
    fid = fid[:900]
    fid = sig.zf(fid)
    assert fid.size == 1024
    assert np.all((fid[900:] == 0.0 + 0.0j))
    assert fid[899] != 0.0 + 0.0j


# TODO:
# --- More rigourous 2D checking in general ---
# test_proc_amp_modulated
# test_proc_phase_modulated
