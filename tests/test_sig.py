import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy.signal import argrelextrema

from nmrespy import ExpInfo, sig


MANUAL_PHASE = True


def same(x, y):
    return np.allclose(x, y, atol=1E-10, rtol=0)


def simple_fid(dim=1):
    pts, sw = 1024, 2000
    expinfo = ExpInfo(pts=pts, sw=sw, dim=dim)
    if dim == 1:
        params = np.array([[1., 0., 500., 10.]])
    elif dim == 2:
        params = np.array([[1., 0., 500., 500., 10., 10.]])
    return sig.make_fid(params, expinfo)[0]

    fid, _ = sig.make_fid(params, expinfo)
    return fid


def test_make_fid():
    # --- 1D ---
    pts, sw = 2048, 2000
    expinfo = ExpInfo(pts=pts, sw=sw)

    # Simple FID with one oscillator
    params = np.array([[1., 0., 500., 10.]])
    fid, tp = sig.make_fid(params, expinfo)
    assert np.round(fid[0], 3) == 1. + 0. * 1j
    # N.B. Signal is at 500Hz with range of -1000 to 1000Hz, i.e. 3/4 along
    assert np.argmax(fftshift(fft(fid))) == int(0.75 * 2048)
    assert np.array_equal(tp[0], np.linspace(0, (pts - 1) / sw, pts))

    # Phase of π/2. Amplitude of first point should be 0 + i
    params[:, 1] = np.pi / 2
    fid, _ = sig.make_fid(params, expinfo)
    assert np.round(fid[0], 3) == 0. + 1. * 1j

    # Three oscillators
    params = np.array([
        [1., 0., -250., 10.],
        [2., 0., 250., 10.],
        [1., 0., 500., 10.]
    ])
    fid, _ = sig.make_fid(params, expinfo)
    assert np.round(fid[0], 3) == 4. + 0. * 1j
    # Peaks should be 3/8, 5/8 and 3/4 aling the spectrum
    spectrum = fftshift(fft(fid))
    assert [x for x in argrelextrema(spectrum, np.greater)[0]] == \
        [int(i / 8 * 2048) for i in (3, 5, 6)]
    # Maximum should be the peak that is 5/8 along
    assert np.argmax(spectrum) == int(5 / 8 * 2048)

    # Include an offset and check signal is identical
    offset = 200.
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset)
    params[:, 2] += offset
    fid_with_offset, _ = sig.make_fid(params, expinfo)
    assert np.array_equal(fid, fid_with_offset)

    # Make noisy version
    snr = 30.
    expected_std = np.std(np.abs(fid)) / (10 ** (snr / 20))
    noisy_fid, _ = sig.make_fid(params, expinfo, snr=snr)
    assert np.allclose(
        np.std(np.abs(noisy_fid - fid)), expected_std, rtol=0, atol=0.01
    )

    # --- 2D ---
    pts, sw, offset = 2048, 2000., 1000.

    # Single oscillator
    params = np.array([[1., 0., 750., 1250., 10., 10.]])
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, dim=2)
    fid, tp = sig.make_fid(params, expinfo)
    assert fid[0, 0] == 1. + 0. * 1j
    # Take absolute value of spectrum, as phase-twisted
    assert np.argmax(abs_spectrum_2d(fid)) == \
        pts * int(0.375 * pts) + int(0.625 * pts)
    assert np.array_equal(tp[0], tp[1])
    assert np.array_equal(tp[0], np.linspace(0, (pts - 1) / sw, pts))

    # Three oscillators
    params = np.array([
        [1., 0., 700., 1300., 10., 10.],
        [2., 0., 1500., 500., 10., 10.],
        [1., 0., 500., 1100., 10., 10.],
    ])
    fid, _ = sig.make_fid(params, expinfo)
    assert fid[0, 0] == 4. + 0. * 1j
    assert np.argmax(abs_spectrum_2d(fid)) == \
        pts * int(0.75 * pts) + int(0.25 * pts)

    # TODO modulation stuff...


def abs_spectrum_2d(fid):
    return np.abs(fftshift(fft(fft(fid, axis=0), axis=1)))


def test_virtual_echo():
    # --- 1D ---
    fid = simple_fid()
    ve = sig.make_virtual_echo([fid])
    assert same(np.imag(fftshift(fft(ve))), np.zeros(ve.shape))
    assert ve.size == 2 * fid.size - 1
    assert np.array_equal(ve[1:fid.size], fid[1:])
    assert ve[0] == np.real(fid[0]) + 0. * 1j
    assert np.array_equal(ve[fid.size:], fid.conj()[1:][::-1])

    # TODO 2D


def test_get_timepoints():
    # --- 1D ---
    pts, sw = 10, 500
    expinfo = ExpInfo(pts=10, sw=500)
    test = np.arange(pts) / sw
    # start at t = 0
    assert np.array_equal(sig.get_timepoints(expinfo)[0], test)

    # non-zero start times, using 'dt' notation
    pts = (-4, 6)
    sts = [[f'{p}dt'] for p in pts]
    for pt, st in zip(pts, sts):
        assert np.array_equal(
            sig.get_timepoints(expinfo, start_time=st)[0],
            test + (pt / sw)
        )

    # non-zero start times, using floats
    for st in ([0.03], [-0.04]):
        assert np.array_equal(
            sig.get_timepoints(expinfo, start_time=st)[0],
            test + st
        )

    # --- 2D ---
    # For these I was getting errors if I didn't round the values, I assume
    # due to imprecision of floating point arithmetic.
    pts, sw = [10, 15], [500, 1000]
    expinfo = ExpInfo(pts=pts, sw=sw)
    test = [np.arange(pts_) / sw_ for pts_, sw_ in zip(pts, sw)]

    # start at t = 0
    assert all(same(a, b) for a, b in zip(sig.get_timepoints(expinfo), test))

    # non-zero start times, using 'dt' notation
    start_time = ['-4dt', '6dt']
    test2 = [t + i / sw_ for i, t, sw_ in zip((-4, 6), test, sw)]
    assert all(same(a, b) for a, b in zip(
        sig.get_timepoints(expinfo, start_time=start_time), test2
    ))

    # non-zero start times, using floats
    start_time = [0.003, -0.005]
    test3 = [t + i for i, t in zip(start_time, test)]
    assert all(same(a, b) for a, b in zip(
        sig.get_timepoints(expinfo, start_time=start_time), test3
    ))


def test_get_shifts():
    # --- 1D ---
    pts, sw, offset, sfo = 10, 500, 100, 500
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)
    test = np.linspace(sw / 2 + offset, -sw / 2 + offset, pts)
    assert same(sig.get_shifts(expinfo)[0], test)
    assert same(sig.get_shifts(expinfo, unit='ppm')[0], test / sfo)
    # Check flipping
    assert same(
        sig.get_shifts(expinfo, flip=True)[0],
        sig.get_shifts(expinfo, flip=False)[0][::-1],
    )

    # --- 2D ---
    pts, sw, offset, sfo = (10, 20), (500, 1000), (100, -100), (500, 125)
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo)
    test = tuple([np.linspace(sw_ / 2 + offset_, -sw_ / 2 + offset_, pts_)
                  for sw_, offset_, pts_ in zip(sw, offset, pts)])

    shifts_hz = sig.get_shifts(expinfo)
    assert all(isinstance(x, tuple) for x in (test, shifts_hz))
    for a, b in zip(test, shifts_hz):
        assert same(a, b)

    shifts_ppm = sig.get_shifts(expinfo, unit='ppm')
    for a, b, sfo_ in zip(test, shifts_ppm, sfo):
        assert same(a / sfo_, b)

    shifts_no_flip = sig.get_shifts(expinfo, flip=False)
    for a, b in zip(test, shifts_no_flip):
        assert same(a[::-1], b)


def test_ft():
    # --- 1D ---
    fid = simple_fid()
    assert same(sig.ft(fid), fftshift(fft(fid))[::-1])
    assert same(sig.ft(fid, flip=False), fftshift(fft(fid)))

    # --- 2D ---
    fid = simple_fid(dim=2)
    assert same(
        sig.ft(fid),
        fftshift(fft(fft(fid, axis=0), axis=1))[::-1, ::-1]
    )
    assert same(
        sig.ft(fid, flip=False),
        fftshift(fft(fft(fid, axis=0), axis=1))
    )


def test_ift():
    # --- 1D ---
    spectrum = sig.ft(simple_fid())
    assert same(sig.ift(spectrum), ifft(ifftshift(spectrum[::-1])))
    assert same(sig.ift(spectrum, flip=False), ifft(ifftshift(spectrum)))

    # --- 2D ---
    spectrum = sig.ft(simple_fid(dim=2))
    assert same(
        sig.ift(spectrum),
        ifft(ifft(ifftshift(spectrum[::-1, ::-1]), axis=0), axis=1)
    )
    assert same(
        sig.ift(spectrum, flip=False),
        ifft(ifft(ifftshift(spectrum), axis=0), axis=1)
    )


def test_phase():
    fid = simple_fid()
    # 90 degree phase. First point should go from (1 + 0i) -> (0 + i)
    assert same(sig.phase(fid, p0=[np.pi / 2], p1=[0.])[0], 0. + 1j)
    # Check two opposing phasings recover the original signal
    assert same(
        fid,
        sig.phase(sig.phase(fid, p0=[1.24], p1=[-0.15]), p0=[-1.24], p1=[0.15])
    )

    # TODO: 2D


def test_maunal_phase():
    if MANUAL_PHASE:
        spectrum = sig.ft(simple_fid())
        p0, p1 = sig.manual_phase_spectrum(spectrum)
        sig.phase(spectrum, [p0], [p1])


def test_oscillator_integral():
    # --- 1D ---
    pts, sw = 300000, 2000
    expinfo = ExpInfo(pts=pts, sw=sw)
    params = np.array([1., 0., 500., 10.])
    integral1 = sig.oscillator_integral(params, expinfo)
    params[0] *= 2.
    integral2 = sig.oscillator_integral(params, expinfo)
    assert same(2. * integral1, integral2)
    params[1] = np.pi
    integral3 = sig.oscillator_integral(params, expinfo, abs_=False)
    assert same(-1. * integral2, integral3)

    # TODO 2D


def test_random_signal():
    # --- 1D ---
    pts, sw, offset = 4048, 2000., 1000.
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset)
    fid, tp, params = sig._generate_random_signal(10, expinfo)
    assert params.shape == (10, 4)
    assert all(-sw / 2 + offset <= f <= sw / 2 + offset for f in params[:, 2])
    assert same(tp[0], np.linspace(0, (pts - 1) / sw, pts))
    assert same(sig.make_fid(params, expinfo)[0], fid)

    # --- 2D ---
    pts, sw, offset = 128, 2000., 1000.
    expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, dim=2)
    fid, tp, params = sig._generate_random_signal(10, expinfo)
    assert params.shape == (10, 6)
    assert all(-sw / 2 + offset <= f <= sw / 2 + offset
               for f in params[:, 2:4].flatten())
    assert same(tp[0], tp[1])
    assert same(tp[0], np.linspace(0, (pts - 1) / sw, pts))
    assert same(sig.make_fid(params, expinfo)[0], fid)


# TODO:
# --- More rigourous 2D checking in general ---
# test_proc_amp_modulated
# test_proc_phase_modulated
