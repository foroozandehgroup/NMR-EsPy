import numpy as np
from nmrespy import sig


def test_get_timepoints():
    # --- 1D ---
    n = [10]
    sw = [500.]
    test = np.arange(n[0]) / sw[0]
    # start at t = 0
    assert np.array_equal(sig.get_timepoints(n, sw)[0], test)

    # non-zero start times, using 'dt' notation
    pts = (-4, 6)
    sts = [[f'{p}dt'] for p in pts]
    for pt, st in zip(pts, sts):
        assert np.array_equal(
            sig.get_timepoints(n, sw, start_time=st)[0],
            test + (pt / sw[0])
        )

    # non-zero start times, using floats
    for st in ([0.03], [-0.04]):
        assert np.array_equal(
            sig.get_timepoints(n, sw, start_time=st)[0],
            test + st
        )

    # --- 2D ---
    # For these I was getting errors if I didn't round the values, I assume
    # due to imprecision of floating point arithmetic.
    n = [10, 15]
    sw = [500., 1000.]
    test = [np.arange(n_) / sw_ for n_, sw_ in zip(n, sw)]

    # start at t = 0
    assert all([np.array_equal(np.round(a, 3), np.round(b, 3))
                for a, b in zip(sig.get_timepoints(n, sw), test)])

    # non-zero start times, using 'dt' notation
    start_time = ['-4dt', '6dt']
    test2 = [t + i / sw_ for i, t, sw_ in zip((-4, 6), test, sw)]
    assert all(
        [np.array_equal(np.round(a, 3), np.round(b, 3))
         for a, b in zip(
            sig.get_timepoints(n, sw, start_time=start_time), test2
        )]
    )

    # non-zero start times, using floats
    start_time = [0.003, -0.005]
    test3 = [t + i for i, t in zip(start_time, test)]
    assert all(
        [np.array_equal(np.round(a, 3), np.round(b, 3))
         for a, b in zip(
            sig.get_timepoints(n, sw, start_time=start_time), test3
        )]
    )
