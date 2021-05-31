# _timing.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Provides timer decorator"""

import functools
import time


def timer(f):
    """Times function f, and prints result once completed."""
    @functools.wraps(f)
    def timed(*args, **kwargs):
        start = time.time()
        if not args[0].fprint:
            return f(*args, **kwargs)
        result = f(*args, **kwargs)
        run_time = convert(time.time() - start)
        print(f'Time elapsed: {run_time}')
        return result
    return timed


def convert(secs):
    """Takes a time in seconds and converts to min:sec:msec"""
    mins = int(secs // 60)
    secs %= 60
    msecs = int(round(((secs - int(secs)) * 1000)))
    secs = int(secs)

    return f'{mins} mins, {secs} secs, {msecs} msecs'
