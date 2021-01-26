# nmrespy.timing
# timing.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# Provides timer decorator

import functools
import time

def timer(f):
    """Times function f, and prints result once completed."""
    @functools.wraps(f)
    def timed(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        run_time = convert(time.time() - start)
        print(f'\tTime elapsed: {run_time}')
        return result
    return timed


def convert(secs):
    """Takes a time in seconds and converts to min:sec:msec"""
    mins = int(secs // 60)
    secs %= 60
    msecs = int(round(((secs - int(secs)) * 1000)))
    secs = int(secs)

    return f'{mins} mins, {secs} secs, {msecs} msecs'

# ————————————————————————————————————————————————————————————————————
# “The story so far:
# In the beginning the Universe was created.
# This has made a lot of people very angry and been widely regarded as
# a bad move.”
# —————————Douglas Adams, The Restaurant at the End of the Universe———

# TODO
# to be deprecated
import numpy as np

def _print_time(time):
    """
    Takes in a float of number of seconds.
    Returns the number of mins, seconds and milliseconds as ints
    """

    mins = int(np.floor((time) / 60))
    secs = int(np.round((time) - (mins * 60)))
    millisecs = int(np.round((time) - np.floor(time), 3) * 1000)

    if mins > 0:
        print(f'Time taken: {mins}:{secs}\n')
    else:
        print(f'Time taken: {secs}.{millisecs}s\n')
