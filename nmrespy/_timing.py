# _timing.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 07 Oct 2021 12:12:06 BST

"""Support for timing routines."""

import functools
import time


def timer(f):
    """Time function f, and prints result once completed."""

    @functools.wraps(f)
    def timed(*args, **kwargs):
        start = time.time()
        if not args[0].fprint:
            return f(*args, **kwargs)
        result = f(*args, **kwargs)
        run_time = convert(time.time() - start)
        print(f"Time elapsed: {run_time}")
        return result

    return timed


def convert(time: float) -> str:
    """Take a time in seconds and convert to formatted string.

    Parameters
    ----------
    time
        Time in seconds

    Returns
    -------
    formatted_time
        Format of the string is ``f'{min} mins, {s} secs, {ms} msecs'``
        where ``min``, ``s``, and ``ms`` are the number of minutes, seconds
        and millisconds, respectively.
    """
    min = int(time // 60)
    s = time % 60
    ms = int(round(((s - int(s)) * 1000)))
    s = int(s)

    return f"{min} mins, {s} secs, {ms} msecs"
