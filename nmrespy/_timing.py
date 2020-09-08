# nmrespy.timing
# timing.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# Provides _print_time(), for formatted time output to terminal.

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

# ————————————————————————————————————————————————————————————————————
# “The story so far:
# In the beginning the Universe was created.
# This has made a lot of people very angry and been widely regarded as
# a bad move.”
# —————————Douglas Adams, The Restaurant at the End of the Universe———
