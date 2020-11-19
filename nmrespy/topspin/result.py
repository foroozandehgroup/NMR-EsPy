#!/usr/bin/python3

import os

from nmrespy.load import pickle_load
from nmrespy.topspin.apps import TOPSPINDIR, ResultApp

if __name__ == '__main__':

    info = pickle_load(os.path.join(TOPSPINDIR, 'tmp.pkl'))
    # os.remo ve(os.path.join(TOPSPINDIR, 'tmp.pkl'))
    result_app = ResultApp(info)
    result_app.mainloop()
