#!/usr/bin/python3
import os
from nmrespy.topspin.apps import TOPSPINDIR, SetupApp

if __name__ == '__main__':
    # extract path information
    infopath = os.path.join(TOPSPINDIR, 'info.txt')

    with open(infopath, 'r') as fh:
        fidpath, pdatapath = fh.read().split(' ')

    setup_app = SetupApp(fidpath, pdatapath)
    setup_app.mainloop()
