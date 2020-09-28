#!/usr/bin/python3

import os
import tkinter as tk

import nmrespy
from nmrespy import load
from nmrespy.topspin.topspin_setup import MainSetup
from nmrespy.topspin.topspin_result import MainResult

if __name__ == '__main__':
    # path to nmrespy directory
    espypath = os.path.dirname(nmrespy.__file__)

    # extract path information
    infopath = os.path.join(espypath, 'topspin/tmp/info.txt')
    try:
        with open(infopath, 'r') as fh:
            from_topspin = fh.read().split(' ')
    except:
        raise IOError(f'No file of path {infopath} found')

    # import dictionary of spectral info
    fidpath = from_topspin[0]
    pdatapath = from_topspin[1]

    # determine the data type to consider (FID or pdata)
    root = tk.Tk()
    root.title('NMR-EsPy - Calculation Setup')
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    window = MainSetup(root, fidpath, pdatapath)
    root.mainloop()


    tmpdir = os.path.join(espypath, 'topspin/tmp')
    info = load.pickle_load('tmp.pkl', tmpdir)

    # load the result GUI
    root = tk.Tk()
    res_app = MainResult(root, info)
    root.mainloop()

    # construct save files
    descrip = res_app.descrip
    file = res_app.file
    dir = res_app.dir

    txt = res_app.txt
    pdf = res_app.pdf
    pickle = res_app.pickle


    if txt == '1':
        info.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True)
    if pdf == '1':
        info.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True, format='pdf')
    if pickle == '1':
        info.pickle_save(fname=file, dir=dir, force_overwrite=True)
