# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 14 Oct 2022 15:27:27 BST

# This is currently only applicable to 1D NMR data.

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from nmrespy import Estimator1D
from nmrespy.app import config as cf
from nmrespy.app.frames import DataType  # TODO: WaitingWindow
from nmrespy.app.stup import SetUp1D
from nmrespy.app.result import Result1D

import matplotlib as mpl
mpl.rcParams["text.usetex"] = False


class NMREsPyApp(tk.Tk):
    """App for using NMR-EsPy.

    path : str
        Path to the specified directory.

    topspin : bool
        Indicates whether or not the app has been invoked via TopSpin
        or not. If it has, the type of data (fid or pdata) the user wishes
        to use will have to be ascertained, using the `DataType` window.
    """

    # This is the "controller"
    # When you see `self.ctrl` in other classes in code for the app, it refers
    # to this class

    def __init__(self, path, res, topspin=False, pdflatex=None):
        super().__init__()
        # Hide the root app window. This is not going to be used. Everything
        # will be built onto Toplevels
        self.withdraw()

        self.pdflatex = pdflatex

        style = ttk.Style(self)
        style.theme_settings(
            "default",
            settings={
                "TNotebook": {
                    "configure": {
                        "tabmargins": [2, 0, 5, 0],
                        "background": cf.BGCOLOR,
                        "bordercolor": "black",
                    }
                },
                "TNotebook.Tab": {
                    "configure": {
                        "padding": [10, 3],
                        "background": cf.NOTEBOOKCOLOR,
                        "font": (cf.MAINFONT, 11),
                    },
                    "map": {
                        "background": [("selected", cf.ACTIVETABCOLOR)],
                        "expand": [("selected", [1, 1, 1, 0])],
                        "font": [("selected", (cf.MAINFONT, 11, "bold"))],
                        "foreground": [("selected", "white")],
                    },
                },
            },
        )
        style.configure("Region.TNotebook", background=cf.NOTEBOOKCOLOR)

        if topspin:
            # Open window to ask user for data type (fid or pdata)
            # from this, self acquires the attirbutes dtype and path
            path = Path(path)
            paths = {"pdata": path, "fid": path.parent.parent}
            data_type_window = DataType(self, paths)
            path = data_type_window.path

        if res:
            # Wish to view result from a previously-generated result.
            # Jump straight to the result window.
            self.estimator = Estimator1D.from_pickle(path)
            self.result()

        else:
            # Create Estimator instance from the provided path
            self.estimator = Estimator1D.new_bruker(path)
            self.setup_window = SetUp1D(self)
            self.wait_window(self.setup_window)

            # TODO: animation window
            # self.waiting_window = WaitingWindow(self)
            # self.waiting_window.withdraw()

    def result(self):
        self.result_window = Result1D(self)
        self.wait_window(self.result_window)
