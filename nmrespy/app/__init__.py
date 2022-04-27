# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 26 Apr 2022 12:24:56 BST

# This is currently only applicable to 1D NMR data.

import pathlib
import tkinter as tk

from nmrespy import Estimator1D
from nmrespy.app.frames import DataType  # TODO: WaitingWindow
from nmrespy.app import stup, result


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
        path = pathlib.Path(path)
        self.pdflatex = pdflatex

        if topspin:
            # Open window to ask user for data type (fid or pdata)
            # from this, self acquires the attirbutes dtype and path
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

            # TODO: animation window
            # self.waiting_window = WaitingWindow(self)
            # self.waiting_window.withdraw()

            self.setup_window = stup.SetUp(self)
            # hold at this point
            # relieved once setup is destroyed
            # see SetUp.run()
            self.wait_window(self.setup_window)

        # TODO
        # For some reason, the program hangs after destroy call
        # ie still in mainloop, even though I have apparently destroyed
        # the application.
        # This should be looked into, it shouldn't be behaving like this.
        # Perhaps there is something still active that means the mainloop
        # isn't ending?
        #
        # For now, I'll just use this force-exit of the program, though it's
        # probably not ideal:
        exit()

    def result(self):
        self.result_window = result.Result(self)
        self.wait_window(self.result_window)


if __name__ == "__main__":

    app = NMREsPyApp()
    app.mainloop()
