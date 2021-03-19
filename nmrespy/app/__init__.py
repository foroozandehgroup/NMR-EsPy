#!/usr/bin/python3

# Application for using NMR-EsPy
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# This is currently only applicable to 1D NMR data.

import pathlib
import tkinter as tk

from nmrespy._errors import TwoDimUnsupportedError
from nmrespy.core import Estimator
from nmrespy.app.frames import DataType
from nmrespy.app import setup, result


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
    # When you see `self.ctrl` in other classes in this file, it refers
    # to this class

    def __init__(self, path, res, topspin=False):
        super().__init__()

        # Hide the root app window. This is not going to be used. Everything
        # will be built onto Toplevels
        self.withdraw()
        path = pathlib.Path(path)

        if topspin:
            # Open window to ask user for data type (fid or pdata)
            # from this, self acquires the attirbutes dtype and path
            paths = {'pdata': path, 'fid': path.parent.parent}
            data_type_window = DataType(self, paths)
            path = data_type_window.path

        if res:
            # Wish to view result from a previously-generated result.
            # Jump straight to the result window.
            self.estimator = Estimator.from_pickle(path)

        else:
            # Create Estimator instance from the provided path
            self.estimator = Estimator.new_bruker(path)

            # App is only applicable to 1D data currently
            if self.estimator.get_dim() > 1:
                raise TwoDimUnsupportedError()

            setup_window = setup.SetUp(self)
            # hold at this point
            # relieved once setup is destroyed
            # see SetUp.run()
            self.wait_window(setup_window)

        result_window = result.Result(self)
        self.wait_window(result_window)
        self.destroy()

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

if __name__ == '__main__':

    app = NMREsPyApp()
    app.mainloop()