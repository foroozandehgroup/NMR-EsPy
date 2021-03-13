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

            self.setup_window = setup.SetUp(self, self.estimator)
            # hold at this point
            # relieved once setup is destroyed
            # see SetUp.run()
            self.wait_window(self.setup_window)

        self.result_window = result.Result(self, self.estimator)
        # # create attributres relating to the result Toplevel
        # self.generate_result_variables()
        #
        # # result Toplevel is for inspecting and saving the estimation result
        # self.result = MyToplevel(self)
        # # same basic configuration as setup
        # self.result.resizable(True, True)
        # self.result.columnconfigure(0, weight=1)
        # self.result.rowconfigure(0, weight=1)
        # # frames contained within the result Toplevel
        # self.result_frames = {}
        # # frame containing the plot
        # self.result_frames['plot_frame'] = PlotFrame(
        #     parent=self.result, figure=self.resultfig['fig'],
        # )
        # # frame containing the navigation toolbar
        # self.result_frames['toolbar_frame'] = RootToolbarFrame(
        #     parent=self.result, canvas=self.result_frames['plot_frame'].canvas,
        #     ctrl=self,
        # )
        # # frame with NMR-EsPy and MF group logos
        # self.result_frames['logo_frame'] = LogoFrame(
        #     parent=self.result, scale=0.06,
        # )
        # # frame with cancel/help/run/edit parameters buttons
        # self.result_frames['button_frame'] = ResultButtonFrame(
        #     parent=self.result, ctrl=self,
        # )
        #
        # # configure frame placements
        # self.result_frames['plot_frame'].grid(
        #     row=0, column=0, columnspan=2, sticky='nsew',
        # )
        # self.result_frames['toolbar_frame'].grid(
        #     row=1, column=0, columnspan=2, sticky='ew',
        # )
        # self.result_frames['logo_frame'].grid(
        #     row=3, column=0, padx=10, pady=10, sticky='w',
        # )
        # self.result_frames['button_frame'].grid(
        #     row=3, column=1, sticky='s',
        # )

if __name__ == '__main__':

    app = NMREsPyApp()
    app.mainloop()
