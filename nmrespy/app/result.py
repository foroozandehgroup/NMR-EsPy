from .config import *
from .custom_widgets import *
from .frames import *


class Result(MyToplevel):
    def __init__(self, parent, estimator):

        # Generate figure of result
        self.result_plot = estimator.plot_result()
        self.result_plot.fig.set_size_inches(6, 3.5)
        self.result_plot.fig.set_dpi(170)

        # Prevent panning outside the selected region
        xlim = self.result_plot.ax.get_xlim()
        Restrictor(self.result_plot.ax, x=lambda x: x<= xlim[0])
        Restrictor(self.result_plot.ax, x=lambda x: x>= xlim[1])

        # --- Construction of the result GUI ------------------------------
        self.ctrl = parent
        super().__init__(self.ctrl)
        
        self.resizable(True, True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Frame containing the plot
        self.plot_frame = MyFrame(self)
        # Make `plot_frame` resizable
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        # Canvas for figure
        self.canvas = backend_tkagg.FigureCanvasTkAgg(
            self.result_plot.fig, master=self.plot_frame,
        )
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        # Frame containing the navigation toolbar and advanced settings
        # button
        self.toolbar_frame = MyFrame(self)
        self.toolbar = MyNavigationToolbar(
            self.canvas, parent=self.toolbar_frame,
        )
        self.toolbar.grid(row=0, column=0, sticky='w', padx=(10,0), pady=(0,5))

        # Frame with NMR-EsPy an MF group logos
        self.logo_frame = LogoFrame(parent=self, scale=0.72)

        # Frame with cancel/help/run/advanced settings buttons
        self.button_frame = ResultButtonFrame(parent=self, ctrl=self)

        self.plot_frame.grid(row=0, column=0, columnspan=2)
        self.toolbar_frame.grid(row=1, column=0)
        self.logo_frame.grid(row=2, column=0)



class ResultButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, parent, ctrl):
        super().__init__(parent, ctrl)
        self.save_button['command'] = self.save_options
        self.save_button['text'] = 'Save'

        self.edit_parameter_button = MyButton(
            self, text='Edit Parameter Estimate', command=self.edit_parameters,
        )
        self.edit_parameter_button.grid(
            row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=(10,0)
        )

    def edit_parameters(self):
        EditParams(parent=self, ctrl=self.ctrl)

    def save_options(self):
        SaveFrame(parent=self, ctrl=self.ctrl)
