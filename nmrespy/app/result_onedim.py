# result_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 16 Jul 2023 13:34:36 BST

from matplotlib import backends

from nmrespy.app.result import Result1DType, ResultButtonFrame, SaveFrame
from nmrespy.app import config as cf, custom_widgets as wd


class Result1D(Result1DType):
    table_titles = [
        "a",
        "ϕ (rad)",
        "f (ppm)",
        "η (s⁻¹)",
    ]

    def __init__(self, ctrl):
        super().__init__(ctrl)

    def construct_gui_frames(self):
        super().construct_gui_frames()
        self.button_frame = ResultButtonFrame1D(self)
        self.button_frame.grid(row=1, column=1, padx=(10, 0), sticky="se")

    def get_region(self, idx):
        return self.estimator.get_results(indices=[idx])[0].get_region(unit="ppm")

    def new_tab(self, idx, replace=False):
        def append(lst, obj):
            if replace:
                lst.pop(idx)
                lst.insert(idx, obj)
            else:
                lst.append(obj)

        super().new_tab(idx, replace)

        fig, ax = self.estimator.plot_result(
            indices=[idx],
            axes_bottom=0.12,
            axes_left=0.02,
            axes_right=0.98,
            axes_top=0.98,
            xaxis_unit="ppm",
            figsize=(6, 3.5),
            dpi=170,
        )
        ax = ax[0][0]
        fig.patch.set_facecolor(cf.NOTEBOOKCOLOR)
        ax.set_facecolor(cf.PLOTCOLOR)
        append(self.figs, fig)
        append(self.axs, ax)
        cf.Restrictor(self.axs[idx], self.get_region(idx))

        append(
            self.canvases,
            backends.backend_tkagg.FigureCanvasTkAgg(
                self.figs[idx],
                master=self.tabs[idx],
            ),
        )
        self.canvases[idx].get_tk_widget().grid(column=0, row=0, sticky="nsew")

        append(
            self.toolbars,
            wd.MyNavigationToolbar(
                self.canvases[idx],
                parent=self.tabs[idx],
                color=cf.NOTEBOOKCOLOR,
            ),
        )
        self.toolbars[idx].grid(row=1, column=0, padx=10, pady=5, sticky="w")


class ResultButtonFrame1D(ResultButtonFrame):
    def __init__(self, master):
        super().__init__(master)

    def save_options(self):
        SaveFrame1D(self.master)


class SaveFrame1D(SaveFrame):
    def __init__(self, master):
        super().__init__(master)

    def generate_figure(self, figsize, dpi):
        return self.ctrl.estimator.plot_result(
            region_unit="ppm",
            figsize=figsize,
            dpi=dpi,
        )
