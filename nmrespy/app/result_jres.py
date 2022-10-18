# result_jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 18 Oct 2022 17:39:27 BST

from matplotlib import backends

from nmrespy.app.result import Result1DType
from nmrespy.app import config as cf, custom_widgets as wd


def get_ax_geom(box):
    w, h = box.x1 - box.x0, box.y1 - box.y0
    return [box.x0, box.y0, w, h]


class Result2DJ(Result1DType):
    table_titles = [
        "a",
        "ϕ (rad)",
        "f₁ (Hz)",
        "f₂ (ppm)",
        "η₁ (s⁻¹)",
        "η₂ (s⁻¹)",
    ]

    def __init__(self, ctrl):
        self.ylim = ctrl.estimator.get_shifts(meshgrid=False)[0][[0, -1]]
        super().__init__(ctrl)

    def new_region(self, idx, replace=False):
        def append(lst, obj):
            if replace:
                lst.pop(idx)
                lst.insert(idx, obj)
            else:
                lst.append(obj)

        append(
            self.xlims,
            self.estimator.get_results(indices=[idx])[0].get_region(unit="ppm")[-1],
        )
        super().new_region(idx, replace)
        fig, (ax_1d, ax_2d) = self.estimator.plot_result(
            indices=[idx],
            axes_bottom=0.12,
            axes_left=0.02,
            axes_right=0.98,
            axes_top=0.98,
            region_unit="ppm",
            figsize=(6, 3.5),
            dpi=170,
        )
        ax_1d, ax_2d = ax_1d[0], ax_2d[0]
        # Manipulate the positions of the axes
        geom2 = get_ax_geom(ax_2d.get_position())
        geom2[1] -= 0.02
        geom2[3] -= 0.02
        ax_2d.set_position(geom2)
        ax_2d.spines["top"].set_visible(True)
        ax_1d.spines["bottom"].set_visible(True)

        fig.patch.set_facecolor(cf.NOTEBOOKCOLOR)
        ax_1d.set_facecolor(cf.PLOTCOLOR)
        ax_2d.set_facecolor(cf.PLOTCOLOR)
        append(self.figs, fig)
        append(self.axs, [ax_1d, ax_2d])
        cf.Restrictor(self.axs[idx][0], self.xlims[idx])
        cf.Restrictor(self.axs[idx][1], self.xlims[idx], self.ylim)
        self.axs[idx][0].callbacks.connect(
            "xlim_changed",
            lambda evt: self.update_ax_xlim(1, idx),
        )
        self.axs[idx][1].callbacks.connect(
            "xlim_changed",
            lambda evt: self.update_ax_xlim(0, idx),
        )

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

    def update_ax_xlim(self, i, idx):
        if self.axs[idx][0].get_xlim() == self.axs[idx][1].get_xlim():
            return
        else:
            j = 1 - i
            self.axs[idx][i].set_xlim(self.axs[idx][j].get_xlim())
            # self.canvases[idx].draw_idle()
