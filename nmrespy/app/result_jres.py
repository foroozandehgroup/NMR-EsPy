# result_jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 20 Oct 2022 12:13:04 BST

from matplotlib import backends
import numpy as np

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
        self.ylim = tuple(
            ctrl.estimator.get_shifts(meshgrid=False)[0][[0, -1]].tolist()
        )
        self.contour_frames = []
        self.nlevs = []
        self.nlev_labels = []
        self.nlev_entries = []
        self.bases = []
        self.base_labels = []
        self.base_entries = []
        self.factors = []
        self.factor_labels = []
        self.factor_entries = []

        super().__init__(ctrl)

        self.multiplet_thold = 0.5 * (
            self.estimator.default_pts[0] / self.estimator.sw()[0]
        )

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

        if replace:
            self.contour_frames[idx].destroy()
            self.nlev_labels[idx].destroy()
            self.nlev_entries[idx].destroy()
            self.base_labels.destroy()
            self.base_entries.destroy()
            self.factor_labels.destroy()
            self.factor_entries.destroy()

        else:
            append(self.nlevs, None)
            append(self.bases, None)
            append(self.factors, None)

        fig, (ax_1d, ax_2d) = self.estimator.plot_result(
            indices=[idx],
            axes_bottom=0.12,
            axes_left=0.1,
            axes_right=0.96,
            axes_top=0.97,
            region_unit="ppm",
            label_peaks=True,
            contour_nlevels=self.nlevs[idx],
            contour_base=self.bases[idx],
            contour_factors=self.factors[idx],
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
        ax_2d.tick_params(axis="both", which="major", labelsize=6)
        ax_2d.yaxis.get_label().set_fontsize(8)
        fig.texts[0].set_fontsize(8)
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

        append(
            self.contour_frames,
            wd.MyFrame(self.tabs[idx], bg=cf.NOTEBOOKCOLOR),
        )
        self.contour_frames[idx].grid(row=1, column=1, padx=10, pady=5, sticky="e")

        append(
            self.nlev_labels,
            wd.MyLabel(
                self.contour_frames[idx],
                text="# levels:",
                bg=cf.NOTEBOOKCOLOR,
            )
        )
        self.nlev_label.grid(row=0, column=0, padx=(0, 5), sticky="w")

        append(
            self.nlev_entries,
            wd.MyEntry(
                self.contour_frames[idx],
                return_command=self.update_nlev,
                return_args=(idx,),
                width=12,
            ),
        )
        self.nlev_entry.grid(row=0, column=1, padx=(0, 10))

        append(
            self.base_labels,
            wd.MyLabel(
                self.contour_frames[idx],
                text="base:",
                bg=cf.NOTEBOOKCOLOR,
            ),
        )
        self.base_labels[idx].grid(row=0, column=2, padx=(0, 5), sticky="w")

        append(
            self.base_entries,
            wd.MyEntry(
                self.contour_frames[idx],
                return_command=self.update_base,
                return_args=(idx,),
                width=10,
            ),
        )
        self.base_entry.grid(row=0, column=3, padx=(0, 10))

        append(
            self.factor_labels,
            wd.MyLabel(
                self.contour_frames[idx],
                text="factor:",
                bg=cf.NOTEBOOKCOLOR,
            ),
        )
        self.factor_labels[idx].grid(row=0, column=4, padx=(0, 5), sticky="w")

        append(
            self.factor_entries,
            wd.MyEntry(
                self.contour_frames[idx],
                return_command=self.update_factor,
                return_args=(idx,),
                width=10,
            ),
        )
        self.factor_entries[idx].grid(row=0, column=5)

        if replace:
            self.nlev_entries[idx].insert(0, str(self.nlev))
            self.base_entries[idx].insert(0, f"{self.base:6g}".replace(" ", ""))
            self.factor_entry.insert(0, f"{self.factor:6g}".replace(" ", ""))

    def update_ax_xlim(self, i, idx):
        if self.axs[idx][0].get_xlim() == self.axs[idx][1].get_xlim():
            return
        else:
            j = 1 - i
            self.axs[idx][i].set_xlim(self.axs[idx][j].get_xlim())
            # self.canvases[idx].draw_idle()
