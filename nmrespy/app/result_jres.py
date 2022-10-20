# result_jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 20 Oct 2022 17:59:00 BST

import tkinter as tk

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
        self.contours = []

        super().__init__(ctrl)
        self.shifts = self.estimator.get_shifts(unit="ppm")

        self.multiplet_thold = 0.5 * (
            self.estimator.default_pts[0] / self.estimator.sw()[0]
        )

    def get_region(self, idx):
        return (
            self.ylim,
            self.estimator.get_results(indices=[idx])[0].get_region(unit="ppm")[1],
        )

    def new_tab(self, idx, replace=False):
        def append(lst, obj):
            if replace:
                lst.pop(idx)
                lst.insert(idx, obj)
            else:
                lst.append(obj)

        super().new_tab(idx, replace)

        if not replace:
            append(self.nlevs, 10)
            append(self.bases, np.amax(np.abs(self.estimator.spectrum).real) / 100)
            append(self.factors, 1.2)

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
            contour_factor=self.factors[idx],
            figsize=(6, 3.5),
            dpi=170,
        )
        ax_1d, ax_2d = ax_1d[0], ax_2d[0]
        append(self.contours, ax_2d.collections)

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

        region = self.get_region(idx)
        cf.Restrictor(self.axs[idx][0], region[1])
        cf.Restrictor(self.axs[idx][1], region[1], region[0])

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
        self.canvases[idx].get_tk_widget().grid(
            column=0, columnspan=2, row=0, sticky="nsew",
        )

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
        self.nlev_labels[idx].grid(row=0, column=0, padx=(0, 5), sticky="w")

        append(
            self.nlev_entries,
            wd.MyEntry(
                self.contour_frames[idx],
                return_command=self.update_nlev,
                return_args=(idx,),
                width=12,
            ),
        )
        self.nlev_entries[idx].grid(row=0, column=1, padx=(0, 10))

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
        self.base_entries[idx].grid(row=0, column=3, padx=(0, 10))

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
            self.nlev_entries[idx].insert(0, str(self.nlevs[idx]))
            self.base_entries[idx].insert(0, f"{self.bases[idx]:6g}".replace(" ", ""))
            self.factor_entries[idx].insert(0, f"{self.factors[idx]:6g}".replace(" ", ""))

    def update_ax_xlim(self, i, idx):
        if self.axs[idx][0].get_xlim() == self.axs[idx][1].get_xlim():
            return
        else:
            j = 1 - i
            self.axs[idx][i].set_xlim(self.axs[idx][j].get_xlim())
            # self.canvases[idx].draw_idle()

    def update_nlev(self, idx):
        entry = self.nlev_entries[idx]
        inpt = entry.get()
        try:
            assert (value := int(inpt)) > 0
            self.nlevs[idx] = value
            self.new_tab(idx, replace=True)

        except Exception:
            entry.delete(0, tk.END)
            entry.insert(0, str(self.nlevs[idx]))

    def update_base(self, idx):
        entry = self.base_entries[idx]
        inpt = entry.get()
        try:
            assert (value := float(inpt)) > 0.
            self.bases[idx] = value
            self.new_tab(idx, replace=True)

        except Exception:
            entry.delete(0, tk.END)
            entry.insert(0, str(self.bases[idx]))

    def update_factor(self, idx):
        entry = self.factor_entries[idx]
        inpt = entry.get()
        try:
            assert (value := float(inpt)) > 1.
            self.factors[idx] = value
            self.new_tab(idx, replace=True)

        except Exception:
            entry.delete(0, tk.END)
            entry.insert(0, str(self.factors[idx]))
