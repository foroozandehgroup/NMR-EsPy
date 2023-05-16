# contour_app.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 16 May 2023 12:30:02 BST

import re
import tkinter as tk
from typing import Iterable

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
import matplotlib.pyplot as plt
import numpy as np

from nmrespy.app.custom_widgets import MyEntry


class ContourApp(tk.Tk):
    """Tk app for viewing 2D spectra as contour plots."""

    def __init__(self, data: np.ndarray, expinfo) -> None:
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.shifts = list(reversed(
            [s.T for s in expinfo.get_shifts(data.shape, unit="ppm")]
        ))
        nuclei = expinfo.nuclei
        units = ["ppm" if sfo is not None else "Hz" for sfo in expinfo.sfo]
        self.f1_label, self.f2_label = [
            f"{nuc} ({unit})" if nuc is not None
            else unit
            for nuc, unit in zip(nuclei, units)
        ]

        self.data = data.T.real

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.fig = plt.figure(dpi=160, frameon=True)
        self._color_fig_frame()

        self.ax = self.fig.add_axes([0.1, 0.1, 0.87, 0.87])
        self.ax.set_xlim(self.shifts[0][0][0], self.shifts[0][-1][0])
        self.ax.set_ylim(self.shifts[1][0][0], self.shifts[1][0][-1])

        self.cmap = tk.StringVar(self, "bwr")
        self.nlevels = 10
        self.factor = 1.3
        self.base = np.amax(np.abs(self.data)) / 10
        self.update_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew",
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            self,
            pack_toolbar=False,
        )
        self.toolbar.grid(row=1, column=0, pady=(0, 10), sticky="w")

        self.widget_frame = tk.Frame(self)
        self._add_widgets()
        self.widget_frame.grid(
            row=2,
            column=0,
            padx=10,
            pady=(0, 10),
            sticky="nsew",
        )
        self.close_button = tk.Button(
            self, text="Close", command=self.quit,
        )
        self.close_button.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")

    def _color_fig_frame(self) -> None:
        r, g, b = [x >> 8 for x in self.winfo_rgb(self.cget("bg"))]
        color = f"#{r:02x}{g:02x}{b:02x}"
        if not re.match(r"^#[0-9a-f]{6}$", color):
            color = "#d9d9d9"

        self.fig.patch.set_facecolor(color)

    def _add_widgets(self) -> None:
        # Colormap selection
        self.cmap_label = tk.Label(self.widget_frame, text="Colormap:")
        self.cmap_label.grid(row=0, column=0, padx=(0, 10))
        self.cmap_widget = tk.OptionMenu(
            self.widget_frame,
            self.cmap,
            self.cmap.get(),
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
            command=lambda x: self.update_plot(),
        )
        self.cmap_widget.grid(row=0, column=1)

        # Number of contour levels
        self.nlevels_label = tk.Label(self.widget_frame, text="levels")
        self.nlevels_label.grid(row=0, column=2, padx=(0, 10))
        self.nlevels_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("nlevels",),
        )
        self.nlevels_box.insert(0, str(self.nlevels))
        self.nlevels_box.grid(row=0, column=3)

        # Base contour level
        self.base_label = tk.Label(self.widget_frame, text="base")
        self.base_label.grid(row=0, column=4, padx=(0, 10))
        self.base_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("base",),
        )
        self.base_box.insert(0, f"{self.base:.2f}")
        self.base_box.grid(row=0, column=5)

        # Contour level scaling factor
        self.factor_label = tk.Label(self.widget_frame, text="factor")
        self.factor_label.grid(row=0, column=6, padx=(0, 10))
        self.factor_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("factor",),
        )
        self.factor_box.insert(0, f"{self.factor:.2f}")
        self.factor_box.grid(row=0, column=7)

    def change_levels(self, var: str) -> None:
        input_ = self.__dict__[f"{var}_box"].get()
        try:
            if var == "nlevels":
                value = int(input_)
                if value <= 0.:
                    raise ValueError
            else:
                value = float(input_)
                if (
                    value <= 1. and var == "factor" or
                    value <= 0. and var == "base"
                ):
                    raise ValueError

            self.__dict__[var] = value
            self.update_plot()

        except ValueError:
            box = self.__dict__[f"{var}_box"]
            box.delete(0, "end")
            box.insert(0, str(self.__dict__[var]))

    def make_levels(self) -> Iterable[float]:
        levels = [self.base * self.factor ** i
                  for i in range(self.nlevels)]
        return [-x for x in reversed(levels)] + levels

    def update_plot(self) -> None:
        levels = self.make_levels()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.contour(
            *self.shifts, self.data, cmap=self.cmap.get(), levels=levels,
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel(self.f2_label)
        self.ax.set_ylabel(self.f1_label)
        self.fig.canvas.draw_idle()
