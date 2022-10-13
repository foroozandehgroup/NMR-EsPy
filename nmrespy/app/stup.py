# stup.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Oct 2022 15:59:22 BST

import collections
import copy
from datetime import datetime
import re
import tkinter as tk
import webbrowser

from matplotlib import patches, pyplot as plt, transforms
from matplotlib.backends import backend_tkagg
import numpy as np

from nmrespy import sig
import nmrespy._paths_and_links as pl
import nmrespy.app.config as cf
import nmrespy.app.custom_widgets as wd

import nmrespy.app.frames as fr


class SetUp(wd.MyToplevel):
    region_colors = [
        "#E74C3C",
        "#E67E22",
        "#F1C40F",
        "#1ABC9C",
        "#2ECC71",
        "#3498DB",
        "#9B59B6",
    ]

    @property
    def estimator(self):
        return self.ctrl.estimator

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.ctrl = ctrl
        self.configure_root()
        self.construct_gui_frames()
        self.place_gui_frames()
        self.configure_gui_frames()
        self.construct_figure()
        self.construct_toolbar()
        self.construct_pre_proc_objects()
        self.construct_region_objects()
        self.construct_advanced_settings_objects()

    def configure_root(self):
        self.title("NMR-EsPy - Setup Calculation")
        self.resizable(True, True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.ctrl.destroy)

    def construct_gui_frames(self):
        self.plot_frame = wd.MyFrame(self)
        self.toolbar_frame = wd.MyFrame(self)
        self.notebook_frame = wd.MyFrame(self)
        self.notebook = wd.MyNotebook(self.notebook_frame)
        self.pre_proc_frame = wd.MyFrame(self.notebook, bg=cf.NOTEBOOKCOLOR)
        self.region_frame = wd.MyFrame(self.notebook, bg=cf.NOTEBOOKCOLOR)
        self.region_notebook = wd.MyNotebook(self.region_frame)
        self.adset_frame = wd.MyFrame(self.notebook, bg=cf.NOTEBOOKCOLOR)
        # self.logo_frame = fr.LogoFrame(self, scale=0.72)
        self.button_frame = SetupButtonFrame(master=self)

    def place_gui_frames(self):
        self.notebook.add(
            self.pre_proc_frame,
            text="Pre-Processing",
            sticky="nsew",
        )
        self.notebook.add(
            self.region_frame,
            text="Region Selection",
            sticky="nsew",
        )
        self.notebook.add(
            self.adset_frame,
            text="Advanced Settings",
            sticky="nsew",
        )
        self.notebook.bind(
            "<<NotebookTabChanged>>",
            lambda event: self.switch_main_tab(),
        )
        self.notebook.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.region_notebook.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.notebook_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        # self.logo_frame.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.button_frame.grid(row=3, column=1, sticky="s")

    def configure_gui_frames(self):
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self.notebook_frame.columnconfigure(0, weight=1)
        self.pre_proc_frame.columnconfigure(0, weight=1)
        self.region_frame.columnconfigure(0, weight=1)

    def construct_figure(self):
        self.fig, self.ax = plt.subplots(
            figsize=(6, 3.5),
            dpi=170,
        )
        shifts = self.estimator.get_shifts(unit="ppm")[0]
        self.xlim = [shifts[i] for i in (0, -1)]
        self.ax.set_xlim(self.xlim)

        # Prevent user panning/zooming beyond spectral window
        # See Restrictor class for more info ↑
        cf.Restrictor(self.ax, x=lambda x: x <= self.xlim[0])
        cf.Restrictor(self.ax, x=lambda x: x >= self.xlim[1])

        # Aesthetic tweaks
        self.fig.patch.set_facecolor(cf.BGCOLOR)
        self.ax.set_facecolor(cf.PLOTCOLOR)
        self.ax.set_xlabel(
            f"{self.estimator.unicode_nuclei[0]} (ppm)", fontsize=8,
        )
        self.ax.locator_params(axis="x", nbins=10)
        self.ax.tick_params(axis="x", which="major", labelsize=6)
        self.ax.set_yticks([])
        for direction in ("top", "bottom", "left", "right"):
            self.ax.spines[direction].set_color("k")

        self.spec_line, = self.ax.plot(
            shifts,
            self.estimator.spectrum.real,
            color="k",
            lw=1.,
        )

        # Create figure canvas
        self.canvas = backend_tkagg.FigureCanvasTkAgg(
            self.fig,
            master=self.plot_frame,
        )
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky="nsew")

    def construct_toolbar(self):
        self.toolbar = wd.MyNavigationToolbar(
            self.canvas,
            parent=self.toolbar_frame,
        )
        self.toolbar.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(10, 0),
            pady=(0, 5),
        )

    def construct_pre_proc_objects(self):
        # --- Exponential damping ---
        self.lb = 0.
        self.lb_wgt = wd.MyLabelScaleEntry(
            self.pre_proc_frame,
            name="lb",
            frame_kw={"bg": cf.NOTEBOOKCOLOR},
            label_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "width": 5,
            },
            scale_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "troughcolor": "white",
                "from_": 0,
                "to": 20,
                "resolution": 0.001,
                "command": lambda value: self.update_lb_scale(value),
            },
            entry_kw={
                "return_command": self.update_lb_entry,
            },
        )
        self.lb_wgt.scale.set(self.lb)
        self.lb_wgt.entry.insert(0, f"{self.lb:.3f}")

        # --- Phase correction ---
        self.pivot = {}
        init_pivot = self.estimator.data.shape[-1] // 2
        for unit in ("idx", "hz", "ppm"):
            self.pivot[unit] = self.conv(init_pivot, f"idx->{unit}")

        for name in ("p0", "p1"):
            self.__dict__[name] = {}
            for unit in ("rad", "deg"):
                self.__dict__[name][unit] = 0.

        # Pivot plot
        self.pivot_line = self.ax.axvline(
            x=self.pivot["ppm"],
            color=cf.PIVOTCOLOR,
            lw=0.8,
        )

        self.pivot_wgt = wd.MyLabelScaleEntry(
            self.pre_proc_frame,
            name="pivot",
            frame_kw={"bg": cf.NOTEBOOKCOLOR},
            label_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "width": 5,
            },
            scale_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "troughcolor": cf.PIVOTCOLOR,
                "from_": 0,
                "to": self.estimator.data.shape[-1] - 1,
                "resolution": 1,
                "command": lambda value: self.update_pivot_scale(value),
            },
            entry_kw={
                "return_command": self.update_pivot_entry,
            },
        )
        self.pivot_wgt.scale.set(self.pivot["idx"])
        self.pivot_wgt.entry.insert(0, f"{self.pivot['ppm']:.3f}")

        self.p0_wgt = wd.MyLabelScaleEntry(
            self.pre_proc_frame,
            name="φ₀",
            frame_kw={"bg": cf.NOTEBOOKCOLOR},
            label_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "width": 5,
            },
            scale_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "troughcolor": "white",
                "from_": -4,
                "to": 4,
                "resolution": 0.0001,
                "command": lambda value: self.update_phase_scale(value, "p0"),
            },
            entry_kw={
                "return_command": self.update_phase_entry,
                "return_args": ("p0",),
            },
        )
        self.p0_wgt.scale.set(self.p0["rad"])
        self.p0_wgt.entry.insert(0, f"{self.p0['rad']:.3f}")

        self.p1_wgt = wd.MyLabelScaleEntry(
            self.pre_proc_frame,
            name="φ₁",
            frame_kw={"bg": cf.NOTEBOOKCOLOR},
            label_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "width": 5,
            },
            scale_kw={
                "bg": cf.NOTEBOOKCOLOR,
                "troughcolor": "white",
                "from_": -40,
                "to": 40,
                "resolution": 0.0001,
                "command": lambda value: self.update_phase_scale(value, "p1"),
            },
            entry_kw={
                "return_command": self.update_phase_entry,
                "return_args": ("p1",),
            },
        )
        self.p1_wgt.scale.set(self.p1["rad"])
        self.p1_wgt.entry.insert(0, f"{self.p1['rad']:.3f}")

        for row, wgt in enumerate(
            (self.lb_wgt, self.pivot_wgt, self.p0_wgt, self.p1_wgt)
        ):
            pady = 10 if row == 0 else (0, 10)
            wgt.grid(row=row, column=0, padx=10, pady=pady, sticky="ew")

    def construct_region_objects(self):
        self.regions = []
        self.region_tabs = []
        self.region_wgts = []
        self.region_patches = []
        self.region_labels = []
        self.noscs_wgts = []

        self.new_region_frame = wd.MyFrame(self.region_notebook)
        self.region_notebook.add(
            self.new_region_frame,
            text="+",
            sticky="nsew",
        )
        self.region_notebook.bind(
            "<<NotebookTabChanged>>",
            lambda event: self.switch_region_tab(),
        )

        # Noise region
        self.new_region(noise=True)
        self.new_region()
        self.region_notebook.select(0)

    def construct_advanced_settings_objects(self):
        self.opt_label = wd.MyLabel(
            self.adset_frame, text="optimisation method:", bg=cf.NOTEBOOKCOLOR,
        )
        self.opt_label.grid(row=0, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        options = ("Gauss-Newton", "Exact Hessian", "L-BFGS")
        self.opt = tk.StringVar()
        self.opt.set(options[0])

        # I was getting funny behaviour when I tried to make a class
        # that inherited from tk.OptionMenu
        # had to customise manually after generating an instance
        self.opt_menu = tk.OptionMenu(
            self.adset_frame,
            self.opt,
            *options,
        )
        self.opt_menu["bg"] = "white"
        self.opt_menu["width"] = 12
        self.opt_menu["highlightbackground"] = "black"
        self.opt_menu["highlightthickness"] = 1
        self.opt_menu["menu"]["bg"] = "white"
        self.opt_menu["menu"]["activebackground"] = cf.ACTIVETABCOLOR
        self.opt_menu["menu"]["activeforeground"] = "white"

        # change the max. number of iterations after changing NLP
        # algorithm
        self.opt.trace("w", self.update_opt)
        self.opt_menu.grid(
            row=0,
            column=1,
            padx=10,
            pady=(10, 0),
            sticky="w",
        )

        self.maxit_label = wd.MyLabel(
            self.adset_frame,
            bg=cf.NOTEBOOKCOLOR,
            text="maximum iterations:",
        )
        self.maxit_label.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.maxit = 200
        self.maxit_entry = wd.MyEntry(
            self.adset_frame,
            return_command=self.update_maxit,
        )
        self.maxit_entry.insert(0, str(self.maxit))
        self.maxit_entry.grid(row=1, column=1, padx=10, pady=(10, 0), sticky="w")

        self.pv_label = wd.MyLabel(
            self.adset_frame,
            text="optimise phase variance:",
            bg=cf.NOTEBOOKCOLOR,
        )
        self.pv_label.grid(row=2, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.pv = tk.IntVar(self)
        self.pv_checkbutton = wd.MyCheckbutton(
            self.adset_frame,
            variable=self.pv,
            bg=cf.NOTEBOOKCOLOR,
        )
        self.pv_checkbutton.select()
        self.pv_checkbutton.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="w")

    def conv(self, value, conversion):
        return self.estimator.convert([value], conversion)[0]

    # --- Tab switching methods ----------------------------------------
    def switch_main_tab(self):
        tab = self.notebook.index(self.notebook.select())
        pivot_alpha, patch_alpha = (1, 0) if tab == 0 else (0, 1)
        self.pivot_line.set_alpha(pivot_alpha)
        for label, patch in zip(self.region_labels, self.region_patches):
            label.set(alpha=patch_alpha)
            patch.set_alpha(patch_alpha)

        self.canvas.draw_idle()

    def switch_region_tab(self):
        tab = self.region_notebook.index(self.region_notebook.select())
        if tab == len(self.regions):
            self.new_region()
            self.region_notebook.select(len(self.regions) - 1)

    # --- Pre-processing methods ---------------------------------------
    def update_lb_scale(self, value):
        self.lb = float(value)
        self.lb_wgt.entry.delete(0, tk.END)
        self.lb_wgt.entry.insert(0, f"{self.lb:.3f}")
        self.update_spectrum()

    def update_lb_entry(self):
        inpt = self.lb_wgt.entry.get()
        try:
            value = float(inpt)
            assert 0. <= value <= 20.
            self.lb = value
            self.lb_wgt.scale.set(value)

        except Exception:
            pass

        self.lb_wgt.entry.delete(0, tk.END)
        self.lb_wgt.entry.insert(0, f"{self.lb:.3f}")

    def update_pivot_scale(self, value):
        self.pivot["idx"] = int(value)
        for unit in ("ppm", "hz"):
            self.pivot[unit] = self.conv(float(value), f"idx->{unit}")
        self.pivot_wgt.entry.delete(0, tk.END)
        self.pivot_wgt.entry.insert(0, f"{self.pivot['ppm']:.3f}")
        self.pivot_line.set_xdata([self.pivot["ppm"], self.pivot["ppm"]])
        self.update_spectrum()

    def update_pivot_entry(self):
        inpt = self.pivot_wgt.entry.get()
        try:
            value = float(inpt)
            assert self.xlim[0] >= value >= self.xlim[1]
            for unit in ("idx", "ppm", "hz"):
                self.pivot[unit] = self.conv(value, f"ppm->{unit}")
            self.pivot_wgt.scale.set(self.pivot["idx"])

        except Exception:
            self.pivot_wgt.entry.delete(0, tk.END)
            self.pivot_wgt.entry.insert(0, f"{self.pivot['ppm']:.3f}")

    def update_phase_scale(self, value, name):
        """Update the GUI after the user changes the slider on a phase scale
        widget"""
        obj = self.__dict__[name]
        rad = float(value)
        deg = rad * 180 / np.pi
        obj["rad"] = rad
        obj["deg"] = deg
        wgt = self.__dict__[f"{name}_wgt"]
        wgt.entry.delete(0, tk.END)
        wgt.entry.insert(0, f"{obj['rad']:.3f}")
        self.update_spectrum()

    def update_phase_entry(self, name):
        obj, wgt = (self.__dict__[s] for s in (name, f"{name}_wgt"))
        inpt = wgt.entry.get()
        limit = 4 if name == "p0" else 40
        try:
            value = float(inpt)
            assert abs(value) <= limit
            obj["rad"] = value
            obj["deg"] = value * 180 / np.pi
            wgt.scale.set(obj["rad"])

        except Exception:
            wgt.entry.delete(0, tk.END)
            wgt.entry.insert(0, f"{obj['rad']:.3f}")

    def update_spectrum(self):
        data = copy.deepcopy(self.estimator.data)
        data = sig.exp_apodisation(data, self.lb)
        data[0] *= 0.5
        self.spec_line.set_ydata(
            sig.phase(
                sig.ft(data),
                (self.p0["rad"],),
                (self.p1["rad"],),
                (self.pivot["idx"],),
            ).real
        )
        self.canvas.draw_idle()

    # --- Region selection methods -------------------------------------
    def new_region(self, noise=False):
        init_bounds = (0.5, 0.55) if not noise else (0.1, 0.15)
        region = {}
        region["idx"] = [
            int(x * self.estimator.data.shape[-1]) for x in init_bounds
        ]
        for unit in ("hz", "ppm"):
            region[unit] = self.conv(region["idx"], f"idx->{unit}")

        self.regions.append(region)

        idx = len(self.regions) - 1
        color = self.region_colors[idx - 1] if not noise else "#808080"

        region_tab = wd.MyFrame(self.region_notebook)
        self.region_notebook.insert(
            idx,
            region_tab,
            text=str(idx - 1) if not noise else "noise",
            sticky="nsew"
        )
        region_tab.columnconfigure(0, weight=1)

        patch = self.ax.axvspan(
            *self.regions[idx]["ppm"],
            facecolor=color,
            alpha=0.7,
        )
        self.region_patches.append(patch)

        trans = transforms.blended_transform_factory(
            self.ax.transData,
            self.ax.transAxes,
        )
        label = self.ax.text(
            self.regions[idx]["ppm"][0],
            0.995,
            str(idx - 1) if not noise else "N",
            verticalalignment="top",
            transform=trans,
            fontsize=7,
        )
        self.region_labels.append(label)

        left_wgt = wd.MyLabelScaleEntry(
            region_tab,
            name="left bound",
            label_kw={
                "width": 10,
            },
            scale_kw={
                "troughcolor": color,
                "from_": self.xlim[0],
                "to": self.xlim[1],
                "resolution": 0.0001,
                "command": lambda value: self.update_region_scale(value, idx, "lb"),
            },
            entry_kw={
                "return_command": self.update_region_entry,
                "return_args": (idx, "lb"),
            }
        )
        left_wgt.scale.set(self.regions[idx]["ppm"][0])
        left_wgt.entry.insert(0, f"{self.regions[idx]['ppm'][0]:.3f}")

        right_wgt = wd.MyLabelScaleEntry(
            region_tab,
            name="right bound",
            label_kw={
                "width": 10,
            },
            scale_kw={
                "troughcolor": color,
                "from_": self.xlim[0],
                "to": self.xlim[1],
                "resolution": 0.0001,
                "command": lambda value: self.update_region_scale(value, idx, "rb"),
            },
            entry_kw={
                "return_command": self.update_region_entry,
                "return_args": (idx, "rb"),
            }
        )
        right_wgt.scale.set(self.regions[idx]["ppm"][1])
        right_wgt.entry.insert(0, f"{self.regions[idx]['ppm'][1]:.3f}")

        self.region_tabs.append(region_tab)
        left_wgt.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        right_wgt.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.region_wgts.append([left_wgt, right_wgt])

        if not noise:
            noscs_wgt = wd.NOscWidget(self.region_tabs[-1])
            self.noscs_wgts.append(noscs_wgt)
            noscs_wgt.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

    def update_region_scale(self, value, idx, bound):
        i = 0 if bound == "lb" else 1
        region = self.regions[idx]
        wgt = self.region_wgts[idx][i]
        value = self.check_valid_region_bound(value, idx, bound)

        if value is not None:
            for unit in ("hz", "ppm", "idx"):
                region[unit][i] = self.conv(value, f"ppm->{unit}")

            self.update_region_patch(idx, bound)

        else:
            wgt.scale.set(region["ppm"][i])

        wgt.entry.delete(0, tk.END)
        wgt.entry.insert(0, f"{region['ppm'][i]:.3f}")

    def update_region_entry(self, idx, bound):
        i = 0 if bound == "lb" else 1
        region = self.regions[idx]
        wgt = self.region_wgts[idx][i]

        inpt = wgt.entry.get()
        value = self.check_valid_region_bound(inpt, idx, bound)

        if value is not None:
            for unit in ("hz", "ppm", "idx"):
                region[unit][i] = self.conv(value, f"ppm->{unit}")

            self.update_region_patch(idx, bound)
            wgt.scale.set(region["ppm"][i])

        wgt.entry.delete(0, tk.END)
        wgt.entry.insert(0, f"{region['ppm'][i]:.3f}")

    def update_region_patch(self, idx, bound):
        i = 0 if bound == "lb" else 1
        patch = self.region_patches[idx]
        slice_ = [0, 1, 4] if i == 0 else [2, 3]
        coords = patch.get_xy()
        coords[slice_, 0] = self.regions[idx]["ppm"][i]
        patch.set_xy(coords)

        if i == 0:
            label = self.region_labels[idx]
            label.set_x(self.regions[idx]["ppm"][0])

        self.canvas.draw_idle()

    def check_valid_region_bound(self, value, idx, bound):
        try:
            value = float(value)
            left, right = self.regions[idx]["ppm"]
            if bound == "lb":
                assert self.xlim[0] >= value > right
            elif bound == "rb":
                assert left > value >= self.xlim[-1]
            return value

        except Exception:
            return None

    # --- Advanced Settings methods ------------------------------------
    def update_opt(self, *args):
        opt = self.opt.get()

        if opt == "Exact Hessian":
            default = "100"
        elif opt == "Gauss-Newton":
            default = "200"
        elif opt == "L-BFGS":
            default = "500"

        self.maxit_entry.delete(0, tk.END)
        self.maxit_entry.insert(0, default)
        self.update_maxit()

    def update_maxit(self):
        inpt = self.maxit_entry.get()
        try:
            value = int(inpt)
            assert value > 0
            self.maxit = value

        except Exception:
            pass

        self.maxit_entry.delete(0, tk.END)
        self.maxit_entry.insert(0, str(self.maxit))

    def run(self):
        # TODO Check no red boxes
        if not cf.check_invalid_entries(self):
            msg = "Some parameters have not been validated."
            fr.WarnWindow(self, msg=msg)
            return

        self.estimator.exp_apodisation(self.lb)
        self.estimator.phase_data(self.p0["rad"], self.p1["rad"], self.pivot["idx"])
        regions = collections.deque([r["ppm"] for r in self.regions])
        noise_region = regions.popleft()
        init_guesses = [
            None if noscs_wgt.mdl_var.get() == 1 else int(noscs_wgt.entry.get())
            for noscs_wgt in self.noscs_wgts
        ]
        for region, init_guess in zip(regions, init_guesses):
            self.estimator.estimate(
                region=region,
                noise_region=noise_region,
                region_unit="ppm",
                initial_guess=init_guess,
                max_iterations=self.maxit,
                phase_variance=bool(self.pv.get()),
            )

        # Pickle result to the temporary directory
        tmppath = cf.TMPPATH / datetime.now().strftime("%y%m%d%H%M%S")
        self.estimator.to_pickle(path=tmppath)

        # TODO: animation window
        # self.ctrl.waiting_window.destroy()
        self.destroy()
        self.ctrl.result()


class SetupButtonFrame(fr.RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, master):
        super().__init__(master)
        self.green_button["text"] = "Run"
        self.green_button["command"] = self.master.run

        self.help_button["command"] = lambda: webbrowser.open_new(
            f"{pl.DOCSLINK}/content/gui/usage/setup.html"
        )

# TODO: Put in run
    # def close(self):
    #     valid = cf.check_invalid_entries(self.main_frame)
    #     if valid:
    #         self.destroy()
    #     else:
    #         msg = "Some parameters have not been validated."
    #         fr.WarnWindow(self, msg=msg)
    #         return
