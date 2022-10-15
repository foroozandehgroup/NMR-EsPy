# result.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 14 Oct 2022 19:28:38 BST

import re
import tkinter as tk
from tkinter import ttk
import webbrowser

from matplotlib.backends import backend_tkagg

import nmrespy._paths_and_links as pl
from nmrespy.app import config as cf, custom_widgets as wd, frames as fr
import numpy as np


class Result1D(wd.MyToplevel):
    @property
    def estimator(self):
        return self.ctrl.estimator

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.ctrl = ctrl
        self.configure_root()
        self.construct_gui_frames()
        self.construct_notebook()

    def configure_root(self):
        self.title("NMR-EsPy - Result")
        self.resizable(True, True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.ctrl.destroy)

    def construct_gui_frames(self):
        self.notebook_frame = wd.MyFrame(self)
        self.notebook_frame.columnconfigure(0, weight=1)
        self.notebook_frame.rowconfigure(0, weight=1)
        self.notebook_frame.grid(
            row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="nsew",
        )

        self.logo_frame = fr.LogoFrame(self, scale=0.6)
        self.logo_frame.grid(row=1, column=0, padx=(10, 0), pady=10, sticky="w")

        self.button_frame = ResultButtonFrame(self)
        self.button_frame.grid(row=1, column=1, padx=(10, 0), sticky="se")

    def construct_notebook(self):
        self.notebook = ttk.Notebook(self.notebook_frame)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # TODO Should probably bundle this stuff into a dataclass
        self.tabs = []
        self.figs = []
        self.axs = []
        self.xlims = []
        self.canvases = []
        self.toolbars = []
        self.table_frames = []
        self.tables = []
        self.histories = []
        self.button_frames = []
        self.add_buttons = []
        self.remove_buttons = []
        self.merge_buttons = []
        self.split_buttons = []
        self.undo_buttons = []
        self.rerun_buttons = []
        self.edit_boxes = []
        self.staged_edits = []
        self.staged_oscs = []

        n_regions = len(self.estimator._results)
        for i in range(n_regions):
            self.tabs.append(wd.MyFrame(self.notebook, bg=cf.NOTEBOOKCOLOR))
            self.tabs[-1].columnconfigure(0, weight=1)
            self.tabs[-1].rowconfigure(0, weight=1)
            self.notebook.add(
                self.tabs[-1],
                text=str(i),
                sticky="nsew",
            )
            fig, ax = self.estimator.plot_result(
                indices=[i],
                axes_bottom=0.12,
                axes_left=0.02,
                axes_right=0.98,
                axes_top=0.98,
                region_unit="ppm",
                figsize=(6, 3.5),
                dpi=170,
            )
            self.figs.append(fig)
            self.figs[-1].patch.set_facecolor(cf.NOTEBOOKCOLOR)
            self.axs.append(ax[0][0])
            self.axs[-1].set_facecolor(cf.PLOTCOLOR)
            self.xlims.append(
                self.estimator.get_results(indices=[i])[0].get_region(unit="ppm")[0]
            )
            self.canvases.append(
                backend_tkagg.FigureCanvasTkAgg(
                    self.figs[-1],
                    master=self.tabs[-1],
                )
            )
            self.canvases[-1].get_tk_widget().grid(column=0, row=0, sticky="nsew")
            self.toolbars.append(
                wd.MyNavigationToolbar(
                    self.canvases[-1],
                    parent=self.tabs[-1],
                    color=cf.NOTEBOOKCOLOR,
                )
            )
            self.toolbars[-1].grid(row=1, column=0, padx=10, pady=5, sticky="w")
            cf.Restrictor(self.axs[-1], self.xlims[-1])

            self.table_frames.append(
                wd.MyFrame(
                    self.tabs[-1], bg=cf.NOTEBOOKCOLOR, highlightbackground="black",
                    highlightthickness=3,
                )
            )
            self.table_frames[-1].columnconfigure(0, weight=1)
            self.table_frames[-1].rowconfigure(0, weight=1)
            self.table_frames[-1].grid(
                row=0, column=1, rowspan=2, padx=(0, 10), pady=10, sticky="ns",
            )
            self.tables.append(
                wd.MyTable(
                    self.table_frames[-1],
                    contents=self.estimator.get_params(indices=[i], funit="ppm"),
                    titles=[
                        "Amplitude",
                        "Phase (rad)",
                        "Frequency (ppm)",
                        "Damping (s⁻¹)",
                    ],
                    region=self.xlims[-1],
                    bg=cf.NOTEBOOKCOLOR,
                )
            )
            self.tables[-1].grid(
                row=0, column=0, columnspan=4, padx=(0, 10), pady=10, sticky="n",
            )

            self.histories.append(
                (
                    self.estimator.get_params(indices=[i]),
                    self.estimator.get_errors(indices=[i]),
                )
            )

            self.button_frames.append(
                wd.MyFrame(self.table_frames[-1], bg=cf.NOTEBOOKCOLOR)
            )
            for r in range(4):
                self.button_frames[-1].columnconfigure(r, weight=1)
            self.button_frames[-1].grid(row=1, column=0, sticky="s")

            self.add_buttons.append(
                wd.MyButton(
                    self.button_frames[-1],
                    text="Add",
                    command=self.add,
                )
            )
            self.add_buttons[-1].grid(
                row=0, column=0, padx=(10, 0), pady=(10, 0), sticky="ew",
            )

            self.remove_buttons.append(
                wd.MyButton(
                    self.button_frames[-1],
                    text="Remove",
                    state="disabled",
                    command=self.remove,
                )
            )
            self.remove_buttons[-1].grid(
                row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="ew",
            )

            self.merge_buttons.append(
                wd.MyButton(
                    self.button_frames[-1],
                    text="Merge",
                    state="disabled",
                    command=self.merge,
                )
            )
            self.merge_buttons[-1].grid(
                row=0, column=2, padx=(10, 0), pady=(10, 0), sticky="ew",
            )

            self.split_buttons.append(
                wd.MyButton(
                    self.button_frames[-1],
                    text="Split",
                    state="disabled",
                    command=self.split,
                )
            )
            self.split_buttons[-1].grid(
                row=0, column=3, padx=10, pady=(10, 0), sticky="ew",
            )

            self.undo_buttons.append(
                wd.MyButton(
                    self.button_frames[-1],
                    text="Undo",
                    state="disabled",
                    command=self.undo,
                )
            )
            self.undo_buttons[-1].grid(
                row=1, column=2, padx=(10, 0), pady=10, sticky="ew",
            )

            self.rerun_buttons.append(
                wd.MyButton(
                    self.button_frames[-1],
                    text="Re-run",
                    state="disabled",
                    command=self.rerun,
                )
            )
            self.rerun_buttons[-1].grid(
                row=1, column=3, padx=10, pady=10, sticky="ew",
            )

            self.tables[-1].selected_number.trace("w", self.configure_button_states)

            self.edit_boxes.append(
                tk.Text(
                    self.table_frames[-1],
                    height=10,
                )
            )
            self.edit_boxes[-1].insert(
                "1.0",
                f"Edits staged for result {i}:\n",
            )
            self.edit_boxes[-1]["state"] = "disabled"
            self.edit_boxes[-1].grid(row=2, column=0, sticky="ew", padx=10, pady=10)

            self.staged_edits.append([])
            self.staged_oscs.append([])

    # --- Edit Parameters Methods ----------------------------------
    def get_idx(self):
        return self.notebook.index(self.notebook.select())

    def configure_button_states(self, *args):
        idx = self.get_idx()

        # Number of curently selected oscillators
        number = self.tables[idx].selected_number.get()

        if number == 0:
            self.add_buttons[idx]["state"] = "normal"
            self.remove_buttons[idx]["state"] = "disabled"
            self.merge_buttons[idx]["state"] = "disabled"
            self.split_buttons[idx]["state"] = "disabled"

        elif number == 1:
            self.add_buttons[idx]["state"] = "disabled"
            self.remove_buttons[idx]["state"] = "normal"
            self.merge_buttons[idx]["state"] = "disabled"
            self.split_buttons[idx]["state"] = "normal"

        else:
            self.add_buttons[idx]["state"] = "disabled"
            self.remove_buttons[idx]["state"] = "normal"
            self.merge_buttons[idx]["state"] = "normal"
            self.split_buttons[idx]["state"] = "disabled"

    def append_text(self, idx, text):
        self.edit_boxes[idx]["state"] = "normal"
        self.edit_boxes[idx].insert(tk.END, text)
        self.edit_boxes[idx]["state"] = "disabled"

    def reset_table(self, idx):
        self.tables[idx].selected_rows = []
        self.tables[idx].activate_rows(top=self.tables[idx].top)

    def activate_run_button(self, idx):
        self.rerun_buttons[idx]["state"] = "normal"

    def check_for_duplicated_oscs(self, new_oscs, idx):
        return bool(set(new_oscs) & set(self.staged_oscs[idx]))

    def add(self):
        idx = self.get_idx()
        add_frame = AddFrame(self, idx)
        self.wait_window(add_frame)
        oscs = add_frame.new_oscillators
        if oscs is not None:
            self.staged_edits[idx].append((0, oscs))
            self.edit_boxes[idx]["state"] = "normal"
            plural = "s" if oscs.shape[0] > 1 else ""
            self.append_text(
                idx,
                f"--> Add {oscs.shape[0]} oscillator{plural}\n",
            )
            self.activate_run_button(idx)

        self.reset_table(idx)

    def remove(self):
        idx = self.get_idx()
        to_rm = self.tables[idx].selected_rows
        if not self.check_for_duplicated_oscs(to_rm, idx):
            self.staged_oscs[idx].extend(to_rm)
            self.staged_edits[idx].append((1, to_rm))
            plural = "s" if len(to_rm) > 1 else ""
            self.append_text(
                idx,
                f"--> Remove oscillator{plural} {', '.join([str(x) for x in to_rm])}\n",
            )
            self.activate_run_button(idx)

        self.reset_table(idx)

    def merge(self):
        idx = self.get_idx()
        to_merge = self.tables[idx].selected_rows
        if not self.check_for_duplicated_oscs(to_merge, idx):
            self.staged_oscs[idx].extend(to_merge)
            self.staged_edits[idx].append((2, to_merge))
            self.append_text(
                idx,
                f"--> Merge oscillators {', '.join([str(x) for x in sorted(to_merge)])}\n",  # noqa: E501
            )
            self.activate_run_button(idx)

        self.reset_table(idx)

    def split(self):
        idx = self.get_idx()
        to_split = self.tables[idx].selected_rows
        if not self.check_for_duplicated_oscs(to_split, idx):
            split_frame = SplitFrame(self, idx, to_split[0])
            self.wait_window(split_frame)
            split_info = split_frame.split_info
            if split_info is not None:
                self.staged_oscs[idx].extend(to_split)
                self.staged_edits[idx].append((3, {to_split[0]: split_info}))
                self.append_text(
                    idx,
                    f"--> Split oscillator {to_split[0]}\n",
                )
                self.activate_run_button(idx)

        self.reset_table(idx)

    def undo(self):
        pass

    def rerun(self):
        idx = self.get_idx()
        add_oscs = None
        rm_oscs = None
        merge_oscs = None
        split_oscs = None
        for typ, info in self.staged_edits[idx]:
            if typ == 0:
                add_oscs = info if add_oscs is None else np.vstack(add_oscs, info)
            elif typ == 1:
                if rm_oscs is None:
                    rm_oscs = [info]
                else:
                    rm_oscs.extend(info)
            elif typ == 2:
                if merge_oscs is None:
                    merge_oscs = [info]
                else:
                    merge_oscs.append(info)
            elif typ == 3:
                if split_oscs is None:
                    split_oscs = {info[0]: info[1]}
                else:
                    split_oscs[info[0]] = info[1]

        self.ctrl.estimator.edit_result(
            index=idx,
            add_oscs=add_oscs,
            rm_oscs=rm_oscs,
            merge_oscs=merge_oscs,
            split_oscs=split_oscs,
        )
        print(self.ctrl.estimator.get_params([0]))



class ResultButtonFrame(fr.RootButtonFrame):
    def __init__(self, master):
        self.ctrl = master.ctrl
        cancel_msg = (
            "Are you sure you want to close NMR-EsPy?\n"
        )
        super().__init__(master, cancel_msg=cancel_msg)
        self.green_button["command"] = self.save_options
        self.green_button["text"] = "Save"

        self.help_button["command"] = lambda: webbrowser.open_new(
            f"{pl.DOCSLINK}/content/gui/usage/result.html"
        )

    def save_options(self):
        SaveFrame(self.master)


class AddFrame(wd.MyToplevel):
    def __init__(self, master, index):
        super().__init__(master)
        self.index = index
        self.new_oscillators = None
        self.title("NMR-EsPy - Add oscillators")
        self.ctrl = self.master.master
        self.grab_set()

        titles = [
            "Amplitude",
            "Phase (rad)",
            "Frequency (ppm)",
            "Damping (s⁻¹)",
        ]

        # Empty entry boxes to begin with
        contents = [["", "", "", ""]]
        region = self.ctrl.estimator.get_results(indices=[index])[0] \
                                    .get_region(unit="ppm")

        self.table = wd.MyTable(
            self,
            contents=contents,
            titles=titles,
            region=region,
        )

        # Turn all widgets red initially to indicate they need filling in
        for entry, value_var in zip(self.table.entries[0], self.table.value_vars[0]):
            entry["state"] = "normal"
            entry.key_press()

        self.table.grid(column=0, row=0, padx=10, pady=(10, 0))

        self.button_frame = wd.MyFrame(self)
        self.button_frame.grid(row=1, column=0, padx=10, pady=10)

        self.add_button = wd.MyButton(
            self.button_frame,
            text="Add",
            command=self.add_row,
        )
        self.add_button.grid(row=0, column=0)

        self.cancel_button = wd.MyButton(
            self.button_frame,
            text="Cancel",
            command=self.destroy,
            bg=cf.BUTTONRED,
        )
        self.cancel_button.grid(row=0, padx=(5, 0), column=1)

        self.confirm_button = wd.MyButton(
            self.button_frame,
            text="Confirm",
            command=self.confirm,
            bg=cf.BUTTONGREEN,
        )
        self.confirm_button.grid(row=0, padx=(5, 0), column=2)

    def add_row(self):
        contents = self.table.get_values()
        contents.append(4 * [""])
        self.table.reconstruct(contents, top=0)
        # Set all entry widgets that are empty to red:
        # Loop over each table row
        for entries in self.table.entries:
            # Loop over each entry in a row
            for entry in entries:
                if entry.get() == "":
                    entry["state"] = "normal"
                    entry.key_press()

    def confirm(self):
        if self.table.check_red_entry():
            msg = "Some parameters have not been validated."
            warn_window = fr.WarnWindow(self, msg=msg)
            self.wait_window(warn_window)
            return

        # Extract parameters from table
        oscs = np.array(self.table.get_values())
        # Convert from ppm to hz
        oscs[:, 2] = self.ctrl.estimator.convert(
            [oscs[:, 2]],
            "ppm->hz",
        )[0]
        self.new_oscillators = oscs
        self.destroy()


class SplitFrame(wd.MyToplevel):
    def __init__(self, master, index, to_split):
        super().__init__(master)
        self.grab_set()
        self.title("NMR-EsPy - Split oscillator")
        self.ctrl = self.master.master
        self.index = index

        # Add a frame with some padding from the window edge
        frame = wd.MyFrame(self)
        frame.grid(row=0, column=0, padx=10, pady=10)

        # Window title and widget labels
        wd.MyLabel(
            frame,
            text=f"Splitting Oscillator {to_split}",
            font=(cf.MAINFONT, 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        wd.MyLabel(frame, text="Number of oscillators:").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(frame, text="Frequency separation:").grid(
            row=2,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(frame, text="Amplitude ratio:").grid(
            row=3,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        # --- Number of child oscillators --------------------------------
        # Goes from 2 to max. of 10
        self.number_chooser = tk.Spinbox(
            frame,
            values=tuple(range(2, 11)),
            width=4,
            command=self.update_number,
            state="readonly",
            readonlybackground="white",
        )
        self.number_chooser.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            padx=(10, 0),
            pady=(10, 0),
        )

        # --- Separation frequnecy ---------------------------------------
        # Set default separation frequency as 2Hz
        # Convert frequency to ppm
        self.sep_freq = {
            "hz": 2.0,
            "ppm": self.ctrl.estimator.convert([2.0], "hz->ppm")[0],
        }
        self.sep_entry = wd.MyEntry(
            frame,
            width=10,
            return_command=self.check_freq_sep,
            return_args=(),
        )
        # By default, use Hz as the unit
        self.update_sep_entry(self.sep_freq["hz"])
        self.sep_entry.grid(
            row=2,
            column=1,
            sticky="w",
            padx=(10, 0),
            pady=(10, 0),
        )

        # Option menu to specify the separation frequency unit to use
        self.sep_unit = tk.StringVar()
        self.sep_unit.set("hz")
        options = ("hz", "ppm")
        self.sep_unit_box = tk.OptionMenu(
            frame, self.sep_unit, *options, command=self.change_unit
        )
        self.sep_unit_box["bg"] = cf.BGCOLOR
        self.sep_unit_box["width"] = 2
        self.sep_unit_box["highlightbackground"] = "black"
        self.sep_unit_box["highlightthickness"] = 1
        self.sep_unit_box["menu"]["bg"] = cf.BGCOLOR
        self.sep_unit_box["menu"]["activebackground"] = cf.ACTIVETABCOLOR
        self.sep_unit_box["menu"]["activeforeground"] = "white"
        self.sep_unit_box.grid(
            row=2,
            column=2,
            sticky="w",
            padx=(10, 0),
            pady=(10, 0),
        )

        # --- Ratio of amplitudes for children ---------------------------
        # Valid values consist of  a string of colon-separated integers
        # with the number of values matching the number specified by
        # the number chooser.

        # By default, set each child with equal amplitude
        self.amp_ratio = cf.value_var_dict("1:1", "1:1")
        self.ratio_entry = wd.MyEntry(
            frame,
            width=16,
            textvariable=self.amp_ratio["var"],
            return_command=self.check_amp_ratio,
            return_args=(),
        )
        self.ratio_entry.grid(
            column=1,
            row=3,
            sticky="w",
            columnspan=2,
            padx=(10, 0),
            pady=(10, 0),
        )

        # --- Confirm and Cancel buttons ---------------------------------
        button_frame = wd.MyFrame(frame)
        button_frame.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="e",
            pady=(10, 0),
        )

        self.cancel_button = wd.MyButton(
            button_frame, bg=cf.BUTTONRED, command=self.cancel, text="Cancel"
        )
        self.cancel_button.grid(row=0, column=0, sticky="e")

        self.save_button = wd.MyButton(
            button_frame, bg=cf.BUTTONGREEN, command=self.confirm, text="Confirm"
        )
        self.save_button.grid(row=0, column=1, sticky="e", padx=(10, 0))

    def update_number(self):
        """Called when the number choosing spinbox is changed. Updates
        the amplitude ratio to match the new number of children. Each child
        oscillator is set to have the same amplitude"""
        number = int(self.number_chooser.get())
        self.amp_ratio["value"] = ":".join(number * ["1"])
        self.amp_ratio["var"].set(self.amp_ratio["value"])

    def update_sep_entry(self, value):
        """Update the separation frwquency entry widget"""
        self.sep_entry.delete(0, "end")
        self.sep_entry.insert(0, cf.strip_zeros(f"{value:.5f}"))

    def change_unit(self, *args):
        """Called when the user updates the separation frequecny unit box.
        Updates the separation frequency entry widget accordingly."""
        unit = self.sep_unit.get()
        self.update_sep_entry(self.sep_freq[unit])

    def check_freq_sep(self):
        """Called upon user entering value into the separation frequency
        entry widget. Validates that the input is valid, and updates values
        as required."""
        unit = self.sep_unit.get()
        str_value = self.sep_entry.get()

        try:
            value = float(str_value)
            if value > 0:
                self.sep_freq[unit] = value
                self.update_sep_entry(self.sep_freq[unit])
            else:
                raise

        except Exception:
            self.update_sep_entry(self.sep_freq[unit])
            return

        # Update values for other unit
        from_, to = ("hz", "ppm") if unit == "hz" else ("ppm", "hz")
        self.sep_freq[to] = self.ctrl.estimator.convert(
            [value], f"{from_}->{to}"
        )[0]

    def check_amp_ratio(self):
        """Determine whether a user-given amplitude ratio is valid, and if so,
        update."""
        ratio = self.amp_ratio["var"].get()
        # Regex for string of ints separated by colons
        number = int(self.number_chooser.get())
        regex = r"^\d+(\.\d+)?(:\d+(\.\d+)?)+$"
        # Check that:
        # a) the ratio fully matches the regex
        # b) the number of values matches the specified number of child
        # oscillators
        if re.fullmatch(regex, ratio) and len(ratio.split(":")) == number:
            self.amp_ratio["value"] = ratio
        # If conditions are not met, revert back the previous valid value
        else:
            self.amp_ratio["var"].set(self.amp_ratio["value"])

    def cancel(self):
        self.split_info = None
        self.destroy()

    def confirm(self):
        """Perform the oscillator split and update the plot and parameter
        table"""
        self.split_info = {
            "separation": self.sep_freq["hz"],
            "number": int(self.number_chooser.get()),
            "amp_ratio": [float(i) for i in self.amp_ratio["var"].get().split(":")],
        }
        self.destroy()


class SaveFrame(wd.MyToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.label = wd.MyLabel(self, text="TODO")
        self.pack()
