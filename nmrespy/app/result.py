# result.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 07 Mar 2023 13:27:13 GMT

import io
from pathlib import Path
import re
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import webbrowser

from matplotlib.backends import backend_tkagg

from nmrespy._colors import GRE, END, USE_COLORAMA
import nmrespy._paths_and_links as pl
from nmrespy.app import config as cf, custom_widgets as wd, frames as fr
import numpy as np

if USE_COLORAMA:
    import colorama
    colorama.init()


class Result1DType(wd.MyToplevel):
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
        self.unstage_buttons = []
        self.rerun_buttons = []
        self.edit_boxes = []
        self.staged_edits = []
        self.staged_oscs = []

        self.n_regions = len(self.estimator._results)

        self.update_all_tabs(0)

    def update_all_tabs(self, current, replace=False):
        for idx in range(self.n_regions):
            self.new_tab(idx, replace)
        self.notebook.select(current)

    def new_tab(self, idx, replace=False):
        def append(lst, obj):
            if replace:
                lst.pop(idx)
                lst.insert(idx, obj)
            else:
                lst.append(obj)

        if replace:
            self.tabs[idx].destroy()

        append(
            self.tabs,
            wd.MyFrame(self.notebook, bg=cf.NOTEBOOKCOLOR),
        )
        self.tabs[idx].columnconfigure(0, weight=1)
        self.tabs[idx].rowconfigure(0, weight=1)

        if replace and idx < self.n_regions - 1:
            self.notebook.insert(
                idx,
                self.tabs[idx],
                text=str(idx),
                sticky="nsew",
            )

        else:
            self.notebook.add(
                self.tabs[idx],
                text=str(idx),
                sticky="nsew",
            )
            self.histories.append(
                [
                    (
                        self.estimator.get_params(indices=[idx]),
                        self.estimator.get_errors(indices=[idx]),
                    )
                ]
            )

        append(
            self.table_frames,
            wd.MyFrame(
                self.tabs[idx],
                bg=cf.NOTEBOOKCOLOR,
                highlightbackground="black",
                highlightthickness=3,
            ),
        )
        self.table_frames[idx].columnconfigure(0, weight=1)
        self.table_frames[idx].rowconfigure(0, weight=1)
        self.table_frames[idx].grid(
            row=0, column=2, rowspan=2, padx=(0, 10), pady=10, sticky="ns",
        )

        append(
            self.tables,
            wd.MyTable(
                self.table_frames[idx],
                contents=self.estimator.get_params(indices=[idx], funit="ppm"),
                titles=self.table_titles,
                region=self.get_region(idx),
                bg=cf.NOTEBOOKCOLOR,
            ),
        )
        self.tables[idx].grid(
            row=0, column=0, columnspan=4, padx=(0, 10), pady=10, sticky="n",
        )

        append(
            self.button_frames,
            wd.MyFrame(self.table_frames[idx], bg=cf.NOTEBOOKCOLOR),
        )
        for r in range(4):
            self.button_frames[idx].columnconfigure(r, weight=1)
        self.button_frames[idx].grid(row=1, column=0, sticky="s")

        append(
            self.add_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Add",
                command=self.add,
            ),
        )
        self.add_buttons[idx].grid(
            row=0, column=0, padx=(10, 0), pady=(10, 0), sticky="ew",
        )

        append(
            self.remove_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Remove",
                state="disabled",
                command=self.remove,
            ),
        )
        self.remove_buttons[idx].grid(
            row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="ew",
        )

        append(
            self.merge_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Merge",
                state="disabled",
                command=self.merge,
            ),
        )
        self.merge_buttons[idx].grid(
            row=0, column=2, padx=(10, 0), pady=(10, 0), sticky="ew",
        )

        append(
            self.split_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Split",
                state="disabled",
                command=self.split,
            ),
        )
        self.split_buttons[idx].grid(
            row=0, column=3, padx=10, pady=(10, 0), sticky="ew",
        )

        append(
            self.unstage_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Unstage",
                state="disabled",
                command=self.unstage,
            ),
        )
        self.unstage_buttons[idx].grid(
            row=1, column=1, padx=(10, 0), pady=10, sticky="ew",
        )

        append(
            self.undo_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Undo",
                state="normal" if len(self.histories[idx]) > 1 else "disabled",
                command=self.undo,
            ),
        )
        self.undo_buttons[idx].grid(
            row=1, column=2, padx=(10, 0), pady=10, sticky="ew",
        )

        append(
            self.rerun_buttons,
            wd.MyButton(
                self.button_frames[idx],
                text="Re-run",
                state="disabled",
                command=self.rerun,
                fg="red",
            ),
        )
        self.rerun_buttons[idx].grid(
            row=1, column=3, padx=10, pady=10, sticky="ew",
        )

        self.tables[idx].selected_number.trace("w", self.configure_button_states)

        append(
            self.edit_boxes,
            tk.Text(
                self.table_frames[idx],
                width=40,
                height=10,
            ),
        )
        self.edit_boxes[idx].grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.append_text(idx, f"Edits staged for result {idx}:\n")

        append(self.staged_edits, [])
        append(self.staged_oscs, [])

        self.notebook.select(idx)

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

    def clear_text(self, idx):
        self.edit_boxes[idx]["state"] = "normal"
        self.edit_boxes[idx].delete("1.0", tk.END)
        self.edit_boxes[idx]["state"] = "disabled"

    def reset_table(self, idx):
        self.tables[idx].selected_rows = []
        self.tables[idx].activate_rows(top=self.tables[idx].top)
        self.tables[idx].selected_number.set(0)

    def activate_rerun_button(self, idx):
        self.rerun_buttons[idx]["state"] = "normal"
        self.unstage_buttons[idx]["state"] = "normal"
        self.button_frame.green_button["state"] = "disabled"

    def deactivate_rerun_button(self, idx):
        self.rerun_buttons[idx]["state"] = "disabled"
        self.unstage_buttons[idx]["state"] = "disabled"
        self.button_frame.green_button["state"] = "normal"

    def check_for_duplicated_oscs(self, new_oscs, idx):
        curr_oscs = [oscs for osc_list in self.staged_oscs[idx] for oscs in osc_list]
        return bool(set(new_oscs) & set(curr_oscs))

    def add(self):
        idx = self.get_idx()
        add_frame = AddFrame(self, self.table_titles, idx)
        self.wait_window(add_frame)
        oscs = add_frame.new_oscillators
        if oscs is not None:
            self.staged_oscs[idx].append([])
            self.staged_edits[idx].append((0, oscs))
            self.edit_boxes[idx]["state"] = "normal"
            plural = "s" if oscs.shape[0] > 1 else ""
            self.append_text(
                idx,
                f"--> Add {oscs.shape[0]} oscillator{plural}\n",
            )
            self.activate_rerun_button(idx)

        self.reset_table(idx)

    def remove(self):
        idx = self.get_idx()
        to_rm = self.tables[idx].selected_rows
        if not self.check_for_duplicated_oscs(to_rm, idx):
            self.staged_oscs[idx].append(to_rm)
            self.staged_edits[idx].append((1, to_rm))
            plural = "s" if len(to_rm) > 1 else ""
            self.append_text(
                idx,
                f"--> Remove oscillator{plural} {', '.join([str(x) for x in to_rm])}\n",
            )
            self.activate_rerun_button(idx)

        self.reset_table(idx)

    def merge(self):
        idx = self.get_idx()
        to_merge = self.tables[idx].selected_rows
        if not self.check_for_duplicated_oscs(to_merge, idx):
            self.staged_oscs[idx].append(to_merge)
            self.staged_edits[idx].append((2, to_merge))
            self.append_text(
                idx,
                f"--> Merge oscillators {', '.join([str(x) for x in sorted(to_merge)])}\n",  # noqa: E501
            )
            self.activate_rerun_button(idx)

        self.reset_table(idx)

    def split(self):
        idx = self.get_idx()
        to_split = self.tables[idx].selected_rows
        if not self.check_for_duplicated_oscs(to_split, idx):
            split_frame = SplitFrame(self, idx, to_split[0])
            self.wait_window(split_frame)
            split_info = split_frame.split_info
            if split_info is not None:
                self.staged_oscs[idx].append([to_split])
                self.staged_edits[idx].append((3, {to_split[0]: split_info}))
                self.append_text(
                    idx,
                    f"--> Split oscillator {to_split[0]}\n",
                )
                self.activate_rerun_button(idx)

        self.reset_table(idx)

    def undo(self):
        idx = self.get_idx()
        self.histories[idx].pop()
        params, errors = self.histories[idx][-1]
        self.estimator._results[idx].params = params
        self.estimator._results[idx].errors = errors
        self.new_tab(idx, replace=True)
        self.notebook.select(idx)
        self.button_frame.green_button["state"] = "normal"

    def unstage(self):
        idx = self.get_idx()
        self.staged_edits[idx].pop()
        self.staged_oscs[idx].pop()
        lines = self.edit_boxes[idx].get("1.0", tk.END)[:-2].split("\n")
        lines.pop()
        self.clear_text(idx)
        self.append_text(idx, "\n".join(lines) + "\n")
        if not self.staged_edits[idx]:
            self.unstage_buttons[idx]["state"] = "disabled"
            self.deactivate_rerun_button(idx)
        self.reset_table(idx)

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
                    rm_oscs = info
                else:
                    rm_oscs.extend(info)
            elif typ == 2:
                if merge_oscs is None:
                    merge_oscs = [info]
                else:
                    merge_oscs.append(info)
            elif typ == 3:
                if split_oscs is None:
                    split_oscs = info
                else:
                    split_oscs = {**split_oscs, **info}

        self.ctrl.estimator.edit_result(
            index=idx,
            add_oscs=add_oscs,
            rm_oscs=rm_oscs,
            merge_oscs=merge_oscs,
            split_oscs=split_oscs,
        )
        self.histories[idx].append(
            (
                self.ctrl.estimator.get_params(indices=[idx]),
                self.ctrl.estimator.get_errors(indices=[idx]),
            )
        )

        self.new_tab(idx, replace=True)
        self.notebook.select(idx)
        self.button_frame.green_button["state"] = "normal"


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
    def __init__(self, master, titles, index):
        super().__init__(master)
        self.titles = titles
        self.cols = len(self.titles)
        self.dim = self.cols // 2 - 1
        self.index = index
        self.new_oscillators = None

        self.title("NMR-EsPy - Add oscillators")
        self.ctrl = self.master.master
        self.grab_set()

        # Empty entry boxes to begin with
        contents = [["" for _ in range(self.cols)]]

        self.table = wd.MyTable(
            self,
            contents=contents,
            titles=titles,
            region=master.get_region(index),
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
        self.title("NMR-EsPy - Save Result")
        self.ctrl = self.master.master
        self.grab_set()

        # --- Result figure ----------------------------------------------
        self.fig_frame = wd.MyFrame(self)
        self.fig_frame.grid(row=0, column=0, pady=(10, 0), padx=10, sticky="w")

        wd.MyLabel(
            self.fig_frame,
            text="Result Figure",
            font=("Helvetica", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        wd.MyLabel(self.fig_frame, text="Save Figure:",).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(self.fig_frame, text="Format:").grid(
            row=2,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(self.fig_frame, text="Filename:").grid(
            row=3,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(self.fig_frame, text="dpi:").grid(
            row=4,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(self.fig_frame, text="Size (cm):").grid(
            row=5,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        self.save_fig = tk.IntVar()
        self.save_fig.set(1)
        self.save_fig_checkbutton = wd.MyCheckbutton(
            self.fig_frame,
            variable=self.save_fig,
            command=self.update_save_fig,
        )
        self.save_fig_checkbutton.grid(
            row=1,
            column=1,
            sticky="w",
            pady=(10, 0),
        )

        self.fig_fmt = tk.StringVar()
        self.fig_fmt.set("pdf")
        self.fig_fmt.trace("w", self.update_fig_fmt)

        options = ("eps", "jpg", "pdf", "png", "ps", "svg")
        self.sep_unit_box = tk.OptionMenu(self.fig_frame, self.fig_fmt, *options)
        self.sep_unit_box["bg"] = cf.BGCOLOR
        self.sep_unit_box["width"] = 5
        self.sep_unit_box["highlightbackground"] = "black"
        self.sep_unit_box["highlightthickness"] = 1
        self.sep_unit_box["menu"]["bg"] = cf.BGCOLOR
        self.sep_unit_box["menu"]["activebackground"] = cf.ACTIVETABCOLOR
        self.sep_unit_box["menu"]["activeforeground"] = "white"
        self.sep_unit_box.grid(
            row=2,
            column=1,
            sticky="w",
            pady=(10, 0),
        )

        self.fig_name_frame = wd.MyFrame(self.fig_frame)
        self.fig_name_frame.grid(row=3, column=1, sticky="w", pady=(10, 0))
        self.fig_name = tk.StringVar()
        self.fig_name.set("nmrespy_figure")
        self.fig_name_entry = wd.MyEntry(
            self.fig_name_frame,
            textvariable=self.fig_name,
            width=18,
            return_command=self.update_file_name,
            return_args=(self.fig_name,),
        )
        self.fig_name_entry.grid(column=0, row=0)

        self.fig_fmt_label = wd.MyLabel(self.fig_name_frame)
        self.update_fig_fmt()
        self.fig_fmt_label.grid(column=1, row=0, padx=(2, 0), pady=(5, 0))

        self.fig_dpi = cf.value_var_dict(300, "300")
        self.fig_dpi_entry = wd.MyEntry(
            self.fig_frame,
            textvariable=self.fig_dpi["var"],
            width=6,
            return_command=self.update_fig_dpi,
            return_args=(),
        )
        self.fig_dpi_entry.grid(row=4, column=1, sticky="w", pady=(10, 0))

        self.fig_width = cf.value_var_dict(15, "15")
        self.fig_height = cf.value_var_dict(10, "10")

        self.fig_size_frame = wd.MyFrame(self.fig_frame)
        self.fig_size_frame.grid(row=5, column=1, sticky="w", pady=(10, 0))

        wd.MyLabel(self.fig_size_frame, text="w:").grid(column=0, row=0)
        wd.MyLabel(self.fig_size_frame, text="h:").grid(column=2, row=0)

        for i, (dim, value) in enumerate(zip(("width", "height"), (15, 10))):

            self.__dict__[f"fig_{dim}"] = cf.value_var_dict(value, str(value))
            self.__dict__[f"fig_{dim}_entry"] = wd.MyEntry(
                self.fig_size_frame,
                textvariable=self.__dict__[f"fig_{dim}"]["var"],
                return_command=self.update_fig_size,
                return_args=(dim,),
            )

            padx, column = ((2, 5), 1) if i == 0 else ((2, 0), 3)
            self.__dict__[f"fig_{dim}_entry"].grid(
                column=column,
                row=0,
                padx=padx,
            )

        # --- Other result files -----------------------------------------
        self.file_frame = wd.MyFrame(self)
        self.file_frame.grid(row=1, column=0, padx=10, sticky="w")

        wd.MyLabel(
            self.file_frame, text="Result Files", font=("Helvetica", 12, "bold")
        ).grid(row=0, column=0, pady=(20, 0), columnspan=4, sticky="w")

        wd.MyLabel(self.file_frame, text="Format:").grid(
            row=1, column=0, pady=(10, 0), columnspan=2, sticky="w"
        )
        wd.MyLabel(self.file_frame, text="Filename:").grid(
            row=1,
            column=2,
            columnspan=2,
            padx=(20, 0),
            pady=(10, 0),
            sticky="w",
        )

        titles = ("Text:", "PDF:")
        self.fmts = ("txt", "pdf")
        for i, (title, tag) in enumerate(zip(titles, self.fmts)):

            wd.MyLabel(self.file_frame, text=title).grid(
                row=i + 2,
                column=0,
                pady=(10, 0),
                sticky="w",
            )

            # Dictates whether to save the file format
            self.__dict__[f"save_{tag}"] = save_var = tk.IntVar()
            save_var.set(1)

            self.__dict__[f"{tag}_check"] = check = wd.MyCheckbutton(
                self.file_frame,
                variable=save_var,
                command=lambda tag=tag: self.update_save_file(tag),
            )
            check.grid(
                row=i + 2,
                column=1,
                padx=(2, 0),
                pady=(10, 0),
                sticky="w",
            )

            self.__dict__[f"name_{tag}"] = fname_var = tk.StringVar()
            fname_var.set("nmrespy_result")

            self.__dict__[f"{tag}_entry"] = entry = wd.MyEntry(
                self.file_frame,
                textvariable=fname_var,
                width=18,
                return_command=self.update_file_name,
                return_args=(fname_var,),
            )
            entry.grid(
                row=i + 2,
                column=2,
                padx=(20, 0),
                pady=(10, 0),
                sticky="w",
            )

            self.__dict__[f"{tag}_ext"] = ext = wd.MyLabel(
                self.file_frame,
                text=f".{tag}",
            )
            ext.grid(row=i + 2, column=3, pady=(15, 0), sticky="w")

        self.pdflatex = self.master.master.pdflatex
        check_latex = subprocess.call(
            f"{self.pdflatex if self.pdflatex is not None else 'pdflatex'} -v",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        )

        # pdflatex could not be found. Deny selecting PDF option
        if check_latex != 0:
            self.save_pdf.set(0)
            self.pdf_check["state"] = "disabled"
            self.pdf_entry["state"] = "disabled"
            self.name_pdf.set("")
            self.pdf_ext["fg"] = "#808080"

        wd.MyLabel(self.file_frame, text="Description:").grid(
            row=5,
            column=0,
            columnspan=4,
            pady=(10, 0),
            sticky="w",
        )

        self.descr_box = wd.MyText(self.file_frame, width=30, height=3)
        self.descr_box.grid(
            row=6,
            column=0,
            columnspan=4,
            pady=(10, 0),
            sticky="ew",
        )

        # --- Pickle Estimator -------------------------------------------
        self.pickle_frame = wd.MyFrame(self)
        self.pickle_frame.grid(row=2, column=0, padx=10, sticky="w")

        wd.MyLabel(
            self.pickle_frame, text="Estimator", font=("Helvetica", 12, "bold")
        ).grid(row=0, column=0, pady=(20, 0), columnspan=4, sticky="w")

        wd.MyLabel(self.pickle_frame, text="Save Estimator:",).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        wd.MyLabel(self.pickle_frame, text="Filename:",).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        self.pickle_estimator = tk.IntVar()
        self.pickle_estimator.set(1)
        self.pickle_estimator_checkbutton = wd.MyCheckbutton(
            self.pickle_frame,
            variable=self.pickle_estimator,
            command=self.update_pickle_estimator,
        )
        self.pickle_estimator_checkbutton.grid(
            row=1,
            column=1,
            sticky="w",
            pady=(10, 0),
        )

        self.pickle_name_frame = wd.MyFrame(self.pickle_frame)
        self.pickle_name_frame.grid(row=2, column=1, sticky="w", pady=(10, 0))
        self.pickle_name = tk.StringVar()
        self.pickle_name.set("estimator")
        self.pickle_name_entry = wd.MyEntry(
            self.pickle_name_frame,
            textvariable=self.pickle_name,
            width=18,
            return_command=self.update_file_name,
            return_args=(self.pickle_name,),
        )
        self.pickle_name_entry.grid(column=0, row=0)

        self.pickle_ext_label = wd.MyLabel(self.pickle_name_frame, text=".pkl")
        self.pickle_ext_label.grid(column=1, row=0, padx=(2, 0), pady=(5, 0))

        # --- Directory selection ----------------------------------------
        self.dir_frame = wd.MyFrame(self)
        self.dir_frame.grid(row=3, column=0, padx=10, sticky="w")

        wd.MyLabel(
            self.dir_frame, text="Directory", font=("Helvetica", 12, "bold"),
        ).grid(
            row=0, column=0, pady=(20, 0), columnspan=2, sticky="w"
        )

        self.dir_name = tk.StringVar()
        path = Path.home()
        self.dir_name = cf.value_var_dict(path, str(path))

        self.dir_entry = wd.MyEntry(
            self.dir_frame,
            textvariable=self.dir_name["var"],
            width=30,
            return_command=self.update_dir,
            return_args=(),
        )
        self.dir_entry.grid(row=1, column=0, pady=(10, 0), sticky="w")

        self.img = cf.get_PhotoImage(cf.FOLDERPATH, scale=0.02)

        self.dir_button = wd.MyButton(
            self.dir_frame,
            command=self.browse,
            image=self.img,
            width=32,
            bg=cf.BGCOLOR,
        )
        self.dir_button.grid(row=1, column=1, padx=(5, 0), pady=(10, 0))

        # --- Save/cancel buttons ----------------------------------------
        # buttons at the bottom of the frame
        self.button_frame = wd.MyFrame(self)
        self.button_frame.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="e")
        # cancel button - returns usere to result toplevel
        self.cancel_button = wd.MyButton(
            self.button_frame,
            text="Cancel",
            bg=cf.BUTTONRED,
            command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=0, pady=(10, 0))

        # save button - determines what file types to save and generates them
        self.save_button = wd.MyButton(
            self.button_frame,
            text="Save",
            width=8,
            bg=cf.BUTTONGREEN,
            command=self.save
        )
        self.save_button.grid(
            row=0,
            column=1,
            padx=(10, 0),
            pady=(10, 0),
        )

    def update_save_fig(self):
        state = "normal" if self.save_fig.get() else "disabled"
        widgets = [
            self.sep_unit_box,
            self.fig_name_entry,
            self.fig_dpi_entry,
            self.fig_width_entry,
            self.fig_height_entry,
        ]

        for widget in widgets:
            widget["state"] = state

        self.fig_fmt_label["fg"] = "#000000" if state == "normal" else "#808080"

    def update_fig_fmt(self, *args):
        self.fig_fmt_label["text"] = f".{self.fig_fmt.get()}"

    def update_fig_dpi(self, *args):
        try:
            # Try to convert dpi text variable as an int. Both ensures
            # the user-given value can be converted to a numerical value
            # and removes any decimal places, if given. Reset as a string
            # afterwards.
            dpi = int(self.fig_dpi["var"].get())
            if not dpi > 0:
                raise
            self.fig_dpi["var"].set(str(dpi))
            self.fig_dpi["value"] = dpi

        except Exception:
            # Failed to convert to int, reset to previous value
            self.fig_dpi["var"].set(str(self.fig_dpi["value"]))

    def update_fig_size(self, dim):
        try:
            length = float(self.__dict__[f"fig_{dim}"]["var"].get())
            if not length > 0:
                raise

            if cf.check_int(length):
                # If length is an integer, remove decimal places.
                length = int(length)

            self.__dict__[f"fig_{dim}"]["value"] = length
            self.__dict__[f"fig_{dim}"]["var"].set(str(length))

        except Exception:
            # Failed to convert to float, reset to previous value
            pass

        self.__dict__[f"fig_{dim}"]["var"].set(
            str(self.__dict__[f"fig_{dim}"]["value"])
        )

    # Result file settings
    def update_save_file(self, tag):
        state = "normal" if self.__dict__[f"save_{tag}"].get() else "disabled"
        self.__dict__[f"{tag}_entry"]["state"] = state
        self.__dict__[f"{tag}_ext"]["fg"] = (
            "#000000" if state == "normal" else "#808080"
        )

    def update_file_name(self, var):
        name = var.get()
        var.set("".join(x for x in name if x.isalnum() or x in " _-"))

    # Pickle estimator
    def update_pickle_estimator(self):
        state = "normal" if self.pickle_estimator.get() else "disabled"
        self.pickle_name_entry["state"] = state
        self.pickle_ext_label["fg"] = "#000000" if state == "normal" else "#808080"

    # Save directory
    def browse(self):
        """Directory selection using tkinter's filedialog"""
        name = tk.filedialog.askdirectory(initialdir=self.dir_name["value"])
        # If user clicks close cross, an empty tuple is returned
        if name:
            self.dir_name["value"] = Path(name).resolve()
            self.dir_name["var"].set(str(self.dir_name["value"]))

    def update_dir(self):
        path = Path(self.dir_name["var"].get()).resolve()
        if path.is_dir():
            self.dir_name["value"] = path
            self.dir_name["var"].set(path)
        else:
            self.dir_name["var"].set(str(self.dir_name["value"]))

    def save(self):
        if not cf.check_invalid_entries(self):
            msg = "Some parameters have not been validated."
            warn_window = fr.WarnWindow(self, msg=msg)
            self.wait_window(warn_window)
            return

        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        dir_ = self.dir_name["value"]

        # Figure
        if self.save_fig.get():
            # Generate figure path
            fig_fmt = self.fig_fmt.get()
            dpi = self.fig_dpi["value"]
            fig_name = self.fig_name.get()
            fig_path = (dir_ / f"{fig_name}").with_suffix(f".{fig_fmt}")

            # Convert size from cm -> inches
            figsize = (
                self.fig_width["value"] / 2.54,
                self.fig_height["value"] / 2.54,
            )
            fig = self.generate_figure(figsize, dpi)[0]
            fig.savefig(fig_path)

        # Result files
        for fmt in ("txt", "pdf"):
            if self.__dict__[f"save_{fmt}"].get():
                name = self.__dict__[f"name_{fmt}"].get()
                path = str(dir_ / name)
                description = self.descr_box.get("1.0", "end-1c")
                if description == "":
                    description = None

                self.ctrl.estimator.write_result(
                    path=path,
                    description=description,
                    fmt=fmt,
                    force_overwrite=True,
                    pdflatex_exe=self.pdflatex,
                )

        if self.pickle_estimator.get():
            name = self.pickle_name.get()
            path = str(dir_ / name)
            self.ctrl.estimator.to_pickle(path=path, force_overwrite=True)

        sys.stdout = old_stdout
        msg = mystdout.getvalue() \
                      .replace(GRE, "") \
                      .replace(END, "") \
                      .replace("Saved", "• Saved")

        wdw = fr.ConfirmWindow(self, msg, inc_no=False)
        self.wait_window(wdw)

        self.ctrl.destroy()
