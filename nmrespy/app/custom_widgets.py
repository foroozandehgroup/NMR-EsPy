# custom_widgets.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 21 Oct 2022 12:48:15 BST

"""
Customised widgets for NMR-EsPy GUI.
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import numpy as np

import nmrespy.app.config as cf


def generate(cls_, keys, values, kwargs):
    """Configure class attributes. Enables flexibility to change default
    behaviour by including kwargs"""
    for key, value in zip(keys, values):
        cls_[key] = value
    # Overwrite default properties with any provided kwargs
    for kwkey, kwvalue in zip(kwargs.keys(), kwargs.values()):
        cls_[kwkey] = kwvalue


class MyFrame(tk.Frame):
    """A Tkinter frame with a white background by default."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        generate(self, ("bg",), (cf.BGCOLOR,), kwargs)


class MyToplevel(tk.Toplevel):
    """A Tkinter toplevel, with the default backgroun for the app, and
    which is not resizable by default.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        self.iconbitmap = cf.ICONPATH

        generate(self, ("bg",), (cf.BGCOLOR,), kwargs)

        self.title("NMR-EsPy")
        self.resizable(False, False)


class MyLabel(tk.Label):
    """Tkinter label with white background by deafult"""

    def __init__(self, parent, bold=False, **kwargs):
        super().__init__(parent)
        keys = ("bg", "font")

        if bold:
            values = (cf.BGCOLOR, (cf.MAINFONT, "11", "bold"))
        else:
            values = (cf.BGCOLOR, (cf.MAINFONT, "11"))

        generate(self, keys, values, kwargs)


class MyButton(tk.Button):
    """Tkinter button with various tweaks"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        keys = (
            "width",
            "highlightbackground",
            "bg",
            "disabledforeground",
        )
        values = (8, "black", cf.BUTTONDEFAULT, cf.BUTTONDEFAULT)

        generate(self, keys, values, kwargs)


class MyTextbox(MyToplevel):

    def __init__(self, parent, text, **kwargs):
        super().__init__(parent)
        label = MyText(self, text)
        label.grid(row=0, column=0, padx=(10, 10), pady=(10, 0))
        close_button = MyButton(
            self, color=cf.BUTTONRED, text="Close", command=self.destroy,
        )
        close_button.grid(row=1, column=1, padx=(10, 10), pady=(10, 10), sticky="e")


class MyCheckbutton(tk.Checkbutton):
    """Tkinter button with various tweaks"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        keys = ("bg", "highlightthickness", "bd")
        values = (cf.BGCOLOR, 0, 0)

        generate(self, keys, values, kwargs)


class MyScale(tk.Scale):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        keys = (
            "orient",
            "showvalue",
            "sliderlength",
            "bd",
            "highlightthickness",
            "highlightbackground",
            "relief",
            "bg",
            "troughcolor",
        )
        values = (tk.HORIZONTAL, 0, 15, 0, 1, "black", "flat", cf.BGCOLOR, "white")

        generate(self, keys, values, kwargs)


class MyEntry(tk.Entry):
    """Entry widget with some aesthetic teweaks.

    ``return_command`` specifies a callable to be bound to the widget
    for when the user presses <Return>. If the callable has any arguments,
    these should be provided as a tuple into ``return_args``.
    The upshot of these commands is the text becomes red once the user
    changes the input, and goes back to black once <Return> has been
    pressed. The idea is to warn the user that they haven't validated their
    changes since altering the entry widget.
    """

    def __init__(self, parent, return_command=None, return_args=None, **kwargs):
        super().__init__(parent)

        self.return_command = return_command
        self.return_args = return_args

        keys = (
            "width",
            "highlightthickness",
            "highlightbackground",
            "bg",
            "readonlybackground",
        )
        values = (7, 1, "black", "white", cf.READONLYENTRYCOLOR)

        generate(self, keys, values, kwargs)

        if self.return_command is not None:
            self.return_args = () if self.return_args is None else self.return_args
            self.bind_command()

    def bind_command(self):
        self.bind("<Key>", lambda event: self.key_press())
        self.bind("<Return>", lambda event: self.return_press())
        self.bind("<Tab>", lambda event: None)

    def key_press(self):
        if self["state"] == "normal":
            self["fg"] = "red"
            self["highlightcolor"] = "red"
            self["highlightbackground"] = "red"

    def black_highlight(self):
        self["fg"] = "black"
        self["highlightcolor"] = "black"
        self["highlightbackground"] = "black"

    def return_press(self):
        self.black_highlight()
        self.return_command(*self.return_args)


class MyOptionMenu(tk.OptionMenu):
    def __init__(self, parent, variable, value, *values, **kwargs):

        super().__init__(parent, variable, value, values, kwargs)

        keys = ("bg", "borderwidth", "width", "highlightcolor")
        values = (cf.BGCOLOR, 1, 10, "black")

        generate(self, keys, values, kwargs)

        if "bg" in kwargs.keys():
            self["menu"]["bg"] = kwargs["bg"]
        else:
            self["menu"]["bg"] = cf.BGCOLOR


class MyText(tk.Text):
    def __init__(self, parent, **kwargs):

        super().__init__(parent)

        keys = ("bg", "highlightcolor", "highlightbackground")
        values = ("white", "black", "black")

        generate(self, keys, values, kwargs)


class MyNavigationToolbar(NavigationToolbar2Tk):
    """Tweak default matplotlib navigation bar to exclude subplot-config
    and save buttons. Also dialogues as cursor goes over plot, and bar
    is set to be white"""

    def __init__(self, canvas, parent, color=cf.BGCOLOR):

        # slice toolitems (this gets rid of the unwanted buttons)
        self.toolitems = self.toolitems[:6]

        super().__init__(canvas, parent, pack_toolbar=False)

        # make everything white
        self["bg"] = color
        self._message_label["bg"] = color
        for button in self.winfo_children():
            button["bg"] = color

    def set_message(self, msg):
        pass


class MyLabelScaleEntry(MyFrame):
    def __init__(
        self,
        master,
        name,
        frame_kw=None,
        label_kw=None,
        scale_kw=None,
        entry_kw=None,
    ):
        frame_kw = {} if frame_kw is None else frame_kw
        super().__init__(master, **frame_kw)
        self.columnconfigure(1, weight=1)
        label_kw = {} if label_kw is None else label_kw
        scale_kw = {} if scale_kw is None else scale_kw
        entry_kw = {} if entry_kw is None else entry_kw
        self.label = MyLabel(self, text=name, **label_kw)
        self.scale = MyScale(self, **scale_kw)
        self.entry = MyEntry(self, **entry_kw)

        for col, (wgt, sticky, padx) in enumerate(zip(
            (self.label, self.scale, self.entry),
            ("w", "ew", "w"),
            ((0, 10), (0, 10), 0),
        )):
            wgt.grid(row=0, column=col, padx=padx, sticky=sticky)


class NOscWidget(MyFrame):
    def __init__(self, master):
        super().__init__(master)
        self.label = MyLabel(self, text="number of oscillators:")
        self.mdl_label = MyLabel(self, text="Use MDL:")
        self.mdl_var = tk.IntVar(self)
        self.mdl_var.set(1)
        self.mdl_box = MyCheckbutton(
            self, command=self.update_mdl_box, variable=self.mdl_var,
        )
        self.noscs = 0
        self.entry = MyEntry(
            self,
            return_command=self.check_noscs,
            state="disabled",
        )

        self.label.grid(row=0, column=0)
        self.entry.grid(row=0, column=1, padx=(5, 0))
        self.mdl_label.grid(row=0, column=2, padx=(15, 0))
        self.mdl_box.grid(row=0, column=3, padx=(5, 0))

    def update_mdl_box(self):
        if self.mdl_var.get():
            self.entry["state"] = "disabled"
            self.entry.black_highlight()
        else:
            self.entry["state"] = "normal"
        self.check_noscs()

    def check_noscs(self):
        inpt = self.entry.get()
        try:
            value = int(inpt)
            assert value > 0
            self.noscs = value

        except Exception:
            pass

        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.noscs) if self.noscs > 0 else "")

        if self.entry.get() == "":
            self.entry.key_press()


class MyTable(MyFrame):
    def __init__(self, master, contents, titles, region, bg=cf.BGCOLOR):
        super().__init__(master, bg=bg)
        self.bg = bg
        self.titles = titles
        self.dim = len(self.titles) // 2 - 1
        self.region = region
        # Number of selected rows
        self.selected_number = tk.IntVar()
        self.selected_number.set(0)
        self.selected_rows = []
        self.max_rows = 10

        self.create_value_vars(contents)
        self.construct(top=0)

    def create_value_vars(self, contents):
        """Create a nested list of dictionaries, each containing a tkinter
        StringVar, and a float"""

        self.value_vars = []
        for osc in contents:
            value_var_row = []
            for param in osc:
                if isinstance(param, (int, float)):
                    value_var = cf.value_var_dict(
                        param,
                        cf.strip_zeros(f"{param:.6g}"),
                    )
                else:
                    # The only occasion when this should occur in the program
                    # is when param is an empty string. This occurs when
                    # oscillators are added in the AddFrame widget.
                    value_var = cf.value_var_dict(param, param)
                value_var_row.append(value_var)
            self.value_vars.append(value_var_row)

    def construct(self, top):
        """Generate a table of the parameters. Creates a maximum of ``max_rows``
        rows, starting from ``top``."""

        self.table_frame = MyFrame(self, bg=self.bg)
        self.table_frame.grid(row=0, column=0)

        # Column titles
        for column, title in enumerate(["#"] + self.titles):
            padx = 0 if column == 0 else (5, 0)
            sticky = "" if column == 0 else "w"
            MyLabel(self.table_frame, text=title, bg=self.bg).grid(
                row=0,
                column=column,
                padx=padx,
                sticky=sticky,
            )

        # Store entry widgets and string variables
        self.labels = []
        self.entries = []
        # Get the value_var dictionaries corresponding to oscillators that
        # will be present in the table, based on `top` and `self.max_rows`.
        value_var_rows = [
            elem
            for i, elem in enumerate(self.value_vars)
            if (top <= i < top + self.max_rows)
        ]

        for i, value_var_row in enumerate(value_var_rows):
            # Oscillator labels.
            # These act as a oscillator selection widgets
            label = MyLabel(self.table_frame, text=str(top + i))
            # Bind to left mouse click: select oscillator
            label.bind(
                "<Button-1>",
                lambda ev, i=i, top=top: self.left_click(i, top),
            )
            # Bind to left mouse click + shift: select oscillator, keep
            # other already selected oscillators still selected.
            label.bind(
                "<Shift-Button-1>",
                lambda ev, i=i, top=top: self.shift_left_click(i, top),
            )
            # Add some internal padding to make selection easy.
            label.grid(row=i + 1, column=0, pady=(5, 0), ipadx=10, ipady=2)
            self.labels.append(label)

            # Row of parameter entry widgets
            entry_row = []

            for j, value_var in enumerate(value_var_row):
                entry = MyEntry(
                    self.table_frame,
                    textvariable=value_var["var"],
                    state="disabled",
                    width=10,
                    return_command=self.check_param,
                    return_args=(i, j),
                )

                entry.grid(row=i + 1, column=j + 1, padx=(5, 0), pady=(5, 0))
                entry_row.append(entry)

            self.entries.append(entry_row)

        # Activate any active oscillators
        # Colours all row labels corresponding to oscillators selected
        self.activate_rows(top)

        # Add naviagtion buttons if more than `self.max_rows` oscillators
        if len(self.value_vars) > self.max_rows:
            self.navigate_frame = MyFrame(self, bg=self.bg)
            self.navigate_frame.grid(row=1, column=0, pady=(10, 0))

            self.arrow_frame = MyFrame(self.navigate_frame, bg=self.bg)
            self.arrow_frame.grid(row=0, column=0)

            self.up_arrow_img = cf.get_PhotoImage(cf.UPARROWPATH, scale=0.5)
            self.down_arrow_img = cf.get_PhotoImage(cf.DOWNARROWPATH, scale=0.5)

            self.up_arrow = MyButton(
                self.arrow_frame,
                image=self.up_arrow_img,
                width=30,
                command=self.up,
            )
            self.up_arrow.grid(row=0, column=0)

            self.down_arrow = MyButton(
                self.arrow_frame,
                image=self.down_arrow_img,
                width=30,
                command=self.down,
            )
            self.down_arrow.grid(row=0, column=1, padx=(5, 0))

            # Check if oscillator 1 is present. If so disable down arrow.
            if self.labels[0]["text"] == "0":
                self.up_arrow["state"] = "disabled"

            # Check if last oscillator is present. If so disable up arrow.
            if self.labels[-1]["text"] == f"{len(self.value_vars) - 1}":
                self.down_arrow["state"] = "disabled"

            self.jump_frame = MyFrame(self.navigate_frame, bg=self.bg)
            self.jump_frame.grid(row=1, column=0)

            self.jump_label = MyLabel(
                self.jump_frame,
                text="Jump to:",
                bg=self.bg,
            )
            self.jump_label.grid(row=0, column=0, pady=(5, 0))

            self.jump_entry = MyEntry(
                self.jump_frame,
                return_command=self.jump,
            )
            self.jump_entry.grid(row=0, column=1, padx=(10, 0), pady=(5, 0))

    def reconstruct(self, contents, top=0):
        """Regenerate table, given a new contents array"""
        # Destroy all contents in self
        for widget in self.winfo_children():
            widget.destroy()

        self.create_value_vars(contents)
        self.construct(top)

    def left_click(self, idx, top):
        """Deals with a <Button-1> event on a label.

        Parameters
        ----------
        idx : int
            Equivalent to oscillator label value - 1.

        Notes
        -----
        This will set the background of the selected label to blue, and
        foreground to white. Entry widgets in the corresponding row are set
        to read-only mode. All other oscillator labels widgets are set to
        "disabled" mode."""
        for i in [i for i in range(len(self.value_vars)) if i != idx]:
            try:
                self.selected_rows.remove(i)
            except ValueError:
                pass

        # Proceed to highlight the selected row
        self.shift_left_click(idx, top)

    def shift_left_click(self, idx, top):
        """Deals with a <Shift-Button-1> event on a label.

        Parameters
        ----------
        index : int
            Equivalent to oscillator label value - 1.

        Notes
        -----
        This will set the background of the selected label to blue, and
        foreground to white.  Entry widgets in the corresponding row are set
        to read-only mode. Other rows are unaffected.
        """
        if idx + top in self.selected_rows:
            self.selected_rows.remove(idx + top)
        else:
            self.selected_rows.append(idx + top)
        self.selected_number.set(len(self.selected_rows))

        self.activate_rows(top)

    def activate_rows(self, top):
        for i, (label, entries) in enumerate(zip(self.labels, self.entries)):
            if i + top in self.selected_rows:
                fg, bg, state = cf.TABLESELECTFGCOLOR, cf.TABLESELECTBGCOLOR, "readonly"
            else:
                fg, bg, state = "#000000", self.bg, "disabled"

            label["fg"] = fg
            label["bg"] = bg
            for entry in entries:
                entry["state"] = state

    def check_param(self, row, column):
        """Given a StringVar, ensure the value corresponds to a valid
        parameter value"""
        value_var = self.value_vars[row][column]
        entry = self.entries[row][column]

        try:
            value = float(value_var["var"].get())
            # Amplitude or damping factor
            if column in [0] + [2 + self.dim + i for i in range(self.dim)]:
                value = value if value > 0 else None
            elif column == 1:
                value = (value + np.pi) % (2 * np.pi) - np.pi
            else:
                d = column - 2
                r = self.region[d]
                value = value if r[0] >= value >= r[1] else None

            if isinstance(value, float):
                value_var["value"] = value

        except Exception:
            pass

        if isinstance(value_var["value"], float):
            value_var["var"].set(f"{value_var['value']:.6g}")
        else:
            value_var["var"].set(value_var["value"])

        if value_var["var"].get() == "":
            entry.key_press()

    def get_values(self):
        """Takes a nested list of value_var dicts and returns a list of the
        same size, just containing the values"""
        value_vars = self.value_vars
        values = []

        for row in value_vars:
            value_row = []
            for element in row:
                value_row.append(element["value"])
            values.append(value_row)

        return values

    def check_red_entry(self):
        """Determines whether any of the entry widgets are red, indicated
        they contain unvalidated contents"""
        for row in self.entries:
            for entry in row:
                if entry["fg"] == "red":
                    return True
        return False

    @property
    def top(self):
        return int(self.labels[0]["text"])

    def up(self):
        """Scroll down one place in the table"""
        self.reconstruct(contents=self.get_values(), top=self.top - 1)

    def down(self):
        """Scroll down one place in the table"""
        self.reconstruct(contents=self.get_values(), top=self.top + 1)

    def jump(self):
        entry = self.jump_entry
        inpt = entry.get()
        nrows = len(self.value_vars)
        try:
            assert 0 <= (value := int(inpt)) <= nrows - 1
            if nrows - value < self.max_rows:
                top = nrows - self.max_rows
            else:
                top = value
            self.reconstruct(contents=self.get_values(), top=top)
        except Exception:
            entry.delete(0, tk.END)
