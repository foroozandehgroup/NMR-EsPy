# custom_widgets.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 11:26:47 GMT

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

        if self.return_command:
            self.bind_command()

    def bind_command(self):
        self.bind("<Key>", lambda event: self.key_press())
        self.bind("<Return>", lambda event: self.return_press())

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


class MyNotebook(ttk.Notebook):
    def __init__(self, parent):
        style = ttk.Style()
        style.theme_create(
            "notebook",
            parent="alt",
            settings={
                "TNotebook": {
                    "configure": {
                        "tabmargins": [2, 0, 5, 0],
                        "background": cf.BGCOLOR,
                        "bordercolor": "black",
                    }
                },
                "TNotebook.Tab": {
                    "configure": {
                        "padding": [10, 3],
                        "background": cf.NOTEBOOKCOLOR,
                        "font": (cf.MAINFONT, 11),
                    },
                    "map": {
                        "background": [("selected", cf.ACTIVETABCOLOR)],
                        "expand": [("selected", [1, 1, 1, 0])],
                        "font": [("selected", (cf.MAINFONT, 11, "bold"))],
                        "foreground": [("selected", "white")],
                    },
                },
            },
        )
        style.theme_use("notebook")

        super().__init__(parent)


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


class MyTable(MyFrame):
    def __init__(self, master, contents, titles, region):

        super().__init__(master)
        self.titles = titles
        self.region = region[0]
        # Number of selected rows
        self.selected_number = tk.IntVar()
        self.selected_number.set(0)
        self.selected_rows = []

        self.max_rows = 12

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
                        cf.strip_zeros(f"{param:.5f}"),
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

        self.table_frame = MyFrame(self)
        self.table_frame.grid(row=0, column=0)

        # Column titles
        for column, title in enumerate(["#"] + self.titles):
            padx = 0 if column == 0 else (5, 0)
            sticky = "" if column == 0 else "w"
            MyLabel(self.table_frame, text=title).grid(
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
            label = MyLabel(self.table_frame, text=str(top + i + 1))
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
            ent_row = []

            for j, value_var in enumerate(value_var_row):
                if j == 0:
                    type_ = "amp"
                elif j == 1:
                    type_ = "phase"
                elif j == 2:
                    type_ = "freq"
                elif j == 3:
                    type_ = "damp"

                ent = MyEntry(
                    self.table_frame,
                    textvariable=value_var["var"],
                    state="disabled",
                    width=14,
                )
                # Ensure that entry widgets are checked after user input
                # to ensure valid parameters.
                ent.return_command = self.check_param
                ent.return_args = (value_var, type_, ent)
                ent.bind_command()

                padx = (5, 0)
                pady = (5, 0)

                ent.grid(row=i + 1, column=j + 1, padx=padx, pady=pady)
                ent_row.append(ent)

            self.entries.append(ent_row)

        # Activate any active oscillators
        # Colours all row labels corresponding to oscillators selected
        self.activate_rows(top)

        # Add naviagtion buttons if more than `self.max_rows` oscillators
        if len(self.value_vars) > self.max_rows:
            self.navigate_frame = MyFrame(self)
            self.navigate_frame.grid(row=1, column=0, pady=(10, 0))

            self.up_arrow_img = cf.get_PhotoImage(cf.UPARROWPATH, scale=0.5)
            self.down_arrow_img = cf.get_PhotoImage(cf.DOWNARROWPATH, scale=0.5)

            self.up_arrow = MyButton(
                self.navigate_frame,
                image=self.up_arrow_img,
                width=30,
                command=self.up,
            )
            self.up_arrow.grid(row=0, column=0)

            self.down_arrow = MyButton(
                self.navigate_frame,
                image=self.down_arrow_img,
                width=30,
                command=self.down,
            )
            self.down_arrow.grid(row=0, column=1, padx=(5, 0))

            # Check if oscillator 1 is present. If so disable down arrow.
            if self.labels[0]["text"] == "1":
                self.up_arrow["state"] = "disabled"

            # Check if last oscillator is present. If so disable up arrow.
            if int(self.labels[-1]["text"]) == len(self.value_vars):
                self.down_arrow["state"] = "disabled"

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
                fg, bg, state = "#000000", cf.BGCOLOR, "disabled"

            label["fg"] = fg
            label["bg"] = bg
            for entry in entries:
                entry["state"] = state

    def check_param(self, value_var, type_, entry):
        """Given a StringVar, ensure the value corresponds to a valid
        parameter value"""

        try:
            value = float(value_var["var"].get())

            if type_ in ["amp", "damp"] and value > 0.0:
                pass
            elif type_ == "phase":
                # Wrap phase
                value = (value + np.pi) % (2 * np.pi) - np.pi
            elif type_ == "freq":
                if min(self.region) <= value <= max(self.region):
                    pass
                else:
                    raise
            else:
                raise

            value_var["value"] = value

        except Exception:
            pass

        if isinstance(value_var["value"], (int, float)):
            value_var["var"].set(cf.strip_zeros(f"{value_var['value']:.5f}"))
        else:
            # The only time the result shouldn't be a numerical value
            # if when it is an empty string (this crops up in result.AddFrame)
            # In this case, want to re-colour red as the entry widget should
            # not be empty
            value_var["var"].set(value_var["value"])
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

    def up(self):
        """Scroll down one place in the table"""
        top = int(self.labels[0]["text"]) - 2
        self.reconstruct(contents=self.get_values(), top=top)

    def down(self):
        """Scroll down one place in the table"""
        top = int(self.labels[0]["text"])
        self.reconstruct(contents=self.get_values(), top=top)
