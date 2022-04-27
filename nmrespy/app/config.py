# config.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 26 Apr 2022 10:18:51 BST

import tkinter as tk
from PIL import ImageTk, Image
import numpy as np

import nmrespy._paths_and_links as pl

# useful paths
TMPPATH = pl.NMRESPYPATH / "app/tmp"

# GUI font
MAINFONT = "Helvetica"

# colors related to plot
BGCOLOR = "#e4eaef"
PLOTCOLOR = "#ffffff"  # plot background
REGIONCOLOR = "#7fd47f"  # region rectangle patch
NOISEREGIONCOLOR = "#66b3ff"  # noise region rectange patch
PIVOTCOLOR = "#ff0000"  # pivot line plot
NOTEBOOKCOLOR = "#c4d1dc"
ACTIVETABCOLOR = "#648ba4"
BUTTONGREEN = "#9eda88"
BUTTONORANGE = "#ffb861"
BUTTONRED = "#ff9894"
BUTTONDEFAULT = "#6699cc"
READONLYENTRYCOLOR = "#cde6ff"
TABLESELECTBGCOLOR = "#0000ff"
TABLESELECTFGCOLOR = "#ffffff"
FIDCOLORS = [
    "#e6261f",
    "#eb7532",
    "#f7d038",
    "#a3e048",
    "#49da9a",
    "#34bbe6",
    "#4355db",
    "#d23be7",
]

# Image paths
FOLDERPATH = pl.IMAGESPATH / "folder_icon.png"
ICONPATH = pl.IMAGESPATH / "icon.ico"
UPARROWPATH = pl.IMAGESPATH / "up_arrow.png"
DOWNARROWPATH = pl.IMAGESPATH / "down_arrow.png"
WARNINGPATH = pl.IMAGESPATH / "warning.png"


def get_PhotoImage(path, scale=1.0):
    """Generate a TKinter-compatible photo image, given a path, and a scaling
    factor.

    Parameters
    ----------
    path : str
        Path to the image file.
    scale : float, default: 1.0
        Scaling factor.

    Returns
    -------
    img : `PIL.ImageTk.PhotoImage <https://pillow.readthedocs.io/en/4.2.x/\
    reference/ImageTk.html#PIL.ImageTk.PhotoImage>`_
        Tkinter-compatible image. This can be incorporated into a GUI using
        tk.Label(parent, image=img)
    """

    image = Image.open(path).convert("RGBA")
    [w, h] = image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)


def value_var_dict(value, var_object):
    """Generates a dict with keys 'value' and 'var'. The value corresponds to
    some quantity. The var is a corresponding StringVar or IntVar.

    Parameters
    ----------
    value: int, float, bool, etc.
        The quantity of interest

    var_object: str or int
        The value to set the tkinter variable to. If a str, the variable will
        be a StringVar, if an int, the variable will be an IntVar

    Returns
    -------
    value_var_dict: dict
    """

    if isinstance(var_object, str):
        var = tk.StringVar()
    elif isinstance(var_object, int):
        var = tk.IntVar()

    var.set(var_object)
    return {"value": value, "var": var}


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class Restrictor:
    """Resict naivgation within a defined range (used to prevent
    panning/zooming) outside spectral window on x-axis.
    Inspiration from
    `here <https://stackoverflow.com/questions/48709873/restricting-panning-\
    range-in-matplotlib-plots>`_"""

    def __init__(self, ax, x=lambda x: True, y=lambda x: True):

        self.res = [x, y]
        self.ax = ax
        self.limits = self.get_lim()
        self.ax.callbacks.connect("xlim_changed", lambda evt: self.lims_change(axis=0))
        self.ax.callbacks.connect("ylim_changed", lambda evt: self.lims_change(axis=1))

    def get_lim(self):
        return [self.ax.get_xlim(), self.ax.get_ylim()]

    def set_lim(self, axis, lim):
        if axis == 0:
            self.ax.set_xlim(lim)
        else:
            self.ax.set_ylim(lim)
        self.limits[axis] = self.get_lim()[axis]

    def lims_change(self, event=None, axis=0):
        curlim = np.array(self.get_lim()[axis])
        if self.limits[axis] != self.get_lim()[axis]:
            # avoid recursion
            if not np.all(self.res[axis](curlim)):
                # if limits are invalid, reset them to previous state
                self.set_lim(axis, self.limits[axis])
            else:
                # if limits are valid, update previous stored limits
                self.limits[axis] = self.get_lim()[axis]


def check_int(value):

    try:
        int_value = int(value)
        float_value = float(value)

        if int_value == float_value:
            return True
        else:
            return False

    except Exception:
        return False


def check_float(value):
    # Note this isn't the same as isinstance(value, float)
    # Want to accept anything that can be interpreted as float, like int,
    # and return as a float.
    try:
        float(value)
        return True

    except Exception:
        return False


def strip_zeros(number):
    return number.rstrip("0").rstrip(".")


def check_invalid_entries(master):
    """Check whether any entry widgets in a certain frame have been
    assigned a red colour. The implication of this is that certain
    entries have not been verified by the user pressing <Return>"""

    def get_widgets(master):
        """Recursively gets all widgets associated with master"""
        widgets = master.winfo_children()
        for widget in widgets:
            if widget.winfo_children():
                # Get intersection of current widgets and children of
                # currently considered widget
                widgets = list(set(get_widgets(widget)) | set(widgets))

        return widgets

    for widget in list(get_widgets(master)):
        if widget.winfo_class() == "Entry" and widget["fg"] == "red":
            return False

    return True
