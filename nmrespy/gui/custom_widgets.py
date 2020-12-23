"""
Customised widgets for NMR-EsPy GUI.
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

BGCOLOR = '#e4eaef'

def overwrite_defaults(self, strings, kwargs):
    for k, v in zip(kwargs.keys(), kwargs.values()):
        for string in strings:
            if k == string:
                self[string] = v


class MyFrame(tk.Frame):
    """A Tkinter frame with a white background by default."""

    def __init__(self, parent, **kwargs):
        tk.Frame.__init__(self, parent, kwargs)

        self['bg'] = BGCOLOR

        overwrite_defaults(self, ('bg'), kwargs)


class MyToplevel(tk.Toplevel):
    """A Tkinter toplevel, which by default:

    * Has a white background
    * CAnnot be resized
    * Has the title NMR-EsPy
    """

    def __init__(self, parent, **kwargs):
        tk.Toplevel.__init__(self, parent)

        self['bg'] = BGCOLOR

        overwrite_defaults(self, ('bg'), kwargs)

        self.title('NMR-EsPy')
        self.resizable(False, False)


class MyLabel(tk.Label):
    """Tkinter label with white background by deafult"""

    def __init__(self, parent, **kwargs):
        tk.Label.__init__(self, parent, kwargs)

        strings = ('bg', 'font')

        self[strings[0]] = BGCOLOR
        self[strings[1]] = ('Helvetica', '11')

        overwrite_defaults(self, strings, kwargs)


class MyButton(tk.Button):
    """Tkinter button with various tweaks"""

    def __init__(self, parent, **kwargs):
        tk.Button.__init__(self, parent, kwargs)

        strings = ('width', 'highlightbackground', 'bg')

        self[strings[0]] = 8
        self[strings[1]] = 'black'
        self[strings[2]] = '#6699cc'

        overwrite_defaults(self, strings, kwargs)


class MyCheckbutton(tk.Checkbutton):
    """Tkinter button with various tweaks"""

    def __init__(self, parent, **kwargs):
        tk.Checkbutton.__init__(self, parent, kwargs)

        strings = ('bg', 'highlightthickness', 'bd')

        self[strings[0]] = BGCOLOR
        self[strings[1]] = 0
        self[strings[2]] = 0

        overwrite_defaults(self, strings, kwargs)


class MyScale(tk.Scale):

    def __init__(self, parent, **kwargs):
        tk.Scale.__init__(self, parent, kwargs)

        strings = (
            'orient', 'showvalue', 'sliderlength', 'bd', 'highlightthickness',
            'highlightbackground', 'relief', 'bg', 'troughcolor',
        )

        self[strings[0]] = tk.HORIZONTAL
        self[strings[1]] = 0
        self[strings[2]] = 15
        self[strings[3]] = 0
        self[strings[4]] = 1
        self[strings[5]] = 'black'
        self[strings[6]] = 'flat'
        self[strings[7]] = BGCOLOR
        self[strings[8]] = 'white'

        overwrite_defaults(self, strings, kwargs)

class MyEntry(tk.Entry):

    def __init__(self, parent, **kwargs):
        tk.Entry.__init__(self, parent, kwargs)

        strings = (
            'width', 'highlightthickness', 'highlightbackground',
            'disabledbackground', 'disabledforeground', 'bg',
        )

        self[strings[0]] = 7
        self[strings[1]] = 1
        self[strings[2]] = 'black'
        self[strings[3]] = '#505050'
        self[strings[4]] = '#505050'
        self[strings[5]] = 'white'

        overwrite_defaults(self, strings, kwargs)


class MyOptionMenu(tk.OptionMenu):

    def __init__(self, parent, variable, value, *values, **kwargs):
        print(value, values, kwargs)
        tk.OptionMenu.__init__(self, parent, variable, value, values, kwargs)


        print(values)
        strings = ('bg', 'borderwidth', 'width', 'highlightcolor',)

        self[strings[0]] = BGCOLOR
        self[strings[1]] = 1
        self[strings[2]] = 10
        self[strings[3]] = 'black'

        # this is not accessible without explicitly accessing after
        # generating an instance of tk.OptionMenu
        self['menu']['bg'] = BGCOLOR

        overwrite_defaults(self, strings, kwargs)


class MyNavigationToolbar(NavigationToolbar2Tk):
    """Tweak default matplotlib navigation bar to exclude subplot-config
    and save buttons. Also dialogues as cursor goes over plot, and bar
    is set to be white"""

    def __init__(self, canvas, parent, color=BGCOLOR):

        # slice toolitems (this gets rid of the unwanted buttons)
        self.toolitems = self.toolitems[:6]

        NavigationToolbar2Tk.__init__(
            self, canvas, parent, pack_toolbar=False
        )

        # make everything white
        self['bg'] = color
        self._message_label['bg'] = color
        for button in self.winfo_children():
            button['bg'] = color

    def set_message(self, msg):
        pass
