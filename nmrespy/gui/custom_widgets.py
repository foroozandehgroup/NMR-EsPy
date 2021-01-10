"""
Customised widgets for NMR-EsPy GUI.
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from .config import BGCOLOR, MAINFONT

def generate(cls, keys, values, kwargs):

    for key, value in zip(keys, values):
        cls[key] = value

    # overwrite default properties with any provided kwargs
    for kwkey, kwvalue in zip(kwargs.keys(), kwargs.values()):
        cls[kwkey] = kwvalue


class MyFrame(tk.Frame):
    """A Tkinter frame with a white background by default."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        generate(self, ('bg',), (BGCOLOR,), kwargs)


class MyToplevel(tk.Toplevel):
    """A Tkinter toplevel, which by default:

    * Has a white background
    * CAnnot be resized
    * Has the title NMR-EsPy
    """

    def __init__(self, parent, **kwargs):
        super().__init__(master=parent)

        generate(self, ('bg',), (BGCOLOR,), kwargs)

        self.title('NMR-EsPy')
        self.resizable(False, False)


class MyLabel(tk.Label):
    """Tkinter label with white background by deafult"""

    def __init__(self, parent, bold=False, **kwargs):
        tk.Label.__init__(self, parent)
        keys = ('bg', 'font')

        if bold:
            values = (BGCOLOR, (MAINFONT, '11', 'bold'))
        else:
            values = (BGCOLOR, (MAINFONT, '11'))

        generate(self, keys, values, kwargs)


class MyButton(tk.Button):
    """Tkinter button with various tweaks"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        keys = ('width', 'highlightbackground', 'bg')
        values = (8, 'black', '#6699cc')

        generate(self, keys, values, kwargs)


class MyCheckbutton(tk.Checkbutton):
    """Tkinter button with various tweaks"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        keys = ('bg', 'highlightthickness', 'bd')
        values = (BGCOLOR, 0, 0)

        generate(self, keys, values, kwargs)


class MyScale(tk.Scale):

    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        keys = (
            'orient', 'showvalue', 'sliderlength', 'bd', 'highlightthickness',
            'highlightbackground', 'relief', 'bg', 'troughcolor',
        )
        values = (
            tk.HORIZONTAL, 0, 15, 0, 1, 'black', 'flat', BGCOLOR, 'white'
        )

        generate(self, keys, values, kwargs)


# class MyEntry(tk.Entry):
#
#     def __init__(self, parent, **kwargs):
#         super().__init__(parent)
#
#         keys = (
#             'width', 'highlightthickness', 'highlightbackground',
#             'disabledbackground', 'disabledforeground', 'bg',
#         )
#         values = (7, 1, 'black', '#505050', '#505050', 'white')
#
#         generate(self, keys, values, kwargs)


# This is a work in progress:
# I am hoping to set up entry widgets so that if a user changes the input,
# the tex becomes red. Only once they have pressed <Return> does the text
# go back to the default black

class MyEntry(tk.Entry):
    """Entry widget with some aesthetic teweaks.

    ``return_command`` specifies a callable to be bound to the widget
    for when the user presses <Return>. If the callable has any arguments,
    these should be provided as a tuple into ``return_args``.
    The upshot of these commands is the text becomes red once the user
    changes the input, and goes back to black once <Return> has been
    pressed. The idea is to warn the user that they haven't saved their
    changes since altering the entry widget.
    """

    def __init__(self, parent, return_command=None, return_args=None, **kwargs):
        super().__init__(parent)

        self.return_command = return_command
        self.return_args = return_args

        keys = (
            'width', 'highlightthickness', 'highlightbackground',
            'disabledbackground', 'disabledforeground', 'bg',
        )
        values = (7, 1, 'black', '#505050', '#505050', 'white')

        generate(self, keys, values, kwargs)

        if self.return_command:
            self.bind('<Key>', lambda event: self.key_press())
            self.bind('<Return>', lambda event: self.return_press())

    def key_press(self):
        self['fg'] = 'red'

    def return_press(self):
        self['fg'] = 'black'
        self.return_command(*self.return_args)

class MyOptionMenu(tk.OptionMenu):

    def __init__(self, parent, variable, value, *values, **kwargs):

        super().__init__(parent, variable, value, values, kwargs)

        keys = ('bg', 'borderwidth', 'width', 'highlightcolor')
        values = (BGCOLOR, 1, 10, 'black')

        generate(self, keys, values, kwargs)

        if 'bg' in kwargs.keys():
            self['menu']['bg'] = kwayrgs['bg']
        else:
            self['menu']['bg'] = BGCOLOR


class MyNavigationToolbar(NavigationToolbar2Tk):
    """Tweak default matplotlib navigation bar to exclude subplot-config
    and save buttons. Also dialogues as cursor goes over plot, and bar
    is set to be white"""

    def __init__(self, canvas, parent, color=BGCOLOR):

        # slice toolitems (this gets rid of the unwanted buttons)
        self.toolitems = self.toolitems[:6]

        super().__init__(canvas, parent, pack_toolbar=False)

        # make everything white
        self['bg'] = color
        self._message_label['bg'] = color
        for button in self.winfo_children():
            button['bg'] = color

    def set_message(self, msg):
        pass
