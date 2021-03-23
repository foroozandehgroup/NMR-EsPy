"""
Customised widgets for NMR-EsPy GUI.
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from .config import *

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
        super().__init__(parent)
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

        keys = (
            'width', 'highlightbackground', 'bg',
            'disabledforeground',
        )
        values = (8, 'black', BUTTONDEFAULT, '#a0a0a0')

        generate(self, keys, values, kwargs)

        # there isnt a disabledbackground variable for tk.Button
        # for some reason
        # have to manually change color upon change of state
        if self['state'] == 'disabled':
            self['bg'] = '#e0e0e0'



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
            'width', 'highlightthickness', 'highlightbackground',
            'bg', 'readonlybackground'
        )
        values = (
            7, 1, 'black', 'white', READONLYENTRYCOLOR
        )

        generate(self, keys, values, kwargs)

        if self.return_command:
            self.bind('<Key>', lambda event: self.key_press())
            self.bind('<Return>', lambda event: self.return_press())

    def key_press(self):
        self['fg'] = 'red'
        self['highlightcolor'] = 'red'
        self['highlightbackground'] = 'red'

    def black_highlight(self):
        self['fg'] = 'black'
        self['highlightcolor'] = 'black'
        self['highlightbackground'] = 'black'

    def return_press(self):
        self.black_highlight()
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


class MyText(tk.Text):

    def __init__(self, parent, **kwargs):

        super().__init__(parent)

        keys = ('bg', 'highlightcolor', 'highlightbackground')
        values = ('white', 'black', 'black')

        generate(self, keys, values, kwargs)


class MyNotebook(ttk.Notebook):

    def __init__(self, parent):
        style = ttk.Style()
        style.theme_create('notebook', parent='alt',
            settings={
                'TNotebook': {
                    'configure': {
                        'tabmargins': [2, 0, 5, 0],
                        'background': BGCOLOR,
                        'bordercolor': 'black'}
                    },
                'TNotebook.Tab': {
                    'configure': {
                        'padding': [10, 3],
                        'background': NOTEBOOKCOLOR,
                        'font': (MAINFONT, 11)
                    },
                    'map': {
                        'background': [('selected', ACTIVETABCOLOR)],
                        'expand': [("selected", [1, 1, 1, 0])],
                        'font': [('selected', (MAINFONT, 11, 'bold'))],
                        'foreground': [('selected', 'white')],
                    }
                }
            }
        )
        style.theme_use("notebook")

        super().__init__(parent)

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


class MyTable(MyFrame):

    def __init__(self, parent, titles, contents, **kwargs):

        super().__init__(parent, **kwargs)

        self.table_frame = MyFrame(self)
        self.table_frame.grid(column=0, row=0, padx=(0,10), pady=10)

        self.title_labels = {}
        self.labels = {}
        self.entries = {}

        rows, columns = len(contents), len(contents[0])

        max_width = 0

        for row in range(rows+1):
            for column in range(columns+1):
                if row != 0 and column == 0:
                    # oscillator labels (1, 2, 3, etc.)
                    self.labels[row] = MyLabel(
                        self.table_frame, text=f"{row}", bold=True,
                    )

                    if row == rows + 1:
                        pady = 2
                    else:
                        pady = (2, 0)

                    self.labels[row].grid(
                        column=column, row=row, ipadx=10, pady=pady, sticky='w',
                    )

                elif row == 0 and column != 0:
                    self.table_frame.columnconfigure(column, weight=1)
                    # column titles
                    text = titles[column-1]
                    self.title_labels[text] = MyLabel(
                        self.table_frame, text=text, bold=True,
                    )

                    self.title_labels[text].grid(
                        column=column, row=row, sticky='w', pady=(2,0),
                    )

                elif row != 0 and column != 0:
                    if column == 1:
                        self.entries[row] = {}

                    self.entries[row][text] = MyEntry(
                        self.table_frame, textvariable=contents[row-1][column-1],
                        width=14, readonlybackground='#a0a0a0', state='readonly',
                    )
                    self.entries[row][text].grid(
                        row=row, column=column, sticky='ew',
                    )
