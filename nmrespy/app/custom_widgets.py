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
            self.bind_command()

    def bind_command(self):
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

    def __init__(self, master, contents, titles, region,
                 entry_state='normal'):

        super().__init__(master)
        self.titles = titles
        self.region = region[0]
        self.entry_state = entry_state

        self.create_value_vars(contents)
        self.construct()


    def create_value_vars(self, contents):
        self.value_vars = []
        for osc in contents:
            value_var_row = []
            for param in osc:
                if isinstance(param, (int, float)):
                    value_var = value_var_dict(param, f"{param:.5f}")
                else:
                    value_var = value_var_dict(param, param)
                value_var_row.append(value_var)
            self.value_vars.append(value_var_row)


    def construct(self):
        """Generate a table of the parameters."""

        # Column titles
        for column, title in enumerate(['#'] + self.titles):
            padx = 0 if column == 0 else (5, 0)
            sticky = '' if column == 0 else 'w'
            MyLabel(self, text=title).grid(
                row=0, column=column, padx=padx, sticky=sticky,
            )

        # Store entry widgets and string variables
        self.labels = []
        self.entries = []

        for i, value_var_row in enumerate(self.value_vars):
            # --- Oscillator labels --------------------------------------
            # These act as a oscillator selection widgets
            label = MyLabel(self, text=str(i+1))
            # Bind to left mouse click: select oscillator
            label.bind("<Button-1>", lambda ev, i=i: self.left_click(i))
            # Bind to left mouse click + shift: select oscillator, keep
            # other already selected oscillators still selected.
            label.bind('<Shift-Button-1>', lambda ev, i=i: self.shift_left_click(i))
            label.grid(row=i+1, column=0, pady=(5,0), ipadx=10, ipady=2)
            self.labels.append(label)

            ent_row = []

            for j, value_var in enumerate(value_var_row):
                if j == 0:
                    type_ = 'amp'
                elif j == 1:
                    type_ = 'phase'
                elif j == 2:
                    type_ = 'freq'
                elif j == 3:
                    type_ = 'damp'

                ent = MyEntry(self, textvariable=value_var['var'],
                              state=self.entry_state, width=14)

                ent.return_command = self.check_param
                ent.return_args = (value_var, type_, ent)
                ent.bind_command()

                padx = (5, 0)
                pady = (5, 0)

                ent.grid(row=i+1, column=j+1, padx=padx, pady=pady)
                ent_row.append(ent)

            self.entries.append(ent_row)


    def reconstruct(self, contents):
        """Regenerate table, given a new contents array"""
        for widget in self.winfo_children():
            widget.destroy()

        self.create_value_vars(contents)
        self.construct()


    def left_click(self, idx):
        """Deals with a <Button-1> event on a label.

        Parameters
        ----------
        idx : int
            Equivalent to oscillator label value - 1.

        Notes
        -----
        This will set the background of the selected label to blue, and
        foreground to white. Entry widgets in the corresponding row are set to
        read-only mode. All other oscillator labels widgets are set to "disabled"
        mode."""

        # Disable all rows that do not match the index
        for i, label in enumerate(self.labels):
            if i != idx and label['bg'] == TABLESELECTBGCOLOR:

                label['bg'] = BGCOLOR
                label['fg'] = '#000000'
                for entry in self.entries[i]:
                    entry['state'] = 'disabled'


        # Proceed to highlight the selected row
        self.shift_left_click(idx)


    def shift_left_click(self, idx):
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
        if self.labels[idx]['fg'] == '#000000':
            fg, bg, state = TABLESELECTFGCOLOR, TABLESELECTBGCOLOR, 'readonly'
        else:
            fg, bg, state  = '#000000', BGCOLOR, 'disabled'


        self.labels[idx]['fg'] = fg
        self.labels[idx]['bg'] = bg

        for entry in self.entries[idx]:
            entry['state'] = state


    def check_param(self, value_var, type_, entry):
        """Given a StringVar, ensure the value corresponds to a valid
        parameter value"""

        try:
            value = float(value_var['var'].get())

            if type_ in ['amp', 'damp'] and value > 0.0:
                pass
            elif type_ == 'phase':
                # Wrap phase
                value = (value + np.pi) % (2 * np.pi) - np.pi
            elif type_ == 'freq':
                if min(self.region) <= value <= max(self.region):
                    pass
                else:
                    raise
            else:
                raise

            value_var['value'] = value

        except:
            pass

        if isinstance(value_var['value'], (int, float)):
            value_var['var'].set(f"{value_var['value']:.5f}")
        else:
            value_var['var'].set(value_var['value'])
            entry.key_press()
