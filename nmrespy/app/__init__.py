#!/usr/bin/python3

# Application for using NMR-EsPy
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# This is currently only applicable to 1D NMR data.

# Look for:
# YOU ARE HERE
# This denotes where I am up to in tweaking things

import ast
from itertools import cycle
import os
import pathlib
import random
import webbrowser

import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from PIL import ImageTk, Image

import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from nmrespy import *
from nmrespy._errors import *
from nmrespy._misc import latex_nucleus
from nmrespy.core import Estimator
from nmrespy.app.config import *
from nmrespy.app.custom_widgets import *
from nmrespy.app.frames import *



class WarnFrame(MyToplevel):
    """A window in case the user does something silly."""

    def __init__(self, parent, msg):
        super().__init__(parent)
        self.title('NMR-EsPy - Error')

        # warning image
        self.img = get_PhotoImage(os.path.join(IMAGESPATH, 'warning.png'), 0.08)
        self.warn_sign = MyLabel(self, image=self.img)
        self.warn_sign.grid(row=0, column=0, padx=(10,0), pady=10)

        # add text explaining the issue
        text = MyLabel(self, text=msg, wraplength=400)
        text.grid(row=0, column=1, padx=10, pady=10)

        # close button
        close_button = MyButton(
            self, text='Close', bg='#ff9894', command=self.destroy,
        )
        close_button.grid(row=1, column=1, padx=10, pady=(0,10))



class NMREsPyApp(tk.Tk):
    """App for using NMR-EsPy.

    path : str
        Path to the specified directory.

    topspin : bool
        Indicates whether or not the app has been invoked via TopSpin
        or not. If it has, the type of data (fid or pdata) the user wishes
        to use will have to be ascertained, using the `DataType` window.
    """

    # This is the "controller"
    # When you see `self.ctrl` in other classes in this file, it refers
    # to this class

    def __init__(self, path, topspin=False):
        super().__init__()

        # Hide the root app window. This is not going to be used. Everything
        # will be built onto Toplevels
        self.withdraw()

        path = pathlib.Path(path)
        if topspin:
            # Open window to ask user for data type (fid or pdata)
            # from this, self acquires the attirbutes dtype and path
            paths = {'pdata': path, 'fid': path.parent.parent}
            data_type_window = DataType(self, paths)
            path = data_type_window.path

        # Create Estimator instance from the provided path
        self.estimator = Estimator.new_bruker(path)

        # App is only applicable to 1D data currently
        if self.estimator.get_dim() > 1:
            raise TwoDimUnsupportedError()

        self.setup = SetUp(self, self.estimator)
        # hold at this point
        # relieved once setup is destroyed
        # see SetUp.run()
        self.wait_window(self.setup)

        # # create attributres relating to the result Toplevel
        # self.generate_result_variables()
        #
        # # result Toplevel is for inspecting and saving the estimation result
        # self.result = MyToplevel(self)
        # # same basic configuration as setup
        # self.result.resizable(True, True)
        # self.result.columnconfigure(0, weight=1)
        # self.result.rowconfigure(0, weight=1)
        # # frames contained within the result Toplevel
        # self.result_frames = {}
        # # frame containing the plot
        # self.result_frames['plot_frame'] = PlotFrame(
        #     parent=self.result, figure=self.resultfig['fig'],
        # )
        # # frame containing the navigation toolbar
        # self.result_frames['toolbar_frame'] = RootToolbarFrame(
        #     parent=self.result, canvas=self.result_frames['plot_frame'].canvas,
        #     ctrl=self,
        # )
        # # frame with NMR-EsPy and MF group logos
        # self.result_frames['logo_frame'] = LogoFrame(
        #     parent=self.result, scale=0.06,
        # )
        # # frame with cancel/help/run/edit parameters buttons
        # self.result_frames['button_frame'] = ResultButtonFrame(
        #     parent=self.result, ctrl=self,
        # )
        #
        # # configure frame placements
        # self.result_frames['plot_frame'].grid(
        #     row=0, column=0, columnspan=2, sticky='nsew',
        # )
        # self.result_frames['toolbar_frame'].grid(
        #     row=1, column=0, columnspan=2, sticky='ew',
        # )
        # self.result_frames['logo_frame'].grid(
        #     row=3, column=0, padx=10, pady=10, sticky='w',
        # )
        # self.result_frames['button_frame'].grid(
        #     row=3, column=1, sticky='s',
        # )





    def generate_result_variables(self):
        """Produce variables that are used in the result Toplevel"""

        # result figure
        # uses core.NMREsPyBruker.plot_result method to produce the result
        self.resultfig = {}
        keys = ('fig', 'ax', 'lines', 'labels')
        for key, value in zip(keys, self.info.plot_result()):
            self.resultfig[key] = value

        self.resultfig['fig'].set_size_inches(6, 3.5)
        self.resultfig['fig'].set_dpi(170)
        self.resultfig['xlim'] = self.resultfig['ax'].get_xlim()

        # prevent panning outside the selected region
        Restrictor(
            self.resultfig['ax'],
            x=lambda x: x<= self.resultfig['xlim'][0],
        )
        Restrictor(
            self.resultfig['ax'],
            x=lambda x: x>= self.resultfig['xlim'][1],
        )

        self.resultfig['fig'].patch.set_facecolor(BGCOLOR)
        self.resultfig['ax'].set_facecolor(PLOTCOLOR)
        self.resultfig['ax'].tick_params(axis='x', which='major', labelsize=6)
        self.resultfig['ax'].locator_params(axis='x', nbins=10)
        self.resultfig['ax'].set_xlabel(
            self.resultfig['ax'].get_xlabel(), fontsize=8,
        )


class ResultButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, parent, ctrl):
        super().__init__(parent, ctrl)
        self.save_button['command'] = self.save_options
        self.save_button['text'] = 'Save'

        self.edit_parameter_button = MyButton(
            self, text='Edit Parameter Estimate', command=self.edit_parameters,
        )
        self.edit_parameter_button.grid(
            row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=(10,0)
        )

    def edit_parameters(self):
        EditParams(parent=self, ctrl=self.ctrl)

    def save_options(self):
        SaveFrame(parent=self, ctrl=self.ctrl)


class RerunSettingsFrame(AdvancedSettingsFrame):

    def __init__(self, parent, ctrl):
        AdvancedSettingsFrame.__init__(self, parent, ctrl)
        self.ctrl = ctrl
        self.rows['0'].grid_forget()
        self.rows['1'].grid_forget()


class EditParams(MyToplevel):
    """Window allowing user to edit the estimation result."""

    def __init__(self, parent, ctrl):

        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.ctrl = ctrl
        self.grab_set()

        # frame to contain the table of parameters
        self.table = MyFrame(self)
        self.table.grid(row=0, column=0)
        # generate table inside frame
        self.construct_table()
        # frame to contain various buttons: mrege, split, manual edit, close
        self.button_frame = MyFrame(self)
        self.button_frame.grid(row=1, column=0, sticky='ew')

        for i in range(4):
            self.button_frame.columnconfigure(i, weight=1)

        # Re-run nonlinear programming
        self.rerun_button = MyButton(
            self.button_frame, text='Re-run NLP', command=self.rerun,
            bg=BUTTONGREEN,
        )
        self.rerun_button.grid(
            column=0, row=0, rowspan=2, padx=(10,0), pady=10, sticky='nsew',
        )

        # add oscillator(s)
        self.add_button = MyButton(
            self.button_frame, text='Add', command=self.add,
        )
        self.add_button.grid(
            row=0, column=1, sticky='ew', padx=(10,0), pady=(10,0),
        )

        # remove oscillator(s)
        self.remove_button = MyButton(
            self.button_frame, text='Remove', state='disabled',
            command=self.remove,
        )
        self.remove_button.grid(
            row=0, column=2, sticky='ew', padx=(10,0), pady=(10,0),
        )

        # manually edit parameters associated with oscillator
        self.manual_button = MyButton(
            self.button_frame, text='Edit Manually', state='disabled',
            command=self.manual_edit,
        )
        self.manual_button.grid(
            row=0, column=3, sticky='ew', padx=10, pady=(10,0),
        )

        # split selected oscillator
        self.split_button = MyButton(
            self.button_frame, text='Split', state='disabled',
            command=self.split,
        )
        self.split_button.grid(
            row=1, column=1, sticky='ew', padx=(10,0), pady=(10,10),
        )

        # merge selected oscillators
        self.merge_button = MyButton(
            self.button_frame, text='Merge', state='disabled',
            command=self.merge,
        )
        self.merge_button.grid(
            row=1, column=2, sticky='ew', padx=(10,0), pady=(10,10),
        )

        # close window
        self.close_button = MyButton(
            self.button_frame, text='Close', command=self.destroy,
            bg=BUTTONRED,
        )
        self.close_button.grid(
            row=1, column=3, sticky='ew', padx=10, pady=(10,10)
        )


    def construct_table(self, reconstruct=False):

        if reconstruct:
            for widget in self.table.winfo_children():
                widget.destroy()

        # column titles
        titles = ('#', 'Amplitude', 'Phase', 'Frequency', 'Damping')
        for column, title in enumerate(titles):
            label = MyLabel(self.table, text=title)

            ipadx = 0
            padx = (5, 0)
            pady = (10, 0)

            if column == 0:
                padx, ipadx = 0, 10
            elif column == 4:
                padx = (5, 10)

            label.grid(
                row=0, column=column, ipadx=ipadx, padx=padx, pady=pady,
                sticky='w'
            )

        # store oscillator labels, entry widgets, and string variables
        self.table_labels = [] # tk.Label instances (M elements)
        self.table_entries = [] # tk.Entry instances (M x 4 elements)
        self.table_variables = [] # tk.StringVar instances (M x 4 elements)

        for i, oscillator in enumerate(self.ctrl.info.get_theta(frequency_unit='ppm')):

            # first column: oscillator number
            label = MyLabel(self.table, text=str(i+1))
            # left click: highlight row (unselect all others)
            label.bind('<Button-1>', lambda entry, i=i: self.left_click(i))

            # left click with shift: highlight row, keep other selected rows
            # highlighted
            label.bind(
                '<Shift-Button-1>', lambda entry, i=i: self.shift_left_click(i),
            )

            label.grid(row=i+1, column=0, ipadx=10, pady=(5,0))
            self.table_labels.append(label)

            # construct entry widgets and parameter variables for current row
            entry_row = []
            variable_row = []
            for j, parameter in enumerate(oscillator):
                var = tk.StringVar()
                var.set(f'{parameter:.5f}')
                variable_row.append(var)

                return_args = (i, j, self.table_variables, self.table_entries)
                entry = MyEntry(
                    self.table, return_command=self.check_param,
                    return_args=return_args, textvariable=var, state='disabled',
                    width=14,
                )

                # conditions affect how entry widget is padded
                padx = (5, 10)
                pady = (5, 0)

                if j == 0:
                    pady = 0
                elif j != 3:
                    padx = (5, 0)

                entry.grid(row=i+1, column=j+1, padx=(5,10), pady=(5,0))
                entry_row.append(entry)

            self.table_entries.append(entry_row)
            self.table_variables.append(variable_row)


    def rerun(self):
        RerunFrame(self, self.ctrl)


    def left_click(self, index):
        """Deals with a <Button-1> event on a label.

        Parameters
        ----------
        index : int
            Rows index (equivaent to oscillator number - 1)

        Notes
        -----
        This will set the background of the selected label to blue, and
        foreground to white. Entry widgets in the corresponding row are set to
        read-only mode. All other entry widgets are set to disabled mode."""

        for i, label in enumerate(self.table_labels):
            # disable all rows that do not match the index
            if i != index:
                if label['bg'] == 'blue':
                    label['bg'] = BGCOLOR
                    label['fg'] = 'black'
                    for entry in self.table_entries[i]:
                        entry['state'] = 'disabled'


        # proceed to highlight the selected row
        self.shift_left_click(index)

    def shift_left_click(self, index):
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

        if self.table_labels[index]['fg'] == 'black':
            fg, bg, state = 'white', 'blue', 'readonly'
        else:
            fg, bg, state  = 'black', BGCOLOR, 'disabled'

        self.table_labels[index]['fg'] = fg
        self.table_labels[index]['bg'] = bg

        for entry in self.table_entries[index]:
            entry['state'] = state

        # based on the number of rows selected, activate/deactivate
        # buttons accordingly
        self.activate_buttons()

    def activate_buttons(self):
        """Deals with a <Shift-Button-1> or <Button-1> event. Based on the
        number of selected rows, activates/deactivates buttons.
        """

        # determine number of selected rows in the table
        activated_number = len(self.get_selected_indices())

        buttons = [
            self.add_button,
            self.remove_button,
            self.split_button,
            self.merge_button,
            self.manual_button,
        ]

        # deactivate all buttons
        if activated_number == 0:
            states = ['normal'] + ['disabled'] * 4

        # activate split and manual edit buttons
        # deactivate merge button (can't merge one oscillator...)
        elif activated_number == 1:
            states = ['normal'] * 3 + ['disabled', 'normal']

        # activate merge button
        # deactivate split and manual edit buttons (ambiguous with multiple
        # oscillators selected)
        else: # activated_number > 1
            states = ['normal', 'normal', 'disabled', 'normal', 'disabled']

        for button, state in zip(buttons, states):
            button['state'] = state
            if state == 'normal':
                button['bg'] = BUTTONDEFAULT
            else:
                button['bg'] = '#e0e0e0'


    def add(self):
        """Add oscillators using
        :py:meth:`~nmrespy.core.NMREsPyBruker.add_oscillators`.
        """
        AddFrame(self, self.ctrl)


    def remove(self):
        """Removes all selected oscillators using
        :py:meth:`~nmrespy.core.NMREsPyBruker.remove_oscillators`.
        """

        indices = self.get_selected_indices()
        self.ctrl.info.remove_oscillators(indices)

        # update the plot
        self.ud_plot()
        # destroy and reconstruct the data table to match the new theta
        # self.reconstruct_table()
        self.construct_table(reconstruct=True)
        # decativate all buttons except close
        self.activate_buttons()

    def merge(self):
        """Merges all selected oscillators into a single oscillator.
        Mereging is performed using
        :py:meth:`~nmrespy.core.NMREsPyBruker.merge_oscillators`.
        """

        indices = self.get_selected_indices()
        self.ctrl.info.merge_oscillators(indices)

        # update the plot
        ud_plot(self.ctrl)

        # destroy and reconstruct the data table to match the new theta
        self.construct_table(reconstruct=True)
        # decativate all buttons except close
        self.activate_buttons()

    def split(self):
        """Opens a Toplevel (:py:class:`SplitFrame`) enabling the user
        to specify how they would like to split the oscillator.
        Splitting is done using
        :py:meth:`~nmrespy.core.NMREsPyBruker.split_oscillator`."""

        # N.B. the corresponding button is only active if the number of
        # selected rows is 1
        i = int(self.get_selected_indices()[0])

        # opens a SplitFrame window
        SplitFrame(self, self.ctrl, i)


    def manual_edit(self):
        """Enables user to manually change the parameters of the selected
        osciullator. I can't think of many times this would be necessary,
        but it's included for completeness."""

        # N.B. the corresponding button is only active if the number of
        # selected rows is 1
        i = int(self.get_selected_indices()[0])
        self.left_click(i)
        self.add_button['state'], self.add_button['bg'] = 'disabled', '#e0e0e0'

        for entry in self.table_entries[i]:
            entry['state'] = 'normal'

        # hacky way to get button with height less than 1:
        # make temporary frame with defined height (pixels), and
        # pack buttons in
        self.tmpframe = MyFrame(self.table, height=22, width=120)
        self.tmpframe.pack_propagate(0) # don't shrink
        self.tmpframe.grid(row=i+1, column=5)

        for c in (0, 1):
            self.tmpframe.columnconfigure(c, weight=1)

        self.udbutton = MyButton(
            self.tmpframe, text='Save', width=2, bg=BUTTONGREEN,
            font=(MAINFONT, 10), command=lambda i=i: self.ud_manual(i),
        )
        self.udbutton.pack(fill=tk.BOTH, expand=1, side=tk.LEFT, pady=(2,0))

        self.cancelbutton = MyButton(
            self.tmpframe, text='Cancel', width=2, bg=BUTTONRED,
            command=lambda i=i: self.cancel_manual(i), font=(MAINFONT, 10),
        )
        self.cancelbutton.pack(fill=tk.BOTH, expand=1, side=tk.RIGHT,
                               padx=(3,10), pady=(2,0))


    def ud_manual(self, i):
        """Replace the current parameters of the selected oscillator with
        the values in the entry boxes"""

        if not self.ctrl.check_invalid_entries(self.table):
            return

        # construct numpy array with oscillator's new parameters.
        oscillator = np.array([
            float(self.table_variables[i][j].get()) for j in range(4)
        ])

        # convert chemical shift from ppm to hz
        oscillator[2] = self.ctrl.convert(oscillator[2], conversion='ppm->hz')


        # replace oscillator with user input
        self.ctrl.info.theta[i] = oscillator
        # sort oscillators in order of frequency
        self.ctrl.info.theta = \
        self.ctrl.info.theta[np.argsort(self.ctrl.info.theta[..., 2])]

        # remove temporary buton frame
        self.tmpframe.destroy()
        # update plot and parameter table
        self.ud_plot()
        self.construct_table(reconstruct=True)

        self.activate_buttons()


    def cancel_manual(self, i):
        """Cancel manually chaning oscillator parameters."""

        # remove temporary button frame
        self.tmpframe.destroy()

        # replace contents of entry widgets with previous values in theta
        # set entry widgets back to read-only mode
        for j in range(4):
            value = self.ctrl.info.get_theta(frequency_unit='ppm')[i][j]
            self.table_variables[i][j].set(f'{value:.5f}')
            self.table_entries[i][j]['fg'] = 'black'
            self.table_entries[i][j]['highlightcolor'] = 'black'
            self.table_entries[i][j]['highlightbackground'] = 'black'

        # deactivate row
        self.left_click(i)


    def get_selected_indices(self):
        """Determine the indices of the rows which are selected."""

        indices = []
        for i, label in enumerate(self.table_labels):
            if label['bg'] == 'blue':
                indices.append(i)

        return indices


    def ud_plot(self):
        """Reconstructs the result plot after a change to the oscillators is
        made
        """

        # get new lines and labels
        # also obtaining ax to acquire new y-limits
        for key, value in zip(('lines', 'labels'), self.ctrl.info.plot_result()[2:]):
            self.ctrl.resultfig[key] = value

        # remove previous lines and labels
        self.ctrl.resultfig['ax'].lines = []
        self.ctrl.resultfig['ax'].texts = []

        # plot data line onto axis
        for line in self.ctrl.resultfig['lines'].values():
            self.ctrl.resultfig['ax'].plot(
            line.get_xdata(), line.get_ydata(), color=line.get_color(),
            lw=line.get_lw(),
        )

        for label in self.ctrl.resultfig['labels'].values():
            self.ctrl.resultfig['ax'].text(
                *label.get_position(), label.get_text(), fontsize=8,
            )

        # draw the new plot!
        self.ctrl.result_frames['plot_frame'].canvas.draw_idle()


    def check_param(self, row, column, vars_, entries, recover=True):

        try:
            value = float(vars_[row][column].get())

            # amplitude
            if column == 0:
                if value <= 0.:
                    raise

            # phase
            elif column == 1:
                if value <= -np.pi or value >= np.pi:
                    # if phase outside acceptable range, then wrap
                    value = (value + np.pi) % (2 * np.pi) - np.pi

            # frequency
            elif column == 2:
                if self.ctrl.bounds['lb']['ppm']['value'] < value \
                or self.ctrl.bounds['rb']['ppm']['value'] > value:
                    raise

            elif column == 3:
                if value <= 0.:
                    raise

            vars_[row][column].set(f'{value:.5f}')

        except:
            vars_[row][column].set('')
            entries[row][column]['fg'] = 'red'
            entries[row][column]['highlightcolor'] = 'red'
            entries[row][column]['highlightbackground'] = 'red'


class RerunFrame(tk.Toplevel):

    def __init__(self, parent, ctrl):
        tk.Toplevel.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'
        self.resizable(False, False)
        self.grab_set()

        with open(self.ctrl.info.logpath, 'r') as fh:
            lines = fh.readlines()[2:]

        for line in reversed(lines):
            if 'nonlinear_programming' in line:
                nlp_call = line
                break
            else:
                continue

        param_dict = ast.literal_eval(nlp_call[25:])
        for k, v in zip(param_dict.keys(), param_dict.values()):
            print(k, v)

        self.adsetframe = RerunSettingsFrame(self, self.ctrl)
        self.adsetframe.grid(row=0, column=0)


class AddFrame(MyToplevel):
    """Window for adding oscillators.
    Opened after calling :py:meth:`EditParams.add`."""

    def __init__(self, parent, ctrl):
        super().__init__(parent)
        self.parent = parent
        self.ctrl = ctrl
        self.grab_set()

        self.table_frame = MyFrame(self)
        self.table_frame.grid(row=0, column=0)

        titles = ('Amplitude', 'Phase', 'Frequency', 'Damping')
        for column, title in enumerate(titles):
            label = MyLabel(self.table_frame, text=title)

            if column == 0:
                padx = (10, 5)
            elif column == 3:
                padx = (0, 10)
            else:
                padx = (0, 5)

            label.grid(row=0, column=column, padx=padx, pady=(10,0), sticky='w')

        self.entries = []
        self.vars = []

        self.add_row()

        self.button_frame = MyFrame(self)
        self.button_frame.grid(row=1, column=0, sticky='e')

        self.add_button = MyButton(
            self.button_frame, text='Add', width=3, command=self.add_row,
        )
        self.add_button.grid(row=0, column=0, padx=(0,10), pady=(10,10))

        self.cancel_button = MyButton(
            self.button_frame, text='Cancel', bg=BUTTONRED,
            command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=1, padx=(0,10), pady=(10,10))

        self.save_button = MyButton(
            self.button_frame, text='Save', width=8, bg=BUTTONGREEN,
            command=self.save,
        )
        self.save_button.grid(row=0, column=2, padx=(0,10), pady=(10,10))


    def add_row(self):

        row = len(self.entries) + 1

        entry_row = []
        var_row = []

        for column in range(4):
            var = tk.StringVar()
            var.set('')

            return_args = (row-1, column, self.vars, self.entries)
            entry = MyEntry(
                self.table_frame, return_command=self.parent.check_param,
                return_args=return_args, width=12, textvariable=var,
                highlightcolor='red', highlightbackground='red',
                fg='red',
            )

            if column == 0:
                padx = (10, 5)
            elif column == 3:
                padx = (0, 10)
            else:
                padx = (0, 5)

            entry.grid(
                row=row, column=column, padx=padx, pady=(5,0), sticky='w',
            )

            var_row.append(var)
            entry_row.append(entry)

        self.entries.append(entry_row)
        self.vars.append(var_row)


    def save(self):

        if not self.ctrl.check_invalid_entries(self.table_frame):
            return

        oscillators = []

        for var_row in self.vars:
            oscillator_row = []
            for i, var in enumerate(var_row):

                # convert chemical shift from ppm to hz
                if i == 2:
                    value = self.ctrl.convert(
                        float(var.get()), conversion='ppm->hz',
                    )

                else:
                    value = float(var.get())

                oscillator_row.append(value)
            oscillators.append(oscillator_row)

        oscillators = np.array(oscillators)

        self.ctrl.info.add_oscillators(oscillators)
        print(self.ctrl.info.get_theta())
        # update the plot
        self.parent.ud_plot()
        # reconstruct the parameter table with the updated oscillators
        self.parent.construct_table(reconstruct=True)
        self.destroy()


class SplitFrame(tk.Toplevel):
    """Window for specifying how to split a certain oscillator.
    Opened after calling :py:meth:`EditParams.split`."""

    def __init__(self, parent, ctrl, index):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.ctrl = ctrl
        # row selected
        self.index = index
        self['bg'] = 'white'
        self.resizable(False, False)
        self.grab_set()

        # create text labels (title, number of oscillators, frequency
        # separatio, amplitude ratio)
        title = tk.Label(self, bg='white',
                         text=f'Splitting Oscillator {index + 1}:',
                         font=(MAINFONT, 12, 'bold'))
        title.grid(row=0, column=0, columnspan=3, sticky='w', padx=10,
                   pady=(10,0))

        number_label = tk.Label(self, bg='white', text='Number of oscillators:')
        number_label.grid(row=1, column=0, sticky='w', padx=(10,0), pady=(10,0))
        separation_label = tk.Label(self, bg='white',
                                    text='Frequency separation:')
        separation_label.grid(row=2, column=0, sticky='w', padx=(10,0),
                              pady=(10,0))
        ratio_label = tk.Label(self, bg='white', text='Amplitude ratio:')
        ratio_label.grid(row=4, column=0, sticky='w', padx=(10,0), pady=(10,0))

        # spinbox enabling user to choose number of oscillators to split into
        self.number = tk.Spinbox(self, values=tuple(range(2, 11)), width=4,
                                 command=self.change_number, state='readonly',
                                 readonlybackground='white')
        self.number.grid(row=1, column=1, columnspan=2, sticky='w', padx=10,
                         pady=(10,0))

        # frequency separation variable (set default to 2Hz)
        self.freq_hz = tk.StringVar()
        self.freq_hz.set(f'{2.0:.5f}')
        # entry widget for separation frequency (Hz)
        self.freq_hz_entry = tk.Entry(self, bg='white', width=10,
                                      textvariable=self.freq_hz,
                                      highlightthickness=0)
        # upon any change to the entry widget, update the ppm entry as well
        self.freq_hz_entry.bind('<KeyRelease>', self.key_press_hz)
        self.freq_hz_entry.grid(row=2, column=1, sticky='w', padx=(10,0),
                                pady=(10,0))

        hz_label = tk.Label(self, bg='white', text='Hz')
        hz_label.grid(row=2, column=2, sticky='w', padx=(0,10), pady=(10,0))

        # repeat frequency separation widgets for ppm units
        ppm = float(self.freq_hz.get()) / self.ctrl.info.get_sfo()[0]
        self.freq_ppm = tk.StringVar()
        self.freq_ppm.set(f'{ppm:.5f}')

        self.freq_ppm_entry = tk.Entry(self, bg='white', width=10,
                                       textvariable=self.freq_ppm,
                                       highlightthickness=0)
        self.freq_ppm_entry.bind('<KeyRelease>', self.key_press_ppm)
        self.freq_ppm_entry.grid(row=3, column=1, sticky='w', padx=(10,0),
                                 pady=(5,0))

        ppm_label = tk.Label(self, bg='white', text='ppm')
        ppm_label.grid(row=3, column=2, sticky='w', padx=(0,10), pady=(5,0))

        # amplitude ratio string variable
        self.amplitude_ratio = tk.StringVar()
        # default: all oscillators have same amplitude (i.e. '1:1:1' if 3)
        self.amplitude_ratio.set('1:' * (int(self.number.get()) - 1) + '1')
        # entry box for amplitude ratio
        self.ratio_entry = tk.Entry(self, bg='white', width=16,
                                    textvariable=self.amplitude_ratio,
                                    highlightthickness=0)
        self.ratio_entry.grid(row=4, column=1, columnspan=3, sticky='w',
                              padx=10, pady=(15,10))
        # frame for containing save and cancel buttons
        self.buttonframe = tk.Frame(self, bg='white')
        self.buttonframe.grid(row=5, column=0, columnspan=4, sticky='e')
        # cancel button: close window without changes
        self.cancel_button = tk.Button(self.buttonframe, text='Cancel', width=8,
                                      bg='#ff9894',
                                      highlightbackground='black',
                                      command=self.destroy)
        self.cancel_button.grid(row=0, column=0, sticky='e', padx=(0,10),
                                pady=(0,10))
        # save button: close window and make changes
        self.save_button = tk.Button(self.buttonframe, text='Save', width=8,
                                     bg='#9eda88',
                                     highlightbackground='black',
                                     command=self.save)
        self.save_button.grid(row=0, column=1, sticky='e', padx=(0,10),
                             pady=(0,10))


    def change_number(self):
        """Upon changing the value of the spinbox, set default string
        for the amplitude ratio (all equal amplitudes)"""
        self.amplitude_ratio.set('1:' * (int(self.number.get()) - 1) + '1')

    def key_press_hz(self, entry):
        """Update the value of the ppm variable upon change to Hz."""
        try:
            ppm = float(self.freq_hz.get()) / self.ctrl.info.get_sfo()[0]
            self.freq_ppm.set(f'{ppm:.5f}')
        # if Hz input could not be understood as a numerical value, clear
        # ppm string variable
        except:
            self.freq_ppm.set('')

    def key_press_ppm(self, entry):
        """Update the value of the Hz variable upon change to ppm."""
        try:
            hz = float(self.freq_ppm.get()) * self.ctrl.info.get_sfo()[0]
            self.freq_hz.set(f'{hz:.5f}')
        # if ppm input could not be understood as a numerical value, clear
        # Hz string variable
        except:
            self.freq_hz.set('')

    def save(self):
        """Apply oscillator splitting and close window"""

        # nnumber of child oscillators
        number = int(self.number.get())
        # amplitude ratio
        ratio = self.amplitude_ratio.get().split(':')
        # check number of values in raio match number of children
        if len(ratio) == number:
            # check all elements in ratio can be converted to floats
            try:
                ratio = [float(i) for i in ratio]
            except:
                # determine identites of component(s) that could not be
                # converted to float
                fails = []
                for i in ratio:
                    try:
                        i = float(i)
                    except:
                        fails.append(i)

                msg = 'Some of the values you specified in the amplitude' \
                      + ' ratio could not be understood a numerical values:\n'
                for fail in fails:
                    msg += f'\'{fail}\','

                WarnFrame(self, msg[:-1])
                return
        else:
            msg = 'The number of values specified by the amplitude ratio' \
                  + f' ({len(ratio)}) does not match the desired number of' \
                  + f' oscillators ({number}).'

            WarnFrame(self, msg)
            return

        # check that the specified freuency separation can be converted to
        # float
        try:
            frequency_sep = float(self.freq_hz.get())
        except:
            msg = f'The frequency separation ({frequency_sep}) could not be' \
                  + 'interpreted as a numerical value.'

            WarnFrame(self, msg)
            return

        self.ctrl.info.split_oscillator(self.index,
                                              frequency_sep=frequency_sep,
                                              split_number=number,
                                              amp_ratio=ratio)
        # update the plot
        ud_plot(self.ctrl)
        # reconstruct the parameter table with the updated oscillators
        self.parent.construct_table(reconstruct=True)
        self.destroy()


class SaveFrame(MyToplevel):
    """Toplevel for choosing how to save estimation result"""

    def __init__(self, parent, ctrl):

        super().__init__(parent)

        self.parent = parent
        self.ctrl = ctrl

        self.grab_set()

        # main contents of the toplevel
        self.file_frame = MyFrame(self)
        self.file_frame.grid(row=0, column=0)
        # buttons at the bottom of the frame
        self.button_frame = MyFrame(self)
        self.button_frame.grid(row=1, column=0, sticky='e')

        # --- RESULT FIGURE ---
        MyLabel(self.file_frame, text='Save Figure').grid(
            row=0, column=0, padx=(10,0), pady=(10,0), sticky='w'
        )
        # specifier for whether or not to save a figure
        self.figure_var = tk.IntVar()
        self.figure_var.set(1)
        # checkbutton to choose whether to save a figure
        self.figure_check = MyCheckbutton(
            self.file_frame, variable=self.figure_var,
            command=(lambda: self.check('figure')),
        )
        self.figure_check.grid(row=0, column=1, padx=(2,0), pady=(10,0))
        # open the figure customiser
        self.figure_button = MyButton(
            self.file_frame, text='Customise Figure',
            command=self.customise_figure,
        )
        self.figure_button.grid(
            row=0, column=2, columnspan=3, padx=10, pady=(10,0), sticky='ew',
        )

        # --- OTHER FILES: PDF, TEXT, PICKLE ---
        titles = ('Save textfile:', 'Save PDF:', 'Pickle result:')
        extensions = ('.txt', '.pdf', '.pkl')
        for i, (title, extension) in enumerate(zip(titles, extensions)):

            MyLabel(self.file_frame, text=title).grid(
                row=i+1, column=0, padx=(10,0), pady=(10,0), sticky='w',
            )

            tag = extension[1:]
            # variable which dictates whether to save the filetype
            self.__dict__[f'{tag}_var'] = save_var = tk.IntVar()
            save_var.set(1)

            self.__dict__[f'{tag}_check'] = check = \
            MyCheckbutton(
                self.file_frame, variable=save_var,
                command=(lambda tag=tag: self.check(tag))
            )
            check.grid(row=i+1, column=1, padx=(2,0), pady=(10,0))

            MyLabel(self.file_frame, text='Filename:').grid(
                row=i+1, column=2, padx=(15,0), pady=(10,0), sticky='w'
            )

            self.__dict__[f'{tag}_name'] = fname_var = tk.StringVar()
            fname_var.set('nmrespy_result')

            self.__dict__[f'{tag}_entry'] = entry = \
            MyEntry(
                self.file_frame, textvariable=fname_var, width=20,
            )
            entry.grid(row=i+1, column=3, padx=(5,0), pady=(10,0))

            MyLabel(self.file_frame, text=extension).grid(
                row=i+1, column=4, padx=(0,10), pady=(10,0), sticky='w',
            )

        # --- DESCRIPTION FOR TEXTFILE AND PDF ---
        MyLabel(self.file_frame, text='Description:').grid(
            row=4, column=0, padx=(10,0), pady=(10,0), sticky='nw',
        )

        self.descr_box = tk.Text(self.file_frame, width=40, height=3)
        self.descr_box.grid(
            row=4, column=1, columnspan=4, padx=10, pady=(10,0), sticky='ew',
        )

        # --- DIRECTORY TO SAVE FILES TO ---
        MyLabel(self.file_frame, text='Directory:').grid(
            row=5, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.dir_var = tk.StringVar()
        self.dir_var.set(os.path.expanduser('~'))

        self.dir_entry = tk.Entry(
            self.file_frame, textvariable=self.dir_var, width=25,
            highlightthickness=0
        )
        self.dir_entry.grid(
            row=5, column=1, columnspan=3, padx=(10,0), pady=(10,0), sticky='ew'
        )

        self.img = get_PhotoImage(
            os.path.join(IMAGESPATH, 'folder_icon.png'), scale=0.02
        )

        self.dir_button = MyButton(
            self.file_frame, command=self.browse, image=self.img, width=40
        )
        self.dir_button.grid(row=5, column=4, padx=(5,5), pady=(10,0))

        # cancel button - returns usere to result toplevel
        self.cancel_button = MyButton(
            self.button_frame, text='Cancel', bg=BUTTONRED,
            command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=0, pady=10)

        # save button - determines what file types to save and generates them
        self.save_button = MyButton(
            self.button_frame, text='Save', width=8, bg=BUTTONGREEN,
            command=self.save
        )
        self.save_button.grid(row=0, column=1, padx=10, pady=10)


    def check(self, tag):
        """Deals with when user clicks a checkbutton"""
        var = self.__dict__[f'{tag}_var'].get()
        state = 'normal' if var else 'disabled'

        if tag == 'figure':
            self.figure_button['state'] = state
        else:
            self.__dict__[f'{tag}_entry']['state'] = state

    def browse(self):
        """Directory selection using tkinter's filedialog"""
        self.dir_var.set(filedialog.askdirectory(initialdir=self.dir_var.get()))

    def save(self):
        # check directory is valid
        dir = self.dir_var.get()
        descr = self.descr_box.get('1.0', 'end-1c')

        # check textfile
        if self.txt_var.get():
            self.ctrl.info.write_result(
                description=descr, fname=self.txt_name.get(), dir=dir,
                format='txt', force_overwrite=True,
            )

        # check PDF
        if self.pdf_var.get():
            self.ctrl.info.write_result(
                description=descr, fname=self.pdf_name.get(), dir=dir,
                format='pdf', force_overwrite=True,
            )

        # check pickle
        if self.pkl_var.get():
            self.ctrl.info.pickle_save(
                fname=self.pkl_name.get(), dir=dir, force_overwrite=True
            )

    def customise_figure(self):
        CustomiseFigureFrame(self, self.ctrl)

class CustomiseFigureFrame(tk.Toplevel):

    def __init__(self, parent, ctrl):
        tk.Toplevel.__init__(self, parent)
        self.title('Customise Figure')
        self.ctrl = ctrl
        self['bg'] = 'white'
        self.grab_set()

        self.mainlist = tk.Listbox(self, width=10, selectmode=tk.SINGLE)
        self.mainlist.bind('<<ListboxSelect>>', self.ud_mainlist)

        self.frames = {}

        for item in ['General', 'Axes', 'Lines', 'Labels']:
            self.mainlist.insert(tk.END, item)

        self.mainlist.grid(
            row=0, column=0, padx=(10,0), pady=(10,0), sticky='n'
        )

        self.optframe = tk.Frame(self, bg='white')
        self.optframe.grid(row=0, column=1)

        self.frames = {}

        for F in (GeneralFrame, AxesFrame, LinesFrame, LabelsFrame):
            self.frames[F.__name__] =  F(self.optframe, self.ctrl)

        self.frames['GeneralFrame'].grid(row=0, column=0)
        self.active = 'GeneralFrame'


    def ud_mainlist(self, event):
        item = self.mainlist.get(self.mainlist.curselection())

        self.frames[self.active].grid_remove()
        self.active = f'{item}Frame'
        self.frames[self.active].grid(row=0, column=0)


class GeneralFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        tk.Label(
            self, text='GeneralFrame', font=(MAINFONT, 20, 'bold'), bg='white'
        ).pack(padx=30, pady=30)


class AxesFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        tk.Label(
            self, text='AxesFrame', font=(MAINFONT, 20, 'bold'), bg='white'
        ).pack(padx=30, pady=30)


class LinesFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        items = ['data']
        items += [f'osc{i+1}' for i in range(self.ctrl.info.theta.shape[0])]

        tk.Label(
            self, text='Color:', font=(MAINFONT, 12, 'bold'), bg='white'
        ).grid(row=0, column=1, padx=10, pady=(10,0), sticky='w')

        self.frames = {}

        for item in items:
            self.frames[item] = frame = tk.Frame(self, bg='white')
            col = self.ctrl.lines[item].get_color()
            colorpicker = ColorPicker(
                frame, self.ctrl, init_color=col, object=self.ctrl.lines[item]
            )
            colorpicker.grid(row=0, column=0)

            lwframe = tk.Frame(frame, bg='white')
            lwframe.grid(row=1, column=0)

            tk.Label(
                lwframe, text='Linewidth:', font=(MAINFONT, 12, 'bold'),
                bg='white'
            ).grid(row=0, column=0, padx=(10,0), pady=(10,0), sticky='w')



        self.active = 'data'
        self.frames[self.active].grid(row=1, column=1, pady=(10,0))

        self.linelist = CustomListBox(self, items)

        self.linelist.grid(row=0, column=0, rowspan=3, pady=10)



    def click_linelist(self):

        self.frames[self.active].grid_remove()

        for item in self.linelist.options.keys():
            if item['bg'] == '#cde6ff':
                self.active = item

        self.frames[self.active].grid(row=1, column=1, pady=(10,0))


    def change_color(self, key):
        try:
            color = self.colorpickers[key].color_var.get()
            self.ctrl.lines[key].set_color(color)
            self.ctrl.frames['PlotFrame'].canvas.draw_idle()
        except KeyError:
            pass


class LabelsFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        tk.Label(
            self, text='LabelsFrame', font=(MAINFONT, 20, 'bold'), bg='white'
        ).pack(padx=30, pady=30)


class LineFrame(tk.Frame):
    def __init__(self, parent, ctrl, idx):

        tk.Frame.__init__(self, parent)
        self['bg'] = 'white'
        self.ctrl = ctrl
        self.idx = idx

        color = self.ctrl.lines[self.idx].get_color()
        self.colorpicker = ColorPicker(self, ctrl, init_color='random')
        self.colorpicker.grid(row=0, column=0)



class ColorPicker(tk.Frame):
    # BUG: If init color has equal RGB values, the entry widget content
    #      is mysteriously linked. Any change to one color will change
    #      the entry widgets to match.

    def __init__(self, parent, ctrl, init_color='random', object=None):

        tk.Frame.__init__(self, parent)
        self['bg'] = 'white'
        self.ctrl = ctrl
        self.object = object
        self.topframe = tk.Frame(self, bg='white')
        self.topframe.grid(row=0, column=0)
        self.bottomframe = tk.Frame(self, bg='white')
        self.bottomframe.grid(row=1, column=0, sticky='w')

        if init_color == 'random':
            r = lambda: random.randint(0,255)
            init_color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())

        print(init_color)

        self.color_var = tk.StringVar()
        self.color_var.set(init_color)
        color_tags = ['r', 'g', 'b']
        color_hexes = ['#ff0000', '#008000', '#0000ff']
        for i, (tag, hexa) in enumerate(zip(color_tags, color_hexes)):

            tk.Label(self.topframe, text=tag.upper(), fg=hexa, bg='white').grid(
                row=i, column=0, padx=(10,0), pady=(10,0), sticky='w'
            )

            self.__dict__[f'{tag}_var'] = var = tk.StringVar()
            var.set(init_color[2*i+1:2*i+3])

            tc = '#' + (i * '00') + var.get() + (abs(i - 2) * '00')
            self.__dict__[f'{tag}_scale'] = scale = \
                tk.Scale(
                    self.topframe, from_=0, to=255, orient=tk.HORIZONTAL,
                    showvalue=0, bg='white', sliderlength=15, bd=0,
                    troughcolor=tc, length=500, highlightthickness=1,
                    highlightbackground='black',
                    command=lambda val, tag=tag: self.ud_scale(val, tag)
                )
            scale.set(int(var.get(), 16))
            scale.grid(row=i, column=1, padx=(10,0), pady=(10,0))

            self.__dict__[f'{tag}_entry'] = entry = \
                tk.Entry(
                    self.topframe, bg='white', text=var.get().upper(),
                    width=3, highlightthickness=0
                )
            entry.bind('<Return>', (lambda event, tag=tag: self.ud_entry(tag)))
            entry.grid(row=i, column=2, padx=(10,0), pady=(10,0))

        self.swatch = tk.Canvas(self.topframe, width=40, height=40, bg='white')
        self.rectangle = self.swatch.create_rectangle(
            0, 0, 40, 40, fill=self.color_var.get()
        )
        self.swatch.grid(row=0, column=3, rowspan=3, pady=(10,0), padx=10)

        tk.Label(self.bottomframe, text='matplotlib colour:', bg='white').grid(
            row=0, column=0, padx=(10,0), pady=10
        )

        self.mpl_entry = tk.Entry(
            self.bottomframe, bg='white', width=12, highlightthickness=0
        )
        self.mpl_entry.bind('<Return>', self.mpl_color)
        self.mpl_entry.grid(row=0, column=1, padx=(5,10), pady=10)


    def ud_scale(self, value, tag):

        hexa = f'{int(value):02x}'
        self.__dict__[f'{tag}_var'].set(hexa)
        self.__dict__[f'{tag}_entry'].delete(0, tk.END)
        self.__dict__[f'{tag}_entry'].insert(0, hexa.upper())

        col = self.color_var.get()
        if tag == 'r':
            self.r_scale['troughcolor'] = f'#{hexa}0000'
            self.color_var.set(f'#{hexa}{col[3:]}')
        elif tag == 'g':
            self.g_scale['troughcolor'] = f'#00{hexa}00'
            self.color_var.set(f'#{col[1:3]}{hexa}{col[5:]}')
        else:
            self.b_scale['troughcolor'] = f'#0000{hexa}'
            self.color_var.set(f'#{col[1:5]}{hexa}')

        self.swatch.itemconfig(self.rectangle, fill=self.color_var.get())

        if self.object:
            self.object.set_color(self.color_var.get())
            self.ctrl.frames['PlotFrame'].canvas.draw_idle()


    def ud_entry(self, tag):

        hexa = self.__dict__[f'{tag}_entry'].get().lower()

        # check valid 2-digit hex value
        try:
            # decimal value
            dec = int(hexa, 16)
            self.__dict__[f'{tag}_scale'].set(dec)
            # all other necessary updates will be handled by ud_scale...

        except:
            # invalid input from user: reset entry widget with current value
            self.__dict__[f'{tag}_entry'].delete(0, tk.END)
            self.__dict__[f'{tag}_entry'].insert(
                0, self.__dict__[f'{tag}_var'].get().upper()
            )

    def mpl_color(self, event):

        color = self.mpl_entry.get()
        self.mpl_entry.delete(0, tk.END)

        try:
            hexa = mcolors.to_hex(color)

        except:
            return

        self.color_var.set(hexa)
        self.r_scale.set(int(hexa[1:3], 16))
        self.g_scale.set(int(hexa[3:5], 16))
        self.b_scale.set(int(hexa[5:], 16))



class CustomListBox(tk.Frame):

    def __init__(self, parent, items, height=12):
        tk.Frame.__init__(self, parent)
        self['bg'] = 'white'
        self['highlightthickness'] = 2

        self.options = {}
        for i, item in enumerate(items):

            self.options[item] = label = \
                tk.Label(self, text=item, bg='white', anchor='w')
            label.bind('<Button-1>', lambda e, item=item: self.click(item))
            label.grid(row=i, column=0, sticky='new', ipadx=2)

        self.active = items[0]
        self.options[self.active]['bg'] = '#cde6ff'


    def click(self, item):
        print('click_called')
        self.options[self.active]['bg'] = 'white'
        self.active = item
        self.options[self.active]['bg'] = '#cde6ff'




class ConfigLines(tk.Toplevel):
    def __init__(self, parent, lines, figcanvas):
        tk.Toplevel.__init__(self, parent)
        self.resizable(True, False)
        self.lines = lines
        self.figcanvas = figcanvas

        self.config(bg='white')

        options = ['All Oscillators']
        keys = self.lines.keys()
        for key in keys:
            opt = key.replace('osc', 'Oscillator ').replace('data', 'Data')
            options.append(opt)

        self.leftframe = tk.Frame(self, bg='white')
        self.rightframe = tk.Frame(self, bg='white')
        self.customframe = tk.Frame(self.rightframe, bg='white')
        self.buttonframe = tk.Frame(self.rightframe, bg='white')

        self.list = tk.Listbox(self.leftframe, height=14, width=14)
        for opt in options:
            self.list.insert(tk.END, opt)

        self.list.bind('<Double-Button-1>', (lambda entry: self.showlineeditor()))

        self.list.activate(0)
        self.list.selection_set(0)



        self.close_but = tk.Button(self.buttonframe, text='Close', width=8,
                                   bg='#ff9894', highlightbackground='black',
                                   command=self.close)

        self.list.grid(row=0, column=0, sticky='nsew')
        self.close_but.grid(row=0, column=0, sticky='e')

        self.leftframe.grid(row=0, column=0, sticky='nsew', padx=(10,0), pady=10)
        self.rightframe.grid(row=0, column=1, sticky='ew')
        self.customframe.grid(row=0, column=0, sticky='ew')
        self.buttonframe.grid(row=1, column=0, sticky='e', padx=10, pady=(0,10))

        self.columnconfigure(1, weight=1)
        self.leftframe.rowconfigure(0, weight=1)
        self.rightframe.columnconfigure(0, weight=1)
        self.rightframe.rowconfigure(0, weight=1)
        self.customframe.columnconfigure(0, weight=1)
        self.buttonframe.columnconfigure(0, weight=1)

        self.showlineeditor()


    def showlineeditor(self):

        index = self.list.curselection()
        name = self.list.get(index)

        for widget in self.customframe.winfo_children():
            widget.destroy()

        if 'Oscillator ' in name:
            number = int(name.strip('Oscillator '))
            self.lineeditframe = LineEdit(self.customframe, self.lines, number,
                                          self.figcanvas)

        elif name == 'Data':
            self.lineeditframe = LineEdit(self.customframe, self.lines, 0,
                                          self.figcanvas)

        elif name == 'All Oscillators':
            self.lineeditframe = LineMultiEdit(self.customframe, self.lines,
                                             self.figcanvas)

    def close(self):
        self.destroy()


class LineEdit(tk.Frame):
    def __init__(self, parent, lines, number, figcanvas):
        tk.Frame.__init__(self, parent)

        self['bg'] = 'white'
        self.grid(row=0, column=0, sticky='ew')

        self.lines = lines
        self.figcanvas = figcanvas

        if number == 0:
            self.line = self.lines[f'data']
        else:
            self.line = self.lines[f'osc{number}']

        init_color = self.line.get_color()
        self.lw = self.line.get_lw() # linewidth
        self.ls = self.line.get_ls() # linstyle

        self.colorframe = ColorPicker(self, init_color) # color title, colors scales and entries
        self.lwframe = tk.Frame(self, bg='white') # linewidth scale and entry
        self.lsframe = tk.Frame(self, bg='white') # linestyle optionbox

        self.colorframe.grid(row=0, column=0, sticky='ew')
        self.lwframe.grid(row=1, column=0, sticky='ew', padx=10, pady=5)
        self.lsframe.grid(row=2, column=0, sticky='ew', padx=10, pady=5)

        self.columnconfigure(0, weight=1)

        # tweak scale and entry commands to dynamically change plot
        scales = [self.colorframe.r_scale, self.colorframe.g_scale,
                  self.colorframe.b_scale]
        commands = [lambda r: self.colorframe.ud_r_sc(r, dynamic=True,
                                     object=self.line, figcanvas=self.figcanvas),
                    lambda g: self.colorframe.ud_g_sc(g, dynamic=True,
                                     object=self.line, figcanvas=self.figcanvas),
                    lambda b: self.colorframe.ud_b_sc(b, dynamic=True,
                                     object=self.line, figcanvas=self.figcanvas)]

        for scale, command in zip(scales, commands):
            scale['command'] = command


        # --- FRAME 3: Linewidth ----------------------------------------------
        self.lw_ttl = tk.Label(self.lwframe, text='Linewidth', bg='white',
                               font=(MAINFONT, 12))

        self.lw_scale = tk.Scale(
            self.lwframe, from_=0, to=2.5, orient=tk.HORIZONTAL,
                                 showvalue=0, bg='white', sliderlength=15, bd=0,
                                 troughcolor=f'#808080', highlightthickness=1,
                                 command=self.ud_lw_sc, resolution=0.001,
                                 highlightbackground='black',)
        self.lw_scale.set(self.lw)

        self.lw_ent = tk.Entry(self.lwframe, bg='white', text=f'{self.lw:.3f}',
                               width=6, highlightthickness=0)

        self.lw_ent.bind('<Return>', (lambda event: self.ud_lw_ent()))

        self.lw_ttl.grid(row=0, column=0, sticky='w')
        self.lw_scale.grid(row=0, column=1, sticky='ew', padx=(10,0))
        self.lw_ent.grid(row=0, column=2, sticky='w', padx=(10,0))

        self.lwframe.columnconfigure(1, weight=1)

        # --- FRAME 4: Linestyle Optionmenu -----------------------------------
        self.ls_ttl = tk.Label(self.lsframe, text='Linestyle', bg='white',
                               font=(MAINFONT, 12))

        self.ls_options = ('solid', 'dotted', 'dashed', 'dashdot')
        self.ls_str = tk.StringVar()
        self.ls_str.set(self.ls_options[0])
        self.ls_optmenu = tk.OptionMenu(self.lsframe, self.ls_str, *self.ls_options,
                                        command=self.ud_ls)

        self.ls_ttl.grid(row=0, column=0, sticky='w')
        self.ls_optmenu.grid(row=0, column=1, sticky='w', padx=(10,0))

    def ud_lw_sc(self, lw):
        self.lw = float(lw)
        self.lw_ent.delete(0, tk.END)
        self.lw_ent.insert(0, f'{self.lw:.3f}')
        self.line.set_lw(self.lw)
        self.figcanvas.draw_idle()

    def ud_lw_ent(self):
        value = self.lw_ent.get()
        valid_lw = self.check_lw(value)

        if valid_lw:
            self.lw = float(value)
            self.lw_scale.set(self.lw)
            self.line.set_lw(self.lw)
            self.figcanvas.draw_idle()

        else:
            self.lw_ent.delete(0, tk.END)
            self.lw_ent.insert(0, self.lw)

    @staticmethod
    def check_lw(value):
        try:
            lw = float(value)
            return 0 <= lw <= 5
        except:
            return False

    def ud_ls(self, ls):
        self.ls = ls
        self.line.set_ls(self.ls)
        self.figcanvas.draw_idle()

    def close(self):
        self.parent.destroy()


class LineMultiEdit(tk.Frame):
    def __init__(self, parent, lines, figcanvas):
        tk.Frame.__init__(self, parent)
        self['bg'] = 'white'

        self.grid(row=0, column=0, sticky='nsew')

        self.lines = lines
        self.figcanvas = figcanvas

        self.colors = []
        for key in [k for k in self.lines.keys() if k != 'data']:
            color = self.lines[key].get_color()
            if color in self.colors:
                break
            else:
                self.colors.append(self.lines[key].get_color())

        self.lw = self.lines['osc1'].get_lw() # linewidth
        self.ls = self.lines['osc1'].get_ls() # linstyle

        self.colorframe = tk.Frame(self, bg='white') # colors
        self.lwframe = tk.Frame(self, bg='white') # linewidths
        self.lsframe = tk.Frame(self, bg='white') # linestyles

        self.colorframe.grid(row=0, column=0, sticky='ew', padx=10, pady=(10,0))
        self.lwframe.grid(row=1, column=0, sticky='ew', padx=10, pady=(10,0))
        self.lsframe.grid(row=2, column=0, sticky='ew', padx=10, pady=10)

        # --- FRAME 1: Color List  --------------------------------------------
        self.colortopframe = tk.Frame(self.colorframe, bg='white')
        self.colorbotframe = tk.Frame(self.colorframe, bg='white')

        self.colortopframe.grid(row=0, column=0, sticky='w')
        self.colorbotframe.grid(row=1, column=0, sticky='ew')

        self.colorbotframe.rowconfigure(1, weight=1)

        self.color_ttl = tk.Label(self.colortopframe, text='Color Cycle',
                                  bg='white', font=(MAINFONT, 12))
        self.color_ttl.grid(row=0, column=0)

        self.colorlist = tk.Listbox(self.colorbotframe)
        for i, color in enumerate(self.colors):
            self.colorlist.insert(tk.END, color)
            self.colorlist.itemconfig(i, foreground=color)

        self.add_but = tk.Button(self.colorbotframe, text='Add', width=8,
                                 bg='#9eda88', highlightbackground='black',
                                 command=self.add_color)
        self.rm_but = tk.Button(self.colorbotframe, text='Remove', width=8,
                                 bg='#ff9894', highlightbackground='black',
                                 command=self.rm_color)

        self.colorlist.grid(row=0, column=0, rowspan=2, sticky='nsew')
        self.add_but.grid(row=0, column=1, sticky='nw', padx=(10,0))
        self.rm_but.grid(row=1, column=1, sticky='nw', padx=(10,0), pady=(10,0))

        # --- FRAME 2: Linewidth ----------------------------------------------
        self.lw_ttl = tk.Label(self.lwframe, text='Linewidth', bg='white',
                               font=(MAINFONT, 12))

        self.lw_scale = tk.Scale(self.lwframe, from_=0, to=5, orient=tk.HORIZONTAL,
                                 showvalue=0, bg='white', sliderlength=15, bd=0,
                                 troughcolor=f'#808080', length=500,
                                 highlightthickness=1, command=self.ud_lw_sc,
                                 resolution=0.001)
        self.lw_scale.set(self.lw)

        self.lw_ent = tk.Entry(self.lwframe, bg='white', text=f'{self.lw:.3f}',
                               width=6, highlightthickness=0)

        self.lw_ent.bind('<Return>', (lambda event: self.ud_lw_ent()))

        self.lw_ttl.grid(row=0, column=0, sticky='w')
        self.lw_scale.grid(row=0, column=1, sticky='ew')
        self.lw_ent.grid(row=0, column=2, sticky='w')

        # --- FRAME 4: Linestyle Optionmenu -----------------------------------
        self.ls_ttl = tk.Label(self.lsframe, text='Linestyle', bg='white',
                               font=(MAINFONT, 12))

        self.ls_options = ('solid', 'dotted', 'dashed', 'dashdot')
        self.ls_str = tk.StringVar()
        self.ls_str.set(self.ls_options[0])
        self.ls_optmenu = tk.OptionMenu(self.lsframe, self.ls_str, *self.ls_options,
                                        command=self.ud_linestyles)

        self.ls_ttl.grid(row=0, column=0, sticky='w')
        self.ls_optmenu.grid(row=0, column=1, sticky='w')


    def add_color(self):
        self.colorwindow = tk.Toplevel(self)
        self.colorwindow['bg'] = 'white'
        self.colorwindow.resizable(True, False)
        # generate random hex color
        r = lambda: random.randint(0,255)
        color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
        self.colorpicker = ColorPicker(self.colorwindow, init_color=color)
        self.buttonframe = tk.Frame(self.colorwindow, bg='white')

        self.colorpicker.grid(row=0, column=0, sticky='ew')
        self.buttonframe.grid(row=1, column=0, sticky='e', padx=10, pady=10)

        self.colorwindow.columnconfigure(0, weight=1)
        self.colorpicker.columnconfigure(0, weight=1)

        self.save_but = tk.Button(self.buttonframe, text='Confirm', width=8,
                                  bg='#9eda88', highlightbackground='black',
                                  command=self.confirm)

        self.cancel_but = tk.Button(self.buttonframe, text='Cancel', width=8,
                                  bg='#ff9894', highlightbackground='black',
                                  command=self.cancel)

        self.cancel_but.grid(row=2, column=0, sticky='e')
        self.save_but.grid(row=2, column=1, sticky='e', padx=(10,0))

    def confirm(self):
        color = self.colorpicker.color
        self.colors.append(color)
        self.colorlist.insert(tk.END, color)
        self.colorlist.itemconfig(len(self.colors) - 1, fg=color)
        self.ud_osccols()
        self.figcanvas.draw_idle()
        self.colorwindow.destroy()

    def cancel(self):
        self.colorwindow.destroy()

    def rm_color(self):
        index = self.colorlist.curselection()[0]
        if len(self.colors) > 1:
            self.colorlist.delete(index)
            del self.colors[index]
            self.ud_osccols()
            self.figcanvas.draw_idle()

    def ud_osccols(self):
        cycler = cycle(self.colors)
        for key in [l for l in self.lines if l != 'data']:
            self.lines[key].set_color(next(cycler))

    def ud_lw_sc(self, lw):
        self.lw = float(lw)
        self.lw_ent.delete(0, tk.END)
        self.lw_ent.insert(0, f'{self.lw:.3f}')
        self.ud_linewidths()
        self.figcanvas.draw_idle()

    def ud_lw_ent(self):
        value = self.lw_ent.get()
        valid_lw = self.check_lw(value)

        if valid_lw:
            self.lw = float(value)
            self.lw_scale.set(self.lw)
            self.ud_linewidths()

        else:
            self.lw_ent.delete(0, tk.END)
            self.lw_ent.insert(0, self.lw)

    @staticmethod
    def check_lw(value):
        try:
            lw = float(value)
            return 0 <= lw <= 5
        except:
            return False

    def ud_linewidths(self):
        for key in [l for l in self.lines if l != 'data']:
            self.lines[key].set_lw(self.lw)
        self.figcanvas.draw_idle()

    def ud_linestyles(self, ls):
        self.ls = ls
        for key in [l for l in self.lines if l != 'data']:
            self.lines[key].set_ls(self.ls)
        self.figcanvas.draw_idle()

    def close(self):
        self.parent.destroy()



class ConfigLabels(tk.Toplevel):
    def __init__(self, parent, labels, ax, figcanvas):
        tk.Toplevel.__init__(self, parent)

        self.labels = labels
        self.ax = ax
        self.figcanvas = figcanvas

        self.config(bg='white')

        options = ['All Labels']
        keys = self.labels.keys()
        for key in keys:
            opt = key.replace('osc', 'Label ')
            options.append(opt)


        self.leftframe = tk.Frame(self, bg='white')
        self.rightframe = tk.Frame(self, bg='white')
        self.customframe = tk.Frame(self.rightframe, bg='white')
        self.buttonframe = tk.Frame(self.rightframe, bg='white')

        self.leftframe.grid(row=0, column=0, sticky='nsew')
        self.rightframe.grid(row=0, column=1, sticky='nsew')
        self.customframe.grid(row=0, column=0, sticky='new')
        self.buttonframe.grid(row=1, column=0, sticky='ew')

        self.columnconfigure(1, weight=1)
        self.leftframe.rowconfigure(0, weight=1)
        self.rightframe.columnconfigure(0, weight=1)
        self.rightframe.rowconfigure(0, weight=1)
        self.buttonframe.columnconfigure(0, weight=1)

        self.list = tk.Listbox(self.leftframe)
        for opt in options:
            self.list.insert(tk.END, opt)

        self.list.bind('<Double-Button-1>', (lambda entry: self.showlabeleditor()))

        self.list.activate(0)
        self.list.selection_set(0)

        self.list.grid(row=0, column=0, sticky='new')

        self.close_but = tk.Button(self.buttonframe, text='Close', width=8,
                                   bg='#ff9894', highlightbackground='black',
                                   command=self.close)
        self.close_but.grid(row=0, column=0, sticky='w')

        # self.showlabeleditor()


    def showlabeleditor(self):

        index = self.list.curselection()
        name = self.list.get(index)

        for widget in self.customframe.winfo_children():
            widget.destroy()

        if 'Label ' in name:
            number = int(name.strip('Label '))
            self.labeleditframe = LabelEdit(self.customframe, self.labels,
                                            number, self.ax, self.figcanvas)

        elif name == 'All Labels':
            self.labeleditframe = LabelMultiEdit(self.customframe, self.lines,
                                                 self.figcanvas)

    def close(self):
        self.destroy()


class LabelEdit(tk.Frame):
    def __init__(self, parent, labels, number, ax, figcanvas):
        tk.Frame.__init__(self, parent)

        self.grid(row=0, column=0, sticky='nsew')

        self.labels = labels
        self.ax = ax
        self.figcanvas = figcanvas

        self.label = self.labels[f'osc{number}']

        init_color = mcolors.to_hex(self.label.get_color())
        self.txt = self.label.get_text()
        self.x, self.y = self.label.get_position() # label position
        self.size = self.label.get_fontsize()

        self.txtframe = tk.Frame(self, bg='white') # text string
        self.colorframe = ColorPicker(self, init_color) # color title, colors scales and entries
        self.sizeframe = tk.Frame(self, bg='white') # text size scale and entry
        self.posframe = tk.Frame(self, bg='white') # position scale and entry

        self.txtframe.grid(row=0, column=0, sticky='ew')
        self.colorframe.grid(row=1, column=0, sticky='ew')
        self.sizeframe.grid(row=2, column=0, sticky='ew')
        self.posframe.grid(row=3, column=0, sticky='ew')

        # tweak scale and entry commands to dynamically change plot
        scales = [self.colorframe.r_scale, self.colorframe.g_scale,
                  self.colorframe.b_scale]
        commands = [lambda r: self.colorframe.ud_r_sc(r, dynamic=True,
                                     object=self.label, figcanvas=self.figcanvas),
                    lambda g: self.colorframe.ud_g_sc(g, dynamic=True,
                                     object=self.label, figcanvas=self.figcanvas),
                    lambda b: self.colorframe.ud_b_sc(b, dynamic=True,
                                     object=self.label, figcanvas=self.figcanvas)]

        for scale, command in zip(scales, commands):
            scale['command'] = command


        # --- LABEL TEXT ----------------------------------------------
        self.txt_ttl = tk.Label(self.txtframe, text='Label Text', bg='white',
                                font=(MAINFONT, 12))

        self.txt_ent = tk.Entry(self.txtframe, bg='white', width=10,
                                highlightthickness=0)
        self.txt_ent.insert(0, self.txt)

        self.txt_ent.bind('<Return>', (lambda event: self.ud_txt_ent()))

        self.txt_ttl.grid(row=0, column=0, sticky='w')
        self.txt_ent.grid(row=0, column=1, sticky='w')

        # --- LABEL SIZE ----------------------------------------------
        self.size_ttl = tk.Label(self.sizeframe, text='Label Size', bg='white',
                                font=(MAINFONT, 12))

        self.size_scale = tk.Scale(self.sizeframe, from_=1, to=48,
                                   orient=tk.HORIZONTAL, showvalue=0,
                                   bg='white', sliderlength=15, bd=0,
                                   troughcolor=f'#808080', length=500,
                                   highlightthickness=1,
                                   command=self.ud_size_sc,
                                   resolution=0.001)
        self.size_scale.set(self.size)

        self.size_ent = tk.Entry(self.sizeframe, bg='white',
                                 text=f'{self.size:.3f}',
                                 width=6, highlightthickness=0)

        self.size_ent.bind('<Return>', (lambda event: self.ud_size_ent()))

        self.size_ttl.grid(row=1, column=0, sticky='w', columnspan=2)
        self.size_scale.grid(row=2, column=0, sticky='ew')
        self.size_ent.grid(row=2, column=1, sticky='w')

        # --- LABEL POSITION ----------------------------------------------
        self.postopframe = tk.Frame(self.posframe, bg='white')
        self.posbotframe = tk.Frame(self.posframe, bg='white')

        self.postopframe.grid(row=0, column=0, sticky='ew')
        self.posbotframe.grid(row=1, column=0, sticky='ew')

        self.pos_ttl = tk.Label(self.postopframe, text='Label Position',
                                bg='white', font=(MAINFONT, 12))

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.x_ttl = tk.Label(self.posbotframe, text='x', bg='white')

        self.x_scale = tk.Scale(self.posbotframe, from_=xlim[0], to=xlim[1],
                                orient=tk.HORIZONTAL, showvalue=0,
                                bg='white', sliderlength=15, bd=0,
                                troughcolor=f'#808080', length=500,
                                highlightthickness=1,
                                command=self.ud_x_sc,
                                resolution=0.0001)
        self.x_scale.set(self.x)

        self.x_ent = tk.Entry(self.posbotframe, bg='white',
                              width=10, highlightthickness=0)
        self.x_ent.insert(0, f'{self.x:.4f}')

        self.x_ent.bind('<Return>', (lambda event: self.ud_x_ent()))

        self.y_ttl = tk.Label(self.posbotframe, text='y', bg='white')

        self.y_scale = tk.Scale(self.posbotframe, from_=ylim[0], to=ylim[1],
                                orient=tk.HORIZONTAL, showvalue=0,
                                bg='white', sliderlength=15, bd=0,
                                troughcolor=f'#808080', length=500,
                                highlightthickness=1,
                                command=self.ud_y_sc,
                                resolution=1)
        self.y_scale.set(self.y)

        self.y_ent = tk.Entry(self.posbotframe, bg='white',
                              width=10, highlightthickness=0)
        self.y_ent.insert(0, f'{self.y:.4E}')

        self.y_ent.bind('<Return>', (lambda event: self.ud_y_ent()))

        self.pos_ttl.grid(row=0, column=0, sticky='w')
        self.x_ttl.grid(row=0, column=0, sticky='w')
        self.x_scale.grid(row=0, column=1, sticky='ew')
        self.x_ent.grid(row=0, column=2, sticky='w')
        self.y_ttl.grid(row=1, column=0, sticky='w')
        self.y_scale.grid(row=1, column=1, sticky='ew')
        self.y_ent.grid(row=1, column=2, sticky='w')


    def ud_txt_ent(self):
        txt = self.txt_ent.get()
        try:
            self.label.set_text(txt)
            self.txt = txt
            self.figcanvas.draw_idle()
        except:
            self.txt_ent.delete(0, tk.END)
            self.txt_ent.insert(0, self.txt)

    def ud_size_sc(self, size):
        self.size = float(size)
        self.size_ent.delete(0, tk.END)
        self.size_ent.insert(0, f'{self.size:.3f}')
        self.label.set_size(self.size)
        self.figcanvas.draw_idle()

    def ud_size_ent(self):
        value = self.size_ent.get()
        valid_size = self.check_size(value)

        if valid_size:
            self.size = float(value)
            self.size_scale.set(self.size)

        else:
            self.size_ent.delete(0, tk.END)
            self.size_ent.insert(0, self.size)

    @staticmethod
    def check_size(value):
        try:
            size = float(value)
            return 0 <= size <= 48
        except:
            return False

    def ud_x_sc(self, x):
        self.x = float(x)
        self.x_ent.delete(0, tk.END)
        self.x_ent.insert(0, f'{self.x:.4f}')
        self.label.set_x(self.x)
        self.figcanvas.draw_idle()

    def ud_x_ent(self):
        value = self.x_ent.get()
        valid_x = self.check_x(value)

        if valid_x:
            self.x = float(value)
            self.x_scale.set(self.x)

        else:
            self.x_ent.delete(0, tk.END)
            self.x_ent.insert(0, self.x)

    def check_x(self, value):
        try:
            size = float(value)
            return self.xlim[1] <= size <= self.xlim[0]
        except:
            return False

    def ud_y_sc(self, y):
        self.y = float(y)
        self.y_ent.delete(0, tk.END)
        self.y_ent.insert(0, f'{self.y:.4E}')
        self.label.set_y(self.y)
        self.figcanvas.draw_idle()

    def ud_y_ent(self):
        value = self.y_ent.get()
        valid_y = self.check_y(value)

        if valid_y:
            self.y = float(value)
            self.y_scale.set(self.y)

        else:
            self.y_ent.delete(0, tk.END)
            self.y_ent.insert(0, self.y)

    def check_y(self, value):
        try:
            size = float(value)
            return self.ylim[0] <= size <= self.ylim[1]
        except:
            return False

    def close(self):
        self.parent.destroy()

    # # construct save files
    # descrip = res_app.descrip
    # file = res_app.file
    # dir = res_app.dir
    #
    # txt = res_app.txt
    # pdf = res_app.pdf
    # pickle = res_app.pickle
    #
    #
    # if txt == '1':
    #     info.write_result(descrip=descrip, fname=file, dir=dir,
    #                            force_overwrite=True)
    # if pdf == '1':
    #     info.write_result(descrip=descrip, fname=file, dir=dir,
    #                            force_overwrite=True, format='pdf')
    # if pickle == '1':
    #     info.pickle_save(fname=file, dir=dir, force_overwrite=True)

if __name__ == '__main__':

    app = NMREsPyApp()
    app.mainloop()
