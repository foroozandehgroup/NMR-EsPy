import ast
import copy
import pathlib
import re
import subprocess
from tkinter import filedialog

from matplotlib.backends import backend_tkagg

from .config import *
from .custom_widgets import *
from .frames import *


class Result(MyToplevel):
    def __init__(self, master):

        super().__init__(master)

        self.resizable(True, True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.title('NMR-EsPy - Result')

        self.protocol("WM_DELETE_WINDOW", self.click_cross)

        self.create_plot()

        # Canvas for figure
        self.canvas = backend_tkagg.FigureCanvasTkAgg(
            self.result_plot.fig, master=self,
        )
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            column=0, row=0, columnspan=2, padx=10, pady=10, sticky='nsew',
        )

        # Frame containing the navigation toolbar and advanced settings
        # button
        self.toolbar_frame = MyFrame(self)
        self.toolbar_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar = MyNavigationToolbar(
            self.canvas, parent=self.toolbar_frame,
        )
        self.toolbar.grid(
            row=0, column=0, sticky='w', padx=(10, 0), pady=(0, 5),
        )

        # Frame with NMR-EsPy an MF group logos
        self.logo_frame = LogoFrame(self, scale=0.72)
        self.logo_frame.grid(row=2, column=0, sticky='w', padx=10, pady=10)

        # Frame with cancel/help/run/advanced settings buttons
        self.button_frame = ResultButtonFrame(self)
        self.button_frame.grid(
            row=1, column=1, rowspan=2, sticky='se', padx=10, pady=10,
        )

    def click_cross(self):
        self.button_frame.cancel()

    def create_plot(self):
        # Generate figure of result
        self.result_plot = self.master.estimator.plot_result()
        self.result_plot.fig.set_size_inches(6, 3.5)
        self.result_plot.fig.set_dpi(170)

        # Prevent panning outside the selected region
        xlim = self.result_plot.ax.get_xlim()
        Restrictor(self.result_plot.ax, x=lambda x: x <= xlim[0])
        Restrictor(self.result_plot.ax, x=lambda x: x >= xlim[1])

    def update_plot(self):
        plot = self.master.estimator.plot_result()
        ax, lines, labels = plot.ax, plot.lines, plot.labels

        # Update y-axis limits
        self.result_plot.ax.set_ylim(ax.get_ylim())
        # Clear current objects
        self.result_plot.ax.lines = []
        self.result_plot.ax.texts = []

        for line in lines.values():
            self.result_plot.ax.plot(
                line.get_xdata(), line.get_ydata(), color=line.get_color(),
                lw=line.get_lw(),
            )

        for label in labels.values():
            self.result_plot.ax.text(
                *label.get_position(), label.get_text(),
                fontsize=label.get_fontsize(),
            )

        self.canvas.draw()


class ResultButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, master):

        cancel_msg = (
            "Are you sure you want to close NMR-EsPy? The estimation result "
            "will be unrecoverable."
        )

        super().__init__(master, cancel_msg=cancel_msg)
        self.green_button['command'] = self.save_options
        self.green_button['text'] = 'Save'

        self.edit_parameter_button = MyButton(
            self, text='Edit Parameter Estimate', command=self.edit_parameters,
        )
        self.edit_parameter_button.grid(
            row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=(10, 0),
        )

        self.help_button['command'] = lambda: webbrowser.open_new(
            f"{DOCSLINK}gui/usage/result.html"
        )

    def edit_parameters(self):
        EditParametersFrame(self.master)

    def save_options(self):
        SaveFrame(self.master)


class EditParametersFrame(MyToplevel):
    """TopLevel for editing an estimation result, and re-running the
    optimiser"""

    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)

        self.title("NMR-EsPy - Edit Parameters")

        self.protocol("WM_DELETE_WINDOW", self.click_cross)

        # Reference to NMREsPyApp class: gives access to estimator
        self.ctrl = self.master.master

        # Prevent access to other windows
        self.grab_set()

        # Store initial parameter array incase the user wants to restore
        # after making changes.
        self.previous = copy.deepcopy(self.ctrl.estimator.get_result())

        # --- Parameter table --------------------------------------------
        titles = [
            'Amplitude',
            'Phase (rad)',
            'Frequency (ppm)',
            'Damping (s⁻¹)',
        ]
        # Parameters in ppm - to be added to the table
        contents = self.ctrl.estimator.get_result(freq_unit='ppm')
        # Region of interest, in ppm. Used to ensure any newly frequency
        # satisfies the selected region of interest
        region = self.ctrl.estimator.get_filter_info().get_region(unit='ppm')

        self.table = MyTable(
            self, contents=contents, titles=titles, region=region,
        )

        self.table.grid(column=0, row=0, padx=10, pady=(10, 0))

        # --- Buttons ----------------------------------------------------
        self.button_frame = MyFrame(self)
        self.button_frame.grid(row=1, column=0, pady=10, padx=10, sticky='ew')
        self.button_frame.grid_columnconfigure(0, weight=1)

        # Construct two rows to place buttons
        # Row 1: Edit parameter estimate:
        # Add, remove, merge, split, manual edit
        self.row1 = MyFrame(self.button_frame)
        self.row1.grid(row=0, column=0, sticky='ew')
        # Row 2: Re-run optimiser, undo changes, close window
        self.row2 = MyFrame(self.button_frame)
        self.row2.grid(row=1, column=0, sticky='e')

        for i in range(4):
            self.row1.columnconfigure(i, weight=1)
            if i < 3:
                self.row2.columnconfigure(i, weight=1)

        # Add oscillator(s)
        self.add_button = MyButton(self.row1, text='Add', command=self.add)
        self.add_button.grid(row=0, column=0, sticky='ew')

        # Remove oscillator(s)
        self.remove_button = MyButton(
            self.row1, text='Remove', state='disabled', command=self.remove,
        )
        self.remove_button.grid(row=0, column=1, sticky='ew', padx=(10, 0))

        # Merge oscillators
        self.merge_button = MyButton(
            self.row1, text='Merge', state='disabled', command=self.merge,
        )
        self.merge_button.grid(row=0, column=2, sticky='ew', padx=(10, 0))

        # Split oscillator
        self.split_button = MyButton(
            self.row1, text='Split', state='disabled', command=self.split,
        )
        self.split_button.grid(row=0, column=3, sticky='ew', padx=(10, 0))

        self.table.selected_number.trace('w', self.configure_button_states)

        # Reset
        self.reset_button = MyButton(
            self.row2, text='Reset', command=self.reset, state='disabled',
            width=10,
        )
        self.reset_button.grid(
            row=0, column=0, sticky='e', pady=(10, 0), padx=(10, 0),
        )

        # Button to close if no changes have been made, and re-run optimiser
        # if changes have been made
        self.close_rerun_button = MyButton(
            self.row2, text='Close', command=self.close_or_rerun,
            width=10,
        )
        self.close_rerun_button.grid(
            row=0, column=1, sticky='e', pady=(10, 0), padx=(10, 0),
        )

    def configure_button_states(self, *args):
        # Number of curently selected oscillators
        number = self.table.selected_number.get()

        if number == 0:
            self.add_button['state'] = 'normal'
            self.remove_button['state'] = 'disabled'
            self.merge_button['state'] = 'disabled'
            self.split_button['state'] = 'disabled'

        elif number == 1:
            self.add_button['state'] = 'disabled'
            self.remove_button['state'] = 'normal'
            self.merge_button['state'] = 'disabled'
            self.split_button['state'] = 'normal'

        else:
            self.add_button['state'] = 'disabled'
            self.remove_button['state'] = 'normal'
            self.merge_button['state'] = 'normal'
            self.split_button['state'] = 'disabled'

    def add(self):
        """Loads a window for adding new oscillators"""
        add_frame = AddFrame(self)
        self.wait_window(add_frame)

    def remove(self):
        """Removes selected oscillators from the result"""
        rm_indices = self.table.selected_rows
        self.ctrl.estimator.remove_oscillators(rm_indices)
        self.changed_result()

    def merge(self):
        """Removes selected oscillators from the result"""
        merge_indices = self.table.selected_rows
        self.ctrl.estimator.merge_oscillators(merge_indices)
        self.changed_result()

    def split(self):
        split_frame = SplitFrame(self, *self.table.selected_rows)
        self.wait_window(split_frame)

    def close_or_rerun(self):
        if self.close_rerun_button['text'] == 'Close':
            self.destroy()
        else:
            self.rerun()

    def changed_result(self):
        """Regenerate parameter table and plot to reflect change in
        estimator.result"""
        # If result doess not match original, provide option to re-run
        # the optimiser. If it does match, provide the option to simply
        # close the window.
        if not np.array_equal(self.ctrl.estimator.result, self.previous):
            self.close_rerun_button['text'] = 'Re-run Optimiser'
            self.reset_button['state'] = 'normal'
        else:
            self.close_rerun_button['text'] = 'Close'
            self.reset_button['state'] = 'disabled'

        # Un-select all table rows, and reconstuct the table.
        self.table.selected_rows = []
        self.table.selected_number.set(0)
        self.table.reconstruct(
            contents=self.ctrl.estimator.get_result(freq_unit='ppm'),
            top=0,
        )

        # Update the plot
        self.master.update_plot()

    def rerun(self):
        # Get info from previous call to nonlinear_programming
        for line in reversed(self.ctrl.estimator._log.split('\n')):
            if "nonlinear_programming" in line:
                # Get the kwargs (these are stored within curly braces)
                nlp_args = ast.literal_eval(re.findall(r"\{.*\}", line)[0])

        # Kill result window
        self.master.destroy()
        # RE-run NLP on current set of parameters
        self.ctrl.estimator.nonlinear_programming(
            trim=nlp_args['trim'],
            max_iterations=nlp_args['max_iterations'],
            method=nlp_args['method'],
            phase_variance=nlp_args['phase_variance'],
            amp_thold=nlp_args['amp_thold'],
        )
        # Load new result window.
        self.ctrl.result()

    def reset(self):
        # Reset result back to original
        self.ctrl.estimator.result = self.previous
        # Re-set the table and plot.
        self.changed_result()

    def click_cross(self):
        if self.close_rerun_button['text'] == 'Close':
            self.destroy()
        else:
            msg = (
                "You have manually changed the estimation result. It is "
                "necessary to re-run the optimiser after doing this. Either "
                "click the <Re-run Optimiser> button, or click the <Reset> "
                "button  to undo the changes you have made, and then click "
                "the <Close> button."
            )
            warn_window = WarnWindow(self, msg=msg)
            self.wait_window(warn_window)


class AddFrame(MyToplevel):
    """Toplevel for adding new oscillators to result"""

    def __init__(self, master):

        super().__init__(master)

        self.title('NMR-EsPy - Add oscillators')

        # NMREsPyApp instance
        self.ctrl = self.master.master.master

        # Prevent interacting with other windows
        self.grab_set()

        titles = [
            'Amplitude',
            'Phase (rad)',
            'Frequency (ppm)',
            'Damping (s⁻¹)',
        ]

        # Empty entry boxes to begin with
        contents = [['', '', '', '']]
        region = self.ctrl.estimator.get_filter_info().get_region(unit='ppm')

        self.table = MyTable(
            self, contents=contents, titles=titles, region=region,
        )

        # Turn all widgets red initially to indicate they need filling in
        for entry, value_var in zip(
            self.table.entries[0], self.table.value_vars[0]
        ):
            entry['state'] = 'normal'
            entry.key_press()

        self.table.grid(column=0, row=0, padx=10, pady=(10, 0))

        self.button_frame = MyFrame(self)
        self.button_frame.grid(row=1, column=0, padx=10, pady=10)

        self.add_button = MyButton(
            self.button_frame, text='Add', command=self.add_row,
        )
        self.add_button.grid(row=0, column=0)

        self.cancel_button = MyButton(
            self.button_frame, text='Cancel', command=self.destroy,
            bg=BUTTONRED,
        )
        self.cancel_button.grid(row=0, padx=(5, 0), column=1)

        self.confirm_button = MyButton(
            self.button_frame, text='Confirm', command=self.confirm,
            bg=BUTTONGREEN,
        )
        self.confirm_button.grid(row=0, padx=(5, 0), column=2)

    def add_row(self):
        contents = self.table.get_values()
        contents.append(4 * [''])
        self.table.reconstruct(contents, top=0)
        # Set all entry widgets that are empty to red:
        # Loop over each table row
        for entries in self.table.entries:
            # Loop over each entry in a row
            for entry in entries:
                if entry.get() == '':
                    entry['state'] = 'normal'
                    entry.key_press()

    def confirm(self):
        if self.table.check_red_entry():
            msg = "Some parameters have not been validated."
            warn_window = WarnWindow(self, msg=msg)
            self.wait_window(warn_window)
            return

        # Extract parameters from table
        new_oscillators = np.array(self.table.get_values())
        # Convert from ppm to hz
        new_oscillators[:, 2] = self.ctrl.estimator._converter.convert(
            [new_oscillators[:, 2]], 'ppm->hz',
        )[0]
        # Add new oscillators to result
        result = np.vstack((self.ctrl.estimator.get_result(), new_oscillators))
        # Order by frequency
        self.ctrl.estimator.result = result[np.argsort(result[:, 2])]

        self.master.changed_result()
        self.destroy()


class SplitFrame(MyToplevel):
    """Toplevel for splitting an oscillator into multiple oscillators"""

    def __init__(self, master, index):

        super().__init__(master)

        self.title('NMR-EsPy - Split oscillator')
        # NMREsPyApp instance
        self.ctrl = self.master.master.master
        self.index = index

        # Prevent interacting with other windows
        self.grab_set()

        # Add a frame with some padding from the window edge
        frame = MyFrame(self)
        frame.grid(row=0, column=0, padx=10, pady=10)

        # Window title and widget labels
        MyLabel(
            frame, text=f"Splitting Oscillator {self.index + 1}",
            font=(MAINFONT, 12, 'bold'),
        ).grid(row=0, column=0, columnspan=3, sticky='w')
        MyLabel(frame, text='Number of oscillators:').grid(
            row=1, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(frame, text='Frequency separation:').grid(
            row=2, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(frame, text='Amplitude ratio:').grid(
            row=3, column=0, sticky='w', pady=(10, 0),
        )

        # --- Number of child oscillators --------------------------------
        # Goes from 2 to max. of 10
        self.number_chooser = tk.Spinbox(
            frame, values=tuple(range(2, 11)), width=4,
            command=self.update_number, state='readonly',
            readonlybackground='white',
        )
        self.number_chooser.grid(
            row=1, column=1, columnspan=2, sticky='w', padx=(10, 0),
            pady=(10, 0),
        )

        # --- Separation frequnecy ---------------------------------------
        # Set default separation frequency as 2Hz
        # Convert frequency to ppm
        self.sep_freq = {
            'hz': 2.,
            'ppm': self.ctrl.estimator._converter.convert([2.], 'hz->ppm')[0],
        }
        self.sep_entry = MyEntry(
            frame, width=10, return_command=self.check_freq_sep,
            return_args=(),
        )
        # By default, use Hz as the unit
        self.update_sep_entry(self.sep_freq['hz'])
        self.sep_entry.grid(
            row=2, column=1, sticky='w', padx=(10, 0), pady=(10, 0),
        )

        # Option menu to specify the separation frequency unit to use
        self.sep_unit = tk.StringVar()
        self.sep_unit.set('hz')
        options = ('hz', 'ppm')
        self.sep_unit_box = tk.OptionMenu(
            frame, self.sep_unit, *options, command=self.change_unit
        )
        self.sep_unit_box['bg'] = BGCOLOR
        self.sep_unit_box['width'] = 2
        self.sep_unit_box['highlightbackground'] = 'black'
        self.sep_unit_box['highlightthickness'] = 1
        self.sep_unit_box['menu']['bg'] = BGCOLOR
        self.sep_unit_box['menu']['activebackground'] = ACTIVETABCOLOR
        self.sep_unit_box['menu']['activeforeground'] = 'white'
        self.sep_unit_box.grid(
            row=2, column=2, sticky='w', padx=(10, 0), pady=(10, 0),
        )

        # --- Ratio of amplitudes for children ---------------------------
        # Valid values consist of  a string of colon-separated integers
        # with the number of values matching the number specified by
        # the number chooser.

        # By default, set each child with equal amplitude
        self.amp_ratio = value_var_dict('1:1', '1:1')
        self.ratio_entry = MyEntry(
            frame, width=16, textvariable=self.amp_ratio['var'],
            return_command=self.check_amp_ratio, return_args=(),
        )
        self.ratio_entry.grid(
            column=1, row=3, sticky='w', columnspan=2, padx=(10, 0),
            pady=(10, 0),
        )

        # --- Confirm and Cancel buttons ---------------------------------
        button_frame = MyFrame(frame)
        button_frame.grid(
            row=4, column=0, columnspan=3, sticky='e', pady=(10, 0),
        )

        self.cancel_button = MyButton(
            button_frame, bg=BUTTONRED, command=self.destroy, text='Cancel'
        )
        self.cancel_button.grid(row=0, column=0, sticky='e')

        self.save_button = MyButton(
            button_frame, bg=BUTTONGREEN, command=self.confirm, text='Confirm'
        )
        self.save_button.grid(row=0, column=1, sticky='e', padx=(10, 0))

    def update_number(self):
        """Called when the number choosing spinbox is changed. Updates
        the amplitude ratio to match the new number of children. Each child
        oscillator is set to have the same amplitude"""
        number = int(self.number_chooser.get())
        self.amp_ratio['value'] = ':'.join(number * ['1'])
        self.amp_ratio['var'].set(self.amp_ratio['value'])

    def update_sep_entry(self, value):
        """Update the separation frwquency entry widget"""
        self.sep_entry.delete(0, "end")
        self.sep_entry.insert(0, strip_zeros(f"{value:.5f}"))

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
        from_, to = ('hz', 'ppm') if unit == 'hz' else ('ppm', 'hz')
        self.sep_freq[to] = \
            self.ctrl.estimator._converter.convert(
                [value], f'{from_}->{to}'
        )[0]

    def check_amp_ratio(self):
        """Determine whether a user-given amplitude ratio is valid, and if so,
        update."""
        ratio = self.amp_ratio['var'].get()
        # Regex for string of ints separated by colons
        number = int(self.number_chooser.get())
        regex = r"^\d+(:\d+)+$"
        # Check that:
        # a) the ratio fully matches the regex
        # b) the number of values matches the specified number of child
        # oscillators
        if re.fullmatch(regex, ratio) and len(ratio.split(':')) == number:
            self.amp_ratio['value'] = ratio
        # If conditions are not met, revert back the previous valid value
        else:
            self.amp_ratio['var'].set(self.amp_ratio['value'])

    def confirm(self):
        """Perform the oscillator split and update the plot and parameter
        table"""
        sep_freq = self.sep_freq['hz']
        split_number = int(self.number_chooser.get())
        amp_ratio = self.amp_ratio['var'].get()
        amp_ratio = [int(i) for i in amp_ratio.split(':')]

        self.ctrl.estimator.split_oscillator(
            self.index, separation_frequency=sep_freq,
            split_number=split_number, amp_ratio=amp_ratio,
        )

        self.master.changed_result()
        self.destroy()


class SaveFrame(MyToplevel):
    """Toplevel for choosing how to save estimation result"""

    def __init__(self, master):
        super().__init__(master)
        self.title('NMR-EsPy - Save Result')
        self.ctrl = self.master.master
        self.grab_set()

        # --- Result figure ----------------------------------------------
        self.fig_frame = MyFrame(self)
        self.fig_frame.grid(row=0, column=0, pady=(10, 0), padx=10, sticky='w')

        MyLabel(
            self.fig_frame, text='Result Figure',
            font=('Helvetica', 12, 'bold'),
        ).grid(row=0, column=0, columnspan=2, sticky='w')
        MyLabel(
            self.fig_frame, text='Save Figure:',
        ).grid(
            row=1, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(self.fig_frame, text='Format:').grid(
            row=2, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(self.fig_frame, text='Filename:').grid(
            row=3, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(self.fig_frame, text='dpi:').grid(
            row=4, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(self.fig_frame, text='Size (cm):').grid(
            row=5, column=0, sticky='w', pady=(10, 0),
        )

        self.save_fig = tk.IntVar()
        self.save_fig.set(1)
        self.save_fig_checkbutton = MyCheckbutton(
            self.fig_frame, variable=self.save_fig, command=self.ud_save_fig,
        )
        self.save_fig_checkbutton.grid(
            row=1, column=1, sticky='w', pady=(10, 0),
        )

        self.fig_fmt = tk.StringVar()
        self.fig_fmt.set('pdf')
        self.fig_fmt.trace('w', self.ud_fig_fmt)

        options = ('eps', 'jpg', 'pdf', 'png', 'ps', 'svg')
        self.sep_unit_box = tk.OptionMenu(
            self.fig_frame, self.fig_fmt, *options
        )
        self.sep_unit_box['bg'] = BGCOLOR
        self.sep_unit_box['width'] = 5
        self.sep_unit_box['highlightbackground'] = 'black'
        self.sep_unit_box['highlightthickness'] = 1
        self.sep_unit_box['menu']['bg'] = BGCOLOR
        self.sep_unit_box['menu']['activebackground'] = ACTIVETABCOLOR
        self.sep_unit_box['menu']['activeforeground'] = 'white'
        self.sep_unit_box.grid(
            row=2, column=1, sticky='w', pady=(10, 0),
        )

        self.fig_name_frame = MyFrame(self.fig_frame)
        self.fig_name_frame.grid(row=3, column=1, sticky='w', pady=(10, 0))
        self.fig_name = tk.StringVar()
        self.fig_name.set('nmrespy_figure')
        self.fig_name_entry = MyEntry(
            self.fig_name_frame, textvariable=self.fig_name, width=18,
            return_command=self.ud_file_name, return_args=(self.fig_name,),
        )
        self.fig_name_entry.grid(column=0, row=0)

        self.fig_fmt_label = MyLabel(self.fig_name_frame)
        self.ud_fig_fmt()
        self.fig_fmt_label.grid(column=1, row=0, padx=(2, 0), pady=(5, 0))

        self.fig_dpi = value_var_dict(300, '300')
        self.fig_dpi_entry = MyEntry(
            self.fig_frame, textvariable=self.fig_dpi['var'], width=6,
            return_command=self.ud_fig_dpi, return_args=(),
        )
        self.fig_dpi_entry.grid(row=4, column=1, sticky='w', pady=(10, 0))

        self.fig_width = value_var_dict(15, '15')
        self.fig_height = value_var_dict(10, '10')

        self.fig_size_frame = MyFrame(self.fig_frame)
        self.fig_size_frame.grid(row=5, column=1, sticky='w', pady=(10, 0))

        MyLabel(self.fig_size_frame, text='w:').grid(column=0, row=0)
        MyLabel(self.fig_size_frame, text='h:').grid(column=2, row=0)

        for i, (dim, value) in enumerate(zip(('width', 'height'), (15, 10))):

            self.__dict__[f"fig_{dim}"] = value_var_dict(value, str(value))
            self.__dict__[f"fig_{dim}_entry"] = MyEntry(
                self.fig_size_frame,
                textvariable=self.__dict__[f"fig_{dim}"]["var"],
                return_command=self.ud_fig_size,
                return_args=(dim,)
            )

            padx, column = ((2, 5), 1) if i == 0 else ((2, 0), 3)
            self.__dict__[f"fig_{dim}_entry"].grid(
                column=column, row=0, padx=padx,
            )

        # --- Other result files -----------------------------------------
        self.file_frame = MyFrame(self)
        self.file_frame.grid(row=1, column=0, padx=10, sticky='w')

        MyLabel(
            self.file_frame, text='Result Files',
            font=('Helvetica', 12, 'bold')
        ).grid(row=0, column=0, pady=(20, 0), columnspan=4, sticky='w')

        MyLabel(
            self.file_frame, text='Format:'
        ).grid(row=1, column=0, pady=(10, 0), columnspan=2, sticky='w')
        MyLabel(
            self.file_frame, text='Filename:'
        ).grid(
            row=1, column=2, columnspan=2, padx=(20, 0), pady=(10, 0),
            sticky='w',
        )

        titles = ('Text:', 'PDF:', 'CSV:')
        self.fmts = ('txt', 'pdf', 'csv')
        for i, (title, tag) in enumerate(zip(titles, self.fmts)):

            MyLabel(self.file_frame, text=title).grid(
                row=i + 2, column=0, pady=(10, 0), sticky='w',
            )

            # Dictates whether to save the file format
            self.__dict__[f'save_{tag}'] = save_var = tk.IntVar()
            save_var.set(1)

            self.__dict__[f'{tag}_check'] = check = \
                MyCheckbutton(
                    self.file_frame, variable=save_var,
                    command=lambda tag=tag: self.ud_save_file(tag)
            )
            check.grid(
                row=i + 2, column=1, padx=(2, 0), pady=(10, 0), sticky='w',
            )

            self.__dict__[f'name_{tag}'] = fname_var = tk.StringVar()
            fname_var.set('nmrespy_result')

            self.__dict__[f'{tag}_entry'] = entry = \
                MyEntry(
                    self.file_frame, textvariable=fname_var, width=18,
                    return_command=self.ud_file_name, return_args=(fname_var,),
            )
            entry.grid(
                row=i + 2, column=2, padx=(20, 0), pady=(10, 0), sticky='w',
            )

            self.__dict__[f'{tag}_ext'] = ext = MyLabel(
                self.file_frame, text=f".{tag}",
            )
            ext.grid(row=i + 2, column=3, pady=(15, 0), sticky='w')

        self.pdflatex = self.master.master.pdflatex
        if self.pdflatex is None:
            self.pdflatex = "pdflatex"

        check_latex = subprocess.call(
            f"{self.pdflatex} -v",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        )

        # pdflatex could not be found. Deny selecting PDF option
        if check_latex != 0:
            self.save_pdf.set(0)
            self.pdf_check['state'] = 'disabled'
            self.pdf_entry['state'] = 'disabled'
            self.name_pdf.set('')
            self.pdf_ext['fg'] = '#808080'

        MyLabel(self.file_frame, text='Description:').grid(
            row=5, column=0, columnspan=4, pady=(10, 0), sticky='w',
        )

        self.descr_box = MyText(self.file_frame, width=30, height=3)
        self.descr_box.grid(
            row=6, column=0, columnspan=4, pady=(10, 0), sticky='ew',
        )

        # --- Pickle Estimator -------------------------------------------
        self.pickle_frame = MyFrame(self)
        self.pickle_frame.grid(row=2, column=0, padx=10, sticky='w')

        MyLabel(
            self.pickle_frame, text='Estimator',
            font=('Helvetica', 12, 'bold')
        ).grid(row=0, column=0, pady=(20, 0), columnspan=4, sticky='w')

        MyLabel(
            self.pickle_frame, text='Save Estimator:',
        ).grid(
            row=1, column=0, sticky='w', pady=(10, 0),
        )
        MyLabel(
            self.pickle_frame, text='Filename:',
        ).grid(
            row=2, column=0, sticky='w', pady=(10, 0),
        )

        self.pickle_estimator = tk.IntVar()
        self.pickle_estimator.set(1)
        self.pickle_estimator_checkbutton = MyCheckbutton(
            self.pickle_frame, variable=self.pickle_estimator,
            command=self.ud_pickle_estimator,
        )
        self.pickle_estimator_checkbutton.grid(
            row=1, column=1, sticky='w', pady=(10, 0),
        )

        self.pickle_name_frame = MyFrame(self.pickle_frame)
        self.pickle_name_frame.grid(row=2, column=1, sticky='w', pady=(10, 0))
        self.pickle_name = tk.StringVar()
        self.pickle_name.set('estimator')
        self.pickle_name_entry = MyEntry(
            self.pickle_name_frame, textvariable=self.pickle_name, width=18,
            return_command=self.ud_file_name, return_args=(self.pickle_name,),
        )
        self.pickle_name_entry.grid(column=0, row=0)

        self.pickle_ext_label = MyLabel(
            self.pickle_name_frame, text='.pkl'
        )
        self.pickle_ext_label.grid(column=1, row=0, padx=(2, 0), pady=(5, 0))

        # --- Directory selection ----------------------------------------
        self.dir_frame = MyFrame(self)
        self.dir_frame.grid(row=3, column=0, padx=10, sticky='w')

        MyLabel(
            self.dir_frame, text='Directory',
            font=('Helvetica', 12, 'bold')
        ).grid(row=0, column=0, pady=(20, 0), columnspan=2, sticky='w')

        self.dir_name = tk.StringVar()
        path = pathlib.Path.home()
        self.dir_name = value_var_dict(path, str(path))

        self.dir_entry = MyEntry(
            self.dir_frame, textvariable=self.dir_name['var'], width=30,
            return_command=self.ud_dir, return_args=(),
        )
        self.dir_entry.grid(
            row=1, column=0, pady=(10, 0), sticky='w'
        )

        self.img = get_PhotoImage(IMAGESPATH / 'folder_icon.png', scale=0.02)

        self.dir_button = MyButton(
            self.dir_frame, command=self.browse, image=self.img, width=32,
            bg=BGCOLOR,
        )
        self.dir_button.grid(row=1, column=1, padx=(5, 0), pady=(10, 0))

        # --- Save/cancel buttons ----------------------------------------
        # buttons at the bottom of the frame
        self.button_frame = MyFrame(self)
        self.button_frame.grid(
            row=4, column=0, padx=10, pady=(0, 10), sticky='e'
        )
        # cancel button - returns usere to result toplevel
        self.cancel_button = MyButton(
            self.button_frame, text='Cancel', bg=BUTTONRED,
            command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=0, pady=(10, 0))

        # save button - determines what file types to save and generates them
        self.save_button = MyButton(
            self.button_frame, text='Save', width=8, bg=BUTTONGREEN,
            command=self.save
        )
        self.save_button.grid(
            row=0, column=1, padx=(10, 0), pady=(10, 0),
        )

    # --- Save window methods --------------------------------------------
    # Figure settings
    def ud_save_fig(self):
        state = 'normal' if self.save_fig.get() else 'disabled'
        widgets = [
            self.sep_unit_box,
            self.fig_name_entry,
            self.fig_dpi_entry,
            self.fig_width_entry,
            self.fig_height_entry,
        ]

        for widget in widgets:
            widget['state'] = state

        self.fig_fmt_label['fg'] = \
            '#000000' if state == 'normal' else '#808080'

    def ud_fig_fmt(self, *args):
        self.fig_fmt_label['text'] = f".{self.fig_fmt.get()}"

    def ud_fig_dpi(self, *args):
        try:
            # Try to convert dpi text variable as an int. Both ensures
            # the user-given value can be converted to a numerical value
            # and removes any decimal places, if given. Reset as a string
            # afterwards.
            dpi = int(self.fig_dpi['var'].get())
            if not dpi > 0:
                raise
            self.fig_dpi['var'].set(str(dpi))
            self.fig_dpi['value'] = dpi

        except Exception:
            # Failed to convert to int, reset to previous value
            self.fig_dpi['var'].set(str(self.fig_dpi['value']))

    def ud_fig_size(self, dim):
        try:
            length = float(self.__dict__[f"fig_{dim}"]["var"].get())
            if not length > 0:
                raise

            if check_int(length):
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
    def ud_save_file(self, tag):
        state = 'normal' if self.__dict__[f'save_{tag}'].get() else 'disabled'
        self.__dict__[f'{tag}_entry']['state'] = state
        self.__dict__[f'{tag}_ext']['fg'] = \
            '#000000' if state == 'normal' else '#808080'

    def ud_file_name(self, var):
        name = var.get()
        var.set("".join(x for x in name if x.isalnum() or x in " _-"))

    # Pickle estimator
    def ud_pickle_estimator(self):
        state = 'normal' if self.pickle_estimator.get() else 'disabled'
        self.pickle_name_entry['state'] = state
        self.pickle_ext_label['fg'] = \
            '#000000' if state == 'normal' else '#808080'

    # Save directory
    def browse(self):
        """Directory selection using tkinter's filedialog"""
        name = filedialog.askdirectory(initialdir=self.dir_name['value'])
        # If user clicks close cross, an empty tuple is returned
        if name:
            self.dir_name['value'] = pathlib.Path(name).resolve()
            self.dir_name['var'].set(str(self.dir_name['value']))

    def ud_dir(self):
        path = pathlib.Path(self.dir_name['var'].get()).resolve()
        if path.is_dir():
            self.dir_name['value'] = path
            self.dir_name['var'].set(path)
        else:
            self.dir_name['var'].set(str(self.dir_name['value']))

    def save(self):
        if not check_invalid_entries(self):
            msg = "Some parameters have not been validated."
            warn_window = WarnWindow(self, msg=msg)
            self.wait_window(warn_window)
            return

        # Directory
        dir = self.dir_name['value']

        # Figure
        if self.save_fig.get():
            # Generate figure path
            fig_fmt = self.fig_fmt.get()
            dpi = self.fig_dpi['value']
            fig_name = self.fig_name.get()
            fig_path = dir / f"{fig_name}.{fig_fmt}"

            # Convert size from cm -> inches
            fig_size = (
                self.fig_width['value'] / 2.54,
                self.fig_height['value'] / 2.54,
            )
            fig = self.master.result_plot.fig
            fig.set_size_inches(*fig_size)
            fig.savefig(fig_path, dpi=dpi)

        # Result files
        for fmt in ('txt', 'pdf', 'csv'):
            if self.__dict__[f'save_{fmt}'].get():
                name = self.__dict__[f'name_{fmt}'].get()
                path = str(dir / name)
                description = self.descr_box.get('1.0', 'end-1c')
                if description == '':
                    description = None

                self.ctrl.estimator.write_result(
                    path=path, description=description, fmt=fmt,
                    force_overwrite=True, pdflatex_exe=self.pdflatex,
                )

        if self.pickle_estimator.get():
            name = self.pickle_name.get()
            path = str(dir / name)
            self.ctrl.estimator.to_pickle(
                path=path, force_overwrite=True
            )

        self.master.destroy()

    def customise_figure(self):
        print('TODO')
        # CustomiseFigureFrame(self, self.master)
