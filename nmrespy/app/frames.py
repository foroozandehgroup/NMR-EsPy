import webbrowser

from .. import *
from .config import *
from .custom_widgets import *


class LogoFrame(MyFrame):
    """Contains the NMR-EsPy and MF groups logos"""

    def __init__(self, parent, logos='both', scale=0.6):

        super().__init__(parent)

        column = 0
        padx = 0

        if logos in ['both', 'nmrespy']:
            # add NMR-EsPy logo
            self.nmrespy_img = get_PhotoImage(
                IMAGESPATH / 'nmrespy_full.png', scale / 2.3
            )
            self.nmrespy_logo = MyLabel(
                self, image=self.nmrespy_img, cursor='hand1'
            )
            # provide link to NMR-EsPy docs
            self.nmrespy_logo.bind(
                '<Button-1>', lambda e: webbrowser.open_new(GITHUBLINK)
            )
            self.nmrespy_logo.grid(row=0, column=column)

            column += 1
            padx = (40, 0)

        if logos in ['both', 'mfgroup']:
            # add MF group logo
            self.mfgroup_img = get_PhotoImage(MFLOGOPATH, scale)
            self.mfgroup_logo = MyLabel(
                self, image=self.mfgroup_img, cursor='hand1'
            )
            # provide link to MF group website
            self.mfgroup_logo.bind(
                '<Button-1>', lambda e: webbrowser.open_new(MFGROUPLINK)
            )
            self.mfgroup_logo.grid(row=0, column=column, padx=padx)


class WarnFrame(MyToplevel):
    """A window in case the user does something silly."""

    def __init__(self, parent, msg):
        super().__init__(parent)
        self.title('NMR-EsPy - Error')

        # warning image
        self.img = get_PhotoImage((IMAGESPATH / 'warning.png'), 0.08)
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


class DataType(MyToplevel):
    """GUI for asking user whether they want to analyse the raw FID or
    pdata

    Parameters
    ----------
    parent : tk.Tk

    paths : dict
        Dictionary with two entries:

        * `'pdata'` - Path to processed data
        * `'fid'`` - Path to raw FID file
    """

    def __init__(self, ctrl, paths):
        self.ctrl = ctrl
        self.paths = paths
        super().__init__(self.ctrl)

        # --- Configure frames -------------------------------------------
        # Frame for the NMR-EsPy logo
        self.logo_frame = LogoFrame(self, logos='nmrespy', scale=0.8)
        # Frame containing path labels and checkboxes
        self.main_frame = MyFrame(self)
        # Frame containing confirm/cancel buttons
        self.button_frame = MyFrame(self)
        # Arrange frames
        self.logo_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.main_frame.grid(row=0, column=1, padx=(0, 10), pady=(10, 0))
        self.button_frame.grid(
            row=1, column=1, padx=(0, 10), pady=(0, 10), sticky='e',
        )

        # --- Frame heading ----------------------------------------------
        msg = MyLabel(
            self.main_frame, text='Which data would you like to analyse?',
            font=(MAINFONT, '12', 'bold'),
        )
        msg.grid(
            column=0, row=0, columnspan=2, padx=10, pady=(10,0)
        )

        # --- Processd data checkbutton and labels -----------------------
        pdata_label = MyLabel(
            self.main_frame, text='Processed Data'
        )
        pdata_label.grid(
            column=0, row=1, padx=(10,0), pady=(10,0), sticky='w'
        )

        pdatapath = MyLabel(
            self.main_frame, text=f"{str(self.paths['pdata'])}/1r",
            font=('Courier', 11),
        )
        pdatapath.grid(column=0, row=2, padx=(10, 0), sticky='w')

        # `self.pdata` can be 0 and 1. Specifies whether to use pdata or not
        # This is directly dependent on `self.fid`. When one is `1`, the
        # other is `0`.
        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = MyCheckbutton(
            self.main_frame, variable=self.pdata, command=self.click_pdata,
        )
        self.pdata_box.grid(
            column=1, row=1, rowspan=2, padx=(10, 0), sticky='nsw'
        )

        # --- FID checkbutton and labels ---------------------------------
        fid_label = MyLabel(self.main_frame, text='Raw FID')
        fid_label.grid(
            column=0, row=3, padx=(10, 0), pady=(10, 0), sticky='w',
        )

        fidpath = MyLabel(
            self.main_frame, text=f"{str(self.paths['fid'])}/fid",
            font=('Courier', 11),
        )
        fidpath.grid(column=0, row=4, padx=(10, 0), sticky='w')

        # Initially have set to `0`, i.e. pdata is set to the default.
        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = MyCheckbutton(
            self.main_frame, variable=self.fid, command=self.click_fid,
        )
        self.fid_box.grid(
            column=1, row=3, rowspan=2, padx=(10,0), sticky='nsw'
        )

        # --- Confirm and Cancel buttons ---------------------------------
        self.confirmbutton = MyButton(
            self.button_frame, text='Confirm', command=self.confirm,
            bg=BUTTONGREEN,
        )
        self.confirmbutton.grid(
            column=1, row=0, padx=(5, 0), pady=(10, 0), sticky='e',
        )

        self.cancelbutton = MyButton(
            self.button_frame, text='Cancel', command=self.ctrl.destroy,
            bg=BUTTONRED,
        )
        self.cancelbutton.grid(column=0, row=0, pady=(10, 0), sticky='e')
        self.ctrl.wait_window(self)

    def click_fid(self):
        fidval = self.fid.get()
        if fidval == 1:
            self.pdata.set(0)
        elif fidval == 0:
            self.pdata.set(1)

    def click_pdata(self):
        pdataval = self.pdata.get()
        if pdataval == 1:
            self.fid.set(0)
        elif pdataval == 0:
            self.fid.set(1)

    def confirm(self):
        if self.fid.get() == 1:
            self.path = self.paths['fid']
        else:
            self.path = self.paths['pdata']
        self.destroy()


class RootButtonFrame(MyFrame):

    def __init__(self, parent, ctrl):
        super().__init__(parent)
        self.parent = parent
        self.ctrl = ctrl

        self.cancel_button = MyButton(
            self, text='Cancel', bg=BUTTONRED, command=self.ctrl.destroy
        )
        self.cancel_button.grid(
            row=1, column=0, padx=(10,0), pady=(10,0), sticky='e',
        )

        self.help_button = MyButton(
            self, text='Help', bg=BUTTONORANGE,
            command=lambda: webbrowser.open_new(DOCSLINK)
        )
        self.help_button.grid(
            row=1, column=1, padx=(10,0), pady=(10,0), sticky='e'
        )

        # Command varies - will need to be defined from the class that
        # inherits from this
        # For example, see SetupButtonFrame
        self.save_button = MyButton(self, text='Run', bg=BUTTONGREEN)
        self.save_button.grid(
            row=1, column=2, padx=10, pady=(10,0), sticky='e',
        )

        contact_info_1 = MyLabel(
            self, text='For queries/feedback, contact',
        )
        contact_info_1.grid(
            row=2, column=0, columnspan=3, padx=10, pady=(10,0), sticky='w',
        )

        contact_info_2 = MyLabel(
            self, text='simon.hulse@chem.ox.ac.uk', font='Courier', fg='blue',
            cursor='hand1',
        )
        contact_info_2.bind(
            '<Button-1>', lambda e: webbrowser.open_new(MAILTOLINK),
        )

        contact_info_2.grid(
            row=3, column=0, columnspan=3, padx=10, pady=(0,10), sticky='w',
        )


class AdvancedSettings(MyToplevel):
    """Frame inside SetupApp notebook - for customising details about the
    optimisation routine"""

    def __init__(self, parent, ctrl):
        super().__init__(parent)
        self.ctrl = ctrl

        self.main_frame = MyFrame(self)
        self.main_frame.grid(row=1, column=0)

        adsettings_title = MyLabel(
            self.main_frame, text='Advanced Settings',
            font=(MAINFONT, 14, 'bold'),
        )
        adsettings_title.grid(
            row=0, column=0, columnspan=2, padx=(10,0), pady=(10,0), sticky='w',
        )

        filter_title = MyLabel(
            self.main_frame, text='Signal Filter Options', bold=True,
        )
        filter_title.grid(
            row=1, column=0, columnspan=2, padx=(10,0), pady=(10,0), sticky='w',
        )

        cut_label = MyLabel(self.main_frame, text='Cut signal:')
        cut_label.grid(row=2, column=0, padx=(10,0), pady=(10,0), sticky='w')

        self.cut_checkbutton = MyCheckbutton(
            self.main_frame, variable=self.ctrl.cut['var'],
            command=self.ud_cut,
        )
        self.cut_checkbutton.grid(
            row=2, column=1, padx=10, pady=(10,0), sticky='w',
        )

        ratio_label = MyLabel(
            self.main_frame, text='Cut width/filter width ratio:',
        )
        ratio_label.grid(row=3, column=0, padx=(10,0), pady=(10,0), sticky='w')

        self.ratio_entry = MyEntry(
            self.main_frame,
            return_command=self.ud_cut_ratio,
            return_args=(),
            textvariable=self.ctrl.cut_ratio['var'],
        )
        self.ratio_entry.grid(row=3, column=1, padx=10, pady=(10,0), sticky='w')

        mpm_title = MyLabel(self.main_frame, text='Matrix Pencil', bold=True)
        mpm_title.grid(
            row=4, column=0, columnspan=2, padx=(10,0), pady=(10,0), sticky='w',
        )

        datapoint_label = MyLabel(self.main_frame, text='Datapoints to consider*:')
        datapoint_label.grid(
            row=5, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.mpm_points_entry = MyEntry(
            self.main_frame,
            return_command=self.ud_points,
            return_args=('mpm',),
            textvariable=self.ctrl.trim['mpm']['var'],
        )
        self.mpm_points_entry.grid(
            row=5, column=1, padx=10, pady=(10,0), sticky='w',
        )

        oscillator_label = MyLabel(
            self.main_frame, text='Number of oscillators:',
        )
        oscillator_label.grid(
            row=6, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.oscillator_entry = MyEntry(
            self.main_frame,
            return_command=self.ud_oscillators, return_args=(),
            state='disabled',
            textvariable=self.ctrl.m['var'],
        )
        self.oscillator_entry.grid(
            row=6, column=1, padx=10, pady=(10,0), sticky='w',
        )

        use_mdl_label = MyLabel(self.main_frame, text='Use MDL:')
        use_mdl_label.grid(
            row=7, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.mdl_checkbutton = MyCheckbutton(
            self.main_frame, variable=self.ctrl.mdl['var'],
            command=self.ud_mdl_button,
        )
        self.mdl_checkbutton.grid(
            row=7, column=1, padx=10, pady=(10,0), sticky='w',
        )

        nlp_title = MyLabel(
            self.main_frame, text='Nonlinear Programming', bold=True,
        )
        nlp_title.grid(
            row=8, column=0, columnspan=2, padx=10, pady=(10,0), sticky='w',
        )

        datapoint_label = MyLabel(
            self.main_frame, text='Datapoints to consider*:',
        )
        datapoint_label.grid(
            row=9, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.nlp_points_entry = MyEntry(
            self.main_frame,
            return_command=self.ud_points,
            return_args=('nlp',),
            textvariable=self.ctrl.trim['nlp']['var'],
        )
        self.nlp_points_entry.grid(
            row=9, column=1, padx=10, pady=(10,0), sticky='w',
        )

        nlp_method_label = MyLabel(self.main_frame, text='NLP algorithm:')
        nlp_method_label.grid(
            row=10, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        options = ('Trust Region', 'L-BFGS')

        # I was getting funny behaviour when I tried to make a class
        # that inherited from tk.OptionMenu
        # had to customise manually after generating an instance
        self.algorithm_menu = tk.OptionMenu(
            self.main_frame, self.ctrl.method['var'], *options
        )

        self.algorithm_menu['bg'] = 'white'
        self.algorithm_menu['width'] = 9
        self.algorithm_menu['highlightbackground'] = 'black'
        self.algorithm_menu['highlightthickness'] = 1
        self.algorithm_menu['menu']['bg'] = 'white'
        self.algorithm_menu['menu']['activebackground'] = ACTIVETABCOLOR
        self.algorithm_menu['menu']['activeforeground'] = 'white'

        # change the max. number of iterations after changing NLP
        # algorithm
        self.ctrl.method['var'].trace('w', self.ud_nlp_algorithm)
        self.algorithm_menu.grid(
            row=10, column=1, padx=10, pady=(10,0), sticky='w',
        )

        max_iterations_label = MyLabel(
            self.main_frame, text='Maximum iterations:',
        )
        max_iterations_label.grid(
            row=11, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.max_iterations_entry = MyEntry(
            self.main_frame,
            return_command=self.ud_max_iterations,
            return_args=(),
            textvariable=self.ctrl.maxit['var'],
        )
        self.max_iterations_entry.grid(
            row=11, column=1, padx=10, pady=(10,0), sticky='w',
        )

        phase_variance_label = MyLabel(
            self.main_frame, text='Optimise phase variance:',
        )
        phase_variance_label.grid(
            row=12, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.phase_var_checkbutton = MyCheckbutton(
            self.main_frame, variable=self.ctrl.phase_variance['var'],
            command=self.ud_phase_variance,
        )
        self.phase_var_checkbutton.grid(
            row=12, column=1, padx=10, pady=(10,0), sticky='w',
        )

        # amplitude/frequency thresholds
        amp_thold_label = MyLabel(self.main_frame, text='Amplitude threshold:')
        amp_thold_label.grid(
            row=13, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.amp_thold_frame = MyFrame(self.main_frame)
        self.amp_thold_frame.columnconfigure(1, weight=1)
        self.amp_thold_frame.grid(row=13, column=1, sticky='ew')

        self.amp_thold_entry = MyEntry(
            self.amp_thold_frame, state='disabled',
            return_command=self.ud_amp_thold, return_args=(),
            textvariable=self.ctrl.amp_thold['var'],
        )
        self.amp_thold_entry.grid(
            row=0, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.amp_thold_checkbutton = MyCheckbutton(
            self.amp_thold_frame, variable=self.ctrl.use_amp_thold['var'],
            command=self.ud_amp_thold_button,
        )
        self.amp_thold_checkbutton.grid(
            row=0, column=1, pady=(10,0), padx=10, sticky='w',
        )

        ## May reincorporate later on ========================================
        # freq_thold_label = MyLabel(self.main_frame, text='Frequency threshold:')
        # freq_thold_label.grid(
        #     row=14, column=0, padx=(10,0), pady=(10,0), sticky='w',
        # )
        #
        # self.freq_thold_frame = MyFrame(self.main_frame)
        # self.freq_thold_frame.columnconfigure(1, weight=1)
        # self.freq_thold_frame.grid(row=14, column=1, sticky='ew')
        #
        # self.freq_thold_entry = MyEntry(
        #     self.freq_thold_frame,
        #     textvariable=self.ctrl.adsettings['freq_thold']['var']['ppm'],
        # )
        # self.freq_thold_entry.grid(
        #     row=0, column=0, padx=(10,0), pady=(10,0), sticky='w',
        # )
        #
        # self.freq_thold_checkbutton = MyCheckbutton(
        #     self.freq_thold_frame,
        #     variable=self.ctrl.adsettings['use_freq_thold']['var'],
        #     command=lambda name='freq': self.ud_thold_button(name),
        # )
        # self.freq_thold_checkbutton.grid(
        #     row=0, column=1, pady=(10,0), padx=10, sticky='w',
        # )
        # # ==================================================================

        self.button_frame = MyFrame(self)
        self.button_frame.columnconfigure(2, weight=1)
        self.button_frame.grid(row=2, column=0, sticky='ew')

        max_label = MyLabel(
            self.button_frame, text='*Max points to consider:',
            font=(MAINFONT, 9),
        )
        max_label.grid(
            row=0, column=0, padx=(10,0), pady=(20,10), sticky='w',
        )

        self.max_points_label_mpm = MyLabel(
            self.button_frame, bold=True, font=(MAINFONT, 9, 'bold'),
            textvariable=self.ctrl.max_points['var'],
        )
        self.max_points_label_mpm.grid(
            row=0, column=1, padx=(3,0), pady=(20,10), sticky='w',
        )

        self.close_button = MyButton(
            self.button_frame, text='Close', command=self.close
        )
        self.close_button.grid(row=0, column=2, padx=10, pady=(20,10), sticky='e')



    def _check_int(self, value):

        try:
            int_value = int(value)
            float_value = float(value)

            if int_value == float_value:
                return True
            else:
                return False

        except:
            return False


    def _check_float(self, value):

        try:
            float_value = float(value)
            return True
        except:
            return False

    def _reset(self, obj):
        obj['var'].set(set(obj['value']))


    def ud_cut(self):

        if int(self.ctrl.cut['var'].get()):
            self.ctrl.cut['value'] = True
            self.ratio_entry['state'] = 'normal'
        else:
            self.ctrl.cut['value'] = False
            self.ratio_entry['state'] = 'disabled'

        self.ctrl.ud_max_points()


    def ud_cut_ratio(self):

        # check the value can be interpreted as a float
        str_value = self.ctrl.cut_ratio['var'].get()
        if self._check_float(str_value) and float(str_value) >= 1.0:
            float_value = float(str_value)
            self.ctrl.cut_ratio['value'] = float_value
            self.ctrl.ud_max_points()

        else:
            self.ctrl.cut_ratio['var'].set(str(self.ctrl.cut_ratio['value']))


    def ud_points(self, name):

        str_value = self.ctrl.trim[name]['var'].get()
        if self._check_int(str_value) and \
        0 < int(str_value) <= self.ctrl.max_points['value']:
            int_value = int(str_value)
            self.ctrl.trim[name]['value'] = int_value
            self.ctrl.trim[name]['var'].set(str(int_value))

        else:
            self.ctrl.trim[name]['var'].set(
                str(self.ctrl.trim[name]['value'])
            )


    def ud_mdl_button(self):
        """For when the user clicks on the checkbutton relating to use the
        MDL"""

        if int(self.ctrl.mdl['var'].get()):
            self.ctrl.mdl['value'] = True
            self.oscillator_entry['state'] = 'disabled'
            self.ctrl.m['value'] = 0
            self.ctrl.m['var'].set('')
        else:
            self.ctrl.mdl['value'] = False
            self.oscillator_entry['state'] = 'normal'


    def ud_oscillators(self):

        str_value = self.ctrl.m['var'].get()
        if self._check_int(str_value) and int(str_value) > 0:
            int_value = int(str_value)
            self.ctrl.m['value'] = int_value
            self.ctrl.m['var'].set(str(int_value))

        else:
            if self.ctrl.m['value'] == 0:
                self.ctrl.m['var'].set('')
            else:
                self.ctrl.m['var'].set(str(self.ctrl.m['value']))


    def ud_max_iterations(self):

        str_value = self.ctrl.maxit['var'].get()
        if self._check_int(str_value) and int(str_value) > 0:
            int_value = int(str_value)
            self.ctrl.maxit['value'] = int_value
            self.ctrl.maxit['var'].set(str(int_value))

        else:
            self.ctrl.maxit['var'].set(str(self.ctrl.maxit['value']))


    def ud_nlp_algorithm(self, *args):
        """Called when user changes the NLP algorithm. Sets the default
        number of maximum iterations for the given method"""

        method = self.ctrl.method['var'].get()
        if method == 'Trust Region':
            self.ctrl.method['value'] = 'trust_region'
            self.ctrl.maxit['value'] = 100
            self.ctrl.maxit['var'].set('100')

        elif method == 'L-BFGS':
            self.ctrl.method['value'] = 'lbfgs'
            self.ctrl.maxit['value'] = 500
            self.ctrl.maxit['var'].set('500')


    def ud_phase_variance(self):

        if int(self.ctrl.phase_variance['var'].get()):
            self.ctrl.phase_variance['value'] = True
        else:
            self.ctrl.phase_variance['value'] = False


    def ud_amp_thold_button(self):
        """For when the user clicks on the checkbutton relating to whether
        or not to impose an amplitude threshold"""

        if int(self.ctrl.use_amp_thold['var'].get()):
            self.ctrl.use_amp_thold['value'] = True
            self.amp_thold_entry['state'] = 'normal'
        else:
            self.ctrl.use_amp_thold['value'] = False
            self.amp_thold_entry['state'] = 'disabled'


    def ud_amp_thold(self):

        str_value = self.ctrl.amp_thold['var'].get()

        if self._check_float(str_value):
            float_value = float(str_value)

            if 0.0 <= float_value < 1.0:
                self.ctrl.amp_thold['value'] = float_value
                self.ctrl.ud_max_points()
                return

        self._reset(self.ctrl.amp_thold)


    def close(self):
        valid = self.ctrl.check_invalid_entries(self.main_frame)
        if valid:
            self.destroy()
