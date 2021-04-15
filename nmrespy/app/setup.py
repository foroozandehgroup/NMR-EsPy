from datetime import datetime
from matplotlib import figure, patches
from matplotlib.backends import backend_tkagg
import re
import threading
import time

from .._misc import latex_nucleus
from .config import *
from .custom_widgets import *
from .frames import *


class SetUp(MyToplevel):
    def __init__(self, parent):
        self.estimator = parent.estimator
        # Shorthand for unit conversion
        self.conv = lambda value, conversion: \
            self.estimator._converter.convert([value], conversion)[0]

        # --- SETUP WINDOW -----------------------------------------------
        # Spectral data
        self.spec = sig.ft(self.estimator.get_data())
        # Number of spectrum points
        self.n = self.spec.size
        # Transmitter frequency
        self.sfo = self.estimator.get_sfo()[0]
        # Nucleus identity
        self.nuc = self.estimator.get_nucleus()[0]

        # Sweep width, offset, and chemical shifts, in both ppm and Hz
        self.sw, self.off, self.shifts = {}, {}, {}
        for unit in ['hz', 'ppm']:
            self.sw[unit] = self.estimator.get_sw(unit=unit)[0]
            self.off[unit] = self.estimator.get_offset(unit=unit)[0]
            self.shifts[unit] = self.estimator.get_shifts(unit=unit)[0]

        # Units that are active in the app.
        # For frequencies, default is ppm (Hz also possible)
        # For degrees, default is radians (degrees also possible)
        self.active_units = {'freq': 'ppm', 'angle': 'rad'}

        # --- Region selection parameters --------------------------------
        # Bounds for filtration and noise regions
        self.bounds = AutoVivification()
        # Initial values for region of interest and noise region
        init_bounds = [int(np.floor(x * self.n / 16)) for x in (7, 9, 1, 2)]

        for name, init in zip(['lb', 'rb', 'lnb', 'rnb'], init_bounds):
            # Save bounds to array index, ppm and hz units
            for unit in ['idx', 'hz', 'ppm']:
                if unit == 'idx':
                    value, var_str = init, str(init)
                else:
                    value = self.conv(init, f'idx->{unit}')
                    var_str = f"{value:.4f}"

                self.bounds[name][unit] = value_var_dict(value, var_str)

        # Number of points the selected region is composed of
        self.region_size = self.bounds['rb']['idx']['value'] - \
                           self.bounds['lb']['idx']['value']

        # --- Phase correction parameters --------------------------------
        self.pivot = {}

        # Initialise pivot at center of spectrum
        pivot = int(self.n // 2)
        for unit in ['idx', 'hz', 'ppm']:
            # Set pivot in units of array index, Hz, and ppm
            if unit == 'idx':
                value, var_str = pivot, str(pivot)
            else:
                value = self.conv(pivot, f'idx->{unit}')
                var_str = f"{value:.4f}"

            self.pivot[unit] = value_var_dict(value, var_str)

        # Zero- and first-order correction parameters
        self.phases = AutoVivification()

        # Initialise correction parameters as zero in both radians and
        # degrees.
        for name in ['p0', 'p1']:
            for unit in ['rad', 'deg']:
                self.phases[name][unit] = value_var_dict(0., f"{0.:.4f}")

        # --- Various advanced settings ----------------------------------
        # Specifies whether or not to cut the filtered spectral data
        self.cut = value_var_dict(True, 1)
        # Specifies the the ratio between cut signal size and filter
        # bandwidth
        self.cut_ratio = value_var_dict(3, '3')

        # Largest number of points permitted
        max_points = self.cut_size()
        self.max_points = value_var_dict(max_points, str(max_points))

        # Number of points to be used for MPM and NLP.
        # By default, number of points will not exceed:
        # 4096 (MPM)
        # 8192 (NLP)
        self.trim = {}
        if max_points <= 4096:
            for name in ['mpm', 'nlp']:
                self.trim[name] = value_var_dict(max_points, str(max_points))
        elif max_points <= 8192:
            self.trim['mpm'] = value_var_dict(4096, '4096')
            self.trim['nlp'] = value_var_dict(max_points, str(max_points))
        else:
            self.trim['mpm'] = value_var_dict(4096, '4096')
            self.trim['nlp'] = value_var_dict(8192, '8192')

        # Number of oscillators
        # Initialise as 0 (use MDL)
        self.m = value_var_dict(0, '')
        # Specifies whether or not to use the MDL to estimate M
        self.mdl = value_var_dict(True, 1)
        # Idenitity of the NLP algorithm to use
        self.method = value_var_dict('trust_region', 'Trust Region')
        # Maximum iterations of NLP algorithm
        self.maxit = value_var_dict(200, '200')
        # Whether or not to include phase variance in NLP cost func
        self.phase_variance = value_var_dict(True, 1)
        # Whether or not to purge negligible oscillators
        self.use_amp_thold = value_var_dict(False, 0)
        # Amplitude threshold for purging negligible oscillators
        self.amp_thold = value_var_dict(0.001, '0.001')

        # --- Figure for setting up estimation ---------------------------
        self.setupfig = {}
        # Figure
        self.setupfig['fig'] = figure.Figure(figsize=(6,3.5), dpi=170)
        # Axis
        self.setupfig['ax'] = \
            self.setupfig['fig'].add_axes([0.05, 0.12, 0.9, 0.83])
        # Plot spectrum
        # Generates a matplotlib.Line.Line2D object
        self.setupfig['plot'] = self.setupfig['ax'].plot(
            self.shifts['ppm'], np.real(self.spec), color='k', lw=0.6,
        )[0] # <- unpack from list

        # Set x-limits as edges of spectral window
        xlim = (self.shifts['ppm'][0], self.shifts['ppm'][-1])
        self.setupfig['ax'].set_xlim(xlim)

        # Prevent user panning/zooming beyond spectral window
        # See Restrictor class for more info ↑
        Restrictor(self.setupfig['ax'], x=lambda x: x <= xlim[0])
        Restrictor(self.setupfig['ax'], x=lambda x: x >= xlim[1])

        # Get current y-limit. Will reset y-limits to this value after the
        # very tall `noise_region` and `filter_region` rectangles have been
        # added to the plot
        ylim = self.setupfig['ax'].get_ylim()

        # Highlight the spectral region of intererst
        # matplotlib.patches.Rectangle's first 3 args:
        # (left, bottom), width, height
        bottom_left = (self.bounds['lb']['ppm']['value'], -20 * ylim[1])
        width = self.bounds['rb']['ppm']['value'] - \
                self.bounds['lb']['ppm']['value']
        height = 40 * ylim[1]

        self.setupfig['region'] = patches.Rectangle(
            bottom_left, width, height, facecolor=REGIONCOLOR,
        )
        self.setupfig['ax'].add_patch(self.setupfig['region'])

        # Highlight the noise region (height same as before)
        bottom_left = (self.bounds['lnb']['ppm']['value'], -20 * ylim[1])
        width = self.bounds['rnb']['ppm']['value'] - \
                self.bounds['lnb']['ppm']['value']

        self.setupfig['noise_region'] = patches.Rectangle(
            bottom_left, width, height, facecolor=NOISEREGIONCOLOR
        )
        self.setupfig['ax'].add_patch(self.setupfig['noise_region'])

        # Plot pivot line
        x = 2 * [self.pivot['ppm']['value']]
        y = [-20 * ylim[1], 20 * ylim[1]]

        # `alpha` is set to 0 to make the pivot invisible initially
        self.setupfig['pivot'] = self.setupfig['ax'].plot(
            x, y, color=PIVOTCOLOR, alpha=0, lw=0.8,
        )[0]

        # Reset y limit
        self.setupfig['ax'].set_ylim(ylim)

        # Aesthetic tweaks to the plot
        self.setupfig['fig'].patch.set_facecolor(BGCOLOR)
        self.setupfig['ax'].set_facecolor(PLOTCOLOR)
        self.setupfig['ax'].tick_params(axis='x', which='major', labelsize=6)
        self.setupfig['ax'].locator_params(axis='x', nbins=10)
        self.setupfig['ax'].set_yticks([])

        for direction in ('top', 'bottom', 'left', 'right'):
            self.setupfig['ax'].spines[direction].set_color('k')

        # xlabel of the form $^{<mass>}$<elem> (ppm)
        self.setupfig['ax'].set_xlabel(
            f"{latex_nucleus(self.nuc)} (ppm)", fontsize=8,
        )

        # --- Construction of the setup GUI ------------------------------
        super().__init__(parent)
        self.resizable(True, True)
        # First column is adjusted upon change of window size
        # (plot_frame, toolbar_frame, tab_frame, logo_frame)
        self.columnconfigure(0, weight=1)
        # First row is adjusted upon change of window size
        # (plot_frame)
        self.rowconfigure(0, weight=1)

        # Frame containing the plot
        self.plot_frame = MyFrame(self)
        # Make `plot_frame` resizable
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        # Canvas for figure
        self.canvas = backend_tkagg.FigureCanvasTkAgg(
            self.setupfig['fig'], master=self.plot_frame,
        )
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        # Frame containing the navigation toolbar and advanced settings
        # button
        self.toolbar_frame = MyFrame(self)
        self.toolbar = MyNavigationToolbar(
            self.canvas, parent=self.toolbar_frame,
        )
        self.toolbar.grid(row=0, column=0, sticky='w', padx=(10,0), pady=(0,5))

        # Frame containing notebook widget: for region selection and
        # phase correction
        self.tab_frame = MyFrame(self)
        # Make notebook adjustable horizontally
        self.tab_frame.columnconfigure(0, weight=1)
        self.notebook = MyNotebook(self.tab_frame)
        self.notebook.grid(row=0, column=0, sticky='ew', padx=10, pady=(0,10))

        # Whenever tab is clicked, change plot so that either region
        # selection patches are visible, or phase pivot, depending on
        # tab selected
        self.notebook.bind(
            '<<NotebookTabChanged>>', lambda event: self.switch_tab(),
        )

        # Frame with scale widgets for region selection
        self.region_frame = MyFrame(self.notebook, bg=NOTEBOOKCOLOR)
        self.notebook.add(
            self.region_frame, text='Region Selection', sticky='nsew',
        )

        # Make scales expandable
        self.region_frame.columnconfigure(1, weight=1)
        # Scale labels
        self.region_labels = {}
        # Scales
        self.region_scales = {}
        # Entry widgets related to scales
        self.region_entries = {}

        for row, name in enumerate(('lb', 'rb', 'lnb', 'rnb')):
            # Construct text strings for scale titles
            text = ''
            for letter in name:
                if letter == 'l':
                    text += 'left '
                elif letter == 'r':
                    text += 'right '
                elif letter == 'n':
                    text += 'noise '
                else:
                    text += 'bound'

            # Scale titles
            self.region_labels[name] = title = MyLabel(
                self.region_frame, text=text, bg=NOTEBOOKCOLOR,
            )

            # Troughcolor of scale
            troughcolor = REGIONCOLOR if row < 2 else NOISEREGIONCOLOR

            self.region_scales[name] = scale = MyScale(
                self.region_frame,
                from_ = 0,
                to = self.n - 1,
                troughcolor = troughcolor,
                bg = NOTEBOOKCOLOR,
                command=(lambda idx, name=name:
                    self.update_region_scale(idx, name)),
            )
            scale.set(self.bounds[name]['idx']['value'])

            self.region_entries[name] = entry = MyEntry(
                self.region_frame,
                return_command=self.update_region_entry,
                return_args=(name,),
                textvariable=self.bounds[name]['ppm']['var'],
            )

            pady = (10, 0) if row != 3 else 10
            title.grid(row=row, column=0, padx=(10,0), pady=pady, sticky='w')
            scale.grid(row=row, column=1, padx=(10,0), pady=pady, sticky='ew')
            entry.grid(row=row, column=2, padx=10, pady=pady, sticky='w')

        # Frame with scale widgets for region selection
        self.phase_frame = MyFrame(self.notebook, bg=NOTEBOOKCOLOR)
        self.notebook.add(
            self.phase_frame, text='Phase Correction', sticky='nsew',
        )

        # Make scales expandable
        self.phase_frame.columnconfigure(1, weight=1)

        self.phase_titles = {}
        self.phase_scales = {}
        self.phase_entries = {}

        for row, (name, title) in enumerate(zip(('pivot', 'p0', 'p1'), ('pivot', 'φ₀', 'φ₁'))):
            # Scale titles

            self.phase_titles[name] = title = MyLabel(
                self.phase_frame, text=title, bg=NOTEBOOKCOLOR
            )

            # Pivot scale
            if name == 'pivot':
                troughcolor = PIVOTCOLOR
                from_ = 0
                to = self.n - 1
                resolution = 1

            # p0 and p1 scales
            else:
                troughcolor = 'white'
                # TODO
                # PHASE SCALE WIDGET ISSUE
                # Would like this to be π or 10π, however tkinter seems to
                # convert the scale range to an int, and adjusts `to` to
                # accommodate this.
                from_ = -3.5 if name == 'p0' else -35.
                to = 3.5 if name == 'p0' else 35.
                resolution = 0.001

            self.phase_scales[name] = scale = MyScale(
                    self.phase_frame,
                    troughcolor = troughcolor,
                    from_ = from_,
                    to = to,
                    resolution = resolution,
                    bg = NOTEBOOKCOLOR,
                    command=(lambda value, name=name:
                        self.update_phase_scale(value, name)),
            )


            if name == 'pivot':
                scale.set(self.pivot['idx']['value'])
                var = self.pivot['ppm']['var']
            else:
                scale.set(0.0)
                var = self.phases[name]['rad']['var']

            self.phase_entries[name] = entry = MyEntry(
                self.phase_frame,
                return_command=self.update_phase_entry,
                return_args=(name,),
                textvariable=var,
            )

            pady = (10,0) if row != 2 else 10

            title.grid(row=row, column=0, padx=(10,0), pady=pady, sticky='w')
            scale.grid(row=row, column=1, padx=(10,0), pady=pady, sticky='ew')
            entry.grid(row=row, column=2, padx=10, pady=pady, sticky='w')

        # Frame with NMR-EsPy an MF group logos
        self.logo_frame = LogoFrame(master=self, scale=0.72)

        # Frame with cancel/help/run/advanced settings buttons
        self.button_frame = SetupButtonFrame(master=self)

        # --- Configure frame placements ---------------------------------
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.tab_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        self.logo_frame.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        self.button_frame.grid(row=3, column=1, sticky='s')

    # --- Region selection methods ---------------------------------------
    def update_region_entry(self, name):
        """Update the GUI after the user presses <Enter> whilst in a
        `self.region_entries` widget"""

        unit = self.active_units['freq']

        try:
            # Get StringVar of widget of interest, and try to convert
            # to a numerical value
            value = self.bounds[name][unit]['var'].get()
            if unit == 'idx':
                idx = int(value)
            else:
                idx = self.conv(float(value), f'{unit}->idx')

            # Check whether `idx` is valid
            if self.check_valid_index(idx, name):
                self.bounds[name]['idx']['value'] = idx
                self.region_scales[name].set(idx)
                self.update_bound(name, idx)
            else:
                raise

        except:
            # Conversion to numerical value failed, revert to previous value
            self.reset_bound(name, unit)


    def update_region_scale(self, idx, name):
        """Update the GUI after the user changes the slider on a scale
        widget"""

        idx = int(idx)

        if not self.check_valid_index(idx, name):
            if name[0] == 'l':
                idx = self.bounds[f'r{name[1:]}']['idx']['value'] - 1
            else:
                idx = self.bounds[f'l{name[1:]}']['idx']['value'] + 1

        self.region_scales[name].set(idx)
        self.update_bound(name, idx)


    def reset_bound(self, name, unit):
        value = self.bounds[name][unit]['value']
        if unit == 'idx':
            self.bounds[name][unit]['var'].set(str(value))
        else:
            self.bounds[name][unit]['var'].set(f'{value:.4f}')


    def check_valid_index(self, idx, name):
        """Given an update index, and the identity of the bound to change,
        determine whether the index is valid."""

        # Determine if we are considering a left or right bound
        if name[0] == 'l':
            left = idx
            right = self.bounds[f'r{name[1:]}']['idx']['value']
        else:
            left = self.bounds[f'l{name[1:]}']['idx']['value']
            right = idx

        if left < right and 0 <= idx <= self.n - 1:
            # All good
            return True
        # All bad
        return False

    def update_bound(self, name, idx):
        """Given a dictionary key, and new value in units of array indices,
        update region bound variables"""

        # Set bound's index value and the corresponding StringVar
        self.bounds[name]['idx']['value'] = idx
        self.bounds[name]['idx']['var'].set(str(idx))

        # convert from index to Hz and ppm and update
        for unit in ['hz', 'ppm']:
            self.bounds[name][unit]['value'] = self.conv(idx, f'idx->{unit}')
            self.bounds[name][unit]['var'].set(
                f"{self.bounds[name][unit]['value']:.4f}"
            )

        # Determine if dealing with a bound relating to the region of interest,
        # or a bound relating to the noise region
        reg = 'region' if name in ['lb', 'rb'] else 'noise_region'

        if reg == 'region':
            # Update region of interest size
            self.region_size = self.bounds['rb']['idx']['value'] - \
                               self.bounds['lb']['idx']['value']
            # Update maximum number of points possible
            self.ud_max_points()

        # Get active frequency unit ('hz' or 'ppm')
        active_unit = self.active_units['freq']

        # Determine left and right bounds of the region changed
        left = self.bounds[f'l{name[1:]}'][active_unit]['value']
        right = self.bounds[f'r{name[1:]}'][active_unit]['value']
        # update region rectangle
        self.setupfig[reg].set_x(left)
        self.setupfig[reg].set_width(right - left)
        # update plot
        self.update_plot()


    def switch_tab(self):
        """Adjusts the appearence of the plot when a new tab is selected.
        Hides/reveals region rectangles and pivot plot as required. Toggles
        alpha between 1 and 0"""

        # detemine the active tab
        tab = self.notebook.index(self.notebook.select())
        # set alpha values for region rectangles and pivot plot
        regions, pivot = (1, 0) if tab == 0 else (0, 1)

        self.setupfig['region'].set_alpha(regions)
        self.setupfig['noise_region'].set_alpha(regions)
        self.setupfig['pivot'].set_alpha(pivot)

        # draw updated figure
        self.update_plot()

    # --- Phase correction methods ---------------------------------------
    def update_phase_scale(self, value, name):
        """Update the GUI after the user changes the slider on a phase scale
        widget"""

        if name == 'pivot':
            self.update_pivot(int(value))
        else:
            self.update_p0_p1(float(value), name)


    def update_phase_entry(self, name):
        """Update the GUI after the user changes and entry widget"""

        if name == 'pivot':
            unit = self.active_units['freq']
            value = self.pivot[unit]['var'].get()

            try:
                if unit == 'idx':
                    idx = int(value)
                else:
                    idx = self.conv(float(value), f'{unit}->idx')

                if 0 <= idx <= self.n - 1:
                    self.phase_scales['pivot'].set(idx)
                    self.update_pivot(idx)
                else:
                    raise

            except:
                self.reset_pivot(unit)

        else:
            unit = self.active_units['angle']
            value = self.phases[name][unit]['var'].get()

            try:
                # Regex that matches numerical values
                regex = r"^[+-]?(\d+(\.\d*)?|\.\d+)$"
                if re.fullmatch(regex, value):
                    x = float(value)#
                # Check if numerical value with pi appended
                # i.e. 1pi, 0.25pi, .25pi, 1.pi etc.
                elif re.fullmatch(regex.replace(r'$', r'pi$'), value):
                    x = float(value[:-2]) * np.pi
                else:
                    raise

                # Convert to radians
                if unit == 'rad':
                    pass
                else:
                    x = float(value) * 180 / np.pi

                # Check that zero-order correction if between -π and π
                # Actually between -3.5 and 3.5
                # TODO: see PHASE SCALE WIDGET ISSUE
                if -3.5 <= x <= 3.5 and name == 'p0':
                    self.phase_scales['p0'].set(x)
                # Check that first-order correction if between -10π and 10π
                elif -35 <= x <= 35 and name == 'p1':
                    self.phase_scales['p1'].set(x)
                else:
                    raise

                self.update_phase(x, name)

            except:
                self.reset_phase(name, unit)


    def reset_pivot(self, unit):
        if unit == 'idx':
            self.pivot['idx']['var'].set(str(self.pivot['idx']['value']))
        else:
            self.pivot[unit]['var'].set(
                f"{self.pivot[unit]['value']:.4f}"
            )


    def reset_phase(self, name, unit):
        self.phases[name][unit]['var'].set(
            f"{self.phases[name][unit]['value']:.4f}"
        )

    def update_pivot(self, idx):
        """Given a value in units of array indices, update the phase
        correction pivot"""

        # Update pivot index value and StringVar
        self.pivot['idx']['value'] = idx
        self.pivot['idx']['var'].set(str(idx))

        # Also update in units of Hz and ppm
        for unit in ['ppm', 'hz']:
            self.pivot[unit]['value'] = self.conv(idx, f'idx->{unit}')
            self.pivot[unit]['var'].set(f"{self.pivot[unit]['value']:.4f}")

        # Redefine x data of pivot plot
        x = 2 * [self.pivot[self.active_units['freq']]['value']]
        self.setupfig['pivot'].set_xdata(x)
        # Perform phase correction based on the updated pivot
        self.phase_correct()


    def update_p0_p1(self, rad, name):
        """Given a name and value in units of radians, update a phase
        correction parameter"""

        self.phases[name]['rad']['value'] = rad
        self.phases[name]['rad']['var'].set(f'{rad:.4f}')
        self.phases[name]['deg']['value'] = rad * 180 / np.pi
        self.phases[name]['deg']['var'].set(f'{rad * 180 / np.pi:.4f}')

        # Perform phase correction based on the updated phase
        self.phase_correct()


    def phase_correct(self):
        """Perform phase correction of spectral data, and update setup
        Toplevel plot"""

        pivot = self.pivot['idx']['value']
        p0 = self.phases['p0']['rad']['value']
        p1 = self.phases['p1']['rad']['value']
        n = self.n

        # Perform phase correction of spectrum
        corrector = np.exp(1j * (p0 + p1 * np.arange(-pivot, -pivot+n, 1) / n))
        self.setupfig['plot'].set_ydata(np.real(self.spec * corrector))

        # Update plot
        self.update_plot()


    def ud_max_points(self):
        """Update the maximum number of points StringVar"""

        if self.cut['value'] == True:
            # Check range is suitable. If not, set it within the spectrum.
            # Divide by two as halving signal in `estimator.frequency_filter`
            cut_size = self.cut_size()

            # Determine respective low and high bounds
            lb = self.bounds['lb']['idx']['value']
            rb = self.bounds['rb']['idx']['value']
            low = int((lb + rb) // 2) - int(np.ceil(cut_size / 2))
            high = low + cut_size

            if low < 0:
                low = 0
            if high > self.n - 1:
                high = self.n - 1

            self.max_points['value'] = high - low

        else:
            self.max_points['value'] = self.n // 2

        self.max_points['var'].set(str(self.max_points['value']))

        # If current trim params are larger than the new max points
        # or they are smaller than the default number of points for
        # MPM and NLP, update
        for name, default in zip(('mpm', 'nlp'), (4096, 8192)):
            if self.trim[name]['value'] > self.max_points['value'] \
            or self.max_points['value'] <= default:
                self.trim[name]['value'] = self.max_points['value']
                self.trim[name]['var'].set(str(self.max_points['value']))


    def update_plot(self):
        """Redraw the plot figure canvas"""
        self.canvas.draw_idle()


    def cut_size(self):
        """Get the theoretical number of points if cutting of the filtered
        spectrum is applied"""
        return int((self.cut_ratio['value'] * self.region_size) // 2)


    def run(self):
        """Set up the estimation routine"""

        # Check whether any entry widgets have not been verified
        if not check_invalid_entries(self):
            msg = "Some parameters have not been validated."
            WarnFrame(self, msg=msg)
            return

        # Get rid of setup window
        # this allows __init__ to proceed beyond wait_window
        self.withdraw()

        # TODO: animation window
        # self.master.waiting_window.deiconify()

        # Phase correction variables
        pivot = self.pivot['idx']['value']
        p0 = self.phases['p0']['rad']['value']
        p1 = self.phases['p1']['rad']['value']
        p0 = p0 - p1 * (pivot / self.n)
        p0, p1 = [p0], [p1]

        region = [
            [self.bounds['lb']['idx']['value'],
             self.bounds['rb']['idx']['value']]
        ]
        noise_region = [
            [self.bounds['lnb']['idx']['value'],
             self.bounds['rnb']['idx']['value']]
        ]

        # Whether of not to cut the frequency-filtered signal
        cut = self.cut['value']
        if cut:
            cut_ratio = float(self.cut_ratio['value'])
        else:
            cut_ratio = None

        # Number of points to consider in MPM and NLP
        trim_mpm = [self.trim['mpm']['value']]
        trim_nlp = [self.trim['nlp']['value']]

        # Get number of oscillators for initial guess (or determine whether
        # to use MDL)
        m = self.m['value']

        # Optimisation method
        method = self.method['value']
        # Maximum number of iterations for optimisation
        maxit = self.maxit['value']
        # Whether or not to use phase variance
        phase_variance = self.phase_variance['value']
        # Whether or not to set an amplitude threshold on estimation result
        if self.use_amp_thold['value']:
            amp_thold = self.amp_thold['value']
        else:
            amp_thold = None

        # --- Run through the estimation ---------------------------------
        # Phase data
        self.estimator.phase_data(p0=p0, p1=p1)
        # Filter signal
        self.estimator.frequency_filter(
            region, noise_region, cut=cut, cut_ratio=cut_ratio,
            region_unit='idx',
        )
        # Initial guess using the matrix pencil
        self.estimator.matrix_pencil(trim=trim_mpm, M=m)
        # Generate final estimation result using nonlinear programming
        self.estimator.nonlinear_programming(
            trim=trim_nlp, max_iterations=maxit, method=method,
            phase_variance=phase_variance, amp_thold=amp_thold,
        )

        # Pickle result class to the temporary directory
        dt = datetime.now()
        timestamp = f"{dt.year}{dt.month}{dt.day}{dt.hour}{dt.minute}{dt.second}"
        tmppath = str(TMPPATH / timestamp)
        self.estimator.to_pickle(path=tmppath, force_overwrite=True)

        # TODO: animation window
        # self.master.waiting_window.destroy()
        self.destroy()



class SetupButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, master):
        super().__init__(master)
        self.green_button['text'] = 'Run'
        self.green_button['command'] = self.master.run

        self.adsettings_button = MyButton(
            parent=self, text='Advanced Settings', width=16,
            command=self.advanced_settings
        )
        self.adsettings_button.grid(
            row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=(0,5),
        )

    def advanced_settings(self):
        AdvancedSettings(master=self.master)
