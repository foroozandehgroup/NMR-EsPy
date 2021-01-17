#!/usr/bin/python3

# Application for using NMR-EsPy
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# This is currently only applicable to 1D NMR data.
# Much of Thomas Moss's project (Part II) will be based around making a
# complementary 2D App

import ast
from itertools import cycle
import os
import random
import re
import subprocess
import sys
import webbrowser

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift

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

import nmrespy.load as load
import nmrespy._misc as _misc
import nmrespy._plot as _plot
from nmrespy.gui.config import *
from nmrespy.gui.custom_widgets import *

# flag for testing the result frame easily
TEST_RESULT = True


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

    image = Image.open(path).convert('RGBA')
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
    return {'value': value, 'var': var}


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class WarnFrame(MyToplevel):
    """A window in case the user does something silly."""

    def __init__(self, parent, msg):
        MyToplevel.__init__(self, parent)
        self.title('NMR-EsPy - Error')

        # warning image
        self.img = get_PhotoImage(os.path.join(IMAGESDIR, 'warning.png'), 0.08)
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



class Restrictor():
    """Resict naivgation within a defined range (used to prevent
    panning/zooming) outside spectral window on x-axis.
    Inspiration from
    `here <https://stackoverflow.com/questions/48709873/restricting-panning-\
    range-in-matplotlib-plots>`_"""

    def __init__(self, ax, x=lambda x: True, y=lambda x: True):

        self.res = [x,y]
        self.ax =ax
        self.limits = self.get_lim()
        self.ax.callbacks.connect(
            'xlim_changed', lambda evt: self.lims_change(axis=0)
        )
        self.ax.callbacks.connect(
            'ylim_changed', lambda evt: self.lims_change(axis=1)
        )

    def get_lim(self):
        return [self.ax.get_xlim(), self.ax.get_ylim()]

    def set_lim(self, axis, lim):
        if axis==0:
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


class NMREsPyApp(tk.Tk):
    """App for using NMR-EsPy."""

    # This is the controller
    # when you see self.ctrl in other classes in this file, it refers
    # to this class
    def __init__(self):

        super().__init__()
        self.title('NMR-EsPy - Setup Calculation')
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.withdraw()

        if TEST_RESULT:
            self.dtype = 'pdata'
            # self.path = '/home/simon/Documents/DPhil/data/Camphor/1/pdata/1'
            self.path = '/opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1'

        else:
            # open window to ask user for data type (fid or pdata)
            # acquires dtype attribute
            DataType(self)

        # creates set of attributes requred for setup window
        self.generate_initial_variables()

        if TEST_RESULT:
            self.run()

        else:
            # self.setup is a toplevel for setting up the calculation
            self.setup = MyToplevel(self)
            self.setup.resizable(True, True)
            self.setup.columnconfigure(0, weight=1)
            self.setup.rowconfigure(0, weight=1)

            self.setup_frames = {}
            self.setup_frames['plot_frame'] = PlotFrame(
                parent=self.setup, figure=self.setupfig['fig'],
            )
            self.setup_frames['toolbar_frame'] = SetupToolbarFrame(
                parent=self.setup, canvas=self.setup_frames['plot_frame'].canvas,
                ctrl=self,
            )
            self.setup_frames['tab_frame'] = TabFrame(
                parent=self.setup, ctrl=self,
            )
            self.setup_frames['logo_frame'] = LogoFrame(
                parent=self.setup, scale=0.06,
            )
            self.setup_frames['button_frame'] = SetupButtonFrame(
                parent=self.setup, ctrl=self,
            )

            self.setup_frames['plot_frame'].grid(
                row=0, column=0, columnspan=2, sticky='nsew',
            )
            self.setup_frames['toolbar_frame'].grid(
                row=1, column=0, columnspan=2, sticky='ew',
            )
            self.setup_frames['tab_frame'].grid(
                row=2, column=0, columnspan=2, sticky='ew',
            )
            self.setup_frames['logo_frame'].grid(
                row=3, column=0, padx=10, pady=10, sticky='w',
            )
            self.setup_frames['button_frame'].grid(
                row=3, column=1, sticky='s',
            )

            self.wait_window(self.setup)


        self.generate_result_variables()

        self.result = MyToplevel(self)
        self.result.resizable(True, True)
        self.result.columnconfigure(0, weight=1)
        self.result.rowconfigure(0, weight=1)

        self.result_frames = {}
        self.result_frames['plot_frame'] = PlotFrame(
            parent=self.result, figure=self.resultfig['fig'],
        )
        self.result_frames['toolbar_frame'] = RootToolbarFrame(
            parent=self.result, canvas=self.result_frames['plot_frame'].canvas,
            ctrl=self,
        )
        self.result_frames['logo_frame'] = LogoFrame(
            parent=self.result, scale=0.06,
        )
        self.result_frames['button_frame'] = ResultButtonFrame(
            parent=self.result, ctrl=self,
        )

        self.result_frames['plot_frame'].grid(
            row=0, column=0, columnspan=2, sticky='nsew',
        )
        self.result_frames['toolbar_frame'].grid(
            row=1, column=0, columnspan=2, sticky='ew',
        )
        self.result_frames['logo_frame'].grid(
            row=3, column=0, padx=10, pady=10, sticky='w',
        )
        self.result_frames['button_frame'].grid(
            row=3, column=1, sticky='s',
        )


    def generate_initial_variables(self):

        # generate self.info: NMREsPyBruker class instance
        # FID data - transform to frequency domain
        if self.dtype == 'fid':
            self.info = load.import_bruker_fid(self.path, ask_convdta=False)
            self.spectrum = np.flip(fftshift(fft(self.info.get_data())))

        # pdata - combine real and imaginary components
        else:
            self.info = load.import_bruker_pdata(self.path)
            self.spectrum = self.info.get_data(pdata_key='1r') \
                            + 1j * self.info.get_data(pdata_key='1i')

        # constants
        self.n = self.spectrum.shape[0]
        self.sfo = self.info.get_sfo()[0]
        self.nucleus = self.info.get_nucleus()[0]
        self.sw, self.off, self.shifts = {}, {}, {}
        for unit in ['hz', 'ppm']:
            self.sw[unit] = self.info.get_sw(unit=unit)[0]
            self.off[unit] = self.info.get_offset(unit=unit)[0]
            self.shifts[unit] = self.info.get_shifts(unit=unit)[0]

        self.active_units = {'freq': 'ppm', 'angle': 'rad'}

        self.bounds = AutoVivification()

        # bounds for filtration and noise regions
        if TEST_RESULT:
            init_bounds = [
                11843, # left bound
                12142, # right bound
                1762, # left noise bound
                2210, # right noise bound
            ]

        else:
            init_bounds = [
                int(np.floor(7 * self.n / 16)), # left bound
                int(np.floor(9 * self.n / 16)), # right bound
                int(np.floor(1 * self.n / 16)), # left noise bound
                int(np.floor(2 * self.n / 16)), # right noise bound
            ]

        for name, init in zip(['lb', 'rb', 'lnb', 'rnb'], init_bounds):
            for unit in ['idx', 'hz', 'ppm']:
                if unit == 'idx':
                    value, var_str = init, str(init)
                else:
                    value = self.convert(init, f'idx->{unit}')
                    var_str = f"{value:.4f}"

                self.bounds[name][unit] = value_var_dict(value, var_str)

        self.region_size = self.bounds['rb']['idx']['value'] - \
                           self.bounds['lb']['idx']['value']

        # phase correction parameters
        self.pivot = {}

        pivot = int(self.n // 2)
        for unit in ['idx', 'hz', 'ppm']:
            if unit == 'idx':
                value, value_str = pivot, str(pivot)
            else:
                value = self.convert(pivot, f'idx->{unit}')
                value_str = f"{value:.4f}"

            self.pivot[unit] = value_var_dict(value, var_str)

        self.phases = AutoVivification()

        for name in ['p0', 'p1']:
            for unit in ['rad', 'deg']:
                self.phases[name][unit] = value_var_dict(0., f"{0.:.4f}")

        self.cut = value_var_dict(True, 1)
        self.cut_ratio = value_var_dict(2.5, '2.5')

        max_points = self.cut_size()
        self.max_points = value_var_dict(max_points, str(max_points))

        # number of points to be used for MPM and NLP
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

        # number of oscillators (M) string variable
        self.M_in = value_var_dict(0, '')

        # specifies whether or not to use the MDL to estimate M
        self.mdl = value_var_dict(True, 1)

        # idnitity of the NLP algorithm to use
        self.method = value_var_dict('trust_region', 'Trust Region')

        # maximum iterations of NLP algorithm
        self.maxit = value_var_dict(100, '100')

        # whether or not to include phase variance in NLP cost func
        self.phase_variance = value_var_dict(True, 1)

        # whether or not to purge negligible oscillators
        self.use_amp_thold = value_var_dict(False, 0)

        # amplitude threshold for purging negligible oscillators
        self.amp_thold = value_var_dict(0.001, '0.001')

        self.setupfig = {}

        # plot spectrum
        self.setupfig['fig'] = Figure(figsize=(6,3.5), dpi=170)
        self.setupfig['ax'] = self.setupfig['fig'].add_subplot(111)
        self.setupfig['plot_line'] = self.setupfig['ax'].plot(
            self.shifts['ppm'], np.real(self.spectrum), color='k', lw=0.6,
        )[0]

        # set x-limits as edges of spectral window
        self.setupfig['xlim'] = (self.shifts['ppm'][0], self.shifts['ppm'][-1])
        self.setupfig['ax'].set_xlim(self.setupfig['xlim'])

        # prevent user panning/zooming beyond spectral window
        Restrictor(self.setupfig['ax'], x=lambda x: x<= self.setupfig['xlim'][0])
        Restrictor(self.setupfig['ax'], x=lambda x: x>= self.setupfig['xlim'][1])

        # Get current y-limit. Will reset y-limits to this value after the
        # very tall noise_region and filter_region rectangles have been added
        # to the plot
        self.setupfig['init_ylim'] = self.setupfig['ax'].get_ylim()

        # highlight the spectral region to be filtered (green)
        # Rectangle's first 3 args: bottom left coords, width, height
        bottom_left = (
            self.bounds['lb']['ppm']['value'],
            -20 * self.setupfig['init_ylim'][1],
        )
        width = self.bounds['rb']['ppm']['value'] - \
                self.bounds['lb']['ppm']['value']
        height = 40 * self.setupfig['init_ylim'][1]

        self.setupfig['region'] = Rectangle(
            bottom_left, width, height, facecolor=REGIONCOLOR,
        )
        self.setupfig['ax'].add_patch(self.setupfig['region'])

        # highlight the noise region (blue)
        bottom_left = (
            self.bounds['lnb']['ppm']['value'],
            -20 * self.setupfig['init_ylim'][1],
        )
        width = self.bounds['rnb']['ppm']['value'] - \
                self.bounds['lnb']['ppm']['value']

        self.setupfig['noise_region'] = Rectangle(
            bottom_left, width, height, facecolor=NOISEREGIONCOLOR
        )
        self.setupfig['ax'].add_patch(self.setupfig['noise_region'])

        # plot pivot line
        # alpha set to 0 to make invisible initially
        x = 2 * [self.pivot['ppm']['value']]
        y = [
            -20 * self.setupfig['init_ylim'][1],
             20 * self.setupfig['init_ylim'][1],
        ]

        self.setupfig['pivot_line'] = self.setupfig['ax'].plot(
            x, y, color=PIVOTCOLOR, alpha=0, lw=1,
        )[0]

        # reset y limit
        self.setupfig['ax'].set_ylim(self.setupfig['init_ylim'])

        # aesthetic tweaks to plot
        self.setupfig['fig'].patch.set_facecolor(BGCOLOR)
        self.setupfig['ax'].set_facecolor(PLOTCOLOR)
        self.setupfig['ax'].tick_params(axis='x', which='major', labelsize=6)
        self.setupfig['ax'].locator_params(axis='x', nbins=10)
        self.setupfig['ax'].set_yticks([])

        for direction in ('top', 'bottom', 'left', 'right'):
            self.setupfig['ax'].spines[direction].set_color('k')

        # label plot x-axis with nucleus identity
        comps = filter(None, re.split(r'(\d+)', self.nucleus))

        # xlabel of the form $^{1}$H (ppm)
        self.setupfig['ax'].set_xlabel(
            f'$^{{{next(comps)}}}${next(comps)} (ppm)', fontsize=8
        )


    def generate_result_variables(self):

        self.theta = self.info.get_theta()

        self.resultfig = {}
        for key, value in zip(('fig', 'ax', 'lines', 'labels'), self.info.plot_result()):
            self.resultfig[key] = value

        self.resultfig['fig'].set_size_inches(6, 3.5)
        self.resultfig['fig'].set_dpi(170)

        Restrictor(
            self.resultfig['ax'],
            x=lambda x: x<= self.bounds['rb']['ppm']['value'],
        )
        Restrictor(
            self.resultfig['ax'],
            x=lambda x: x>= self.bounds['lb']['ppm']['value'],
        )

        self.resultfig['fig'].patch.set_facecolor(BGCOLOR)
        self.resultfig['ax'].set_facecolor(PLOTCOLOR)
        self.resultfig['ax'].tick_params(axis='x', which='major', labelsize=6)
        self.resultfig['ax'].locator_params(axis='x', nbins=10)
        self.resultfig['ax'].set_xlabel(
            self.resultfig['ax'].get_xlabel(), fontsize=8,
        )



    def convert(self, value, conversion):

        sw = self.sw['hz']
        off = self.off['hz']
        n = self.n
        sfo = self.sfo

        if conversion == 'idx->hz':
            return float(off + (sw / 2) - ((value * sw) / n))

        elif conversion == 'idx->ppm':
            return float((off + sw / 2 - value * sw / n) / sfo)

        elif conversion == 'ppm->idx':
            return int(round((off + (sw / 2) - sfo * value) * (n / sw)))

        elif conversion == 'ppm->hz':
            return value * sfo

        elif conversion == 'hz->idx':
            return int((n / sw) * (off + (sw / 2) - value))

        elif conversion == 'hz->ppm':
            return value / sfo


    def update_bound(self, name, idx):

        self.bounds[name]['idx']['value'] = idx
        self.bounds[name]['idx']['var'].set(str(idx))

        for unit in ['hz', 'ppm']:
            self.bounds[name][unit]['value'] = self.convert(idx, f'idx->{unit}')
            self.bounds[name][unit]['var'].set(
                f"{self.bounds[name][unit]['value']:.4f}"
            )

        if 'n' in name:
            reg = 'noise_region'
        else:
            reg = 'region'

        if reg == 'region':
            self.region_size = self.bounds['rb']['idx']['value'] - \
                               self.bounds['lb']['idx']['value']
            self.ud_max_points()

        active_unit = self.active_units['freq']
        left = self.bounds[f'l{name[1:]}'][active_unit]['value']
        right = self.bounds[f'r{name[1:]}'][active_unit]['value']

        self.setupfig[reg].set_x(left)
        self.setupfig[reg].set_width(right - left)

        self.update_plot()


    def update_pivot(self, idx):

        self.pivot['idx']['value'] = idx
        self.pivot['idx']['var'].set(str(idx))

        for unit in ['ppm', 'hz']:
            self.pivot[unit]['value'] = self.convert(idx, f'idx->{unit}')
            self.pivot[unit]['var'].set(f"{self.pivot[unit]['value']:.4f}")

        active_unit = self.active_units['freq']

        x = 2 * [self.pivot[active_unit]['value']]
        self.setupfig['pivot_line'].set_xdata(x)

        self.phase_correct()


    def update_p0_p1(self, rad, name):

        self.phases[name]['rad']['value'] = rad
        self.phases[name]['rad']['var'].set(f'{rad:.4f}')
        self.phases[name]['deg']['value'] = rad*180 / np.pi
        self.phases[name]['deg']['var'].set(f'{rad*180 / np.pi:.4f}')

        self.phase_correct()


    def phase_correct(self):

        pivot = self.pivot['idx']['value']
        p0 = self.phases['p0']['rad']['value']
        p1 = self.phases['p1']['rad']['value']
        n = self.n

        corrector = np.exp(1j * (p0 + p1 * np.arange(-pivot, -pivot+n, 1) / n))
        data = np.real(self.spectrum * corrector)
        self.setupfig['plot_line'].set_ydata(data)

        self.update_plot()


    def ud_max_points(self):
        """Update the maximum number of points StringVar"""

        if self.cut['value'] == True:
            # check range is suitable. If not, set it within the spectrum
            # divide by two as halving signal in frequency_filter
            cut_size = self.cut_size()

            # determine respective low and high bounds
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

        for name, default in zip(('mpm', 'nlp'), (4096, 8192)):
            if self.trim[name]['value'] > self.max_points['value'] \
            or self.max_points['value'] <= default:
                self.trim[name]['value'] = self.max_points['value']
                self.trim[name]['var'].set(str(self.max_points['value']))


    def update_plot(self):
        self.setup_frames['plot_frame'].canvas.draw_idle()


    def cut_size(self):
        return int((self.cut_ratio['value'] * self.region_size) // 2)


    def run(self):
        """Set up the estimation routine"""

        # get parameters
        region = (
            self.bounds['lb']['idx']['value'],
            self.bounds['rb']['idx']['value'],
        )

        noise_region = (
            self.bounds['lnb']['idx']['value'],
            self.bounds['rnb']['idx']['value'],
        )

        pivot = self.pivot['ppm']['value']
        p0 = self.phases['p0']['rad']['value']
        p1 = self.phases['p1']['rad']['value']


        cut = self.cut['value']
        if cut:
            cut_ratio = self.cut_ratio['value']
        else:
            cut_ratio = None

        trim_mpm = (self.trim['mpm']['value'],)
        trim_nlp = (self.trim['nlp']['value'],)

        # get number of oscillators for initial guess (or determine whether
        # to use MDL)
        M_in = self.M_in['value']

        method = self.method['value']
        maxit = self.maxit['value']
        phase_variance = self.phase_variance['value']

        if self.use_amp_thold['value']:
            amp_thold = self.amp_thold['value']
        else:
            amp_thold = None

        if TEST_RESULT:
            pass
        else:
            self.setup.withdraw()

        self.info.frequency_filter(
            region=region, noise_region=noise_region, p0=p0, p1=p1,
            cut=cut, cut_ratio=cut_ratio, region_units='idx',
        )

        self.info.matrix_pencil(trim=trim_mpm, M_in=M_in)

        self.info.nonlinear_programming(
            trim=trim_nlp, maxit=maxit, method=method,
            phase_variance=phase_variance, amp_thold=amp_thold,
        )

        self.info.pickle_save(
            fname='tmp.pkl', dir=TMPDIR, force_overwrite=True
        )

        if TEST_RESULT:
            pass
        else:
            self.setup.destroy()


class LogoFrame(MyFrame):
    """Contains the NMR-EsPy logo (who doesn't like a bit of publicity)"""

    def __init__(self, parent, logos='both', scale=0.08):

        super().__init__(parent)

        col = 0
        padx = 0

        if logos in ['both', 'nmrespy']:
            self.nmrespy_img = get_PhotoImage(
                os.path.join(IMAGESDIR, 'nmrespy_full.png'), scale
            )
            self.nmrespy_logo = MyLabel(
                self, image=self.nmrespy_img, cursor='hand1'
            )
            self.nmrespy_logo.bind(
                '<Button-1>', lambda e: webbrowser.open_new(NMRESPYLINK)
            )
            self.nmrespy_logo.grid(row=0, column=col)

            col += 1
            padx = (40, 0)

        if logos in ['both', 'mfgroup']:
            self.mfgroup_img = get_PhotoImage(
                os.path.join(IMAGESDIR, 'mf_logo.png'), scale*10
            )
            self.mfgroup_logo = MyLabel(
                self, image=self.mfgroup_img, cursor='hand1'
            )

            self.mfgroup_logo.bind(
                '<Button-1>', lambda e: webbrowser.open_new(MFGROUPLINK)
            )
            self.mfgroup_logo.grid(row=0, column=col, padx=padx)


class DataType(MyToplevel):
    """GUI for asking user whether they want to analyse the raw FID or
    pdata"""

    def __init__(self, ctrl):

        super().__init__(ctrl)

        self.ctrl = ctrl

        # open info file. Gives paths to fid file and pdata directory
        with open(os.path.join(GUIDIR, 'tmp/info.txt'), 'r') as fh:
            self.fidpath, self.pdatapath = fh.read().split(' ')

        # frame for the NMR-EsPy logo
        self.logo_frame = LogoFrame(self, logos='nmrespy', scale=0.07)

        self.main_frame = MyFrame(self)

        message = MyLabel(
            self.main_frame, text='Which data would you like to analyse?',
            font=(MAINFONT, '12', 'bold'),
        )
        message.grid(
            column=0, row=0, columnspan=2, padx=10, pady=(10,0)
        )

        pdata_label = MyLabel(
            self.main_frame, text='Processed Data'
        )
        pdata_label.grid(
            column=0, row=1, padx=(10,0), pady=(10,0), sticky='w'
        )

        pdatapath = MyLabel(
            self.main_frame, text=f'{self.pdatapath}/1r', font=('Courier', 11),
        )
        pdatapath.grid(column=0, row=2, padx=(10, 0), sticky='w')

        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = MyCheckbutton(
            self.main_frame, variable=self.pdata, command=self.click_pdata,
        )
        self.pdata_box.grid(
            column=1, row=1, rowspan=2, padx=10, sticky='nsw'
        )

        fid_label = MyLabel(self.main_frame, text='Raw FID')
        fid_label.grid(
            column=0, row=3, padx=(10, 0), pady=(10, 0), sticky='w',
        )

        fidpath = MyLabel(
            self.main_frame, text=f'{self.fidpath}/fid', font=('Courier', 11),
        )
        fidpath.grid(column=0, row=4, padx=(10, 0), sticky='w')

        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = MyCheckbutton(
            self.main_frame, variable=self.fid, command=self.click_fid,
        )
        self.fid_box.grid(
            column=1, row=3, rowspan=2, padx=10, sticky='nsw'
        )

        self.button_frame = MyFrame(self)

        self.confirmbutton = MyButton(
            self.button_frame, text='Confirm', command=self.confirm,
            bg=BUTTONGREEN,
        )
        self.confirmbutton.grid(
            column=1, row=0, padx=(5, 10), pady=10, sticky='e',
        )

        self.cancelbutton = MyButton(
            self.button_frame, text='Cancel', command=self.ctrl.destroy,
            bg=BUTTONRED,
        )
        self.cancelbutton.grid(column=0, row=0, pady=10, sticky='e')

        self.logo_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.main_frame.grid(row=0, column=1)
        self.button_frame.grid(row=1, column=1, sticky='e')

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
            self.ctrl.dtype = 'fid'
            self.ctrl.path = self.fidpath
        else:
            self.ctrl.dtype = 'pdata'
            self.ctrl.path = self.pdatapath

        self.destroy()



class PlotFrame(MyFrame):
    """Contains a plot, along with navigation toolbar"""

    def __init__(self, parent, figure):
        super().__init__(parent)

        # make figure canvas expandable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')


class RootToolbarFrame(MyFrame):

    def __init__(self, parent, canvas, ctrl):
        super().__init__(parent)

        self.parent = parent
        self.ctrl = ctrl

        self.columnconfigure(0, weight=1)

        self.toolbar = MyNavigationToolbar(canvas, parent=self)
        self.toolbar.grid(row=0, column=0, sticky='w', padx=(10,0), pady=(0,5))


class SetupToolbarFrame(RootToolbarFrame):

    def __init__(self, parent, canvas, ctrl):
        super().__init__(parent, canvas, ctrl)

        self.adsettings_button = MyButton(
            parent=self, text='Advanced Settings', width=16,
            command=self.advanced_settings
        )
        self.adsettings_button.grid(
            row=0, column=1, sticky='e', padx=10, pady=(0,5),
        )

    def advanced_settings(self):
        AdvancedSettingsFrame(parent=self.parent, ctrl=self.ctrl)


class TabFrame(MyFrame):
    """Contains a notebook for region selection, phase correction, and
    advanced settings"""

    def __init__(self, parent, ctrl):
        MyFrame.__init__(self, parent)
        self.ctrl = ctrl

        # make column containing scales adjustable
        self.columnconfigure(0, weight=1)

        #customise notebook style
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

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky='ew', padx=10, pady=(0,10))
        # # whenever tab clicked, change plot so that either region selection
        # # rectangles are visible, or phase pivot, depending on tab selected
        self.notebook.bind(
            '<<NotebookTabChanged>>', lambda event: self.switch_tab()
        )

        self.region_frame = RegionFrame(parent=self.notebook, ctrl=self.ctrl)
        self.notebook.add(
            self.region_frame, text='Region Selection', sticky='nsew',
        )

        self.phase_frame = PhaseFrame(parent=self.notebook, ctrl=self.ctrl)
        self.notebook.add(
            self.phase_frame, text='Phase Correction', sticky='nsew',
        )


    def switch_tab(self):
        """Adjusts the appearence of the plot when a new tab is selected.
        Hides/reveals region rectangles and pivot plot as required. Toggles
        alpha between 1 and 0"""

        # detemine the active tab
        tab = self.notebook.index(self.notebook.select())
        # set alpha values for region rectangles and pivot plot
        if tab == 0: regions, pivot = 1, 0
        else: regions, pivot = 0, 1

        self.ctrl.setupfig['region'].set_alpha(regions)
        self.ctrl.setupfig['noise_region'].set_alpha(regions)
        self.ctrl.setupfig['pivot_line'].set_alpha(pivot)

        # draw updated figure
        self.ctrl.update_plot()


class RegionFrame(MyFrame):
    """Frame inside SetupApp notebook - for altering region boundaries"""

    def __init__(self, parent, ctrl):
        super().__init__(parent, bg=NOTEBOOKCOLOR)
        self.ctrl = ctrl

        # make scales expandable
        self.columnconfigure(1, weight=1)

        self.labels = {}
        self.scales = {}
        self.entries = {}

        for row, name in enumerate(('lb', 'rb', 'lnb', 'rnb')):
            # construct text strings for scale titles
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

            # scale titles
            self.labels[name] = title = MyLabel(
                self, text=text, bg=NOTEBOOKCOLOR,
            )

            # determine troughcolor of scale (0, 1: green; 2, 3: blue)
            if row < 2:
                troughcolor = REGIONCOLOR
            else:
                troughcolor = NOISEREGIONCOLOR

            self.scales[name] = scale = MyScale(
                    self, from_=0, to=self.ctrl.n - 1, troughcolor=troughcolor,
                    bg=NOTEBOOKCOLOR, highlightthickness=1,
                    command=lambda idx, name=name: self.update_scale(idx, name),
                )
            scale.set(self.ctrl.bounds[name]['idx']['value'])

            self.entries[name] = entry = MyEntry(
                self, textvariable=self.ctrl.bounds[name]['ppm']['var'],
            )
            entry.bind(
                '<Return>', (lambda event, name=name: self.update_entry(name)),
            )

            # only pad above if not bottom row
            pady = (10,0)
            if row == 3:
                # pad below AND above if bottom widget
                pady = 10

            title.grid(row=row, column=0, padx=(10,0), pady=pady, sticky='w')
            scale.grid(row=row, column=1, padx=(10,0), pady=pady, sticky='ew')
            entry.grid(row=row, column=2, padx=10, pady=pady, sticky='w')


    def update_entry(self, name):
        """Update the GUI after the user presses <Enter> whilst in an entry
        widget"""

        unit = self.ctrl.active_units['freq']

        try:
            value = self.ctrl.bounds[name][unit]['var'].get()
            if unit == 'idx':
                idx = int(value)
            else:
                idx = self.ctrl.convert(float(value), f'{unit}->idx')

        except:
            self.reset_bound(name, unit)
            return

        if self.check_valid_index(idx, name):
            self.scales[name].set(idx)
            self.ctrl.update_bound(name, idx)
        else:
            self.reset_bound(name, unit)


    def update_scale(self, idx, name):
        """Update the GUI after the user changes the slider on a scale
        widget"""

        idx = int(idx)
        if not self.check_valid_index(idx, name):
            if name[0] == 'l':
                idx = self.ctrl.bounds[f'r{name[1:]}']['idx']['value'] - 1
            else:
                idx = self.ctrl.bounds[f'l{name[1:]}']['idx']['value'] + 1
            self.scales[name].set(idx)

        self.ctrl.update_bound(name, idx)


    def reset_bound(self, name, unit):
        value = self.ctrl.bounds[name][unit]['value']
        if unit == 'idx':
            self.ctrl.bounds[name][unit]['var'].set(str(value))
        else:
            self.ctrl.bounds[name][unit]['var'].set(f'{value:.4f}')


    def check_valid_index(self, idx, name):
        """Given an update index, and the identity of the bound to change,
        adjust relavent region parameters, and update the GUI."""

        # determine if we are considering a left or right bound
        # twist is the thing that is changing
        if name[0] == 'l':
            left = idx
            right = self.ctrl.bounds[f'r{name[1:]}']['idx']['value']
        else:
            left = self.ctrl.bounds[f'l{name[1:]}']['idx']['value']
            right = idx

        if left < right and 0 <= idx <= self.ctrl.n - 1:
            # all good, update bound attribute
            self.ctrl.bounds[name]['idx']['value'] = idx
            return True
        return False


class PhaseFrame(MyFrame):
    """Frame inside SetupApp notebook - for phase correction of data"""

    def __init__(self, parent, ctrl):
        super().__init__(parent, bg=NOTEBOOKCOLOR)
        self.ctrl = ctrl

        # make scales expandable
        self.columnconfigure(1, weight=1)

        self.titles = {}
        self.scales = {}
        self.entries = {}

        for row, name in enumerate(('pivot', 'p0', 'p1')):
            # scale titles
            self.titles[name] = title = MyLabel(
                self, text=name, bg=NOTEBOOKCOLOR
            )

            # pivot scale
            if name == 'pivot':
                troughcolor = PIVOTCOLOR
                from_ = 0
                to = self.ctrl.n - 1
                resolution = 1

            # p0 and p1 scales
            else:
                troughcolor = 'white'
                from_ = -np.pi
                to = np.pi
                resolution = 0.001

                # p1: set between -10π and 10π rad
                if name == 'p1':
                    from_ *= 10
                    to *= 10

            self.scales[name] = scale = MyScale(
                    self, troughcolor=troughcolor, from_=from_, to=to,
                    resolution=resolution, bg=NOTEBOOKCOLOR,
                    command=lambda value, name=name: self.update_scale(value, name),
            )

            if name == 'pivot':
                scale.set(self.ctrl.pivot['idx']['value'])
                var = self.ctrl.pivot['ppm']['var']
            else:
                scale.set(0)
                var = self.ctrl.phases[name]['rad']['var']

            self.entries[name] = entry = MyEntry(self, textvariable=var)
            entry.bind(
                '<Return>', (lambda event, name=name: self.update_entry(name)),
            )

            pady = (10,0)
            if row == 2:
                pady = 10

            title.grid(row=row, column=0, padx=(10,0), pady=pady, sticky='w')
            scale.grid(row=row, column=1, padx=(10,0), pady=pady, sticky='ew')
            entry.grid(row=row, column=2, padx=10, pady=pady, sticky='w')


    def update_scale(self, value, name):

        """Update the GUI after the user changes the slider on a scale
        widget"""

        if name == 'pivot':
            self.ctrl.update_pivot(int(value))

        else:
            self.ctrl.update_p0_p1(float(value), name)


    def update_entry(self, name):
        """Update the GUI after the user changes and entry widget"""


        if name == 'pivot':
            unit = self.ctrl.active_units['freq']
            value = self.ctrl.pivot[unit]['var'].get()

            try:
                if unit == 'idx':
                    idx = int(value)
                else:
                    idx = self.ctrl.convert(float(value), f'{unit}->idx')


                if 0 <= idx <= self.ctrl.n - 1:
                    self.scales['pivot'].set(idx)
                    self.crtl.update_pivot(idx)
                else:
                    raise

            except:
                self.reset_pivot(unit)

        else:
            unit = self.ctrl.active_units['angle']
            value = self.ctrl.phases[name][unit]['var'].get()

            try:
                if unit == 'rad':
                    rad = float(value)
                else:
                    rad = float(value) * 180 / np.pi

                if -np.pi <= rad <= np.pi and name == 'p0':
                    self.scales['p0'].set(rad)

                elif -10 * np.pi <= rad <= 10 * np.pi and name == 'p1':
                    self.scales['p1'].set(rad)

                else:
                    raise

                self.ctrl.update_phase(rad, name)

            except:
                self.reset_phase(name, unit)


    def reset_pivot(self, unit):
        if unit == 'idx':
            self.ctrl.pivot['idx']['var'].set(
                str(self.ctrl.pivot['idx']['value'])
            )
        else:
            self.ctrl.pivot[unit]['var'].set(
                f"{self.ctrl.pivot[unit]['value']:.4f}"
            )


    def reset_phase(self, name, unit):
        self.ctrl.phases[name][unit]['var'].set(
            f"{self.ctrl.phases[name][unit]['value']:.4f}"
        )


class AdvancedSettingsFrame(MyToplevel):
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
            textvariable=self.ctrl.M_in['var'],
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
            self.button_frame, text='Close', command=self.destroy
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
            self.ctrl.M_in['value'] = 0
            self.ctrl.M_in['var'].set('')
        else:
            self.ctrl.mdl['value'] = False
            self.oscillator_entry['state'] = 'normal'


    def ud_oscillators(self):

        str_value = self.ctrl.M_in['var'].get()
        if self._check_int(str_value) and int(str_value) > 0:
            int_value = int(str_value)
            self.ctrl.M_in['value'] = int_value
            self.ctrl.M_in['var'].set(str(int_value))

        else:
            if self.ctrl.M_in['value'] == 0:
                self.ctrl.M_in['var'].set('')
            else:
                self.ctrl.M_in['var'].set(str(self.ctrl.M_in['value']))


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
            command=lambda: webbrowser.open_new(GUIDOCLINK)
        )
        self.help_button.grid(
            row=1, column=1, padx=(10,0), pady=(10,0), sticky='e'
        )

        # command varies - will need to be defined from the class that
        # inherits from this
        # for example, see SetupButtonFrame
        self.save_button = MyButton(
            self, text='Run', bg=BUTTONGREEN, command=self.ctrl.run
        )
        self.save_button.grid(
            row=1, column=2, padx=10, pady=(10,0), sticky='e'
        )

        contact_info_1 = MyLabel(
            self, text='For queries/feedback, contact',
        )
        contact_info_1.grid(
            row=2, column=0, columnspan=3, padx=10, pady=(10,0), sticky='w'
        )

        contact_info_2 = MyLabel(
            self, text='simon.hulse@chem.ox.ac.uk', font='Courier', fg='blue',
            cursor='hand1',
        )
        contact_info_2.bind(
            '<Button-1>', lambda e: webbrowser.open_new(EMAILLINK)
        )

        contact_info_2.grid(
            row=3, column=0, columnspan=3, padx=10, pady=(0,10), sticky='w'
        )


class SetupButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, parent, ctrl):
        super().__init__(parent, ctrl)
        self.save_button['command'] = self.ctrl.run


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

                # make read-only color a pale blue (easy to see when row
                # seleceted)
                entry = MyEntry(
                    self.table, textvariable=var, state='disabled', width=14,
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

        # deactivate all buttons
        if activated_number == 0:
            self.remove_button['state'] = 'disabled'
            self.split_button['state'] = 'disabled'
            self.merge_button['state'] = 'disabled'
            self.manual_button['state'] = 'disabled'

        # activate split and manual edit buttons
        # deactivate merge button (can't merge one oscillator...)
        elif activated_number == 1:
            self.remove_button['state'] = 'normal'
            self.split_button['state'] = 'normal'
            self.merge_button['state'] = 'disabled'
            self.manual_button['state'] = 'normal'

        # activate merge button
        # deactivate split and manual edit buttons (ambiguous with multiple
        # oscillators selected)
        else: # activated_number > 1
            self.remove_button['state'] = 'normal'
            self.split_button['state'] = 'disabled'
            self.merge_button['state'] = 'normal'
            self.manual_button['state'] = 'disabled'

        for button in [self.remove_button, self.split_button, self.merge_button, self.manual_button]:
            if button['state'] == 'normal':
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
        for entry in self.table_entries[i]:
            entry['state'] = 'normal'

        # hacky way to get button with height less than 1:
        # make temporary frame with defined height (pixels), and
        # pack buttons in
        self.tmpframe = tk.Frame(self.table, height=22, width=120, bg='white')
        self.tmpframe.pack_propagate(0) # don't shrink
        self.tmpframe.grid(row=i+1, column=5)

        for c in (0, 1):
            self.tmpframe.columnconfigure(c, weight=1)

        self.udbutton = tk.Button(self.tmpframe, text='Save', width=3,
                                  bg='#9eda88', highlightbackground='black',
                                  command=lambda i=i: self.ud_manual(i),
                                  font=(MAINFONT, 10))
        self.udbutton.pack(fill=tk.BOTH, expand=1, side=tk.LEFT, pady=(2,0))

        self.cancelbutton = tk.Button(self.tmpframe, text='Cancel', width=3,
                                    bg='#ff9894', highlightbackground='black',
                                    command=lambda i=i: self.cancel_manual(i),
                                    font=(MAINFONT, 10))
        self.cancelbutton.pack(fill=tk.BOTH, expand=1, side=tk.RIGHT,
                               padx=(3,10), pady=(2,0))


    def ud_manual(self, i):
        """Replace the current parameters of the selected oscillator with
        the values in the entry boxes"""

        # construct numpy array with oscillator's new parameters.
        # If any of the values cannot be converted to floats, the
        # entry widgets will be reset, and the user will be warned.
        try:
            oscillator = np.array([
                float(self.table_variables[i][j].get()) for j in range(4)
            ])

        except:
            # reset variables back to original values
            for j in range(4):
                self.table_variables[i][j].set(
                                f'{self.ctrl.info.theta[i][j]:.5f}')

            msg = 'At least one of the parameter values specified could not' \
                  + ' be converted to a numerical value!'
            WarnFrame(self, msg)
            return

        # replace oscillator with user input
        self.ctrl.info.theta[i] = oscillator
        # sort oscillators in order of frequency
        self.ctrl.info.theta = \
        self.ctrl.info.theta[np.argsort(
                                    self.ctrl.info.theta[..., 2])]

        # remove temporary buton frame
        self.tmpframe.destroy()
        # update plot and parameter table
        ud_plot(self.ctrl)
        self.construct_table(reconstruct=True)
        self.activate_buttons()


    def cancel_manual(self, i):
        """Cancel manually chaning oscillator parameters."""

        # remove temporary buton frame
        self.tmpframe.destroy()

        # replace contents of entry widgets with previous values in theta
        # set entry widgets back to read-only mode
        for j in range(4):
            self.table_variables[i][j].set(
                            f'{self.ctrl.info.theta[i][j]:.5f}')

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
                *label.get_position(), label.get_text()
            )

        # draw the new plot!
        self.ctrl.result_frames['plot_frame'].canvas.draw_idle()

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

            entry = MyEntry(
                self.table_frame, return_command=self.check_param,
                return_args=(row, column), width=12, textvariable=var,
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


    def check_param(self, row, column):

        try:
            value = float(self.vars[row][column].get())

        except:
            self.vars[row][column].set('')

        # amplitude
        if column == 0:
            if value < 0.:
                raise

        # phase
        elif column == 1:
            if value <= -np.pi or value >= np.pi:
                # if phase outside acceptable range, then wrap
                value = (value + np.pi) % (2 * np.pi) - np.pi

        # frequency
        elif column == 2:
            pass

        elif column == 3:
            pass




    def save(self):

        oscillators = []
        try:
            for var_row in self.vars:
                oscillator_row = []
                for i, var in enumerate(var_row):
                    value = var.get()

                    oscillator_row.append(float(var.get()))
                oscillators.append(oscillator_row)

        except:
            msg = 'Not all parameters could be converted to floats. Make sure' \
                  + ' all the parameters given are valid numerical values.'
            WarnFrame(self, msg)
            return

        oscillators = np.array(oscillators)

        self.ctrl.info.add_oscillators(oscillators)

        # update the plot
        ud_plot(self.ctrl)
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


class SaveFrame(tk.Toplevel):

    def __init__(self, parent, ctrl):

        tk.Toplevel.__init__(self, parent)
        self.title('Save Options')
        self.parent = parent
        self.ctrl = ctrl
        self['bg'] = 'white'
        self.resizable(False, False)
        self.grab_set()

        self.fileframe = tk.Frame(self, bg='white')
        self.fileframe.grid(row=0, column=0)
        self.buttonframe = tk.Frame(self, bg='white')
        self.buttonframe.grid(row=2, column=0, sticky='e')

        tk.Label(self.fileframe, text='Save Figure', bg='white').grid(
            row=0, column=0, padx=(10,0), pady=(10,0), sticky='w'
        )

        self.fig_var = tk.IntVar()
        self.fig_var.set(1)
        self.fig_check = tk.Checkbutton(
            self.fileframe, variable=self.fig_var, bg='white',
            highlightthickness=0, bd=0,
            command=(lambda: self.check('fig'))
        )
        self.fig_check.grid(row=0, column=1, padx=(2,0), pady=(10,0))

        self.fig_button = tk.Button(
            self.fileframe, text='Customise Figure',
            highlightbackground='black', command=self.customise_figure
        )
        self.fig_button.grid(
            row=0, column=2, columnspan=3, padx=10, pady=(10,0), sticky='ew'
        )

        titles = ('Save textfile:', 'Save PDF:', 'Pickle result:')
        extensions = ('.txt', '.pdf', '.pkl')
        for i, (title, extension) in enumerate(zip(titles, extensions)):

            tk.Label(self.fileframe, text=title, bg='white').grid(
                row=i+1, column=0, padx=(10,0), pady=(10,0), sticky='w'
            )

            tag = extension[1:]
            # variable which dictates whether to save the filetype
            self.__dict__[f'{tag}_var'] = savevar = tk.IntVar()
            savevar.set(1)

            self.__dict__[f'{tag}_check'] = check = \
                tk.Checkbutton(
                    self.fileframe, variable=savevar, bg='white',
                    highlightthickness=0, bd=0,
                    command=(lambda tag=tag: self.check(tag))
                )
            check.grid(row=i+1, column=1, padx=(2,0), pady=(10,0))

            tk.Label(self.fileframe, text='Filename:', bg='white').grid(
                row=i+1, column=2, padx=(15,0), pady=(10,0), sticky='w'
            )

            self.__dict__[f'{tag}_name'] = fnamevar = tk.StringVar()
            fnamevar.set('nmrespy_result')

            self.__dict__[f'{tag}_entry'] = entry = \
                tk.Entry(
                    self.fileframe, textvariable=fnamevar, width=20,
                    highlightthickness=0
                )
            entry.grid(
                row=i+1, column=3, padx=(5,0), pady=(10,0)
            )

            tk.Label(self.fileframe, text=extension, bg='white').grid(
                row=i+1, column=4, padx=(0,10), pady=(10,0), sticky='w'
            )

        tk.Label(self.fileframe, text='Description:', bg='white').grid(
            row=4, column=0, padx=(10,0), pady=(10,0), sticky='nw'
        )

        self.descr_box = tk.Text(self.fileframe, width=40, height=3)
        self.descr_box.grid(
            row=4, column=1, columnspan=4, padx=10, pady=(10,0), sticky='ew'
        )

        tk.Label(self.fileframe, text='Directory:', bg='white').grid(
            row=5, column=0, padx=(10,0), pady=(10,0), sticky='w'
        )

        self.dir_var = tk.StringVar()
        self.dir_var.set(os.path.expanduser('~'))

        self.dir_entry = tk.Entry(
            self.fileframe, textvariable=self.dir_var, width=30,
            highlightthickness=0
        )
        self.dir_entry.grid(
            row=5, column=1, columnspan=3, padx=(10,0), pady=(10,0), sticky='ew'
        )

        self.img = get_PhotoImage(
            os.path.join(IMAGESDIR, 'folder_icon.png'), scale=0.02
        )

        self.dir_button = tk.Button(
            self.fileframe, command=self.browse, bg='white',
            highlightbackground='black', image=self.img
        )
        self.dir_button.grid(row=5, column=4, padx=(5,10), pady=(10,0))


        self.cancel_button = tk.Button(
            self.buttonframe, text='Cancel', width=8, bg='#ff9894',
            highlightbackground='black', command=self.destroy
        )
        self.cancel_button.grid(row=0, column=0, pady=10)

        self.save_button = tk.Button(
            self.buttonframe, text='Save', width=8, bg='#9eda88',
            highlightbackground='black', command=self.save
        )
        self.save_button.grid(row=0, column=1, padx=10, pady=10)


    def check(self, tag):

        var = self.__dict__[f'{tag}_var'].get()

        state = 'disabled'
        if var: # 1 or 0
            state = 'normal'

        if tag == 'fig':
            self.fig_button['state'] = state
        else:
            self.__dict__[f'{tag}_entry']['state'] = state

    def browse(self):
        self.dir_var.set(filedialog.askdirectory(initialdir=self.dir_var.get()))

    def save(self):

        # check directory is valid
        dir = self.dir_var.get()
        if os.path.isdir(dir):
            pass
        else:
            msg = f'The specified directory doesn\'t exist!\n{dir}'
            WarnFrame(self, msg)

        descr = self.descr_box.get('1.0', 'end-1c')

        # check textfile
        if self.txt_var.get():
            fname = self.txt_name.get()
            self.ctrl.info.write_result(
                description=descr, fname=fname, dir=dir, format='txt',
                force_overwrite=True
            )

        # check PDF
        if self.pdf_var.get():
            fname = self.pdf_name.get()
            self.ctrl.info.write_result(
                description=descr, fname=fname, dir=dir, format='pdf',
                force_overwrite=True
            )

        # check pickle
        if self.pkl_var.get():
            fname = self.pkl_name.get()
            self.ctrl.info.pickle_save(
                fname=fname, dir=dir, force_overwrite=True
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

        print(options)

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
