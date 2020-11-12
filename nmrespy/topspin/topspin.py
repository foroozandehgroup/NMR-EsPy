#!/usr/bin/env python3

from copy import deepcopy
from itertools import cycle
import os
import random
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import nmrespy
import nmrespy.load as load
import nmrespy._misc as _misc
from nmrespy._plot import _generate_xlabel


def get_PhotoImage(path, scale):

    image = Image.open(path)
    [w, h] = image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)

class WarnFrame(tk.Toplevel):
    """A window in case the user does something silly."""

    def __init__(self, master, msg):
        tk.Toplevel.__init__(self, master)
        self['bg'] = 'white'
        self.resizable(False, False)

        # warning icon
        path = os.path.join(os.path.dirname(nmrespy.__file__),
                            'topspin/images/warning.png')
        self.img = get_PhotoImage(path, 0.08)
        warn_icon = tk.Label(self, image=self.img, bg='white')
        warn_icon.grid(row=0, column=0, padx=(10,0), pady=10)

        # add text explaining the issue
        msg_label = tk.Label(self, text=msg, wraplength=400, bg='white')
        msg_label.grid(row=0, column=1, padx=10, pady=10)

        # close button
        close_button = tk.Button(self, text='Close', width=8,
                                 bg='#ff9894',
                                 highlightbackground='black',
                                 command=self.destroy)
        close_button.grid(row=1, column=1, padx=10, pady=(0,10))


class CustomNavigationToolbar(NavigationToolbar2Tk):
    """Tweak default matplotlib navigation bar to exclude subplot-config
    and save buttons. Also remove co-ordiantes as cursor goes over plot"""
    def __init__(self, canvas_, parent_):
        self.toolitems = self.toolitems[:6]
        NavigationToolbar2Tk.__init__(self, canvas_, parent_, pack_toolbar=False)

    def set_message(self, msg):
        pass

# https://stackoverflow.com/questions/48709873/restricting-panning-range-in-matplotlib-plots
class Restrictor():
    """Resict naivgation within a defined range (used to prevent
    panning/zooming) outside spectral window on x-axis"""
    def __init__(self, ax, x=lambda x: True, y=lambda x: True):
        self.res = [x,y]
        self.ax =ax
        self.limits = self.get_lim()
        self.ax.callbacks.connect('xlim_changed', lambda evt: self.lims_change(axis=0))
        self.ax.callbacks.connect('ylim_changed', lambda evt: self.lims_change(axis=1))

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


class DataType(tk.Toplevel):
    """GUI for asking user whether they want to analyse the raw FID or
    pdata assocaited with the opened data"""
    def __init__(self, master, fidpath, pdatapath):
        tk.Toplevel.__init__(self, master)

        self.title('NMR-EsPy - Choose Data')
        self.resizable(False, False)
        self.master = master
        self['bg'] = 'white'

        # --- FRAMES ----------------------------------------------------------
        self.logoframe = tk.Frame(self, bg='white')
        self.mainframe = tk.Frame(self, bg='white')
        self.buttonframe = tk.Frame(self, bg='white')

        self.logoframe.grid(row=0, column=0, rowspan=2)
        self.mainframe.grid(row=0, column=1)
        self.buttonframe.grid(row=1, column=1, sticky='e')

        self.pad = 10

        # -- logoframe --------------------------------------------------------
        path = os.path.dirname(nmrespy.__file__)
        image = Image.open(os.path.join(path, 'topspin/images/nmrespy_full.png'))
        scale = 0.07
        [w, h] = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        self.logo = tk.Label(self.logoframe, image=img, bg='white')
        self.logo.grid(row=0, column=0, padx=self.pad, pady=self.pad)

        # --- mainframe --------------------------------------------------------
        self.message = tk.Label(self.mainframe,
                                text='Which data would you like to analyse?',
                                font=('Helvetica', '14'), bg='white')
        self.pdata_label = tk.Label(self.mainframe, text='Processed Data',
                                    bg='white')
        self.pdatapath = tk.Label(self.mainframe, text=f'{pdatapath}/1r',
                                  font='Courier', bg='white')
        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = tk.Checkbutton(self.mainframe, variable=self.pdata,
                                        command=self.click_pdata, bg='white',
                                        highlightthickness=0, bd=0)
        self.fid_label = tk.Label(self.mainframe, text='Raw FID', bg='white')
        self.fidpath = tk.Label(self.mainframe, text=f'{fidpath}/fid',
                                font='Courier', bg='white')
        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = tk.Checkbutton(self.mainframe, variable=self.fid,
                                      command=self.click_fid, bg='white',
                                      highlightthickness=0, bd=0)


        self.message.grid(column=0, row=0, columnspan=2, padx=self.pad,
                          pady=(self.pad, 0))
        self.pdata_label.grid(column=0, row=1, padx=(self.pad, 0),
                              pady=(self.pad, 0), sticky='w')
        self.pdatapath.grid(column=0, row=2, padx=(self.pad, 0), sticky='w')
        self.pdata_box.grid(column=1, row=1, rowspan=2, padx=self.pad,
                            sticky='nsw')
        self.fid_label.grid(column=0, row=3, padx=(self.pad, 0),
                            pady=(self.pad, 0), sticky='w')
        self.fidpath.grid(column=0, row=4, padx=(self.pad, 0), sticky='w')
        self.fid_box.grid(column=1, row=3, rowspan=2, padx=self.pad,
                          sticky='nsw')

        # --- buttonframe ------------------------------------------------------
        self.confirmbutton = tk.Button(self.buttonframe, text='Confirm',
                                       command=self.confirm, width=8,
                                       bg='#9eda88', highlightbackground='black')
        self.cancelbutton = tk.Button(self.buttonframe, text='Cancel',
                                      command=self.cancel, width=8,
                                      bg='#ff9894', highlightbackground='black')

        self.confirmbutton.grid(column=1, row=0, padx=(self.pad/2, self.pad),
                                pady=(self.pad, self.pad), sticky='e')
        self.cancelbutton.grid(column=0, row=0, pady=(self.pad, self.pad),
                               sticky='e')

        self.master.wait_window(self)

    # --- COMMANDS ------------------------------------------------------------
    # click_fid and click_pdata ensure only one checkbutton is selected at
    # any time
    def click_fid(self):
        fidval = self.fid.get()
        if fidval == 1:
            self.pdata.set(0)
        elif fidval == 0:
            self.pdata.set(1)

    def click_pdata(self):
        fidval = self.pdata.get()
        if fidval == 1:
            self.fid.set(0)
        elif fidval == 0:
            self.fid.set(1)

    def cancel(self):
        self.destroy()
        print('NMR-EsPy Cancelled :\'(')
        exit()

    def confirm(self):
        if self.fid.get() == 1:
            self.master.dtype = 'fid'
        elif self.pdata.get() == 1:
            self.master.dtype = 'pdata'
        self.master.master.deiconify()
        self.destroy()

# -----------------------------
# GUI FOR SETTING UP ESTIMATION
# -----------------------------

class Setup(tk.Frame):
    def __init__(self, master, fidpath, pdatapath):

        tk.Frame.__init__(self, master)
        self['bg'] = 'white'

        # --- PADDING VALUE ---------------------------------------------------
        self.pad = 10

        # --- DETERMINE DATA-TYPE TO USE --------------------------------------
        self.fidpath = fidpath
        self.pdatapath = pdatapath
        self.dtype_popup()

        if self.dtype == 'fid':
            self.info = load.import_bruker_fid(self.fidpath, ask_convdta=False)
            self.spec = np.flip(fftshift(fft(self.info.get_data())))
        elif self.dtype == 'pdata':
            self.info = load.import_bruker_pdata(self.pdatapath)
            self.spec = self.info.get_data(pdata_key='1r') \
                        + 1j * self.info.get_data(pdata_key='1i')

        # --- EXTRACT BASIC EXPERIMENT INFO -----------------------------------
        self.shifts = self.info.get_shifts(unit='ppm')[0]
        self.sw_p = self.info.get_sw(unit='ppm')[0]
        self.off_p = self.info.get_offset(unit='ppm')[0]
        self.n = self.spec.shape[0]
        self.nuc = self.info.get_nuc()

        # --- LEFT AND RIGHT BOUNDS -------------------------------------------
        # values in array indices
        self.lb = int(np.floor(7 * self.n / 16))
        self.rb = int(np.floor(9 * self.n / 16))
        self.lnb = int(np.floor(1 * self.n / 16))
        self.rnb = int(np.floor(2 * self.n / 16))

        # values in ppm
        self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                         self.n, direction='idx->ppm')
        self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                         self.n, direction='idx->ppm')
        self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                          self.n, direction='idx->ppm')
        self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                          self.n, direction='idx->ppm')

        # --- PHASE PARAMETERS ------------------------------------------------
        self.pivot = int(np.floor(self.n / 2))
        self.pivot_ppm = _misc.conv_ppm_idx(self.pivot, self.sw_p, self.off_p,
                                            self.n, direction='idx->ppm')
        self.p0 = 0.
        self.p1 = 0.

        # --- CONSTRUCT SPECTRUM PLOT -----------------------------------------
        self.fig = Figure(figsize=(6,3.5), dpi=170)
        self.ax = self.fig.add_subplot(111)
        self.specplot = self.ax.plot(self.shifts, np.real(self.spec), color='k',
                                     lw=0.6)[0]
        self.xlim = (self.shifts[0], self.shifts[-1])

        self.ax.set_xlim(self.xlim)
        self.ylim_init = self.ax.get_ylim()

        # highlight the spectral region to be filtered
        self.filtregion = Rectangle((self.rb_ppm, -20*self.ylim_init[1]),
                                     self.lb_ppm - self.rb_ppm,
                                     40*self.ylim_init[1],
                                     facecolor='#7fd47f')

        self.ax.add_patch(self.filtregion)

        # highlight the noise region
        self.noiseregion = Rectangle((self.rnb_ppm, -20*self.ylim_init[1]),
                                      self.lnb_ppm - self.rnb_ppm,
                                      40*self.ylim_init[1],
                                      facecolor='#66b3ff')

        self.ax.add_patch(self.noiseregion)

        # plot pivot line (alpha=0 to make invisible initially)
        x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
        y = np.linspace(-20*self.ylim_init[1], 20*self.ylim_init[1], 1000)
        self.pivotplot = self.ax.plot(x, y, color='r', alpha=0, lw=1)[0]

        self.ax.set_ylim(self.ylim_init)
        self.ax.tick_params(axis='x', which='major', labelsize=8)
        self.ax.set_yticks([])
        self.ax.spines['top'].set_color('k')
        self.ax.spines['bottom'].set_color('k')
        self.ax.spines['left'].set_color('k')
        self.ax.spines['right'].set_color('k')

        # prevent user panning/zooming beyond spectral window
        self.restrict_left = Restrictor(self.ax, x=lambda x: x<= self.xlim[0])
        self.restrict_right = Restrictor(self.ax, x=lambda x: x>= self.xlim[1])

        # --- LEFT AND RIGHT FRAMES -------------------------------------------
        # leftframe: spectrum plot, navi. toolbar and region scales
        self.leftframe = tk.Frame(self, bg='white')

        # rightframe: logo, ad. settings, save/help/quit butts, contact info.
        self.rightframe = tk.Frame(self, bg='white')

        # --- CANVAS FOR PLOT -------------------------------------------------
        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.leftframe)
        self.canvas.draw()

        # --- NAVIGATION TOOLBAR ----------------------------------------------
        self.toolbar = CustomNavigationToolbar(self.canvas, self.leftframe)
        self.toolbar['bg'] = 'white'
        self.toolbar._message_label['bg'] = 'white'
        for button in self.toolbar.winfo_children():
            button['bg'] = 'white'

        # --- NOTEBOOK FOR REGION/PHASE SCALES --------------------------------
        # customise notebook style
        style = ttk.Style()
        style.theme_create('notebook', parent='alt',
            settings={
                'TNotebook': {
                    'configure': {
                        'tabmargins': [2, 5, 2, 0],
                        'background': 'white',
                        'bordercolor': 'black'}},
                'TNotebook.Tab': {
                    'configure': {
                        'padding': [5, 1],
                        'background': '#d0d0d0'},
                    'map': {
                        'background': [('selected', 'black')],
                        'foreground': [('selected', 'white')],
                        'expand': [("selected", [1, 1, 1, 0])]}}})

        style.theme_use("notebook")

        # scaleframe -> tabs for region selection and phase correction
        self.notebook = ttk.Notebook(self.leftframe)
        self.notebook.bind('<<NotebookTabChanged>>',
                             (lambda event: self.ud_plot()))

        # --- TAB FOR REGION SELECTION ----------------------------------------
        self.regionframe = tk.Frame(self.notebook, bg='white')

        # Bound titles
        lb_title = tk.Label(self.regionframe, text='left bound', bg='white')
        rb_title = tk.Label(self.regionframe, text='right bound', bg='white')
        lnb_title = tk.Label(self.regionframe, text='left noise bound', bg='white')
        rnb_title = tk.Label(self.regionframe, text='right noise bound', bg='white')

        # Scales
        self.lb_scale = tk.Scale(self.regionframe, troughcolor='#cbedcb',
                                 command=self.ud_lb_scale)
        self.rb_scale = tk.Scale(self.regionframe, troughcolor='#cbedcb',
                                 command=self.ud_rb_scale)
        self.lnb_scale = tk.Scale(self.regionframe, troughcolor='#cde6ff',
                                  command=self.ud_lnb_scale)
        self.rnb_scale = tk.Scale(self.regionframe, troughcolor='#cde6ff',
                                  command=self.ud_rnb_scale)

        for scale in [self.lb_scale, self.rb_scale, self.lnb_scale, self.rnb_scale]:
            scale['from'] = 1
            scale['to'] = self.n
            scale['orient'] = tk.HORIZONTAL,
            scale['showvalue'] = 0,
            scale['bg'] = 'white',
            scale['sliderlength'] = 15,
            scale['bd'] = 0,
            scale['highlightthickness'] = 0

        self.lb_scale.set(self.lb)
        self.rb_scale.set(self.rb)
        self.lnb_scale.set(self.lnb)
        self.rnb_scale.set(self.rnb)

        # current values
        self.lb_label = tk.StringVar()
        self.lb_label.set(f'{self.lb_ppm:.3f}')
        self.rb_label = tk.StringVar()
        self.rb_label.set(f'{self.rb_ppm:.3f}')
        self.lnb_label = tk.StringVar()
        self.lnb_label.set(f'{self.lnb_ppm:.3f}')
        self.rnb_label = tk.StringVar()
        self.rnb_label.set(f'{self.rnb_ppm:.3f}')

        self.lb_entry = tk.Entry(self.regionframe, textvariable=self.lb_label)
        self.rb_entry = tk.Entry(self.regionframe, textvariable=self.rb_label)
        self.lnb_entry = tk.Entry(self.regionframe, textvariable=self.lnb_label)
        self.rnb_entry = tk.Entry(self.regionframe, textvariable=self.rnb_label)

        self.lb_entry.bind('<Return>', (lambda event: self.ud_lb_entry()))
        self.rb_entry.bind('<Return>', (lambda event: self.ud_rb_entry()))
        self.lnb_entry.bind('<Return>', (lambda event: self.ud_lnb_entry()))
        self.rnb_entry.bind('<Return>', (lambda event: self.ud_rnb_entry()))

        for entry in [self.lb_entry, self.rb_entry, self.lnb_entry, self.rnb_entry]:
            entry['width'] = 6
            entry['highlightthickness'] = 0

        # organise region selection frame elements
        lb_title.grid(row=0, column=0, padx=(self.pad/2, 0),
                      pady=(self.pad/2, 0), sticky='nsw')
        rb_title.grid(row=1, column=0, padx=(self.pad/2, 0),
                      pady=(self.pad/2, 0), sticky='nsw')
        lnb_title.grid(row=2, column=0, padx=(self.pad/2, 0),
                       pady=(self.pad/2, 0), sticky='nsw')
        rnb_title.grid(row=3, column=0, padx=(self.pad/2, 0),
                       pady=self.pad/2, sticky='nsw')
        self.lb_scale.grid(row=0, column=1, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='ew')
        self.rb_scale.grid(row=1, column=1, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='ew')
        self.lnb_scale.grid(row=2, column=1, padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0), sticky='ew')
        self.rnb_scale.grid(row=3, column=1, padx=(self.pad/2, 0),
                            pady=self.pad/2, sticky='ew')
        self.lb_entry.grid(row=0, column=2, padx=self.pad/2,
                           pady=(self.pad/2, 0), sticky='nsw')
        self.rb_entry.grid(row=1, column=2, padx=self.pad/2,
                           pady=(self.pad/2, 0), sticky='nsw')
        self.lnb_entry.grid(row=2, column=2, padx=self.pad/2,
                            pady=(self.pad/2, 0), sticky='nsw')
        self.rnb_entry.grid(row=3, column=2, padx=self.pad/2, pady=self.pad/2,
                            sticky='nsw')

        # --- TAB FOR PHASE CORRECTION ----------------------------------------
        self.phaseframe = tk.Frame(self.notebook, bg='white')

        # Pivot and phase titles
        pivot_title = tk.Label(self.phaseframe, text='pivot', bg='white')
        p0_title = tk.Label(self.phaseframe, text='p0', bg='white')
        p1_title = tk.Label(self.phaseframe, text='p1', bg='white')

        # scales for pivot, zero order, and first order phases
        self.pivot_scale = tk.Scale(self.phaseframe, troughcolor='#ffb0b0',
                                    command=self.ud_pivot_scale, from_=1,
                                    to=self.n)
        self.p0_scale = tk.Scale(self.phaseframe, resolution=0.0001,
                                 troughcolor='#e0e0e0',
                                 command=self.ud_p0_scale, from_=-np.pi,
                                 to=np.pi)
        self.p1_scale = tk.Scale(self.phaseframe, resolution=0.0001,
                                 troughcolor='#e0e0e0',
                                 command=self.ud_p1_scale, from_=-4*np.pi,
                                 to=4*np.pi)

        for scale in [self.pivot_scale, self.p0_scale, self.p1_scale]:
            scale['orient'] = tk.HORIZONTAL
            scale['bg'] = 'white'
            scale['sliderlength'] = 15
            scale['bd'] = 0
            scale['highlightthickness'] = 0
            scale['relief'] = 'flat'
            scale['showvalue'] = 0

        self.pivot_scale.set(self.pivot)
        self.p0_scale.set(self.p0)
        self.p1_scale.set(self.p1)

        self.pivot_label = tk.StringVar()
        self.pivot_label.set(f'{self.pivot_ppm:.3f}')
        self.p0_label = tk.StringVar()
        self.p0_label.set(f'{self.p0:.3f}')
        self.p1_label = tk.StringVar()
        self.p1_label.set(f'{self.p1:.3f}')

        self.pivot_entry = tk.Entry(self.phaseframe, textvariable=self.pivot_label)
        self.p0_entry = tk.Entry(self.phaseframe, textvariable=self.p0_label)
        self.p1_entry = tk.Entry(self.phaseframe, textvariable=self.p1_label)

        self.pivot_entry.bind('<Return>', (lambda event: self.ud_pivot_entry()))
        self.p0_entry.bind('<Return>', (lambda event: self.ud_p0_entry()))
        self.p1_entry.bind('<Return>', (lambda event: self.ud_p1_entry()))

        for entry in [self.pivot_entry, self.p0_entry, self.p1_entry]:
            entry['width'] = 6
            entry['highlightthickness'] = 0

        # organise phase frame elements
        pivot_title.grid(row=0, column=0, padx=(self.pad/2, 0),
                         pady=(self.pad/2, 0), sticky='w')
        p0_title.grid(row=1, column=0, padx=(self.pad/2, 0),
                      pady=(self.pad/2, 0), sticky='w')
        p1_title.grid(row=2, column=0, padx=(self.pad/2, 0),
                      pady=self.pad/2, sticky='w')
        self.pivot_scale.grid(row=0, column=1, padx=(self.pad/2, 0),
                              pady=(self.pad/2, 0), sticky='ew')
        self.p0_scale.grid(row=1, column=1, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='ew')
        self.p1_scale.grid(row=2, column=1, padx=(self.pad/2, 0),
                           pady=self.pad/2, sticky='ew')
        self.pivot_entry.grid(row=0, column=2, padx=(self.pad/2, self.pad/2),
                              pady=(self.pad/2, 0), sticky='w')
        self.p0_entry.grid(row=1, column=2, padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0), sticky='w')
        self.p1_entry.grid(row=2, column=2, padx=(self.pad/2, self.pad/2),
                           pady=self.pad/2, sticky='w')

        # --- NMR-EsPy LOGO ---------------------------------------------------
        self.logoframe = tk.Frame(self.rightframe, bg='white')

        path = os.path.dirname(nmrespy.__file__)
        image = Image.open(os.path.join(path, 'topspin/images/nmrespy_full.png'))
        scale = 0.08
        [w, h] = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)

        # make img an attribute of the class to prevent garbage collection
        self.img = ImageTk.PhotoImage(image)
        self.logo = tk.Label(self.logoframe, image=self.img, bg='white')
        self.logo.grid(row=0, column=0, padx=self.pad, pady=(self.pad, 0))

        # --- ADVANCED SETTINGS FRAME -----------------------------------------
        self.adsetframe = tk.Frame(self.rightframe, bg='white')

        adset_title = tk.Label(self.adsetframe, text='Advanced Settings',
                               font=('Helvetica', 14), bg='white')

        # Oscillators in intial guess
        mpm_osc_label = tk.Label(self.adsetframe, text='Oscillators for MPM:',
                                  bg='white')

        use_mdl_label = tk.Label(self.adsetframe, text='Use MDL:', bg='white')

        self.mdl = tk.StringVar()
        self.mdl.set('1')
        self.mdl_box = tk.Checkbutton(self.adsetframe, variable=self.mdl,
                                      bg='white', highlightthickness=0, bd=0,
                                      command=self.change_mdl)

        self.osc_num = tk.StringVar()
        self.osc_num.set('')
        self.osc_entry = tk.Entry(self.adsetframe, width=8,
                                  highlightthickness=0,
                                  textvariable=self.osc_num, state='disabled')

        # number of MPM points
        mpm_label = tk.Label(self.adsetframe, text='Points for MPM:',
                             bg='white')
        self.mpm = tk.StringVar()

        if self.n <= 4096:
            self.mpm.set(str(self.n))
        else:
            self.mpm.set('4096')

        self.mpm_entry = tk.Entry(self.adsetframe, width=12,
                                  highlightthickness=0, textvariable=self.mpm)

        maxval = int(np.floor(self.n/2))
        mpm_max_label = tk.Label(self.adsetframe, text=f'Max. value: {maxval}',
                                 bg='white')

        # number of NLP points
        nlp_label = tk.Label(self.adsetframe, text='Points for NLP:', bg='white')
        self.nlp = tk.StringVar()

        if self.n <= 8192:
            self.nlp.set(str(self.n))
        else:
            self.nlp.set('8192')

        self.nlp_entry = tk.Entry(self.adsetframe, width=12,
                                  highlightthickness=0, textvariable=self.nlp)

        nlp_max_label = tk.Label(self.adsetframe, text=f'Max. value: {maxval}',
                                 bg='white')

        # maximum NLP iterations
        maxit_label = tk.Label(self.adsetframe, text='Max. Iterations:',
                               bg='white')

        self.maxit_entry = tk.Entry(self.adsetframe, width=12,
                                    highlightthickness=0)
        self.maxit_entry.insert(0, '100')

        # NLP algorithm
        alg_label = tk.Label(self.adsetframe, text='NLP Method:', bg='white')

        self.algorithm = tk.StringVar(self.adsetframe)
        self.algorithm.set('Trust Region')
        self.algoptions = tk.OptionMenu(self.adsetframe, self.algorithm,
                                        'Trust Region', 'L-BFGS')
        self.algoptions.config(bg='white', borderwidth=0)
        self.algoptions['menu'].configure(bg='white')

        # opt phase variance?
        phasevar_label = tk.Label(self.adsetframe, text='Opt. Phase Variance:',
                                  bg='white')

        self.phasevar = tk.StringVar()
        self.phasevar.set('1')

        self.phasevar_box = tk.Checkbutton(self.adsetframe,
                                           variable=self.phasevar, bg='white',
                                           highlightthickness=0, bd=0)


        adset_title.grid(row=0, column=0, columnspan=3, padx=(self.pad/2, 0),
                         pady=(self.pad/2, 0), sticky='w')
        mpm_osc_label.grid(row=1, column=0, padx=(self.pad/2, 0),
                            pady=(2*self.pad, 0), sticky='w')
        self.osc_entry.grid(row=1, column=1, padx=self.pad/2,
                            pady=(2*self.pad, 0), sticky='w')
        use_mdl_label.grid(row=1, column=2, padx=(self.pad/4, 0),
                          pady=(2*self.pad, 0), sticky='w')
        self.mdl_box.grid(row=1, column=3, padx=(0, self.pad/2),
                          pady=(2*self.pad, 0), sticky='w')
        mpm_label.grid(row=3, column=0, padx=(self.pad/2, 0),
                       pady=(2*self.pad, 0), sticky='nsw')
        self.mpm_entry.grid(row=3, column=1, columnspan=3, padx=(self.pad/2, 0),
                            pady=(2*self.pad, 0), sticky='w')
        mpm_max_label.grid(row=4, column=1, columnspan=3, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='nw')
        nlp_label.grid(row=5, column=0, padx=(self.pad/2, 0),
                       pady=(2*self.pad, 0), sticky='w')
        self.nlp_entry.grid(row=5, column=1, columnspan=3, padx=(self.pad/2, 0),
                            pady=(2*self.pad, 0), sticky='w')
        nlp_max_label.grid(row=6, column=1, columnspan=3, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='nw')
        maxit_label.grid(row=7, column=0, padx=(self.pad/2, 0),
                         pady=(2*self.pad, 0), sticky='w')
        self.maxit_entry.grid(row=7, column=1, columnspan=3,
                              padx=(self.pad/2, 0), pady=(2*self.pad, 0),
                              sticky='w')
        alg_label.grid(row=8, column=0, padx=(self.pad/2, 0),
                       pady=(2*self.pad, 0), sticky='w')
        self.algoptions.grid(row=8, column=1, columnspan=3,
                             padx=(self.pad/2, 0), pady=(2*self.pad, 0),
                             sticky='w')
        phasevar_label.grid(row=9, column=0, padx=(self.pad/2, 0),
                            pady=(2*self.pad, 0), sticky='w')
        self.phasevar_box.grid(row=9, column=1, columnspan=3,
                               padx=(self.pad/2, 0), pady=(2*self.pad, 0),
                               sticky='w')

        # --- BUTTON FRAME ----------------------------------------------------
        self.buttonframe = tk.Frame(self.rightframe, bg='white')

        self.cancel_button = tk.Button(self.buttonframe, text='Cancel',
                                       command=self.cancel, bg='#ff9894')
        self.help_button = tk.Button(self.buttonframe, text='Help',
                                     command=self.load_help, bg='#ffb861')
        self.run_button = tk.Button(self.buttonframe, text='Run',
                                    command=self.run, bg='#9eda88')

        for button in [self.cancel_button, self.help_button, self.run_button]:
            button['highlightbackground'] = 'black'
            button['width'] = 6

        self.cancel_button.grid(row=0, column=0, padx=(self.pad, 0))
        self.help_button.grid(row=0, column=1, padx=(self.pad, 0))
        self.run_button.grid(row=0, column=2, padx=self.pad)

        # --- CONTACT FRAME ---------------------------------------------------
        self.contactframe = tk.Frame(self.rightframe, bg='white')
        feedback = tk.Label(self.contactframe,
                            text='For queries/feedback, contact', bg='white')
        email = tk.Label(self.contactframe, text='simon.hulse@chem.ox.ac.uk',
                         font='Courier', bg='white')
        feedback.grid(row=0, column=0, sticky='w', padx=(self.pad, 0),
                      pady=(self.pad,0))
        email.grid(row=1, column=0, sticky='w', padx=(self.pad, 0),
                   pady=(0, self.pad))

        # --- ORGANISE FRAMES -------------------------------------------------
        # main window
        self.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # left and right frames
        self.leftframe.grid(column=0, row=0, sticky='nsew')
        self.rightframe.grid(column=1, row=0, sticky='nsew')

        # leftframe always fills any extra space upon scaling
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # leftframe contents
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')
        self.toolbar.grid(column=0, row=1, sticky='e')
        self.notebook.grid(column=0, row=2, padx=(self.pad,0), pady=self.pad,
                            sticky='ew')

        # add region selection and phase correction frames to notebook
        self.notebook.add(self.regionframe, text='Region Selection',
                            sticky='ew')
        self.notebook.add(self.phaseframe, text='Phase Correction',
                            sticky='ew')

        # All leftframe widgets expand horizontally
        self.leftframe.columnconfigure(0, weight=1)
        # Only spectrum plot expands vertically
        self.leftframe.rowconfigure(0, weight=1)

        # only allow scales to expand/contract when x-dimension of GUI changed
        self.regionframe.columnconfigure(1, weight=1)
        self.phaseframe.columnconfigure(1, weight=1)


        # rightframe contents
        self.logoframe.grid(row=0, column=0, sticky='ew')
        self.adsetframe.grid(row=1, column=0, sticky='ew')
        self.buttonframe.grid(row=2, column=0, sticky='ew')
        self.contactframe.grid(row=3, column=0, sticky='ew')

        # ad. settings frame acquires any extra space if scaling occurs
        # note that as sticky = 'new', spacing between ad. set. & logo will
        # remain constant. Space will be filled between ad. set. & buttons
        self.rightframe.rowconfigure(1, weight=1)


    def dtype_popup(self):
        self.master.withdraw()
        self.dtypewindow = DataType(self, self.fidpath, self.pdatapath)

    def ud_plot(self):
        tab = self.notebook.index(self.notebook.select())

        # region selection tab selected
        if tab == 0:
            self.filtregion.set_alpha(1)
            self.noiseregion.set_alpha(1)
            self.pivotplot.set_alpha(0)
            self.canvas.draw_idle()

        # phase correction tab selected
        elif tab == 1:
            self.filtregion.set_alpha(0)
            self.noiseregion.set_alpha(0)
            self.pivotplot.set_alpha(1)
            self.canvas.draw_idle()

    def ud_lb_scale(self, lb):
        lb = int(lb.split('.')[0])
        if lb < self.rb:
            self.lb = lb
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.filtregion.set_bounds(self.rb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       40*self.ylim_init[1])
            self.lb_label.set(f'{self.lb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lb = self.rb - 1
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.lb_scale.set(self.lb)
            self.canvas.draw_idle()

    def ud_lb_entry(self):
        lb_ppm = float(self.lb_label.get())
        lb = _misc.conv_ppm_idx(lb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')

        if lb < self.rb:
            self.lb = lb
            self.lb_ppm = lb_ppm
            self.filtregion.set_bounds(self.rb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       40*self.ylim_init[1])
            self.lb_scale.set(self.lb)
            self.canvas.draw_idle()
        else:
            self.lb_label.set(f'{self.lb_ppm:.3f}')

    def ud_rb_scale(self, rb):
        rb = int(rb.split('.')[0])
        if rb > self.lb:
            self.rb = rb
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.filtregion.set_bounds(self.rb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       40*self.ylim_init[1])
            self.rb_label.set(f'{self.rb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rb = self.lb + 1
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.rb_scale.set(self.rb)
            self.canvas.draw_idle()

    def ud_rb_entry(self):
        rb_ppm = float(self.rb_label.get())
        rb = _misc.conv_ppm_idx(rb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')
        if rb > self.lb:
            self.rb = rb
            self.rb_ppm = rb_ppm
            self.filtregion.set_bounds(self.rb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       40*self.ylim_init[1])
            self.rb_scale.set(self.rb)
            self.canvas.draw_idle()
        else:
            self.rb_label.set(f'{self.rb_ppm:.3f}')

    def ud_lnb_scale(self, lnb):
        lnb = int(lnb.split('.')[0])
        if lnb < self.rnb:
            self.lnb = lnb
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.noiseregion.set_bounds(self.rnb_ppm,
                                        -20*self.ylim_init[1],
                                        self.lnb_ppm - self.rnb_ppm,
                                        40*self.ylim_init[1])
            self.lnb_label.set(f'{self.lnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lnb = self.rnb - 1
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.lnb_scale.set(self.lnb)
            self.canvas.draw_idle()

    def ud_lnb_entry(self):
        lnb_ppm = float(self.lnb_label.get())
        lnb = _misc.conv_ppm_idx(lnb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')
        if lnb < self.rnb:
            self.lnb = lnb
            self.lnb_ppm = lnb_ppm
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       40*self.ylim_init[1])
            self.lnb_scale.set(self.lnb)
            self.canvas.draw_idle()
        else:
            self.lnb_label.set(f'{self.lnb_ppm:.3f}')

    def ud_rnb_scale(self, rnb):
        rnb = int(rnb.split('.')[0])
        if rnb > self.lnb:
            self.rnb = rnb
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       40*self.ylim_init[1])
            self.rnb_label.set(f'{self.rnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rnb = self.lnb + 1
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.rnb_scale.set(self.rnb)
            self.canvas.draw_idle()

    def ud_rnb_entry(self):
        rnb_ppm = float(self.rnb_label.get())
        rnb = _misc.conv_ppm_idx(rnb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')
        if rnb > self.lb:
            self.rnb = rnb
            self.rnb_ppm = rnb_ppm
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -20*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       40*self.ylim_init[1])
            self.rnb_scale.set(self.rnb)
            self.canvas.draw_idle()
        else:
            self.rnb_label.set(f'{self.rnb_ppm:.3f}')


    def ud_pivot_scale(self, pivot):

        self.pivot = int(pivot)
        self.pivot_ppm = _misc.conv_ppm_idx(self.pivot, self.sw_p, self.off_p,
                                            self.n, direction='idx->ppm')
        self.pivot_label.set(f'{self.pivot_ppm:.3f}')
        self.ud_phase()
        x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
        self.pivotplot.set_xdata(x)

        self.canvas.draw_idle()


    def ud_pivot_entry(self):

        try:
            self.pivot_ppm = float(self.pivot_label.get())
            self.pivot = _misc.conv_ppm_idx(self.pivot_ppm,
                                            self.sw_p,
                                            self.off_p,
                                            self.n,
                                            direction='ppm->idx')
            self.ud_phase()
            x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
            self.pivotplot.set_xdata(x)
            self.pivot_scale.set(self.pivot)
            self.canvas.draw_idle()

        except:
            self.pivot_label.set(f'{self.pivot_ppm:.3f}')

    def ud_p0_scale(self, p0):
        self.p0 = float(p0)
        self.p0_label.set(f'{self.p0:.3f}')
        self.ud_phase()

    def ud_p0_entry(self):
        try:
            self.p0 = float(self.p0_label.get())
            self.p0_scale.set(self.p0)
            self.ud_phase()

        except:
            self.p0_label.set(f'{self.p0:.3f}')

    def ud_p1_scale(self, p1):
        self.p1 = float(p1)
        self.p1_label.set(f'{self.p1:.3f}')
        self.ud_phase()

    def ud_p1_entry(self):
        try:
            self.p1 = float(self.p1_label.get())
            self.p1_scale.set(self.p1)
            self.ud_phase()

        except:
            self.p1_label.set(f'{self.p1:.3f}')


    def ud_phase(self):

        newspec = np.real(self.spec * np.exp(1j * (self.p0 + (self.p1 * \
        np.arange(-self.pivot, -self.pivot + self.n, 1) / self.n))))

        self.specplot.set_ydata(newspec)
        self.canvas.draw_idle()


    def change_mdl(self):
        if self.mdl.get() == '1':
            self.osc_entry['state'] = 'disabled'
        elif self.mdl.get() == '0':
            self.osc_entry['state'] = 'normal'


    def cancel(self):
        exit()

    def load_help(self):
        webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')

    def run(self):
        p0 = float(self.p0_entry.get())
        p1 = float(self.p1_entry.get())
        pivot_ppm = float(self.pivot_entry.get())
        pivot = _misc.conv_ppm_idx(pivot_ppm, self.sw_p, self.off_p, self.n,
                                   direction='ppm->idx')
        p0 - (p1 * pivot / self.n)

        lb = float(self.lb_entry.get()),
        rb = float(self.rb_entry.get()),
        lnb = float(self.lnb_entry.get()),
        rnb = float(self.rnb_entry.get()),

        mdl = self.mdl.get()
        if mdl == '1':
            M = 0
        else:
            M = int(self.osc_entry.get())

        mpm_points = int(self.mpm_entry.get())
        nlp_points = int(self.nlp_entry.get())
        maxiter = int(self.maxit_entry.get())
        alg = self.algorithm.get()
        pv = self.phasevar.get()
        self.master.destroy()

        if alg == 'Trust Region':
            alg = 'trust_region'
        elif alg == 'L-BFGS':
            alg = 'lbfgs'

        if pv == '1':
            pv = True
        else:
            pv = False

        self.info.virtual_echo(highs=lb, lows=rb, highs_n=lnb, lows_n=rnb,
                               p0=p0, p1=p1)
        self.info.matrix_pencil(trim=mpm_points, M_in=M)
        self.info.nonlinear_programming(trim=nlp_points, maxit=maxiter,
                                        method=alg, phase_variance=pv)

        tmpdir = os.path.join(os.path.dirname(nmrespy.__file__), 'topspin/tmp')

        self.info.pickle_save(fname='tmp.pkl', dir=tmpdir,
                              force_overwrite=True)


def ud_plot(cnt):
    """Reconstructs the result plot after a change to the oscillators is
    made

    Parameters
    ----------

    cnt : nmrespy.topspin.ResultApp
        controller
    """

    # get new lines and labels
    # also obtaining ax to acquire new y-limits
    _, ax, lines, labels = cnt.info.plot_result()

    # wipe lines and text instances from the axis
    cnt.ax.lines = []
    cnt.ax.texts = []
    # wipe lines and text labels from the controller
    cnt.lines = {}
    cnt.labels = {}

    # plot data line onto axis
    cnt.lines['data'] = cnt.ax.plot(lines['data'].get_xdata(),
                                    lines['data'].get_ydata(),
                                    color=lines['data'].get_color(),
                                    lw=lines['data'].get_lw())

    # plot oscillator lines and add oscillator text labels
    # append these to the lines and labels attributes of the controller
    lines_and_labels = zip(list(lines.values())[1:], list(labels.values()))
    for i, (line, label) in enumerate(lines_and_labels):

        key = f'osc{i+1}'
        # plot oscillator
        cnt.lines[key] = cnt.ax.plot(line.get_xdata(), line.get_ydata(),
                                     color=line.get_color(), lw=line.get_lw())
        # add oscillator label
        x, y = label.get_position()
        cnt.labels[key] = cnt.ax.text(x, y, label.get_text())

    # update y-limits to fit new lines
    cnt.ax.set_xlim(ax.get_xlim())
    cnt.ax.set_ylim(ax.get_ylim())
    # draw the new plot!
    cnt.frames['PlotFrame'].canvas.draw_idle()


class ResultApp(tk.Tk):
    """App for dealing with result of estimation. Enables user to tweak
    oscillators (and re-run the optimisation with the updated parameter array),
    customise the final plot, and save results."""

    def __init__(self, info):
        tk.Tk.__init__(self)
        # main container: everything goes inside here
        container = tk.Frame(self, bg='white')
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # left frame will contain the PlotFrame and navigation toolbar
        leftframe = tk.Frame(container, bg='white')
        leftframe.grid(column=0, row=0, sticky='nsew')
        # leftframe expands/contracts if user adjusts window size
        leftframe.rowconfigure(0, weight=1)
        leftframe.columnconfigure(0, weight=1)

        # right frame contains all widgets for editing/saving
        rightframe = tk.Frame(container, bg='white')
        rightframe.grid(column=1, row=0, sticky='nsew')
        rightframe.rowconfigure(2, weight=1)

        # instance of nmrespy.core.NMREsPyBruker
        self.info = info

        # plot result
        self.fig, self.ax, self.lines, self.labels = self.info.plot_result()
        # edit figure resolution and size
        self.fig.set_dpi(170)
        self.fig.set_size_inches(6, 3.5)

        # restrict x-axis to spectral window of interest
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        Restrictor(self.ax, x=lambda x: x<= xlim[0]) # restrict left
        Restrictor(self.ax, x=lambda x: x>= xlim[1]) # restrict right

        # contain all frames inside dictionary (makes it easy to acquire
        # attributes whilst in different classes)
        self.frames = {}

        left = True
        # append all frames to the window
        for F in (PlotFrame, LogoFrame, EditFrame):
            frame_name = F.__name__
            if left:
                frame = F(parent=leftframe, controller=self)
                left = False
            else:
                frame = F(parent=rightframe, controller=self)

            self.frames[frame_name] = frame

        # --- RESULT PLOT ------------------------------------------------------
        # self.plotframe = PlotFrame(self, self.fig)
        # self.logoframe = LogoFrame(self.rightframe)
        # self.editframe = EditFrame(self.rightframe, self.lines, self.labels,
        #                            self.ax, self.plotframe.canvas, self.info)
        # self.saveframe = SaveFrame(self.rightframe)
        # self.buttonframe = ButtonFrame(self.rightframe, self.saveframe,
        #                                self.info)
        # self.contactframe = ContactFrame(self.rightframe)
        #
        # self.logoframe.columnconfigure(0, weight=1)


class PlotFrame(tk.Frame):
    """Contains the result plot, along with navigation toolbar"""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller # ResultApp instance
        self['bg'] = 'white'
        self.grid(row=0, column=0, sticky='nsew')

        # make canvas (see below) expandable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.controller.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        # construct navigation toolbar
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.toolbar['bg'] = 'white'
        self.toolbar._message_label['bg'] = 'white'
        for button in self.toolbar.winfo_children():
            button['bg'] = 'white'
        self.toolbar.grid(column=0, row=1, sticky='e')


class LogoFrame(tk.Frame):
    """Contains the NMR-EsPy logo (who doesn't like a bit of publicity)"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller # ResultApp instance
        self['bg'] = 'white'
        self.grid(row=0, column=0, sticky='ew')

        # path to NMR-EsPy PNG
        path = os.path.join(os.path.dirname(nmrespy.__file__),
                            'topspin/images/nmrespy_full.png')
        # construct Label with image
        self.img = get_PhotoImage(path, 0.08)
        logo = tk.Label(self, image=self.img, bg='white')
        logo.grid(row=0, column=0, padx=10, pady=10)


class EditFrame(tk.Frame):
    # TODO
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        self['bg'] = 'white'
        self.grid(row=1, column=0, sticky='ew')
        self.columnconfigure(0, weight=1)

        self.editoptions = tk.Label(self, text='Edit Options', bg='white',
                                    font=('Helvetica', 14))
        self.edit_params = tk.Button(self, text='Edit Parameters',
                                     command=self.config_params, width=12,
                                     bg='#e0e0e0', highlightbackground='black')
        self.edit_lines = tk.Button(self, text='Edit Lines',
                                    command=self.config_lines, width=12,
                                    bg='#e0e0e0', highlightbackground='black')
        self.edit_labels = tk.Button(self, text='Edit Labels',
                                     command=self.config_labels, width=12,
                                     bg='#e0e0e0', highlightbackground='black')

        self.editoptions.grid(row=0, column=0, sticky='w')
        self.edit_params.grid(row=1, column=0)
        self.edit_lines.grid(row=2, column=0)
        self.edit_labels.grid(row=3, column=0)

    def config_params(self):
        EditParams(self, self.controller)

    def config_lines(self):
        pass
        # ConfigLines(self, self.controller)

    def config_labels(self):
        pass
        # ConfigLabels(self, self.controller)


class SaveFrame(tk.Frame):
    #TODO
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.config(bg='white')
        self.grid(row=2, column=0, sticky='new')
        self.columnconfigure(0, weight=1)

        self.saveoptions = tk.Label(self, text='Save Options', bg='white',
                                    font=('Helvetica', 14))
        self.descrip_lab = tk.Label(self, text='Description:', bg='white')
        self.descrip_box = tk.Text(self, height=4, width=20)
        self.fname_lab = tk.Label(self, text='Filename:', bg='white')
        self.fname_box = tk.Entry(self, width=20, highlightthickness=0)
        self.fname_box.insert(0, 'NMREsPy_result')
        self.dir_lab = tk.Label(self, text='Directory:', bg='white')
        self.dir = tk.StringVar()
        self.dir.set(os.path.expanduser('~'))
        self.dir_box = tk.Entry(self, width=16, text=self.dir, highlightthickness=0)

        path = os.path.join(os.path.dirname(nmrespy.__file__),
                            'topspin/images/folder_icon.png')
        self.img = get_PhotoImage(path, 0.02)

        self.dir_but = tk.Button(self, command=self.browse, bg='white',
                                 highlightbackground='black', image=self.img)
        self.txt_lab = tk.Label(self, text='Save Textfile:', bg='white')
        self.txt = tk.StringVar()
        self.txt.set('1')
        self.txt_box = tk.Checkbutton(self, variable=self.txt, bg='white',
                                      highlightthickness=0, bd=0)

        self.pdf_lab = tk.Label(self, text='Save PDF:', bg='white')
        self.pdf = tk.StringVar()
        self.pdf.set('0')
        self.pdf_box = tk.Checkbutton(self, variable=self.pdf, bg='white',
                                      highlightthickness=0, bd=0)

        self.pickle_lab = tk.Label(self, text='Pickle Result:', bg='white')
        self.pickle = tk.StringVar()
        self.pickle.set('1')
        self.pickle_box = tk.Checkbutton(self, variable=self.pickle, bg='white',
                                         highlightthickness=0, bd=0)

        self.saveoptions.grid(row=0, column=0, columnspan=3, sticky='w')
        self.descrip_lab.grid(row=1, column=0, sticky='nw')
        self.descrip_box.grid(row=1, column=1, columnspan=2, sticky='w')
        self.fname_lab.grid(row=2, column=0, sticky='w')
        self.fname_box.grid(row=2, column=1, columnspan=2, sticky='w')
        self.dir_lab.grid(row=3, column=0, sticky='w')
        self.dir_box.grid(row=3, column=1, sticky='w')
        self.dir_but.grid(row=3, column=2, sticky='w')
        self.txt_lab.grid(row=4, column=0, sticky='w')
        self.txt_box.grid(row=4, column=1, columnspan=2)
        self.pdf_lab.grid(row=5, column=0, sticky='w')
        self.pdf_box.grid(row=5, column=1, columnspan=2)
        self.pickle_lab.grid(row=6, column=0, sticky='w')
        self.pickle_box.grid(row=6, column=1, columnspan=2)

    def browse(self):
        self.dir = filedialog.askdirectory(initialdir=os.path.expanduser('~'))
        self.dir_box.delete(0, 'end')
        self.dir_box.insert(0, self.dir)


class ButtonFrame(tk.Frame):
    #TODO
    def __init__(self, master, sframe, info):
        tk.Frame.__init__(self, master)

        self.sframe = sframe
        self.info = info

        self.config(bg='white')
        self.grid(row=3, column=0, sticky='new')
        self.columnconfigure(0, weight=1)

        self.cancel_but = tk.Button(self, text='Cancel', width=8,
                                    bg='#ff9894', highlightbackground='black',
                                    command=self.cancel)
        self.help_but = tk.Button(self, text='Help', width=8, bg='#ffb861',
                                  highlightbackground='black', command=self.help)
        self.rerun_but = tk.Button(self, text='Re-run', width=8, bg='#f9f683',
                                   highlightbackground='black', command=self.re_run)
        self.save_but = tk.Button(self, text='Save', width=8, bg='#9eda88',
                                  highlightbackground='black', command=self.save)

        self.cancel_but.grid(row=0, column=0)
        self.help_but.grid(row=0, column=1)
        self.rerun_but.grid(row=0, column=2)
        self.save_but.grid(row=0, column=3)

    def cancel(self):
        exit()

    def help(self):
        import webbrowser
        webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')

    def re_run(self):
        print('TODO')

    def save(self):
        descrip = self.sframe.descrip_box.get('1.0', tk.END)
        file = self.sframe.fname_box.get()
        dir = self.sframe.dir_box.get()
        txt = self.sframe.txt.get()
        pdf = self.sframe.pdf.get()
        pickle = self.sframe.pickle.get()

        if txt == '1':
            self.info.write_result(descrip=descrip, fname=file, dir=dir,
                                   force_overwrite=True)
        if pdf == '1':
            self.info.write_result(descrip=descrip, fname=file, dir=dir,
                                   force_overwrite=True, format='pdf')
        if pickle == '1':
            self.info.pickle_save(fname=file, dir=dir, force_overwrite=True)

        exit()


class ContactFrame(tk.Frame):
    #TODO
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.config(bg='white')
        self.grid(row=4, column=0, sticky='new')
        self.columnconfigure(0, weight=1)

        self.msg1 = tk.Label(self, text='For queries/feedback, contact',
                             bg='white')
        self.msg2 = tk.Label(self, text='simon.hulse@chem.ox.ac.uk',
                             font='Courier', bg='white')

        self.msg1.grid(row=0, column=0, sticky='w')
        self.msg2.grid(row=1, column=0, sticky='w')


class EditParams(tk.Toplevel):
    """Window allowing user to edit the estimation result."""

    def __init__(self, parent, controller):

        tk.Toplevel.__init__(self, parent)
        self.controller = controller
        self['bg'] = 'white'
        self.resizable(False, False)

        # frame to contain the table of parameters
        self.table = tk.Frame(self, bg='white')
        self.table.grid(row=0, column=0)
        # generate table inside frame
        self.construct_table()
        # frame to contain various buttons: mrege, split, manual edit, close
        self.buttonframe = tk.Frame(self, bg='white')
        self.buttonframe.grid(row=1, column=0, sticky='e')

        # split selected oscillator
        self.splitbutton = tk.Button(self.buttonframe, text='Split Oscillator',
                                     highlightbackground='black',
                                     state='disabled', command=self.split)
        # merge selected oscillators
        self.mergebutton = tk.Button(self.buttonframe, text='Merge Oscillators',
                                     highlightbackground='black',
                                     state='disabled', command=self.merge)
        # manually edit parameters associated with oscillator
        self.manualbutton = tk.Button(self.buttonframe, text='Edit Manually',
                                      highlightbackground='black',
                                      state='disabled',
                                      command=self.manual_edit)
        # close window
        self.closebutton = tk.Button(self.buttonframe, text='Close',
                                     highlightbackground='black',
                                     command=self.destroy)


        self.splitbutton.grid(row=0, column=0, sticky='e', padx=10, pady=10)
        self.mergebutton.grid(row=0, column=1, sticky='e', padx=(0,10), pady=10)
        self.manualbutton.grid(row=0, column=2, sticky='e', padx=(0,10),
                               pady=10)
        self.closebutton.grid(row=0, column=4, sticky='e', padx=(0,10),
                              pady=(10,10))


    def construct_table(self):
        # column titles
        osc = tk.Label(self.table, bg='white', text='#')
        amp = tk.Label(self.table, bg='white', text='Amplitude')
        phase = tk.Label(self.table, bg='white', text='Phase')
        freq = tk.Label(self.table, bg='white', text='Frequency')
        damp = tk.Label(self.table, bg='white', text='Damping')

        osc.grid(row=0, column=0, ipadx=10, pady=(10,0))
        amp.grid(row=0, column=1, padx=(5,0), pady=(10,0), sticky='w')
        phase.grid(row=0, column=2, padx=(5,0), pady=(10,0), sticky='w')
        freq.grid(row=0, column=3, padx=(5,0), pady=(10,0), sticky='w')
        damp.grid(row=0, column=4, padx=(5,10), pady=(10,0), sticky='w')

        # store oscillator labels, entry widgets, and string variables
        self.table_labels = [] # tk.Label instances (M elements)
        self.table_entries = [] # tk.Entry instances (M x 4 elements)
        self.table_variables = [] # tk.StringVar instances (M x 4 elements)

        for i, oscillator in enumerate(self.controller.info.get_theta()):

            # first column: oscillator number
            label = tk.Label(self.table, bg='white', text=f'{i + 1}')
            # left click: highlight row (unselect all others)
            label.bind('<Button-1>', lambda entry, i=i: self.left_click(i))

            # left click with shift: highlight row, keep other selected rows
            # highlighted
            label.bind('<Shift-Button-1>',
                         lambda entry, i=i: self.shift_left_click(i))

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
                entry = tk.Entry(self.table, textvariable=var, state='disabled',
                                 readonlybackground='#cde6ff', bg='white',
                                 width=14)

                # conditions affect how entry widget is padded
                if j == 0:
                    entry.grid(row=i+1, column=j+1, pady=(5,0))
                elif j in [1, 2]:
                    entry.grid(row=i+1, column=j+1, padx=(5,0),pady=(5,0))
                else: # j = 3
                    entry.grid(row=i+1, column=j+1, padx=(5,10), pady=(5,0))

                entry_row.append(entry)

            self.table_entries.append(entry_row)
            self.table_variables.append(variable_row)


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
            if i == index:
                pass
            # disable all rows that do not match the index
            else:
                if label['bg'] == 'blue':
                    label['bg'] = 'white'
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

        fg = 'white'
        bg = 'blue'
        state = 'readonly'

        # if row is already selected, change parameters so it becomes
        # unselected
        if self.table_labels[index]['fg'] == 'white':
            fg = 'black'
            bg = 'white'
            state = 'disabled'

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
            self.splitbutton['state'] = 'disabled'
            self.mergebutton['state'] = 'disabled'
            self.manualbutton['state'] = 'disabled'

        # activate split and manual edit buttons
        # deactivate merge button (can't merge one oscillator...)
        elif activated_number == 1:
            self.splitbutton['state'] = 'normal'
            self.mergebutton['state'] = 'disabled'
            self.manualbutton['state'] = 'normal'

        # activate merge button
        # deactivate split and manual edit buttons (ambiguous with multiple
        # oscillators selected)
        else: # activated_number > 1
            self.splitbutton['state'] = 'disabled'
            self.mergebutton['state'] = 'normal'
            self.manualbutton['state'] = 'disabled'


    def merge(self):
        """Merges all selected oscillators into a single oscillator.
        Mereging is performed using
        :py:meth:`~nmrespy.core.NMREsPyBruker.merge_oscillators`.
        """

        indices = self.get_selected_indices()
        self.controller.info.merge_oscillators(indices)

        # update the plot
        ud_plot(self.controller)

        # destroy and reconstruct the data table to match the new theta
        for widget in self.table.winfo_children():
            widget.destroy()
        self.construct_table()
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
        SplitFrame(self, self.controller, i)

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
                                  font=('Helvetica', 10))
        self.udbutton.pack(fill=tk.BOTH, expand=1, side=tk.LEFT, pady=(2,0))

        self.cancelbutton = tk.Button(self.tmpframe, text='Cancel', width=3,
                                    bg='#ff9894', highlightbackground='black',
                                    command=lambda i=i: self.cancel_manual(i),
                                    font=('Helvetica', 10))
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
                                f'{self.controller.info.theta[i][j]:.5f}')

            msg = 'At least one of the parameter values specified could not' \
                  + ' be converted to a numerical value!'
            WarnFrame(self, msg)
            return

        # replace oscillator with user input
        self.controller.info.theta[i] = oscillator
        # sort oscillators in order of frequency
        self.controller.info.theta = \
        self.controller.info.theta[np.argsort(
                                    self.controller.info.theta[..., 2])]

        # remove temporary buton frame
        self.tmpframe.destroy()
        # update plot and parameter table
        ud_plot(self.controller)
        self.construct_table()
        self.activate_buttons()

    def cancel_manual(self, i):
        """Cancel manually chaning oscillator parameters."""

        # remove temporary buton frame
        self.tmpframe.destroy()

        # replace contents of entry widgets with previous values in theta
        # set entry widgets back to read-only mode
        for j in range(4):
            self.table_variables[i][j].set(
                            f'{self.controller.info.theta[i][j]:.5f}')
        # recontruct table (all rows will become deselected)
        self.construct_table()
        self.activate_buttons()

    def get_selected_indices(self):
        """Determine the indices of the rows which are selected."""

        indices = []
        for i, label in enumerate(self.table_labels):
            if label['bg'] == 'blue':
                indices.append(i)

        return indices


class SplitFrame(tk.Toplevel):
    """Window for specifying how to split a certain oscillator.
    Opened after calling :py:meth:`EditParams.split`."""

    def __init__(self, parent, controller, index):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        # row selected
        self.index = index
        self['bg'] = 'white'
        self.resizable(False, False)

        # create text labels (title, number of oscillators, frequency
        # separatio, amplitude ratio)
        title = tk.Label(self, bg='white',
                         text=f'Splitting Oscillator {index + 1}:',
                         font=('Helvetica', 12, 'bold'))
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
        ppm = float(self.freq_hz.get()) / self.controller.info.get_sfo()[0]
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
            ppm = float(self.freq_hz.get()) / self.controller.info.get_sfo()[0]
            self.freq_ppm.set(f'{ppm:.5f}')
        # if Hz input could not be understood as a numerical value, clear
        # ppm string variable
        except:
            self.freq_ppm.set('')

    def key_press_ppm(self, entry):
        """Update the value of the Hz variable upon change to ppm."""
        try:
            hz = float(self.freq_ppm.get()) * self.controller.info.get_sfo()[0]
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

        self.controller.info.split_oscillator(self.index,
                                              frequency_sep=frequency_sep,
                                              split_number=number,
                                              amp_ratio=ratio)
        # update the plot
        ud_plot(self.controller)
        # reconstruct the parameter table with the updated oscillators
        self.parent.construct_table()
        self.destroy()


class ConfigLines(tk.Toplevel):
    def __init__(self, master, lines, figcanvas):
        tk.Toplevel.__init__(self, master)
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
    def __init__(self, master, lines, number, figcanvas):
        tk.Frame.__init__(self, master)

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
                               font=('Helvetica', 12))

        self.lw_scale = tk.Scale(self.lwframe, from_=0, to=2.5,
                                 orient=tk.HORIZONTAL,
                                 showvalue=0, bg='white', sliderlength=15, bd=0,
                                 troughcolor=f'#808080', highlightthickness=0,
                                 command=self.ud_lw_sc, resolution=0.001)
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
                               font=('Helvetica', 12))

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
        self.master.destroy()


class LineMultiEdit(tk.Frame):
    def __init__(self, master, lines, figcanvas):
        tk.Frame.__init__(self, master)
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
                                  bg='white', font=('Helvetica', 12))
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
                               font=('Helvetica', 12))

        self.lw_scale = tk.Scale(self.lwframe, from_=0, to=5, orient=tk.HORIZONTAL,
                                 showvalue=0, bg='white', sliderlength=15, bd=0,
                                 troughcolor=f'#808080', length=500,
                                 highlightthickness=0, command=self.ud_lw_sc,
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
                               font=('Helvetica', 12))

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
        self.master.destroy()


class ColorPicker(tk.Frame):
    def __init__(self, master, init_color):
        tk.Frame.__init__(self, master)

        self['bg'] = 'white'

        self.color = init_color
        self.r = self.color[1:3] # red hex
        self.g = self.color[3:5] # green hex
        self.b = self.color[5:] # blue hex

        self.topframe = tk.Frame(self, bg='white')
        self.botframe = tk.Frame(self, bg='white')
        self.topframe.grid(row=0, column=0, sticky='ew', padx=10, pady=(5,0))
        self.botframe.grid(row=1, column=0, sticky='e', padx=10, pady=(10,0))
        self.columnconfigure(0, weight=1)

        # --- FRAME 1: Color scales and entries  ------------------------------
        self.color_ttl = tk.Label(self.topframe, text='Color', bg='white',
                                  font=('Helvetica', 12))
        self.r_lab = tk.Label(self.topframe, text='R', fg='#ff0000', bg='white')
        self.g_lab = tk.Label(self.topframe, text='G', fg='#008000', bg='white')
        self.b_lab = tk.Label(self.topframe, text='B', fg='#0000ff', bg='white')

        self.r_scale = tk.Scale(self.topframe, from_=0, to=255,
                                orient=tk.HORIZONTAL, showvalue=0, bg='white',
                                sliderlength=15, bd=0,
                                troughcolor=f'#{self.r}0000', length=500,
                                highlightthickness=0,
                                command=lambda r: self.ud_r_sc(r))

        self.g_scale = tk.Scale(self.topframe, from_=0, to=255,
                                orient=tk.HORIZONTAL, showvalue=0, bg='white',
                                sliderlength=15, bd=0,
                                troughcolor=f'#00{self.g}00', length=500,
                                highlightthickness=0, command=self.ud_g_sc)

        self.b_scale = tk.Scale(self.topframe, from_=0, to=255,
                                orient=tk.HORIZONTAL, showvalue=0, bg='white',
                                sliderlength=15, bd=0,
                                troughcolor=f'#0000{self.b}', length=500,
                                highlightthickness=0,
                                command=self.ud_b_sc)

        self.r_scale.set(int(self.r, 16))
        self.g_scale.set(int(self.g, 16))
        self.b_scale.set(int(self.b, 16))

        self.r_ent = tk.Entry(self.topframe, bg='white', text=f'{self.r.upper()}',
                              width=3, highlightthickness=0)
        self.g_ent = tk.Entry(self.topframe, bg='white', text=f'{self.g.upper()}',
                              width=3, highlightthickness=0)
        self.b_ent = tk.Entry(self.topframe, bg='white', text=f'{self.b.upper()}',
                              width=3, highlightthickness=0)

        self.r_ent.bind('<Return>', (lambda event: self.ud_r_ent()))
        self.g_ent.bind('<Return>', (lambda event: self.ud_g_ent()))
        self.b_ent.bind('<Return>', (lambda event: self.ud_b_ent()))

        self.swatch = tk.Canvas(self.topframe, width=40, height=40, bg='white')
        self.rect = self.swatch.create_rectangle(0, 0, 40, 40, fill=self.color)

        self.color_ttl.grid(row=0, column=0, columnspan=4, sticky='w')
        self.r_lab.grid(row=1, column=0, sticky='w', pady=(5,0))
        self.g_lab.grid(row=2, column=0, sticky='w', pady=(5,0))
        self.b_lab.grid(row=3, column=0, sticky='w', pady=(5,0))
        self.r_scale.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=(5,0))
        self.g_scale.grid(row=2, column=1, sticky='ew', padx=(10,0), pady=(5,0))
        self.b_scale.grid(row=3, column=1, sticky='ew', padx=(10,0), pady=(5,0))
        self.r_ent.grid(row=1, column=2, sticky='w', padx=(10,0), pady=(5,0))
        self.g_ent.grid(row=2, column=2, sticky='w', padx=(10,0), pady=(5,0))
        self.b_ent.grid(row=3, column=2, sticky='w', padx=(10,0), pady=(5,0))
        self.swatch.grid(row=1, rowspan=3, column=3, padx=(10,0))

        self.topframe.columnconfigure(1, weight=1)

        # --- FRAME 2: Matplotlib color entry ---------------------------------
        self.mplcol_lab = tk.Label(self.botframe, text='Matplotlib color:',
                                   bg='white')
        self.mplcol_ent = tk.Entry(self.botframe, bg='white', width=20,
                                   highlightthickness=0)
        self.mplcol_ent.bind('<Return>', (lambda event: self.ud_mplcol_ent()))

        self.mplcol_lab.grid(row=0, column=0, sticky='e')
        self.mplcol_ent.grid(row=0, column=1, sticky='e', padx=(10,0))


    def ud_r_sc(self, r, dynamic=False, object=None, figcanvas=None):
        self.r = '{:02x}'.format(int(r))
        self.color = '#' + self.r + self.g + self.b
        self.r_scale['troughcolor'] = f'#{self.r}0000'
        self.r_ent.delete(0, tk.END)
        self.r_ent.insert(0, self.r.upper())
        self.swatch.itemconfig(self.rect, fill=self.color)

        if dynamic:
            object.set_color(self.color)
            figcanvas.draw_idle()


    def ud_g_sc(self, g, dynamic=False, object=None, figcanvas=None):
        self.g = '{:02x}'.format(int(g))
        self.color = '#' + self.r + self.g + self.b
        self.g_scale['troughcolor'] = f'#00{self.g}00'
        self.g_ent.delete(0, tk.END)
        self.g_ent.insert(0, self.g.upper())
        self.swatch.itemconfig(self.rect, fill=self.color)

        if dynamic:
            object.set_color(self.color)
            figcanvas.draw_idle()

    def ud_b_sc(self, b, dynamic=False, object=None, figcanvas=None):
        self.b = '{:02x}'.format(int(b))
        self.color = '#' + self.r + self.g + self.b
        self.b_scale['troughcolor'] = f'#0000{self.b}'
        self.b_ent.delete(0, tk.END)
        self.b_ent.insert(0, self.b.upper())
        self.swatch.itemconfig(self.rect, fill=self.color)

        if dynamic:
            object.set_color(self.color)
            figcanvas.draw_idle()

    def ud_r_ent(self):
        value = self.r_ent.get()
        valid_hex = self.check_hex(value)

        if valid_hex:
            self.r = value.lower()
            self.color = '#' + self.r + self.g + self.b
            self.r_scale.set(int(value, 16))
            self.r_scale['troughcolor'] = f'#{self.r}0000'
            self.r_ent.delete(0, tk.END)
            self.r_ent.insert(0, self.r.upper())
            self.swatch.itemconfig(self.rect, fill=self.color)

        else:
            self.r_ent.delete(0, tk.END)
            self.r_ent.insert(0, self.r.upper())

    def ud_g_ent(self):
        value = self.g_ent.get()
        valid_hex = self.check_hex(value)

        if valid_hex:
            self.g = value.lower()
            self.color = '#' + self.r + self.g + self.b
            self.g_scale.set(int(value, 16))
            self.g_scale['troughcolor'] = f'#00{self.g}00'
            self.g_ent.delete(0, tk.END)
            self.g_ent.insert(0, self.g.upper())
            self.swatch.itemconfig(self.rect, fill=self.color)

        else:
            self.g_ent.delete(0, tk.END)
            self.g_ent.insert(0, self.g.upper())

    def ud_b_ent(self):
        value = self.b_ent.get()
        valid_hex = self.check_hex(value)

        if valid_hex:
            self.b = value.lower()
            self.color = '#' + self.r + self.g + self.b
            self.b_scale.set(int(value, 16))
            self.b_scale['troughcolor'] = f'#0000{self.b}'
            self.b_ent.delete(0, tk.END)
            self.b_ent.insert(0, self.b.upper())
            self.swatch.itemconfig(self.rect, fill=self.color)

        else:
            self.b_ent.delete(0, tk.END)
            self.b_ent.insert(0, self.b.upper())


    @staticmethod
    def check_hex(value):
        if len(value) == 2:
            try:
                hex_ = int(value, 16)
                return True
            except:
                return False

        return False


    def ud_mplcol_ent(self):
        value = self.mplcol_ent.get()
        self.mplcol_ent.delete(0, tk.END)

        try:
            self.color = mcolors.to_hex(value)
            self.r = self.color[1:3]
            self.g = self.color[3:5]
            self.b = self.color[5:]
            self.r_scale.set(int(self.r, 16))
            self.g_scale.set(int(self.g, 16))
            self.b_scale.set(int(self.b, 16))
            self.r_ent.delete(0, tk.END)
            self.r_ent.insert(0, self.r.upper())
            self.g_ent.delete(0, tk.END)
            self.g_ent.insert(0, self.g.upper())
            self.b_ent.delete(0, tk.END)
            self.b_ent.insert(0, self.b.upper())

        except:
            pass


class ConfigLabels(tk.Toplevel):
    def __init__(self, master, labels, ax, figcanvas):
        tk.Toplevel.__init__(self, master)

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
    def __init__(self, master, labels, number, ax, figcanvas):
        tk.Frame.__init__(self, master)

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
                                font=('Helvetica', 12))

        self.txt_ent = tk.Entry(self.txtframe, bg='white', width=10,
                                highlightthickness=0)
        self.txt_ent.insert(0, self.txt)

        self.txt_ent.bind('<Return>', (lambda event: self.ud_txt_ent()))

        self.txt_ttl.grid(row=0, column=0, sticky='w')
        self.txt_ent.grid(row=0, column=1, sticky='w')

        # --- LABEL SIZE ----------------------------------------------
        self.size_ttl = tk.Label(self.sizeframe, text='Label Size', bg='white',
                                font=('Helvetica', 12))

        self.size_scale = tk.Scale(self.sizeframe, from_=1, to=48,
                                   orient=tk.HORIZONTAL, showvalue=0,
                                   bg='white', sliderlength=15, bd=0,
                                   troughcolor=f'#808080', length=500,
                                   highlightthickness=0,
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
                                bg='white', font=('Helvetica', 12))

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.x_ttl = tk.Label(self.posbotframe, text='x', bg='white')

        self.x_scale = tk.Scale(self.posbotframe, from_=xlim[0], to=xlim[1],
                                orient=tk.HORIZONTAL, showvalue=0,
                                bg='white', sliderlength=15, bd=0,
                                troughcolor=f'#808080', length=500,
                                highlightthickness=0,
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
                                highlightthickness=0,
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
        self.master.destroy()

# Tweaked using most upvoted answer as template:
# https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python
class ToolTip:

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x1, y1, x2, y2 = self.widget.bbox("insert")
        x1 = x1 + self.widget.winfo_rootx() + 57
        y1 = y1 + y2 + self.widget.winfo_rooty() +27
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(1)
        self.tipwindow.wm_geometry(f'+{x1}+{y1}')
        label = tk.Label(self.tipwindow, text=self.text, justify=tk.LEFT,
                      background="#f0f0f0", relief=tk.SOLID, borderwidth=1,
                      font=('Helvetica', '8'), wraplength=400)
        label.pack(ipadx=1, ipady=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


if __name__ == '__main__':
    # path to nmrespy directory
    espypath = os.path.dirname(nmrespy.__file__)

    # # extract path information
    # infopath = os.path.join(espypath, 'topspin/tmp/info.txt')
    # try:
    #     with open(infopath, 'r') as fh:
    #         from_topspin = fh.read().split(' ')
    # except:
    #     raise IOError(f'No file of path {infopath} found')
    #
    # # import dictionary of spectral info
    # fidpath = from_topspin[0]
    # pdatapath = from_topspin[1]
    #
    # # determine the data type to consider (FID or pdata)
    # root = tk.Tk()
    # root.title('NMR-EsPy - Calculation Setup')
    # root.rowconfigure(0, weight=1)
    # root.columnconfigure(0, weight=1)
    # window = Setup(root, fidpath, pdatapath)
    # root.mainloop()

    try:
        tmpdir = os.path.join(espypath, 'topspin/tmp')
        info = load.pickle_load('tmp.pkl', tmpdir)
        # os.remove(os.path.join(tmpdir, 'tmp.pkl'))
    except FileNotFoundError:
        exit()

    # load the result GUI
    res_app = ResultApp(info)
    res_app.mainloop()

    # construct save files
    descrip = res_app.descrip
    file = res_app.file
    dir = res_app.dir

    txt = res_app.txt
    pdf = res_app.pdf
    pickle = res_app.pickle


    if txt == '1':
        info.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True)
    if pdf == '1':
        info.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True, format='pdf')
    if pickle == '1':
        info.pickle_save(fname=file, dir=dir, force_overwrite=True)
