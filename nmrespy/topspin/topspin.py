#!/usr/bin/env python3
# Sorry for the gross violation of PEP 8 Guido
# I think keeping commands on one line is more readable in this case

import os
import shutil
from copy import copy

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift

import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import filedialog
from PIL import ImageTk, Image

import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import nmrespy
import nmrespy.load as load
import nmrespy._misc as _misc

def nmrespy_image(scale):
    """Generates the NMR-EsPy logo in appropriate format for tkinter use,
    with option of scaling"""
    espypath = os.path.dirname(nmrespy.__file__)
    espy_image = Image.open(os.path.join(espypath, 'topspin/images/nmrespy_full.png'))
    [w, h] = espy_image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    espy_image = espy_image.resize((new_w,new_h), Image.ANTIALIAS)
    return ImageTk.PhotoImage(espy_image)



class dtypeGUI(tk.Toplevel):
    """GUI for asking user whether they want to analyse the raw FID or
    pdata assocaited with the opened data"""
    def __init__(self, master, fidpath, pdatapath):

        self.master = master
        self.master.title('NMR-EsPy - Data Selection')
        self.master.resizable(0, 0)
        self.master['bg'] = 'white'

        # --- FRAMES ----------------------------------------------------------
        self.logoframe = tk.Frame(self.master,
                                  bg='white')
        self.logoframe.grid(row=0,
                            column=0,
                            rowspan=2)

        self.mainframe = tk.Frame(self.master,
                                  bg='white')
        self.mainframe.grid(row=0,
                            column=1)

        self.buttonframe = tk.Frame(self.master,
                                    bg='white')
        self.buttonframe.grid(row=1,
                              column=1,
                              sticky='e')

        self.pad = 10

        # -- NMR-EsPy LOGO ----------------------------------------------------
        self.espypath = os.path.dirname(nmrespy.__file__)
        self.nmrespy_image = Image.open(os.path.join(self.espypath,
                                        'topspin/images/nmrespy_full.png'))

        scale = 0.07
        [w, h] = self.nmrespy_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.nmrespy_image = self.nmrespy_image.resize((new_w, new_h),
                                                       Image.ANTIALIAS)
        self.nmrespy_img = ImageTk.PhotoImage(self.nmrespy_image)

        self.nmrespy_logo = tk.Label(self.logoframe,
                                      image=self.nmrespy_img,
                                      bg='white')

        self.nmrespy_logo.pack(padx=self.pad,
                               pady=self.pad)

        # --- LABELS AND CHECKBOXES -------------------------------------------
        self.message = tk.Label(self.mainframe,
                                 text='Which data would you like to analyse?',
                                 font=('Helvetica', '14'),
                                 bg='white')
        self.message.grid(column=0,
                          row=0,
                          columnspan=2,
                          padx=self.pad,
                          pady=(self.pad, 0))

        self.pdata_label = tk.Label(self.mainframe,
                                     text='Processed Data',
                                     bg='white')
        self.pdata_label.grid(column=0,
                              row=1,
                              padx=(self.pad, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.pdatapath = tk.Label(self.mainframe,
                                   text=f'{pdatapath}/1r',
                                   font='Courier',
                                   bg='white')
        self.pdatapath.grid(column=0,
                            row=2,
                            padx=(self.pad, 0),
                            sticky='w')

        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = tk.Checkbutton(self.mainframe,
                                         variable=self.pdata,
                                         command=self.click_pdata,
                                         bg='white',
                                         highlightthickness=0,
                                         bd=0)
        self.pdata_box.grid(column=1,
                            row=1,
                            rowspan=2,
                            padx=self.pad,
                            sticky='nsw')

        self.fid_label = tk.Label(self.mainframe,
                                     text='Raw FID',
                                     bg='white')
        self.fid_label.grid(column=0,
                              row=3,
                              padx=(self.pad, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.fidpath = tk.Label(self.mainframe,
                                   text=f'{fidpath}/fid',
                                   font='Courier',
                                   bg='white')
        self.fidpath.grid(column=0,
                            row=4,
                            padx=(self.pad, 0),
                            sticky='w')

        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = tk.Checkbutton(self.mainframe,
                                       variable=self.fid,
                                       command=self.click_fid,
                                       bg='white',
                                       highlightthickness=0,
                                       bd=0)
        self.fid_box.grid(column=1,
                          row=3,
                          rowspan=2,
                          padx=self.pad,
                          sticky='nsw')

        # --- BUTTONS ---------------------------------------------------------
        self.confirmbutton = tk.Button(self.buttonframe,
                                        text='Confirm',
                                        command=self.confirm,
                                        width=8,
                                        bg='#9eda88',
                                        highlightbackground='black')

        self.confirmbutton.grid(column=1,
                                row=0,
                                padx=(self.pad/2, self.pad),
                                pady=(self.pad, self.pad),
                                sticky='e')

        self.cancelbutton = tk.Button(self.buttonframe,
                                        text='Cancel',
                                        command=self.cancel,
                                        width=8,
                                        bg='#ff9894',
                                        highlightbackground='black')
        self.cancelbutton.grid(column=0,
                                row=0,
                                pady=(self.pad, self.pad),
                                sticky='e')

    # --- COMMANDS ------------------------------------------------------------
    # click_fid and click_pdata ensure the checkbuttons are mutually exclusive
    # (only one selected at any time)
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
        self.master.destroy()
        print('NMR-EsPy Cancelled :\'(')
        exit()

    def confirm(self):
        if self.fid.get() == 1:
            self.dtype = 'fid'
        elif self.pdata.get() == 1:
            self.dtype = 'pdata'
        self.master.destroy()


class CustomNavigationToolbar(NavigationToolbar2Tk):
    """Tweak default matplotlib navigation bar to exclude subplot-config
    and save buttons. Also remove co-ordiantes as cursor goes over plot"""
    def __init__(self, canvas_, parent_):
        self.toolitems = self.toolitems[:6]
        NavigationToolbar2Tk.__init__(self, canvas_, parent_)

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



class NMREsPyGUI:
    """Main GUI: Phasing and region selection + configuration of routine"""
    def __init__(self, master, info):

        self.master = master
        self.master.title('NMR-ESPY - Calculation Setup')

        # --- EXTRACT SPECTRAL DATA -------------------------------------------
        self.info = info
        self.dtype = self.info.get_dtype()

        if self.dtype == 'raw':
            # raw FID needs to be foFouriere transformed and flipped
            self.spec = np.flip(fftshift(fft(self.info.get_data())))
        elif self.dtype == 'pdata':
            self.spec = self.info.get_data(pdata_key='1r') \
                        + 1j * self.info.get_data(pdata_key='1i')

        # --- EXTRACT BASIC EXPERIMENT INFO -----------------------------------
        self.shifts = info.get_shifts(unit='ppm')[0]
        self.sw_p = info.get_sw(unit='ppm')[0]
        self.off_p = info.get_offset(unit='ppm')[0]
        self.n = self.spec.shape[0]

        # --- LEFT AND RIGHT BOUNDS -------------------------------------------
        self.lb = int(np.floor(7 * self.n / 16))
        self.rb = int(np.floor(9 * self.n / 16))
        self.lnb = int(np.floor(1 * self.n / 16))
        self.rnb = int(np.floor(2 * self.n / 16))
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

        # --- PADDING VALUE ---------------------------------------------------
        self.pad = 10

        # --- FRAMES ----------------------------------------------------------
        # leftframe -> spectrum plot and region scales
        self.leftframe = tk.Frame(self.master, bg='white')
        self.leftframe.grid(column=0, row=0, sticky='nsew')

        # rightframe -> logo, advanced settings, save/help/quit buttons,
        #               contact info.
        self.rightframe = tk.Frame(self.master, bg='white')
        self.rightframe.grid(column=1, row=0, sticky='nsew')

        # plotframe -> contains mpl plot
        self.plotframe = tk.Frame(self.leftframe, bg='white')
        self.plotframe.grid(column=0, row=0, padx=(self.pad,0),
                            pady=(self.pad,0), sticky='nsew')

        # toolbarframe ->
        self.toolbarframe = tk.Frame(self.plotframe, bg='white')
        self.toolbarframe.grid(row=1, column=0, sticky='e')


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
        self.scaleframe = ttk.Notebook(self.leftframe)
        self.scaleframe.grid(column=0, row=1, padx=(self.pad,0), pady=self.pad,
                             sticky='ew')

        # regionframe -> contains scales to enable user to select spectral region
        self.regionframe = tk.Frame(self.scaleframe, bg='white')

        # phaseframe -> enables user to phase data before submitting
        self.phaseframe = tk.Frame(self.scaleframe, bg='white')


        self.scaleframe.bind('<<NotebookTabChanged>>', self.ud_plot)
        self.scaleframe.add(self.regionframe, text='Region Selection',
                            sticky='ew')
        self.scaleframe.add(self.phaseframe, text='Phase Correction',
                            sticky='ew')

        # logoframe -> contains NMR-EsPy logo
        self.logoframe = tk.Frame(self.rightframe, bg='white')
        self.logoframe.grid(column=0, row=0, padx=self.pad, pady=(self.pad, 0),
                            sticky='ew')

        # adsetframe -> customise features of the optimisation procedure
        self.adsetframe = tk.Frame(self.rightframe, bg='white',
                                   highlightbackground='black',
                                   highlightthickness=2)
        self.adsetframe.grid(column=0, row=1, padx=self.pad, pady=(self.pad, 0),
                             sticky='ew')

        # buttonframe -> cancel/help/save & run buttons
        self.buttonframe = tk.Frame(self.rightframe, bg='white')
        self.buttonframe.grid(column=0, row=2, padx=self.pad,
                              pady=(self.pad, 0))

        # contactframe -> contact details
        self.contactframe = tk.Frame(self.rightframe, bg='white')
        self.contactframe.grid(column=0, row=3, padx=self.pad, pady=self.pad,
                               sticky='sw')

        # --- FRAME CONFIGURATION ---------------------------------------------
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        self.leftframe.grid_columnconfigure(0, weight=1)
        self.leftframe.grid_rowconfigure(0, weight=1)
        self.rightframe.grid_rowconfigure(1, weight=1)
        self.plotframe.grid_columnconfigure(0, weight=1)
        self.plotframe.grid_rowconfigure(0, weight=1)
        self.logoframe.grid_columnconfigure(0, weight=1)
        self.logoframe.grid_rowconfigure(0, weight=1)
        self.regionframe.columnconfigure(1, weight=1)
        self.phaseframe.columnconfigure(1, weight=1)

        # --- NMR-EsPy LOGO ---------------------------------------------------
        self.espypath = os.path.dirname(nmrespy.__file__)
        self.nmrespy_image = Image.open(os.path.join(self.espypath,
                                        'topspin/images/nmrespy_full.png'))

        scale = 0.08
        [w, h] = self.nmrespy_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.nmrespy_image = self.nmrespy_image.resize((new_w, new_h),
                                                       Image.ANTIALIAS)
        self.nmrespy_img = ImageTk.PhotoImage(self.nmrespy_image)

        self.nmrespy_logo = tk.Label(self.logoframe, image=self.nmrespy_img,
                                     bg='white')

        self.nmrespy_logo.pack(padx=self.pad, pady=self.pad)

        # --- SPECTRUM PLOT ---------------------------------------------------
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

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plotframe)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.canvas.draw()

        # custom mpl toolar, lacking save button and subplot-config manager
        self.toolbar = CustomNavigationToolbar(self.canvas, self.toolbarframe)
        self.toolbar.config(background='white')
        self.toolbar._message_label.config(bg='white')
        for button in self.toolbar.winfo_children():
            button.config(bg='white')

        # restrict zooming and panning into regions beyond the spectral window
        restrict1 = Restrictor(self.ax, x=lambda x: x<= self.shifts[0])
        restrict2 = Restrictor(self.ax, x=lambda x: x>= self.shifts[-1])


        # --- REGION SELECTION ------------------------------------------------
        rows = range(4)
        # Bound titles
        lb_title = tk.Label(self.regionframe, text='left bound', bg='white')
        rb_title = tk.Label(self.regionframe, text='right bound', bg='white')
        lnb_title = tk.Label(self.regionframe, text='left noise bound', bg='white')
        rnb_title = tk.Label(self.regionframe, text='right noise bound', bg='white')

        titles = [lb_title, rb_title, lnb_title, rnb_title]
        for title, row in zip(titles, rows):
            title.grid(row=row, column=0, padx=(self.pad/2, 0),
                       pady=(self.pad/2, 0), sticky='nsw')
        rnb_title.grid(row=row, column=0, padx=(self.pad/2, 0), pady=self.pad/2,
                       sticky='nsw')

        # Scales
        self.lb_scale = tk.Scale(self.regionframe)
        self.rb_scale = tk.Scale(self.regionframe)
        self.lnb_scale = tk.Scale(self.regionframe)
        self.rnb_scale = tk.Scale(self.regionframe)

        scales = [self.lb_scale, self.rb_scale, self.lnb_scale, self.rnb_scale]
        values = [self.lb, self.rb, self.lnb, self.rnb]
        colors = ['#cbedcb', '#cbedcb', '#cde6ff', '#cde6ff']
        commands = [self.ud_lb_scale, self.ud_rb_scale, self.ud_lnb_scale,
                    self.ud_rnb_scale]
        for scale, value, color, command, row in zip(scales, values, colors, commands, rows):
            scale['from'] = 1
            scale['to'] = self.n
            scale['orient'] = tk.HORIZONTAL,
            scale['showvalue'] = 0,
            scale['bg'] = 'white',
            scale['sliderlength'] = 15,
            scale['bd'] = 0,
            scale['highlightthickness'] = 0
            scale['command'] = command
            scale['troughcolor'] = color
            scale.set(value)
            scale.grid(row=row, column=1, padx=(self.pad/2, 0),
                       pady=(self.pad/2, 0), sticky='ew')
        self.rnb_scale.grid(row=row, column=1, padx=(self.pad/2, 0),
                            pady=self.pad/2, sticky='ew')

        # current values
        self.lb_label = tk.StringVar()
        self.rb_label = tk.StringVar()
        self.lnb_label = tk.StringVar()
        self.rnb_label = tk.StringVar()

        self.lb_entry = tk.Entry(self.regionframe)
        self.rb_entry = tk.Entry(self.regionframe)
        self.lnb_entry = tk.Entry(self.regionframe)
        self.rnb_entry = tk.Entry(self.regionframe)

        labels = [self.lb_label, self.rb_label, self.lnb_label, self.rnb_label]
        values_ppm = [self.lb_ppm, self.rb_ppm, self.lnb_ppm, self.rnb_ppm]
        entries = [self.lb_entry, self.rb_entry, self.lnb_entry, self.rnb_entry]
        commands = [self.ud_lb_entry, self.ud_rb_entry, self.ud_lnb_entry,
                    self.ud_rnb_entry]

        for label, value, entry, command, row in zip(labels, values_ppm, entries, commands, rows):
            label.set(f'{value:.3f}')
            entry['textvariable'] = label
            entry['width'] = 6
            entry['highlightthickness'] = 0
            entry.bind('<Return>', (lambda event: command()))
            entry.grid(row=row, column=2, padx=self.pad/2, pady=(self.pad/2, 0),
                       sticky='nsw')
        self.rnb_entry.grid(row=row, column=2, padx=self.pad/2, pady=self.pad/2,
                            sticky='nsw')


        # --- PHASE CORRECTION ------------------------------------------------
        rows = range(3)
        # Pivot
        self.pivot_title = tk.Label(self.phaseframe, text='pivot', bg='white')
        self.p0_title = tk.Label(self.phaseframe, text='p0', bg='white')
        self.p1_title = tk.Label(self.phaseframe, text='p1', bg='white')

        titles = [self.pivot_title, self.p0_title, self.p1_title]
        for title, row in zip(titles, rows):
            title.grid(row=row, column=0, padx=(self.pad/2, 0),
                       pady=(self.pad/2, 0), sticky='w')
        self.p1_title.grid(row=row, column=0, padx=(self.pad/2, 0),
                           pady=self.pad/2, sticky='w')

        self.pivot_scale = tk.Scale(self.phaseframe)
        self.p0_scale = tk.Scale(self.phaseframe)
        self.p1_scale = tk.Scale(self.phaseframe)

        scales = [self.pivot_scale, self.p0_scale, self.p1_scale]
        values = [self.pivot, self.p0, self.p1]
        colors = ['#ffb0b0', '#e0e0e0', '#e0e0e0']
        commands = [self.ud_pivot_scale, self.ud_p0_scale, self.ud_p1_scale]
        froms = [1, -np.pi, -4 * np.pi]
        tos = [self.n, np.pi, 4 * np.pi]

        for scale, value, color, command, from_, to, row in zip(scales, values, colors, commands, froms, tos, rows):
            scale['from'] = from_
            scale['to'] = to
            scale['orient'] = tk.HORIZONTAL
            scale['command'] = command
            scale['bg'] = 'white'
            scale['troughcolor'] = color
            scale['sliderlength'] = 15
            scale['bd'] = 0
            scale['highlightthickness'] = 0
            scale['relief'] = 'flat'
            scale.set(value)
            scale.grid(row=row, column=1, padx=(self.pad/2, 0),
                       pady=(self.pad/2, 0), sticky='ew')
        self.p1_scale.grid(row=row, column=1, padx=(self.pad/2, 0),
                           pady=self.pad/2, sticky='ew')

        self.pivot_label = tk.StringVar()
        self.p0_label = tk.StringVar()
        self.p1_label = tk.StringVar()

        self.pivot_entry = tk.Entry(self.phaseframe)
        self.p0_entry = tk.Entry(self.phaseframe)
        self.p1_entry = tk.Entry(self.phaseframe)

        labels = [self.pivot_label, self.p0_label, self.p1_label]
        values[0] = self.pivot_ppm
        commands = [self.ud_pivot_entry, self.ud_p0_entry,
                    self.ud_p1_entry]
        entries = [self.pivot_entry, self.p0_entry, self.p1_entry]

        for label, value, command, entry, row in zip(labels, values, commands, entries, rows):
            print(value)
            label.set(f'{value:.3f}')
            entry['textvariable'] = label
            entry['width'] = 6
            entry['highlightthickness'] = 0
            entry.bind('<Return>', (lambda event: command()))
            entry.grid(row=row, column=2, padx=(self.pad/2, self.pad/2),
                       pady=(self.pad/2, 0), sticky='w')
        self.p1_entry.grid(row=row, column=2, padx=(self.pad/2, self.pad/2),
                           pady=self.pad/2, sticky='w')

        # --- ADVANCED SETTINGS -----------------------------------------------
        self.adset_title = tk.Label(self.adsetframe, text='Advanced Settings',
                                    font=('Helvetica', 14), bg='white')
        self.adset_title.grid(row=0, column=0, columnspan=3,
                              padx=(self.pad/2, 0), pady=(self.pad/2, 0),
                              sticky='w')

        # number of points to consider in ITMPM
        self.mpm_label = tk.Label(self.adsetframe, text='Points for MPM:',
                                  bg='white')
        self.mpm_label.grid(row=1, column=0, padx=(self.pad/2, 0),
                            pady=(self.pad, 0), sticky='nsw')

        self.n_mpm = tk.Entry(self.adsetframe, width=12, highlightthickness=0)

        if self.n <= 4096:
            self.n_mpm.insert(0, str(self.n))
        else:
            self.n_mpm.insert(0, '4096')

        self.n_mpm.grid(row=1, column=1, columnspan=2, padx=(self.pad/2, 0),
                        pady=(self.pad, 0), sticky='w')

        maxval = int(np.floor(self.n/2))
        self.mpm_max_label = tk.Label(self.adsetframe,
                                      text=f'Max. value: {maxval}', bg='white')
        self.mpm_max_label.grid(row=2, column=1, columnspan=2,
                                padx=(self.pad/2, 0), pady=(self.pad/2, 0),
                                sticky='nw')

        # number of points to consider in NLP
        self.nlp_label = tk.Label(self.adsetframe, text='Points for NLP:',
                                  bg='white')
        self.nlp_label.grid(row=3, column=0, padx=(self.pad/2, 0),
                            pady=(self.pad, 0), sticky='w')

        self.n_nlp = tk.Entry(self.adsetframe, width=12, highlightthickness=0)

        if self.n <= 8192:
            self.n_nlp.insert(0, str(self.n))
        else:
            self.n_nlp.insert(0, '8192')

        self.n_nlp.grid(row=3, column=1, columnspan=2, padx=(self.pad/2, 0),
                        pady=(self.pad, 0), sticky='w')

        self.nlp_max_label = tk.Label(self.adsetframe,
                                      text=f'Max. value: {maxval}', bg='white')
        self.nlp_max_label.grid(row=4, column=1, columnspan=2,
                                padx=(self.pad/2, 0), pady=(self.pad/2, 0),
                                sticky='nw')

        # maximum number of NLP iterations
        self.maxit_label = tk.Label(self.adsetframe, text='Max. Iterations:',
                                    bg='white')
        self.maxit_label.grid(row=5, column=0, padx=(self.pad/2, 0),
                              pady=(self.pad, 0), sticky='w')

        self.maxiter = tk.Entry(self.adsetframe, width=12, highlightthickness=0)
        self.maxiter.insert(0, '100')
        self.maxiter.grid(row=5, column=1, columnspan=2, padx=(self.pad/2, 0),
                          pady=(self.pad, 0), sticky='w')

        # NLP algorithm
        self.alg_label = tk.Label(self.adsetframe, text='NLP Method:',
                                  bg='white')
        self.alg_label.grid(row=6, column=0, padx=(self.pad/2, 0),
                            pady=(self.pad, 0), sticky='w')

        self.algorithm = tk.StringVar(self.adsetframe)
        self.algorithm.set('Trust Region')
        self.algoptions = tk.OptionMenu(self.adsetframe, self.algorithm,
                                        'Trust Region', 'L-BFGS')

        self.algoptions.config(bg='white', borderwidth=0)
        self.algoptions['menu'].configure(bg='white')

        self.algoptions.grid(row=6, column=1, columnspan=2,
                             padx=(self.pad/2, 0), pady=(self.pad, 0))

        # Phase Variance
        self.phasevar_label = tk.Label(self.adsetframe,
                                       text='Inc. Phase Variance:', bg='white')
        self.phasevar_label.grid(row=7, column=0, padx=(self.pad/2, 0),
                                 pady=(self.pad, 0), sticky='w')

        self.phasevar = tk.StringVar()
        self.phasevar.set('1')

        self.phasevar_box = tk.Checkbutton(self.adsetframe,
                                           variable=self.phasevar, bg='white',
                                           highlightthickness=0, bd=0)
        self.phasevar_box.grid(row=7, column=1, columnspan=2,
                               padx=(self.pad/2, 0), pady=(self.pad, 0),
                               sticky='w')

        # --- SAVE/HELP/QUIT BUTTONS ------------------------------------------
        self.cancel_button = tk.Button(self.buttonframe, text='Cancel',
                                       command=self.cancel, width=8,
                                       bg='#ff9894',
                                       highlightbackground='black')
        self.cancel_button.grid(row=0, column=0)

        self.help_button = tk.Button(self.buttonframe, text='Help',
                                     command=self.load_help, width=8,
                                     bg='#ffb861',
                                     highlightbackground='black')
        self.help_button.grid(row=0, column=1, padx=(self.pad, 0))

        self.save_button = tk.Button(self.buttonframe, text='Save & Run',
                                     command=self.save, width=8,
                                     bg='#9eda88',
                                     highlightbackground='black')
        self.save_button.grid(row=0, column=2, padx=self.pad)

        self.feedback = tk.Label(self.contactframe,
                                 text='For queries/feedback, contact',
                                 bg='white')
        self.feedback.grid(row=0, column=0, sticky='w')

        self.email = tk.Label(self.contactframe,
                              text='simon.hulse@chem.ox.ac.uk', font='Courier',
                              bg='white')
        self.email.grid(row=1, column=0, sticky='w')

    #TODO: these functions feel a bit longwinded. Could probably
    # achieve the same behaviour with fewer, more general ones
    # not particularly high priority though...

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

    def ud_plot(self, _):
        tab = self.scaleframe.index(self.scaleframe.select())

        # region selection tab selected
        if tab == 0:
            self.filtregion.set_alpha(1)
            self.noiseregion.set_alpha(1)
            self.pivotplot.set_alpha(0)

        # phase correction tab selected
        elif tab == 1:
            self.filtregion.set_alpha(0)
            self.noiseregion.set_alpha(0)
            self.pivotplot.set_alpha(1)

        self.canvas.draw_idle()

    def load_help(self):
        import webbrowser
        webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')

    def save(self):
        self.p0 = self.p0 - (self.p1 * self.pivot / self.n)
        self.mpm_points = int(self.n_mpm.get())
        self.nlp_points = int(self.n_nlp.get())
        self.maxiter = int(self.maxiter.get())
        self.alg = self.algorithm.get()
        self.pv = self.phasevar.get()
        self.master.destroy()

    def cancel(self):
        self.master.destroy()
        print("NMR-ESPY Cancelled :'(")
        exit()



class ResultGUI:

    def __init__(self, master, info):

        self.master = master
        self.master.title('NMR-EsPy - Result Settings')
        self.master['bg'] = 'white'
        self.info = info
        self.pad = 10

        # --- FRAMES ----------------------------------------------------------
        # leftframe -> spectrum plot and mpl toolbar
        self.leftframe = tk.Frame(self.master, bg='white')
        self.leftframe.grid(column=0,
                            row=0,
                            sticky='nsew')

        # rightframe -> logo, edit and save options, save/quit buttons
        self.rightframe = tk.Frame(self.master, bg='white')

        self.rightframe.grid(column=1,
                             row=0,
                             sticky='nsew')

        self.plotframe = tk.Frame(self.leftframe, bg='white')

        self.plotframe.grid(column=0,
                            row=0,
                            padx=(self.pad, 0),
                            pady=(self.pad, 0),
                            sticky='nsew')

        self.toolbarframe = tk.Frame(self.plotframe, bg='white')
        self.toolbarframe.grid(row=1,
                               column=0,
                               sticky='e')

        self.logoframe = tk.Frame(self.rightframe, bg='white')

        self.logoframe.grid(column=0,
                            row=0,
                            padx=(self.pad, self.pad),
                            pady=(self.pad, 0),
                            sticky='ew')

        self.editframe = tk.Frame(self.rightframe,
                                  bg='white',
                                  highlightbackground='black',
                                  highlightthickness=2)

        self.editframe.grid(column=0,
                            row=1,
                            padx=(self.pad, self.pad),
                            pady=(self.pad, 0),
                            sticky='ew')

        self.saveframe = tk.Frame(self.rightframe,
                                  bg='white',
                                  highlightbackground='black',
                                  highlightthickness=2)

        self.saveframe.grid(column=0,
                            row=2,
                            padx=(self.pad, self.pad),
                            pady=(self.pad, 0),
                            sticky='ew')

        self.buttonframe = tk.Frame(self.rightframe, bg='white')

        self.buttonframe.grid(column=0,
                              row=3,
                              padx=(self.pad, self.pad),
                              pady=(self.pad, 0))

        self.contactframe = tk.Frame(self.rightframe, bg='white')

        self.contactframe.grid(column=0,
                               row=4,
                               padx=(self.pad, self.pad),
                               pady=(self.pad, self.pad),
                               sticky='sw')

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.leftframe.rowconfigure(0, weight=1)
        self.leftframe.columnconfigure(0, weight=1)
        self.rightframe.grid_rowconfigure(1, weight=1)
        self.plotframe.columnconfigure(0, weight=1)
        self.plotframe.rowconfigure(0, weight=1)
        self.editframe.columnconfigure(0, weight=1)


        # --- RESULT PLOT -----------------------------------------------------
        self.fig, self.ax, self.lines, self.labels = self.info.plot_result()
        self.fig.set_dpi(170)
        self.fig.set_size_inches(6, 3.5)

        self.xlim = self.ax.get_xlim()

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plotframe)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        # custom mpl toolar, lacking save button and subplot-config manager
        self.toolbar = CustomNavigationToolbar(self.canvas, self.toolbarframe)
        self.toolbar.config(background='white')
        self.toolbar._message_label.config(bg='white')
        for button in self.toolbar.winfo_children():
            button.config(bg='white')

        # restrict zooming and panning into regions beyond the specified region
        # see Restrictor class above
        self.restrict1 = Restrictor(self.ax,
                                    x=lambda x: x<= self.xlim[0])
        self.restrict2 = Restrictor(self.ax,
                                    x=lambda x: x>= self.xlim[1])

        # --- NMR-EsPy LOGO ---------------------------------------------------
        self.nmrespy_img = get_logo(0.08)
        self.nmrespy_logo = tk.Label(self.logoframe,
                                      image=self.nmrespy_img,
                                      bg='white')
        self.nmrespy_logo.pack(pady=(0, self.pad))

        # --- EDIT BUTTONS ----------------------------------------------------
        self.editoptions = tk.Label(self.editframe,
                                    text='Edit Options',
                                    bg='white',
                                    font=('Helvetica', 14))
        self.editoptions.grid(row=0,
                              column=0,
                              columnspan=3,
                              padx=(self.pad/2, 0),
                              sticky='w')

        self.edit_params = tk.Button(self.editframe,
                                     text='Edit Parameters',
                                     command=lambda x: print('TODO'),
                                     width=12,
                                     bg='#e0e0e0',
                                     highlightbackground='black')

        self.edit_params.grid(column=0,
                              row=1,
                              padx=self.pad/2,
                              pady=(self.pad/2, 0))

        self.edit_lines = tk.Button(self.editframe,
                                     text='Edit Lines',
                                     command=self.gotoLineEdit,
                                     width=12,
                                     bg='#e0e0e0',
                                     highlightbackground='black')
        self.edit_lines.grid(column=0,
                             row=2,
                             padx=self.pad/2,
                             pady=(self.pad/2, 0))

        self.edit_labels = tk.Button(self.editframe,
                                     text='Edit Labels',
                                     command=lambda x: print('TODO'),
                                     width=12,
                                     bg='#e0e0e0',
                                     highlightbackground='black')
        self.edit_labels.grid(column=0,
                              row=3,
                              padx=self.pad/2,
                              pady=self.pad/2)

        # --- SAVE OPTIONS ----------------------------------------------------
        self.saveoptions = tk.Label(self.saveframe,
                                    text='Save Options',
                                    bg='white',
                                    font=('Helvetica', 14))
        self.saveoptions.grid(row=0,
                             column=0,
                             columnspan=3,
                             padx=(self.pad/2, 0),
                             sticky='w')

        self.descrip_label = tk.Label(self.saveframe,
                                       text='Description:',
                                       bg='white')

        self.descrip_label.grid(row=1,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='nw')

        self.descrip = tk.Text(self.saveframe,
                               height=4,
                               width=16)

        self.descrip.grid(row=1,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.fname_label = tk.Label(self.saveframe,
                                     text='Filename:',
                                     bg='white')

        self.fname_label.grid(row=2,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.fname = tk.Entry(self.saveframe,
                               width=16,
                               highlightthickness=0)

        self.fname.insert(0, 'NMREsPy_result')

        self.fname.grid(row=2,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        self.dir_label = tk.Label(self.saveframe,
                                   text='Directory:',
                                   bg='white')

        self.dir_label.grid(row=3,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.dir = tk.StringVar()
        self.dir.set(os.path.expanduser('~'))

        self.dir_bar = tk.Entry(self.saveframe,
                                 width=16,
                                 text=self.dir,
                                 highlightthickness=0)

        self.dir_bar.grid(row=3,
                          column=1,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.espypath = os.path.dirname(nmrespy.__file__)
        self.folder_image = Image.open(os.path.join(self.espypath,
                                       'topspin/images/folder_icon.png'))

        scale = 0.02
        [w, h] = self.folder_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.folder_image = self.folder_image.resize((new_w, new_h),
                                                       Image.ANTIALIAS)
        self.folder_img = ImageTk.PhotoImage(self.folder_image)

        self.dir_button = tk.Button(self.saveframe,
                                     command=self.browse,
                                     bg='white',
                                     highlightbackground='black',
                                     image=self.folder_img)

        self.dir_button.grid(row=3,
                             padx=(self.pad/2, self.pad/2),
                             pady=(self.pad, 0),
                             column=2)

        self.txtfile_label = tk.Label(self.saveframe,
                                       text='Save Textfile:',
                                       bg='white')

        self.txtfile_label.grid(row=4,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='w')

        self.txtfile = tk.StringVar()
        self.txtfile.set('1')

        self.txt_box = tk.Checkbutton(self.saveframe,
                                       variable=self.txtfile,
                                       bg='white',
                                       highlightthickness=0,
                                       bd=0)

        self.txt_box.grid(row=4,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.pdffile_label = tk.Label(self.saveframe,
                                       text='Save PDF:',
                                       bg='white')

        self.pdffile_label.grid(row=5,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, self.pad/2),
                                sticky='w')

        self.pdffile = tk.StringVar()
        self.pdffile.set('0')

        self.pdf_box = tk.Checkbutton(self.saveframe,
                                       variable=self.pdffile,
                                       bg='white',
                                       highlightthickness=0,
                                       bd=0)

        self.pdf_box.grid(row=5,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.pickle_label = tk.Label(self.saveframe,
                                     text='Pickle Result:',
                                     bg='white')

        self.pickle_label.grid(row=6,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, self.pad/2),
                                sticky='w')

        self.picklefile = tk.StringVar()
        self.picklefile.set('0')

        self.pickle_box = tk.Checkbutton(self.saveframe,
                                       variable=self.picklefile,
                                       bg='white',
                                       highlightthickness=0,
                                       bd=0)

        self.pickle_box.grid(row=6,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, self.pad/2),
                          sticky='w')

        # --- RUN/SAVE/CANCEL BUTTONS -----------------------------------------
        self.cancel_button = tk.Button(self.buttonframe,
                                        text='Cancel',
                                        width=8,
                                        bg='#ff9894',
                                        highlightbackground='black',
                                        command=self.cancel)

        self.cancel_button.grid(row=0,
                                column=0)

        self.help_button = tk.Button(self.buttonframe,
                                      text='Help',
                                      width=8,
                                      bg='#ffb861',
                                      highlightbackground='black',
                                      command=self.load_help)

        self.help_button.grid(row=0,
                              column=1,
                              padx=(self.pad, 0))

        self.rerun_button = tk.Button(self.buttonframe,
                                      text='Re-run',
                                      width=8,
                                      bg='#f9f683',
                                      highlightbackground='black',
                                      command=self.re_run)

        self.rerun_button.grid(row=0,
                              column=2,
                              padx=(self.pad, 0))

        self.save_button = tk.Button(self.buttonframe,
                                      text='Save',
                                      width=8,
                                      bg='#9eda88',
                                      highlightbackground='black',
                                      command=self.save)

        self.save_button.grid(row=0,
                              column=3,
                              padx=(self.pad, 0))

        self.feedback = tk.Label(self.contactframe,
                                  text='For queries/feedback, contact',
                                  bg='white')

        self.feedback.grid(row=0,
                           column=0,
                           sticky='w')

        self.email = tk.Label(self.contactframe,
                              text='simon.hulse@chem.ox.ac.uk',
                              font='Courier',
                              bg='white')

        self.email.grid(row=1,
                        column=0,
                        sticky='w')

    def gotoLineEdit(self):
        root2 = tk.Toplevel(self.master)
        lineedit = LineEdit(root2, self.lines)
        self.master.wait_window()
        self.lines = lineedit.lines
        print(self.lines)

    def browse(self):
        self.dir = filedialog.askdirectory(initialdir=os.path.expanduser('~'))
        self.dir_bar.insert(0, self.dir)

    def re_run(self):
        print('TODO')

    def load_help(self):
        import webbrowser
        webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')

    def save(self):
        self.descrip = self.descrip.get('1.0', tk.END)
        self.file = self.fname.get()
        self.dir = self.dir_bar.get()
        self.txt = self.txtfile.get()
        self.pdf = self.pdffile.get()
        self.pickle = self.picklefile.get()
        self.master.quit()

    def cancel(self):
        self.master.quit()
        print("NMR-ESPY Cancelled :'(")
        exit()



def get_logo(scale):

    espypath = os.path.dirname(nmrespy.__file__)
    nmrespy_image = Image.open(os.path.join(espypath,
                               'topspin/images/nmrespy_full.png'))

    [w, h] = nmrespy_image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    nmrespy_image = nmrespy_image.resize((new_w, new_h),
                                         Image.ANTIALIAS)
    return ImageTk.PhotoImage(nmrespy_image)


class LineEdit:

    def __init__(self, master, lines):

        self.master = master
        self.master.title('NMR-EsPy - Edit Lines')
        self.master['bg'] = 'white'

        self.lines = lines

        tk.Label(self.master, text='hello').pack()
        tk.Button(self.master, text='quit', command=self.destroy).pack()



if __name__ == '__main__':
    # # path to nmrespy directory
    # espypath = os.path.dirname(nmrespy.__file__)
    #
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
    # dtype_app = dtypeGUI(root, fidpath, pdatapath)
    # root.mainloop()
    #
    # dtype = dtype_app.dtype
    # if dtype == 'fid':
    #     info = load.import_bruker_fid(fidpath, ask_convdta=False)
    # elif dtype == 'pdata':
    #     info = load.import_bruker_pdata(pdatapath)
    #
    # root = tk.Tk()
    # main_app = NMREsPyGUI(root, info)
    # root.mainloop()
    #
    # lb_ppm = main_app.lb_ppm,
    # rb_ppm = main_app.rb_ppm,
    # lnb_ppm = main_app.lnb_ppm,
    # rnb_ppm = main_app.rnb_ppm,
    # p0 = main_app.p0
    # p1 = main_app.p1
    # mpm_points = main_app.mpm_points,
    # nlp_points = main_app.nlp_points,
    # maxit = main_app.maxiter
    # alg = main_app.alg
    #
    # if alg == 'Trust Region':
    #     alg = 'trust_region'
    # elif alg == 'L-BFGS':
    #     alg = 'lbfgs'
    #
    # pv = main_app.pv
    # if pv == '1':
    #     pv = True
    # else:
    #     pv = False
    #
    #
    # info.virtual_echo(highs=lb_ppm, lows=rb_ppm, highs_n=lnb_ppm,
    #                   lows_n=rnb_ppm, p0=p0, p1=p1)
    #
    # info.matrix_pencil(trim=mpm_points)
    #
    # info.nonlinear_programming(trim=nlp_points, maxit=maxit, method=alg,
    #                            phase_variance=pv)

    info = load.pickle_load('result', dir=os.path.expanduser('~'))

    # load the result GUI
    root = tk.Tk()
    res_app = ResultGUI(root, info)
    root.mainloop()

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
