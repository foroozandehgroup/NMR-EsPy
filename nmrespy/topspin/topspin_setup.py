#!/usr/bin/env python3

from copy import copy
import os
import shutil
import webbrowser

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import nmrespy
import nmrespy.load as load
import nmrespy._misc as _misc


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


class MainSetup(tk.Frame):
    def __init__(self, master, fidpath, pdatapath):
        tk.Frame.__init__(self, master)

        self['bg'] = 'white'
        self.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

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

        # --- LEFT AND RIGHT BOUNDS -------------------------------------------
        self.lb = int(np.floor(7 * self.n / 16))
        self.rb = int(np.floor(9 * self.n / 16))
        self.lnb = int(np.floor(1 * self.n / 16))
        self.rnb = int(np.floor(2 * self.n / 16))
        bounds_idx = [self.lb, self.rb, self.lnb, self.rnb]

        self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                         self.n, direction='idx->ppm')
        self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                         self.n, direction='idx->ppm')
        self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                          self.n, direction='idx->ppm')
        self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                          self.n, direction='idx->ppm')
        bounds_ppm = [self.lb_ppm, self.rb_ppm, self.lnb_ppm, self.rnb_ppm]

        # --- PHASE PARAMETERS ------------------------------------------------
        self.pivot = int(np.floor(self.n / 2))
        self.pivot_ppm = _misc.conv_ppm_idx(self.pivot, self.sw_p, self.off_p,
                                            self.n, direction='idx->ppm')
        self.p0 = 0.
        self.p1 = 0.

        # --- PADDING VALUE ---------------------------------------------------
        self.pad = 10

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

        # plot pivot line (alpha=0 to make invisible )
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

        # --- FRAMES ----------------------------------------------------------
        # leftframe -> spectrum plot and region scales
        self.leftframe = tk.Frame(self, bg='white')

        # rightframe -> logo, advanced settings, save/help/quit buttons,
        #               contact info.
        self.rightframe = tk.Frame(self, bg='white')

        self.leftframe.grid(column=0, row=0, sticky='nsew')
        self.rightframe.grid(column=1, row=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)



        self.plotframe = PlotFrame(self.leftframe, self.fig)
        self.scaleframe = ScaleFrame(self.leftframe, self.pad, bounds_idx,
                                     bounds_ppm, self.pivot, self.pivot_ppm,
                                     self.p0, self.p1, self.sw_p, self.off_p,
                                     self.n, self.ylim_init, self.filtregion,
                                     self.noiseregion, self.pivotplot,
                                     self.spec, self.specplot,
                                     self.plotframe.canvas)
        self.logoframe = LogoFrame(self.rightframe)
        self.adsetframe = AdvancedSettingsFrame(self.rightframe, self.pad,
                                                self.n)
        self.buttonframe = ButtonFrame(self.rightframe, self.pad, bounds_ppm,
                                       self.pivot, self.p0, self.p1, self.n,
                                       self.scaleframe.regionframe,
                                       self.scaleframe.phaseframe,
                                       self.adsetframe, self.info, self.sw_p,
                                       self.off_p)
        self.contactrame = ContactFrame(self.rightframe)

        self.leftframe.columnconfigure(0, weight=1)
        self.leftframe.rowconfigure(0, weight=1)
        self.rightframe.rowconfigure(1, weight=1)


    def dtype_popup(self):
        self.master.withdraw()
        self.dtypewindow = DataType(self, self.fidpath, self.pdatapath)


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


class LogoFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.config(bg='white')
        self.grid(row=0, column=0, sticky='ew')

        path = os.path.dirname(nmrespy.__file__)
        image = Image.open(os.path.join(path, 'topspin/images/nmrespy_full.png'))
        scale = 0.08
        [w, h] = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)

        # make img an attribute of the class to prevent garbage collection
        self.img = ImageTk.PhotoImage(image)
        logo = tk.Label(self, image=self.img, bg='white')
        logo.grid(row=0, column=0)


class PlotFrame(tk.Frame):
    def __init__(self, master, fig):
        tk.Frame.__init__(self, master)

        self.fig = fig
        self['bg'] = 'white'
        self.grid(row=0, column=0, sticky='nsew')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.toolbar.config(background='white')
        self.toolbar._message_label.config(bg='white')
        for button in self.toolbar.winfo_children():
            button.config(bg='white')
        self.toolbar.grid(column=0, row=1, sticky='e')


class ScaleFrame(tk.Frame):
    def __init__(self, master, pad, bounds_idx, bounds_ppm, pivot, pivot_ppm,
                 p0, p1, sw_p, off_p, n, ylim_init, filtregion, noiseregion,
                 pivotplot, spec, specplot, figcanvas):

        tk.Frame.__init__(self, master)
        self.master = master

        self.pad = pad
        self.filtregion = filtregion
        self.noiseregion = noiseregion
        self.pivotplot = pivotplot
        self.figcanvas = figcanvas

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
        self.notebook = ttk.Notebook(self.master)
        self.notebook.bind('<<NotebookTabChanged>>',
                           (lambda event: self.ud_plot()))

        self.regionframe = RegionFrame(self.notebook, pad, bounds_idx,
                                       bounds_ppm, sw_p, off_p,
                                       n, ylim_init, self.filtregion,
                                       self.noiseregion, self.figcanvas)

        self.phaseframe = PhaseFrame(self.notebook, pad, pivot, pivot_ppm, p0,
                                     p1, sw_p, off_p, n, self.pivotplot,
                                     self.figcanvas, spec, specplot)

        self.notebook.add(self.regionframe, text='Region Selection',
                            sticky='ew')
        self.notebook.add(self.phaseframe, text='Phase Correction',
                            sticky='ew')
        self.notebook.grid(column=0, row=1, padx=(self.pad,0), pady=self.pad,
                           sticky='ew')

    def ud_plot(self):
        tab = self.notebook.index(self.notebook.select())

        # region selection tab selected
        if tab == 0:
            self.filtregion.set_alpha(1)
            self.noiseregion.set_alpha(1)
            self.pivotplot.set_alpha(0)
            self.figcanvas.draw_idle()

        # phase correction tab selected
        elif tab == 1:
            self.filtregion.set_alpha(0)
            self.noiseregion.set_alpha(0)
            self.pivotplot.set_alpha(1)
            self.figcanvas.draw_idle()


class RegionFrame(tk.Frame):
    def __init__(self, master, pad, bounds_idx, bounds_ppm, sw_p,
                 off_p, n, ylim_init, filtregion, noiseregion,
                 figcanvas):
        tk.Frame.__init__(self, master)

        self.lb = bounds_idx[0]
        self.rb = bounds_idx[1]
        self.lnb = bounds_idx[2]
        self.rnb = bounds_idx[3]
        self.lb_ppm = bounds_ppm[0]
        self.rb_ppm = bounds_ppm[1]
        self.lnb_ppm = bounds_ppm[2]
        self.rnb_ppm = bounds_ppm[3]
        self.sw_p = sw_p
        self.off_p = off_p
        self.n = n
        self.ylim_init = ylim_init
        self.filtregion = filtregion
        self.noiseregion = noiseregion
        self.figcanvas = figcanvas

        self.pad = pad
        self['bg'] = 'white'

        # Bound titles
        lb_title = tk.Label(self, text='left bound', bg='white')
        rb_title = tk.Label(self, text='right bound', bg='white')
        lnb_title = tk.Label(self, text='left noise bound', bg='white')
        rnb_title = tk.Label(self, text='right noise bound', bg='white')

        # Scales
        self.lb_scale = tk.Scale(self, troughcolor='#cbedcb',
                                 command=self.ud_lb_scale)
        self.rb_scale = tk.Scale(self, troughcolor='#cbedcb',
                                 command=self.ud_rb_scale)
        self.lnb_scale = tk.Scale(self, troughcolor='#cde6ff',
                                  command=self.ud_lnb_scale)
        self.rnb_scale = tk.Scale(self, troughcolor='#cde6ff',
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

        self.lb_entry = tk.Entry(self, textvariable=self.lb_label)
        self.rb_entry = tk.Entry(self, textvariable=self.rb_label)
        self.lnb_entry = tk.Entry(self, textvariable=self.lnb_label)
        self.rnb_entry = tk.Entry(self, textvariable=self.rnb_label)

        self.lb_entry.bind('<Return>', (lambda event: ud_lb_entry()))
        self.rb_entry.bind('<Return>', (lambda event: ud_rb_entry()))
        self.lnb_entry.bind('<Return>', (lambda event: ud_lnb_entry()))
        self.rnb_entry.bind('<Return>', (lambda event: ud_rnb_entry()))

        for entry in [self.lb_entry, self.rb_entry, self.lnb_entry, self.rnb_entry]:
            entry['width'] = 6
            entry['highlightthickness'] = 0

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

        self.columnconfigure(1, weight=1)

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
            self.figcanvas.draw_idle()
        else:
            self.lb = self.rb - 1
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.lb_scale.set(self.lb)
            self.figcanvas.draw_idle()

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
            self.figcanvas.draw_idle()
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
            self.figcanvas.draw_idle()
        else:
            self.rb = self.lb + 1
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.rb_scale.set(self.rb)
            self.figcanvas.draw_idle()

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
            self.figcanvas.draw_idle()
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
            self.figcanvas.draw_idle()
        else:
            self.lnb = self.rnb - 1
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.lnb_scale.set(self.lnb)
            self.figcanvas.draw_idle()

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
            self.figcanvas.draw_idle()
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
            self.figcanvas.draw_idle()
        else:
            self.rnb = self.lnb + 1
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.rnb_scale.set(self.rnb)
            self.figcanvas.draw_idle()

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
            self.figcanvas.draw_idle()
        else:
            self.rnb_label.set(f'{self.rnb_ppm:.3f}')


class PhaseFrame(tk.Frame):
    def __init__(self, master, pad, pivot, pivot_ppm, p0, p1, sw_p, off_p, n,
                 pivotplot, figcanvas, spec, specplot):
        tk.Frame.__init__(self, master)

        self['bg'] = 'white'

        self.pad = pad
        self.pivot = pivot
        self.pivot_ppm = pivot_ppm
        self.p0 = p0
        self.p1 = p1
        self.sw_p = sw_p
        self.off_p = off_p
        self.n = n
        self.pivotplot = pivotplot
        self.figcanvas = figcanvas
        self.spec = spec
        self.specplot = specplot

        # Pivot
        pivot_title = tk.Label(self, text='pivot', bg='white')
        p0_title = tk.Label(self, text='p0', bg='white')
        p1_title = tk.Label(self, text='p1', bg='white')

        self.pivot_scale = tk.Scale(self, troughcolor='#ffb0b0',
                                    command=self.ud_pivot_scale, from_=1,
                                    to=self.n)
        self.p0_scale = tk.Scale(self, resolution=0.0001, troughcolor='#e0e0e0',
                                 command=self.ud_p0_scale, from_=-np.pi,
                                 to=np.pi)
        self.p1_scale = tk.Scale(self, resolution=0.0001, troughcolor='#e0e0e0',
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

        self.pivot_entry = tk.Entry(self, textvariable=self.pivot_label)
        self.p0_entry = tk.Entry(self, textvariable=self.p0_label)
        self.p1_entry = tk.Entry(self, textvariable=self.p1_label)

        self.pivot_entry.bind('<Return>', (lambda event: self.ud_pivot_entry()))
        self.p0_entry.bind('<Return>', (lambda event: self.ud_p0_entry()))
        self.p1_entry.bind('<Return>', (lambda event: self.ud_p1_entry()))

        for entry in [self.pivot_entry, self.p0_entry, self.p1_entry]:
            entry['width'] = 6
            entry['highlightthickness'] = 0



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

        self.columnconfigure(1, weight=1)


    def ud_pivot_scale(self, pivot):

        self.pivot = int(pivot)
        self.pivot_ppm = _misc.conv_ppm_idx(self.pivot, self.sw_p, self.off_p,
                                            self.n, direction='idx->ppm')
        self.pivot_label.set(f'{self.pivot_ppm:.3f}')
        self.ud_phase()
        x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
        self.pivotplot.set_xdata(x)

        self.figcanvas.draw_idle()


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
            self.figcanvas.draw_idle()

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
        self.figcanvas.draw_idle()


class AdvancedSettingsFrame(tk.Frame):
    def __init__(self, master, pad, n):
        tk.Frame.__init__(self, master)

        self['bg'] = 'white'
        self.pad = pad
        self.n = n
        self.grid(row=1, column=0, sticky='ew')

        adset_title = tk.Label(self, text='Advanced Settings',
                                    font=('Helvetica', 14), bg='white')

        # --- number of MPM points ---------------------------------------------
        mpm_label = tk.Label(self, text='Points for MPM:', bg='white')

        self.mpm = tk.StringVar()

        if self.n <= 4096:
            self.mpm.set(str(self.n))
        else:
            self.mpm.set('4096')

        self.mpm_entry = tk.Entry(self, width=12, highlightthickness=0,
                                  textvariable=self.mpm)

        maxval = int(np.floor(self.n/2))
        mpm_max_label = tk.Label(self, text=f'Max. value: {maxval}', bg='white')

        # --- number of NLP points ---------------------------------------------
        nlp_label = tk.Label(self, text='Points for NLP:', bg='white')

        self.nlp = tk.StringVar()

        if self.n <= 8192:
            self.nlp.set(str(self.n))
        else:
            self.nlp.set('8192')

        self.nlp_entry = tk.Entry(self, width=12, highlightthickness=0,
                                  textvariable=self.nlp)

        nlp_max_label = tk.Label(self, text=f'Max. value: {maxval}', bg='white')

        # --- maximum NLP iterations -------------------------------------------
        maxit_label = tk.Label(self, text='Max. Iterations:', bg='white')

        self.maxit_entry = tk.Entry(self, width=12, highlightthickness=0)
        self.maxit_entry.insert(0, '100')

        # --- NLP algorithm ----------------------------------------------------
        alg_label = tk.Label(self, text='NLP Method:', bg='white')

        self.algorithm = tk.StringVar(self)
        self.algorithm.set('Trust Region')
        self.algoptions = tk.OptionMenu(self, self.algorithm, 'Trust Region',
                                        'L-BFGS')
        self.algoptions.config(bg='white', borderwidth=0)
        self.algoptions['menu'].configure(bg='white')

        # --- opt phase variance? ----------------------------------------------
        phasevar_label = tk.Label(self, text='Opt. Phase Variance:', bg='white')

        self.phasevar = tk.StringVar()
        self.phasevar.set('1')

        self.phasevar_box = tk.Checkbutton(self, variable=self.phasevar,
                                           bg='white', highlightthickness=0,
                                           bd=0)


        adset_title.grid(row=0, column=0, columnspan=3, padx=(self.pad/2, 0),
                         pady=(self.pad/2, 0), sticky='w')
        mpm_label.grid(row=1, column=0, padx=(self.pad/2, 0),
                       pady=(self.pad, 0), sticky='nsw')
        self.mpm_entry.grid(row=1, column=1, columnspan=2, padx=(self.pad/2, 0),
                            pady=(self.pad, 0), sticky='w')
        mpm_max_label.grid(row=2, column=1, columnspan=2, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='nw')
        nlp_label.grid(row=3, column=0, padx=(self.pad/2, 0),
                       pady=(self.pad, 0), sticky='w')
        self.nlp_entry.grid(row=3, column=1, columnspan=2, padx=(self.pad/2, 0),
                            pady=(self.pad, 0), sticky='w')
        nlp_max_label.grid(row=4, column=1, columnspan=2, padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0), sticky='nw')
        maxit_label.grid(row=5, column=0, padx=(self.pad/2, 0),
                         pady=(self.pad, 0), sticky='w')
        self.maxit_entry.grid(row=5, column=1, columnspan=2,
                              padx=(self.pad/2, 0), pady=(self.pad, 0),
                              sticky='w')
        alg_label.grid(row=6, column=0, padx=(self.pad/2, 0),
                       pady=(self.pad, 0), sticky='w')
        self.algoptions.grid(row=6, column=1, columnspan=2,
                             padx=(self.pad/2, 0), pady=(self.pad, 0))
        phasevar_label.grid(row=7, column=0, padx=(self.pad/2, 0),
                            pady=(self.pad, 0), sticky='w')
        self.phasevar_box.grid(row=7, column=1, columnspan=2,
                               padx=(self.pad/2, 0), pady=(self.pad, 0),
                               sticky='w')


class ButtonFrame(tk.Frame):
    def __init__(self, master, pad, bounds_ppm, pivot, p0, p1, n,
                 regionframe, phaseframe, adsetframe, info, sw_p,
                 off_p):
        tk.Frame.__init__(self, master)

        self.pad = pad
        self.lb_ppm = bounds_ppm[0]
        self.rb_ppm = bounds_ppm[1]
        self.lnb_ppm = bounds_ppm[2]
        self.rnb_ppm = bounds_ppm[3]
        self.pivot = pivot
        self.p0 = p0
        self.p1 = p1
        self.n = n
        self.regionframe = regionframe
        self.phaseframe = phaseframe
        self.adsetframe = adsetframe
        self.info = info
        self.sw_p = sw_p
        self.off_p = off_p

        self['bg'] = 'white'
        self.grid(row=2, column=0, sticky='e')

        self.cancel_button = tk.Button(self, text='Cancel', command=self.cancel,
                                       bg='#ff9894')
        self.help_button = tk.Button(self, text='Help', command=self.load_help,
                                     bg='#ffb861')
        self.run_button = tk.Button(self, text='Run', command=self.run,
                                     bg='#9eda88')

        for button in [self.cancel_button, self.help_button, self.run_button]:
            button['highlightbackground'] = 'black'
            button['width'] = 8

        self.cancel_button.grid(row=0, column=0)
        self.help_button.grid(row=0, column=1, padx=(self.pad, 0))
        self.run_button.grid(row=0, column=2, padx=self.pad)

    def cancel(self):
        exit()

    def load_help(self):
        webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')

    def run(self):
        p0 = float(self.phaseframe.p0_entry.get())
        p1 = float(self.phaseframe.p1_entry.get())
        pivot_ppm = float(self.phaseframe.pivot_entry.get())
        pivot = _misc.conv_ppm_idx(pivot_ppm, self.sw_p, self.off_p, self.n,
                                   direction='ppm->idx')
        p0 - (p1 * pivot / self.n)

        lb = float(self.regionframe.lb_entry.get()),
        rb = float(self.regionframe.rb_entry.get()),
        lnb = float(self.regionframe.lnb_entry.get()),
        rnb = float(self.regionframe.rnb_entry.get()),

        mpm_points = int(self.adsetframe.mpm_entry.get())
        nlp_points = int(self.adsetframe.nlp_entry.get())
        maxiter = int(self.adsetframe.maxit_entry.get())
        alg = self.adsetframe.algorithm.get()
        pv = self.adsetframe.phasevar.get()
        self.master.master.master.destroy()

        if alg == 'Trust Region':
            alg = 'trust_region'
        elif alg == 'L-BFGS':
            alg = 'lbfgs'

        if pv == '1':
            pv = True
        else:
            pv = False
        print(self.info)
        self.info.virtual_echo(highs=lb, lows=rb, highs_n=lnb, lows_n=rnb,
                               p0=p0, p1=p1)
        self.info.matrix_pencil(trim=mpm_points)
        self.info.nonlinear_programming(trim=nlp_points, maxit=maxiter,
                                        method=alg, phase_variance=pv)

        tmpdir = os.path.join(os.path.dirname(nmrespy.__file__), 'topspin/tmp')

        self.info.pickle_save(fname='tmp.pkl', dir=tmpdir,
                              force_overwrite=True)

class ContactFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self['bg'] = 'white'
        self.grid(row=3, column=0, sticky='ew')

        feedback = tk.Label(self, text='For queries/feedback, contact',
                            bg='white')
        email = tk.Label(self, text='simon.hulse@chem.ox.ac.uk', font='Courier',
                         bg='white')
        feedback.grid(row=0, column=0, sticky='w')
        email.grid(row=1, column=0, sticky='w')





# class ResultGUI:
#
#     def __init__(self, master, info):
#
#         self.master = master
#         self.master.title('NMR-EsPy - Result Settings')
#         self.master['bg'] = 'white'
#         self.info = info
#         self.pad = 10
#
#         # --- FRAMES ----------------------------------------------------------
#         # leftframe -> spectrum plot and mpl toolbar
#         self.leftframe = tk.Frame(self.master, bg='white')
#         self.leftframe.grid(column=0,
#                             row=0,
#                             sticky='nsew')
#
#         # rightframe -> logo, edit and save options, save/quit buttons
#         self.rightframe = tk.Frame(self.master, bg='white')
#
#         self.rightframe.grid(column=1,
#                              row=0,
#                              sticky='nsew')
#
#         self.plotframe = tk.Frame(self.leftframe, bg='white')
#
#         self.plotframe.grid(column=0,
#                             row=0,
#                             padx=(self.pad, 0),
#                             pady=(self.pad, 0),
#                             sticky='nsew')
#
#         self.toolbarframe = tk.Frame(self.plotframe, bg='white')
#         self.toolbarframe.grid(row=1,
#                                column=0,
#                                sticky='e')
#
#         self.logoframe = tk.Frame(self.rightframe, bg='white')
#
#         self.logoframe.grid(column=0,
#                             row=0,
#                             padx=(self.pad, self.pad),
#                             pady=(self.pad, 0),
#                             sticky='ew')
#
#         self.editframe = tk.Frame(self.rightframe,
#                                   bg='white',
#                                   highlightbackground='black',
#                                   highlightthickness=2)
#
#         self.editframe.grid(column=0,
#                             row=1,
#                             padx=(self.pad, self.pad),
#                             pady=(self.pad, 0),
#                             sticky='ew')
#
#         self.saveframe = tk.Frame(self.rightframe,
#                                   bg='white',
#                                   highlightbackground='black',
#                                   highlightthickness=2)
#
#         self.saveframe.grid(column=0,
#                             row=2,
#                             padx=(self.pad, self.pad),
#                             pady=(self.pad, 0),
#                             sticky='ew')
#
#         self.buttonframe = tk.Frame(self.rightframe, bg='white')
#
#         self.buttonframe.grid(column=0,
#                               row=3,
#                               padx=(self.pad, self.pad),
#                               pady=(self.pad, 0))
#
#         self.contactframe = tk.Frame(self.rightframe, bg='white')
#
#         self.contactframe.grid(column=0,
#                                row=4,
#                                padx=(self.pad, self.pad),
#                                pady=(self.pad, self.pad),
#                                sticky='sw')
#
#         self.master.columnconfigure(0, weight=1)
#         self.master.rowconfigure(0, weight=1)
#         self.leftframe.rowconfigure(0, weight=1)
#         self.leftframe.columnconfigure(0, weight=1)
#         self.rightframe.grid_rowconfigure(1, weight=1)
#         self.plotframe.columnconfigure(0, weight=1)
#         self.plotframe.rowconfigure(0, weight=1)
#         self.editframe.columnconfigure(0, weight=1)
#
#
#         # --- RESULT PLOT -----------------------------------------------------
#         self.fig, self.ax, self.lines, self.labels = self.info.plot_result()
#         self.fig.set_dpi(170)
#         self.fig.set_size_inches(6, 3.5)
#
#         self.xlim = self.ax.get_xlim()
#
#         # place figure into canvas
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.plotframe)
#         self.canvas.draw()
#         self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')
#
#         # custom mpl toolar, lacking save button and subplot-config manager
#         self.toolbar = CustomNavigationToolbar(self.canvas, self.toolbarframe)
#         self.toolbar.config(background='white')
#         self.toolbar._message_label.config(bg='white')
#         for button in self.toolbar.winfo_children():
#             button.config(bg='white')
#
#         # restrict zooming and panning into regions beyond the specified region
#         # see Restrictor class above
#         self.restrict1 = Restrictor(self.ax,
#                                     x=lambda x: x<= self.xlim[0])
#         self.restrict2 = Restrictor(self.ax,
#                                     x=lambda x: x>= self.xlim[1])
#
#         # --- NMR-EsPy LOGO ---------------------------------------------------
#         self.nmrespy_img = get_logo(0.08)
#         self.nmrespy_logo = tk.Label(self.logoframe,
#                                       image=self.nmrespy_img,
#                                       bg='white')
#         self.nmrespy_logo.pack(pady=(0, self.pad))
#
#         # --- EDIT BUTTONS ----------------------------------------------------
#         self.editoptions = tk.Label(self.editframe,
#                                     text='Edit Options',
#                                     bg='white',
#                                     font=('Helvetica', 14))
#         self.editoptions.grid(row=0,
#                               column=0,
#                               columnspan=3,
#                               padx=(self.pad/2, 0),
#                               sticky='w')
#
#         self.edit_params = tk.Button(self.editframe,
#                                      text='Edit Parameters',
#                                      command=lambda x: print('TODO'),
#                                      width=12,
#                                      bg='#e0e0e0',
#                                      highlightbackground='black')
#
#         self.edit_params.grid(column=0,
#                               row=1,
#                               padx=self.pad/2,
#                               pady=(self.pad/2, 0))
#
#         self.edit_lines = tk.Button(self.editframe,
#                                      text='Edit Lines',
#                                      command=self.gotoLineEdit,
#                                      width=12,
#                                      bg='#e0e0e0',
#                                      highlightbackground='black')
#         self.edit_lines.grid(column=0,
#                              row=2,
#                              padx=self.pad/2,
#                              pady=(self.pad/2, 0))
#
#         self.edit_labels = tk.Button(self.editframe,
#                                      text='Edit Labels',
#                                      command=lambda x: print('TODO'),
#                                      width=12,
#                                      bg='#e0e0e0',
#                                      highlightbackground='black')
#         self.edit_labels.grid(column=0,
#                               row=3,
#                               padx=self.pad/2,
#                               pady=self.pad/2)
#
#         # --- SAVE OPTIONS ----------------------------------------------------
#         self.saveoptions = tk.Label(self.saveframe,
#                                     text='Save Options',
#                                     bg='white',
#                                     font=('Helvetica', 14))
#         self.saveoptions.grid(row=0,
#                              column=0,
#                              columnspan=3,
#                              padx=(self.pad/2, 0),
#                              sticky='w')
#
#         self.descrip_label = tk.Label(self.saveframe,
#                                        text='Description:',
#                                        bg='white')
#
#         self.descrip_label.grid(row=1,
#                                 column=0,
#                                 padx=(self.pad/2, 0),
#                                 pady=(self.pad, 0),
#                                 sticky='nw')
#
#         self.descrip = tk.Text(self.saveframe,
#                                height=4,
#                                width=16)
#
#         self.descrip.grid(row=1,
#                           column=1,
#                           columnspan=2,
#                           padx=(self.pad/2, 0),
#                           pady=(self.pad, 0),
#                           sticky='w')
#
#         self.fname_label = tk.Label(self.saveframe,
#                                      text='Filename:',
#                                      bg='white')
#
#         self.fname_label.grid(row=2,
#                               column=0,
#                               padx=(self.pad/2, 0),
#                               pady=(self.pad, 0),
#                               sticky='w')
#
#         self.fname = tk.Entry(self.saveframe,
#                                width=16,
#                                highlightthickness=0)
#
#         self.fname.insert(0, 'NMREsPy_result')
#
#         self.fname.grid(row=2,
#                         column=1,
#                         columnspan=2,
#                         padx=(self.pad/2, 0),
#                         pady=(self.pad, 0),
#                         sticky='w')
#
#         self.dir_label = tk.Label(self.saveframe,
#                                    text='Directory:',
#                                    bg='white')
#
#         self.dir_label.grid(row=3,
#                             column=0,
#                             padx=(self.pad/2, 0),
#                             pady=(self.pad, 0),
#                             sticky='w')
#
#         self.dir = tk.StringVar()
#         self.dir.set(os.path.expanduser('~'))
#
#         self.dir_bar = tk.Entry(self.saveframe,
#                                  width=16,
#                                  text=self.dir,
#                                  highlightthickness=0)
#
#         self.dir_bar.grid(row=3,
#                           column=1,
#                           padx=(self.pad/2, 0),
#                           pady=(self.pad, 0),
#                           sticky='w')
#
#         self.espypath = os.path.dirname(nmrespy.__file__)
#         self.folder_image = Image.open(os.path.join(self.espypath,
#                                        'topspin/images/folder_icon.png'))
#
#         scale = 0.02
#         [w, h] = self.folder_image.size
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         self.folder_image = self.folder_image.resize((new_w, new_h),
#                                                        Image.ANTIALIAS)
#         self.folder_img = ImageTk.PhotoImage(self.folder_image)
#
#         self.dir_button = tk.Button(self.saveframe,
#                                      command=self.browse,
#                                      bg='white',
#                                      highlightbackground='black',
#                                      image=self.folder_img)
#
#         self.dir_button.grid(row=3,
#                              padx=(self.pad/2, self.pad/2),
#                              pady=(self.pad, 0),
#                              column=2)
#
#         self.txtfile_label = tk.Label(self.saveframe,
#                                        text='Save Textfile:',
#                                        bg='white')
#
#         self.txtfile_label.grid(row=4,
#                                 column=0,
#                                 padx=(self.pad/2, 0),
#                                 pady=(self.pad, 0),
#                                 sticky='w')
#
#         self.txtfile = tk.StringVar()
#         self.txtfile.set('1')
#
#         self.txt_box = tk.Checkbutton(self.saveframe,
#                                        variable=self.txtfile,
#                                        bg='white',
#                                        highlightthickness=0,
#                                        bd=0)
#
#         self.txt_box.grid(row=4,
#                           column=1,
#                           columnspan=2,
#                           padx=(self.pad/2, 0),
#                           pady=(self.pad, 0),
#                           sticky='w')
#
#         self.pdffile_label = tk.Label(self.saveframe,
#                                        text='Save PDF:',
#                                        bg='white')
#
#         self.pdffile_label.grid(row=5,
#                                 column=0,
#                                 padx=(self.pad/2, 0),
#                                 pady=(self.pad, self.pad/2),
#                                 sticky='w')
#
#         self.pdffile = tk.StringVar()
#         self.pdffile.set('0')
#
#         self.pdf_box = tk.Checkbutton(self.saveframe,
#                                        variable=self.pdffile,
#                                        bg='white',
#                                        highlightthickness=0,
#                                        bd=0)
#
#         self.pdf_box.grid(row=5,
#                           column=1,
#                           columnspan=2,
#                           padx=(self.pad/2, 0),
#                           pady=(self.pad, 0),
#                           sticky='w')
#
#         self.pickle_label = tk.Label(self.saveframe,
#                                      text='Pickle Result:',
#                                      bg='white')
#
#         self.pickle_label.grid(row=6,
#                                 column=0,
#                                 padx=(self.pad/2, 0),
#                                 pady=(self.pad, self.pad/2),
#                                 sticky='w')
#
#         self.picklefile = tk.StringVar()
#         self.picklefile.set('0')
#
#         self.pickle_box = tk.Checkbutton(self.saveframe,
#                                        variable=self.picklefile,
#                                        bg='white',
#                                        highlightthickness=0,
#                                        bd=0)
#
#         self.pickle_box.grid(row=6,
#                           column=1,
#                           columnspan=2,
#                           padx=(self.pad/2, 0),
#                           pady=(self.pad, self.pad/2),
#                           sticky='w')
#
#         # --- RUN/SAVE/CANCEL BUTTONS -----------------------------------------
#         self.cancel_button = tk.Button(self.buttonframe,
#                                         text='Cancel',
#                                         width=8,
#                                         bg='#ff9894',
#                                         highlightbackground='black',
#                                         command=self.cancel)
#
#         self.cancel_button.grid(row=0,
#                                 column=0)
#
#         self.help_button = tk.Button(self.buttonframe,
#                                       text='Help',
#                                       width=8,
#                                       bg='#ffb861',
#                                       highlightbackground='black',
#                                       command=self.load_help)
#
#         self.help_button.grid(row=0,
#                               column=1,
#                               padx=(self.pad, 0))
#
#         self.rerun_button = tk.Button(self.buttonframe,
#                                       text='Re-run',
#                                       width=8,
#                                       bg='#f9f683',
#                                       highlightbackground='black',
#                                       command=self.re_run)
#
#         self.rerun_button.grid(row=0,
#                               column=2,
#                               padx=(self.pad, 0))
#
#         self.save_button = tk.Button(self.buttonframe,
#                                       text='Save',
#                                       width=8,
#                                       bg='#9eda88',
#                                       highlightbackground='black',
#                                       command=self.save)
#
#         self.save_button.grid(row=0,
#                               column=3,
#                               padx=(self.pad, 0))
#
#         self.feedback = tk.Label(self.contactframe,
#                                   text='For queries/feedback, contact',
#                                   bg='white')
#
#         self.feedback.grid(row=0,
#                            column=0,
#                            sticky='w')
#
#         self.email = tk.Label(self.contactframe,
#                               text='simon.hulse@chem.ox.ac.uk',
#                               font='Courier',
#                               bg='white')
#
#         self.email.grid(row=1,
#                         column=0,
#                         sticky='w')
#
#     def gotoLineEdit(self):
#         root2 = tk.Toplevel(self.master)
#         lineedit = LineEdit(root2, self.lines)
#         self.master.wait_window()
#         self.lines = lineedit.lines
#         print(self.lines)
#
#     def browse(self):
#         self.dir = filedialog.askdirectory(initialdir=os.path.expanduser('~'))
#         self.dir_bar.insert(0, self.dir)
#
#     def re_run(self):
#         print('TODO')
#
#     def load_help(self):
#         import webbrowser
#         webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')
#
#     def save(self):
#         self.descrip = self.descrip.get('1.0', tk.END)
#         self.file = self.fname.get()
#         self.dir = self.dir_bar.get()
#         self.txt = self.txtfile.get()
#         self.pdf = self.pdffile.get()
#         self.pickle = self.picklefile.get()
#         self.master.quit()
#
#     def cancel(self):
#         self.master.quit()
#         print("NMR-ESPY Cancelled :'(")
#         exit()
#
#
#
# def get_logo(scale):
#
#     espypath = os.path.dirname(nmrespy.__file__)
#     nmrespy_image = Image.open(os.path.join(espypath,
#                                'topspin/images/nmrespy_full.png'))
#
#     [w, h] = nmrespy_image.size
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     nmrespy_image = nmrespy_image.resize((new_w, new_h),
#                                          Image.ANTIALIAS)
#     return ImageTk.PhotoImage(nmrespy_image)
#
#
# class LineEdit:
#
#     def __init__(self, master, lines):
#
#         self.master = master
#         self.master.title('NMR-EsPy - Edit Lines')
#         self.master['bg'] = 'white'
#
#         self.lines = lines
#
#         tk.Label(self.master, text='hello').pack()
#         tk.Button(self.master, text='quit', command=self.destroy).pack()
