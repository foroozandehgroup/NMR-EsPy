#!/usr/bin/env python3

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import nmrespy
import nmrespy.load as load
import nmrespy._misc as _misc





class NMREsPyGUI:
    def __init__(self, master, info):

        self.master = master
        self.master.title('NMR-ESPY')

        # acquire data
        self.info = info
        self.dtype = self.info.get_dtype()

        if self.dtype == 'raw':
            self.spec = np.flip(fftshift(fft(self.info.get_data())))
        elif self.dtype == 'pdata':
            self.spec = self.info.get_data(pdata_key='1r') \
                        + 1j * self.info.get_data(pdata_key='1i')

        # basic info
        self.shifts = info.get_shifts(unit='ppm')[0]
        self.sw_p = info.get_sw(unit='ppm')[0]
        self.off_p = info.get_offset(unit='ppm')[0]
        self.n = self.spec.shape[0]
        self.n = self.spec.shape[0]

        # initialsie left and right bounds (idx and ppm)
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

        # initialise phase parameters
        self.pivot = int(np.floor(self.n / 2))
        self.pivot_ppm = _misc.conv_ppm_idx(self.pivot, self.sw_p, self.off_p,
                                            self.n, direction='idx->ppm')
        self.p0 = 0.
        self.p1 = 0.

        # constant padding value
        self.pad = 10

        # --- FRAMES ----------------------------------------------------------
        # leftframe -> spectrum plot and region scales
        self.leftframe = tk.Frame(self.master)

        self.leftframe.grid(column=0,
                            row=0,
                            sticky='nsew')

        # rightframe -> logo, advbanced settings, save/quit buttons,
        #               contact info.
        self.rightframe = tk.Frame(self.master)

        self.rightframe.grid(column=1,
                             row=0,
                             sticky='nsew')

        self.plotframe = tk.Frame(self.leftframe)

        self.plotframe.grid(column=0,
                            row=0,
                            padx=(self.pad, 0),
                            pady=(self.pad, 0),
                            sticky='nsew')

        # scaleframe -> tabs for region selection and phase correction
        self.scaleframe = ttk.Notebook(self.leftframe)

        self.scaleframe.grid(column=0,
                             row=1,
                             padx=(self.pad, 0),
                             pady=(self.pad, self.pad),
                             sticky='ew')

        self.regionframe = ttk.Frame(self.scaleframe)
        self.regionframe.columnconfigure(1, weight=1)
        self.phaseframe = ttk.Frame(self.scaleframe)
        self.phaseframe.columnconfigure(1, weight=1)

        self.scaleframe.bind('<<NotebookTabChanged>>', self.update_plot)

        self.scaleframe.add(self.regionframe,
                            text='Region Selection',
                            sticky='ew')

        self.scaleframe.add(self.phaseframe,
                            text='Phase Correction',
                            sticky='ew')

        self.logoframe = tk.Frame(self.rightframe)

        self.logoframe.grid(column=0,
                            row=0,
                            padx=(self.pad, self.pad),
                            pady=(self.pad, 0),
                            sticky='ew')

        self.adsetframe = tk.Frame(self.rightframe)

        self.adsetframe.grid(column=0,
                             row=1,
                             padx=(self.pad, self.pad),
                             pady=(self.pad, 0),
                             sticky='new')

        self.buttonframe = tk.Frame(self.rightframe)

        self.buttonframe.grid(column=0,
                              row=2,
                              padx=(self.pad, self.pad),
                              pady=(self.pad, 0),
                              sticky='se')

        self.contactframe = tk.Frame(self.rightframe)

        self.contactframe.grid(column=0,
                               row=3,
                               padx=(self.pad, self.pad),
                               pady=(self.pad, self.pad),
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

        self.nmrespy_logo = tk.Label(self.logoframe,
                                      image=self.nmrespy_img)

        self.nmrespy_logo.pack()

        # --- SPECTRUM PLOT ---------------------------------------------------
        # create plot
        self.fig = Figure(figsize=(6,3.5), dpi=170)
        self.ax = self.fig.add_subplot(111)
        self.specplot = self.ax.plot(self.shifts,
                                     np.real(self.spec),
                                     color='k',
                                     lw=0.6)[0]

        self.xlim = (self.shifts[0], self.shifts[-1])
        self.ax.set_xlim(self.xlim)
        self.ylim = self.ax.get_ylim()

        # set up variables to enable returning to previous views
        self.xlim_init = copy(self.xlim)
        self.ylim_init = copy(self.ylim)
        self.xlimit_history = []
        self.ylimit_history = []
        self.xlimit_history.append(self.xlim_init)
        self.ylimit_history.append(self.ylim_init)

        # highlight the spectral region to be filtered
        self.filtregion = Rectangle((self.rb_ppm, -10*self.ylim_init[1]),
                                     self.lb_ppm - self.rb_ppm,
                                     20*self.ylim_init[1],
                                     facecolor='#7fd47f')

        self.ax.add_patch(self.filtregion)

        # highlight the noise region
        self.noiseregion = Rectangle((self.rnb_ppm, -10*self.ylim_init[1]),
                                      self.lnb_ppm - self.rnb_ppm,
                                      20*self.ylim_init[1],
                                      facecolor='#66b3ff')

        self.ax.add_patch(self.noiseregion)

        x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
        y = np.linspace(-10*self.ylim_init[1], 10*self.ylim_init[1], 1000)
        self.pivotplot = self.ax.plot(x, y, color='r', alpha=0)[0]

        self.ax.set_ylim(self.ylim)
        self.ax.tick_params(axis='x', which='major', labelsize=6)
        self.ax.set_yticks([])
        self.ax.spines['top'].set_color('k')
        self.ax.spines['bottom'].set_color('k')
        self.ax.spines['left'].set_color('k')
        self.ax.spines['right'].set_color('k')

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plotframe)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # function for when mouse is clicked inside plot
        def on_click(event):
            # left click
            if event.button == 1:
                if event.inaxes is not None:
                    # get x and y values of click location
                    # this will be one vertex of the zoom region
                    self.click = (event.xdata, event.ydata)

            # right click
            elif event.button == 3:
                # if no limit adjustment has already been made, ignore
                if len(self.xlimit_history) == 1:
                    pass
                # if limit adjustment has been made, go back...
                else:
                    # if double click, go back to original
                    if event.dblclick:
                        self.xlimit_history = [self.xlimit_history[0]]
                        self.ylimit_history = [self.ylimit_history[0]]
                        self.xlim = self.xlimit_history[0]
                        self.ylim = self.ylimit_history[0]
                        self.ax.set_xlim(self.xlim)
                        self.ax.set_ylim(self.ylim)
                        self.canvas.draw_idle()

                    # if single click, go back to previous view
                    else:
                        del self.xlimit_history[-1]
                        del self.ylimit_history[-1]
                        self.xlim = self.xlimit_history[-1]
                        self.ylim = self.ylimit_history[-1]
                        self.ax.set_xlim(self.xlim)
                        self.ax.set_ylim(self.ylim)
                        self.canvas.draw_idle()

        def on_release(event):
            if event.button == 1:
                if event.inaxes is not None:
                    update = True
                    # determine the x co-ordindates of the zoomed region
                    self.release = (event.xdata, event.ydata)
                    if self.click[0] > self.release[0]:
                        self.xlim = (self.click[0], self.release[0])
                    elif self.click[0] < self.release[0]:
                        self.xlim = (self.release[0], self.click[0])
                    else:
                        update = False

                    # determine the y co-ordindates of the zoomed region
                    if self.click[1] < self.release[1]:
                        self.ylim = (self.click[1], self.release[1])
                    elif self.click[1] > self.release[1]:
                        self.ylim = (self.release[1], self.click[1])
                    else:
                        update = False

                    if update:
                        self.xlimit_history.append(self.xlim)
                        self.ylimit_history.append(self.ylim)
                        self.ax.set_xlim(self.xlim)
                        self.ax.set_ylim(self.ylim)
                        self.canvas.draw_idle()

        self.canvas.callbacks.connect('button_press_event', on_click)
        self.canvas.callbacks.connect('button_release_event', on_release)

        # --- REGION SELECTION ------------------------------------------------
        # Left bound
        self.lb_title = tk.Label(self.regionframe,
                                 text='left bound')

        self.lb_title.grid(row=0,
                           column=0,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        self.lb_scale = tk.Scale(self.regionframe,
                                  from_=1,
                                  to=self.n,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_lb_scale,
                                  showvalue=0)

        self.lb_scale.set(self.lb)

        self.lb_scale.grid(row=0,
                           column=1,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsew')

        self.lb_label = tk.StringVar()
        self.lb_label.set(f'{self.lb_ppm:.3f}')

        self.lb_entry = tk.Entry(self.regionframe,
                           textvariable=self.lb_label,
                           width=6)
        self.lb_entry.bind('<Return>', (lambda event: self.update_lb_entry()))
        self.lb_entry.grid(row=0,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # right bound
        self.rb_title = tk.Label(self.regionframe,
                                 text='right bound')

        self.rb_title.grid(row=1,
                           column=0,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        self.rb_scale = tk.Scale(self.regionframe,
                                 from_=1,
                                 to=self.n,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_rb_scale,
                                 showvalue=0)

        self.rb_scale.set(self.rb)
        self.rb_scale.grid(row=1,
                           column=1,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsew')


        self.rb_label = tk.StringVar()
        self.rb_label.set(f'{self.rb_ppm:.3f}')

        self.rb_entry = tk.Entry(self.regionframe,
                                 textvariable=self.rb_label,
                                 width=6)
        self.rb_entry.bind('<Return>', (lambda event: self.update_rb_entry()))
        self.rb_entry.grid(row=1,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # left noise bound
        self.lnb_title = tk.Label(self.regionframe,
                                  text='left noise bound')

        self.lnb_title.grid(row=2,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0),
                            sticky='nsw')

        self.lnb_scale = tk.Scale(self.regionframe,
                                   from_=1,
                                   to=self.n,
                                   orient=tk.HORIZONTAL,
                                   command=self.update_lnb_scale,
                                   showvalue=0,
                                   resolution=-1)

        self.lnb_scale.set(self.lnb)

        self.lnb_scale.grid(row=2,
                            column=1,
                            padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0),
                            sticky='nsew')


        self.lnb_label = tk.StringVar()
        self.lnb_label.set(f'{self.lnb_ppm:.3f}')

        self.lnb_entry = tk.Entry(self.regionframe,
                                 textvariable=self.lnb_label,
                                 width=6)
        self.lnb_entry.bind('<Return>', (lambda event: self.update_lnb_entry()))
        self.lnb_entry.grid(row=2,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # right noise bound
        self.rnb_title = tk.Label(self.regionframe,
                                  text='right noise bound')

        self.rnb_title.grid(row=3,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0),
                            sticky='nsw')

        self.rnb_scale = tk.Scale(self.regionframe,
                                  from_=1,
                                  to=self.n,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_rnb_scale,
                                  showvalue=0)

        self.rnb_scale.set(self.rnb)

        self.rnb_scale.grid(row=3,
                            column=1,
                            padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0),
                            sticky='nsew')


        self.rnb_label = tk.StringVar()
        self.rnb_label.set(f'{self.rnb_ppm:.3f}')

        self.rnb_entry = tk.Entry(self.regionframe,
                                 textvariable=self.rnb_label,
                                 width=6)
        self.rnb_entry.bind('<Return>', (lambda event: self.update_rnb_entry()))
        self.rnb_entry.grid(row=3,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # --- PHASE CORRECTION ------------------------------------------------
        # Pivot
        self.pivot_title = tk.Label(self.phaseframe,
                                    text='pivot')

        self.pivot_title.grid(row=0,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad/2, 0),
                              sticky='nsw')

        self.pivot_scale = tk.Scale(self.phaseframe,
                                    from_=1,
                                    to=self.n,
                                    orient=tk.HORIZONTAL,
                                    command=self.update_pivot_scale,
                                    showvalue=0)

        self.pivot_scale.set(self.pivot)

        self.pivot_scale.grid(row=0,
                              column=1,
                              padx=(self.pad/2, 0),
                              pady=(self.pad/2, 0),
                              sticky='nsew')

        self.pivot_label = tk.StringVar()
        self.pivot_label.set(f'{self.pivot_ppm:.3f}')

        self.pivot_entry = tk.Entry(self.phaseframe,
                                    textvariable=self.pivot_label,
                                    width=6)
        self.pivot_entry.bind('<Return>',
                              (lambda event: self.update_pivot_entry()))
        self.pivot_entry.grid(row=0,
                              column=2,
                              padx=(self.pad/2, self.pad/2),
                              pady=(self.pad/2, 0),
                              sticky='nsw')

        # zero-order phase
        self.p0_title = tk.Label(self.phaseframe,
                                 text='p0')

        self.p0_title.grid(row=1,
                           column=0,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        self.p0_scale = tk.Scale(self.phaseframe,
                                 from_=-np.pi,
                                 to=np.pi,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_p0_scale,
                                 showvalue=0,
                                 resolution=0.0001)

        self.p0_scale.set(self.p0)
        self.p0_scale.grid(row=1,
                           column=1,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsew')


        self.p0_label = tk.StringVar()
        self.p0_label.set(f'{self.p0:.3f}')

        self.p0_entry = tk.Entry(self.phaseframe,
                                 textvariable=self.p0_label,
                                 width=6)
        self.p0_entry.bind('<Return>', (lambda event: self.update_p0_entry()))
        self.p0_entry.grid(row=1,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # first-order phase
        self.p1_title = tk.Label(self.phaseframe,
                                 text='p1')

        self.p1_title.grid(row=2,
                           column=0,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        self.p1_scale = tk.Scale(self.phaseframe,
                                 from_=-4*np.pi,
                                 to=4*np.pi,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_p1_scale,
                                 showvalue=0,
                                 resolution=0.0001)

        self.p1_scale.set(self.p1)
        self.p1_scale.grid(row=2,
                           column=1,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsew')


        self.p1_label = tk.StringVar()
        self.p1_label.set(f'{self.p1:.3f}')

        self.p1_entry = tk.Entry(self.phaseframe,
                                 textvariable=self.p1_label,
                                 width=6)
        self.p1_entry.bind('<Return>', (lambda event: self.update_p1_entry()))
        self.p1_entry.grid(row=2,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # --- ADVANCED SETTINGS -----------------------------------------------
        self.adset_title = tk.Label(self.adsetframe,
                                    text='Advanced Settings',
                                    font=('Helvetica', 14))

        self.adset_title.grid(row=0,
                             column=0,
                             columnspan=3,
                             padx=(self.pad/2, 0),
                             pady=(self.pad, 0),
                             sticky='w')

        # number of points to consider in ITMPM
        self.mpm_label = tk.Label(self.adsetframe,
                                   text='Points for MPM:')

        self.mpm_label.grid(row=1,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='nsw')

        self.n_mpm = tk.Entry(self.adsetframe,
                               width=12)

        if self.n <= 4096:
            self.n_mpm.insert(0, str(self.n))
        else:
            self.n_mpm.insert(0, 4096)

        self.n_mpm.grid(row=1,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        val = int(np.floor(self.n/2))

        self.mpm_max_label = tk.Label(self.adsetframe,
                                      text=f'Max. value: {val}')

        self.mpm_max_label.grid(row=2,
                                column=1,
                                columnspan=2,
                                padx=(self.pad/2, 0),
                                pady=(self.pad/2, 0),
                                sticky='nw')

        # number of points to consider in NLP
        self.nlp_label = tk.Label(self.adsetframe,
                                  text='Points for NLP:')

        self.nlp_label.grid(row=3,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.n_nlp = tk.Entry(self.adsetframe,
                                 width=12)

        if self.n <= 8192:
            self.n_nlp.insert(0, str(self.n))
        else:
            self.n_nlp.insert(0, 8192)

        self.n_nlp.grid(row=3,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        self.nlp_max_label = tk.Label(self.adsetframe,
                                      text=f'Max. value: {val}')

        self.nlp_max_label.grid(row=4,
                                column=1,
                                columnspan=2,
                                padx=(self.pad/2, 0),
                                pady=(self.pad/2, 0),
                                sticky='nw')

        # maximum number of NLP iterations
        self.maxit_label = tk.Label(self.adsetframe,
                                     text='Max. Iterations:')

        self.maxit_label.grid(row=5,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.maxiter = tk.Entry(self.adsetframe,
                                 width=12)

        self.maxiter.insert(0, str(100))

        self.maxiter.grid(row=5,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        # NLP algorithm
        self.alg_label = tk.Label(self.adsetframe,
                                   text='NLP Method:')

        self.alg_label.grid(row=6,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.algorithm = tk.StringVar(self.adsetframe)
        self.algorithm.set('Trust Region')

        self.algoptions = tk.OptionMenu(self.adsetframe,
                                         self.algorithm,
                                         'Trust Region',
                                         'L-BFGS')

        self.algoptions.grid(row=6,
                             column=1,
                             columnspan=2,
                             padx=(self.pad/2, 0),
                             pady=(self.pad, 0))

        self.phasevar_label = tk.Label(self.adsetframe,
                                     text='Inc. Phase Variance:')

        self.phasevar_label.grid(row=7,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.phasevar = tk.StringVar()
        self.phasevar.set('1')
        self.phasevar_box = tk.Checkbutton(self.adsetframe,
                                         variable=self.phasevar)

        self.phasevar_box.grid(row=7,
                            column=1,
                            columnspan=2,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.descrip_label = tk.Label(self.adsetframe,
                                       text='Description:')

        self.descrip_label.grid(row=8,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='nw')

        self.descrip = tk.Text(self.adsetframe,
                               height=4,
                               width=16)

        self.descrip.grid(row=8,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.fname_label = tk.Label(self.adsetframe,
                                     text='Filename:')

        self.fname_label.grid(row=9,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.fname = tk.Entry(self.adsetframe,
                               width=16)

        self.fname.insert(0, 'NMREsPy_result')

        self.fname.grid(row=9,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        self.dir_label = tk.Label(self.adsetframe,
                                   text='Directory:')

        self.dir_label.grid(row=10,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.dir = tk.StringVar()

        self.dir_bar = tk.Entry(self.adsetframe,
                                 width=16)

        self.dir_bar.grid(row=10,
                          column=1,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.dir_button = tk.Button(self.adsetframe,
                                     command=self.browse,
                                     width=1)

        self.dir_button.grid(row=10,
                             padx=(self.pad/2, 0),
                             pady=(self.pad, 0),
                             column=2)

        self.txtfile_label = tk.Label(self.adsetframe,
                                       text='Save Textfile:')

        self.txtfile_label.grid(row=11,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='w')

        self.txtfile = tk.StringVar()
        self.txtfile.set('1')

        self.txt_box = tk.Checkbutton(self.adsetframe,
                                       variable=self.txtfile)

        self.txt_box.grid(row=11,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.pdffile_label = tk.Label(self.adsetframe,
                                       text='Save PDF:')

        self.pdffile_label.grid(row=12,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='w')

        self.pdffile = tk.StringVar()
        self.pdffile.set('0')

        self.pdf_box = tk.Checkbutton(self.adsetframe,
                                       variable=self.pdffile)

        self.pdf_box.grid(row=12,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        for i in range(1, 13):
            self.adsetframe.grid_rowconfigure(i, weight=1)
        self.adsetframe.columnconfigure(1, weight=1)

        # ======================
        # SAVE/HELP/QUIT BUTTONS
        # ======================

        self.cancel_button = tk.Button(self.buttonframe,
                                        text='Cancel',
                                        command=self.cancel,
                                        width=6)

        self.cancel_button.grid(row=0,
                                column=0)

        self.help_button = tk.Button(self.buttonframe,
                                      text='Help',
                                      command=self.load_help,
                                      width=6)

        self.help_button.grid(row=0,
                              column=1,
                              padx=(self.pad, 0))

        self.save_button = tk.Button(self.buttonframe,
                                      text='Save',
                                      command=self.save,
                                      width=6)

        self.save_button.grid(row=0,
                              column=2,
                              padx=(self.pad, 0))

        self.feedback = tk.Label(self.contactframe,
                                  text='For queries/feedback, contact')

        self.feedback.grid(row=0,
                           column=0,
                           sticky='w')

        self.email = tk.Label(self.contactframe,
                              text='simon.hulse@chem.ox.ac.uk',
                              font='Courier')

        self.email.grid(row=1,
                        column=0,
                        sticky='w')

    #TODO: these functions feel a bit longwinded. Could probably
    # achieve the same behaviour with fewer, more general ones
    # not particularly high priority though...

    def update_lb_scale(self, lb):
        lb = int(lb.split('.')[0])
        if lb < self.rb:
            self.lb = lb
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.lb_label.set(f'{self.lb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lb = self.rb - 1
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.lb_scale.set(self.lb)
            self.canvas.draw_idle()

    def update_lb_entry(self):
        lb_ppm = float(self.lb_label.get())
        lb = _misc.conv_ppm_idx(lb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')

        if lb < self.rb:
            self.lb = lb
            self.lb_ppm = lb_ppm
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.lb_scale.set(self.lb)
            self.canvas.draw_idle()
        else:
            self.lb_label.set(f'{self.lb_ppm:.3f}')

    def update_rb_scale(self, rb):
        rb = int(rb.split('.')[0])
        if rb > self.lb:
            self.rb = rb
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.rb_label.set(f'{self.rb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rb = self.lb + 1
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.rb_scale.set(self.rb)
            self.canvas.draw_idle()

    def update_rb_entry(self):
        rb_ppm = float(self.rb_label.get())
        rb = _misc.conv_ppm_idx(rb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')
        if rb > self.lb:
            self.rb = rb
            self.rb_ppm = rb_ppm
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.rb_scale.set(self.rb)
            self.canvas.draw_idle()
        else:
            self.rb_label.set(f'{self.rb_ppm:.3f}')

    def update_lnb_scale(self, lnb):
        lnb = int(lnb.split('.')[0])
        if lnb < self.rnb:
            self.lnb = lnb
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       4*self.ylim_init[1])
            self.lnb_label.set(f'{self.lnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lnb = self.rnb - 1
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.lnb_scale.set(self.lnb)
            self.canvas.draw_idle()

    def update_lnb_entry(self):
        lnb_ppm = float(self.lnb_label.get())
        lnb = _misc.conv_ppm_idx(lnb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')
        if lnb < self.rnb:
            self.lnb = lnb
            self.lnb_ppm = lnb_ppm
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       4*self.ylim_init[1])
            self.lnb_scale.set(self.lnb)
            self.canvas.draw_idle()
        else:
            self.lnb_label.set(f'{self.lnb_ppm:.3f}')

    def update_rnb_scale(self, rnb):
        rnb = int(rnb.split('.')[0])
        if rnb > self.lnb:
            self.rnb = rnb
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       4*self.ylim_init[1])
            self.rnb_label.set(f'{self.rnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rnb = self.lnb + 1
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                             self.n, direction='idx->ppm')
            self.rnb_scale.set(self.rnb)
            self.canvas.draw_idle()

    def update_rnb_entry(self):
        rnb_ppm = float(self.rnb_label.get())
        rnb = _misc.conv_ppm_idx(rnb_ppm, self.sw_p, self.off_p, self.n,
                                direction='ppm->idx')
        if rnb > self.lb:
            self.rnb = rnb
            self.rnb_ppm = rnb_ppm
            self.noiseregion.set_bounds(self.rnb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lnb_ppm - self.rnb_ppm,
                                       4*self.ylim_init[1])
            self.rnb_scale.set(self.rnb)
            self.canvas.draw_idle()
        else:
            self.rnb_label.set(f'{self.rnb_ppm:.3f}')

    def update_pivot_scale(self, pivot):

        self.pivot = int(pivot)
        self.pivot_ppm = _misc.conv_ppm_idx(self.pivot, self.sw_p, self.off_p,
                                            self.n, direction='idx->ppm')
        self.pivot_label.set(f'{self.pivot_ppm:.3f}')
        self.update_phase()
        x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
        self.pivotplot.set_xdata(x)

        self.canvas.draw_idle()


    def update_pivot_entry(self):

        try:
            self.pivot_ppm = float(self.pivot_label.get())
            self.pivot = _misc.conv_ppm_idx(self.pivot_ppm,
                                            self.sw_p,
                                            self.off_p,
                                            self.n,
                                            direction='ppm->idx')
            self.update_phase()
            x = np.linspace(self.pivot_ppm, self.pivot_ppm, 1000)
            self.pivotplot.set_xdata(x)
            self.pivot_scale.set(self.pivot)
            self.canvas.draw_idle()

        except:
            self.pivot_label.set(f'{self.pivot_ppm:.3f}')

    def update_p0_scale(self, p0):
        self.p0 = float(p0)
        self.p0_label.set(f'{self.p0:.3f}')
        self.update_phase()

    def update_p0_entry(self):
        try:
            self.p0 = float(self.p0_label.get())
            self.p0_scale.set(self.p0)
            self.update_phase()

        except:
            self.p0_label.set(f'{self.p0:.3f}')

    def update_p1_scale(self, p1):
        self.p1 = float(p1)
        self.p1_label.set(f'{self.p1:.3f}')
        self.update_phase()

    def update_p1_entry(self):
        try:
            self.p1 = float(self.p1_label.get())
            self.p1_scale.set(self.p1)
            self.update_phase()

        except:
            self.p1_label.set(f'{self.p1:.3f}')


    def update_phase(self):

        newspec = np.real(self.spec * np.exp(1j * (self.p0 + (self.p1 * \
        np.arange(-self.pivot, -self.pivot + self.n, 1) / self.n))))

        self.specplot.set_ydata(newspec)
        self.canvas.draw_idle()

    def update_plot(self, _):
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

    def browse(self):
        self.dir = filedialog.askdirectory()
        self.dir_bar.insert(0, self.dir)

    def load_help(self):
        import webbrowser
        webbrowser.open('http://foroozandeh.chem.ox.ac.uk/home')

    def save(self):
        self.lb = int(np.floor(self.lb))
        self.rb = int(np.ceil(self.rb))
        self.lnb = int(np.floor(self.lnb))
        self.rnb = int(np.ceil(self.rnb))
        self.mpm_points = int(self.n_mpm.get())
        self.nlp_points = int(self.n_nlp.get())
        self.maxiter = int(self.maxiter.get())
        self.alg = self.algorithm.get()
        self.pv = self.phasevar.get()
        self.descrip = self.descrip.get('1.0', tk.END)
        self.file = self.fname.get()
        self.dir = self.dir_bar.get()
        self.txt = self.txtfile.get()
        self.pdf = self.pdffile.get()
        self.master.destroy()

    def cancel(self):
        self.master.destroy()
        print("NMR-ESPY Cancelled :'(")
        exit()


class ResultGUI:

    def __init__(self, master, info):

        self.master = master
        self.info = info

        self.fig, self.ax, self.lines, self.labels = self.info.plot_result()
        self.fig.set_dpi(170)
        self.fig.set_size_inches(6, 3.5)

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


class dtypeGUI:

    def __init__(self, master, fidpath, pdatapath):

        self.master = master
        self.master.resizable(0, 0)

        self.logoframe = tk.Frame(self.master)
        self.logoframe.grid(row=0,
                            column=0,
                            rowspan=2)

        self.mainframe = tk.Frame(self.master)
        self.mainframe.grid(row=0,
                            column=1)

        self.buttonframe = tk.Frame(self.master)
        self.buttonframe.grid(row=1,
                              column=1,
                              sticky='e')

        self.pad = 10

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
                                      image=self.nmrespy_img)

        self.nmrespy_logo.pack(padx=(self.pad, self.pad),
                               pady=self.pad)

        self.message = tk.Label(self.mainframe,
                                 text='Which data would you like to analyse?',
                                 font=('Helvetica', '14'))
        self.message.grid(column=0,
                          row=0,
                          columnspan=2,
                          padx=self.pad,
                          pady=(self.pad, 0))

        self.pdata_label = tk.Label(self.mainframe,
                                     text='Processed Data')
        self.pdata_label.grid(column=0,
                              row=1,
                              padx=(self.pad, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.pdatapath = tk.Label(self.mainframe,
                                   text=f'{pdatapath}/1r',
                                   font='Courier')
        self.pdatapath.grid(column=0,
                            row=2,
                            padx=(self.pad, 0),
                            sticky='w')

        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = tk.Checkbutton(self.mainframe,
                                         variable=self.pdata,
                                         command=self.click_pdata)
        self.pdata_box.grid(column=1,
                            row=1,
                            rowspan=2,
                            padx=self.pad,
                            sticky='nsw')

        self.fid_label = tk.Label(self.mainframe,
                                     text='Raw FID')
        self.fid_label.grid(column=0,
                              row=3,
                              padx=(self.pad, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.fidpath = tk.Label(self.mainframe,
                                   text=f'{fidpath}/fid',
                                   font='Courier')
        self.fidpath.grid(column=0,
                            row=4,
                            padx=(self.pad, 0),
                            sticky='w')

        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = tk.Checkbutton(self.mainframe,
                                       variable=self.fid,
                                       command=self.click_fid)
        self.fid_box.grid(column=1,
                          row=3,
                          rowspan=2,
                          padx=self.pad,
                          sticky='nsw')

        self.confirmbutton = tk.Button(self.buttonframe,
                                        text='Confirm',
                                        command=self.confirm)

        self.confirmbutton.grid(column=1,
                                row=0,
                                padx=(self.pad/2, self.pad),
                                pady=(self.pad, self.pad),
                                sticky='e')

        self.confirmbutton = tk.Button(self.buttonframe,
                                        text='Cancel',
                                        command=self.cancel)
        self.confirmbutton.grid(column=0,
                                row=0,
                                pady=(self.pad, self.pad),
                                sticky='e')

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


if __name__ == '__main__':

    # path to nmrespy directory
    espypath = os.path.dirname(nmrespy.__file__)

    # extract path information
    infopath = os.path.join(espypath, 'topspin/tmp/info.txt')
    try:
        with open(infopath, 'r') as fh:
            from_topspin = fh.read().split(' ')
    except:
        raise IOError(f'No file of path {infopath} found')

    # import dictionary of spectral info
    fidpath = from_topspin[0]
    pdatapath = from_topspin[1]

    root = tk.Tk()
    dtype_app = dtypeGUI(root, fidpath, pdatapath)
    root.mainloop()

    dtype = dtype_app.dtype
    if dtype == 'fid':
        info = load.import_bruker_fid(fidpath, ask_convdta=False)
    elif dtype == 'pdata':
        info = load.import_bruker_pdata(pdatapath)

    print(info.get_n())
    root = tk.Tk()
    main_app = NMREsPyGUI(root, info)
    root.mainloop()

    lb_ppm = main_app.lb_ppm,
    rb_ppm = main_app.rb_ppm,
    lnb_ppm = main_app.lnb_ppm,
    rnb_ppm = main_app.rnb_ppm,
    mpm_points = main_app.mpm_points,
    nlp_points = main_app.nlp_points,
    maxit = main_app.maxiter
    alg = main_app.alg

    if alg == 'Trust Region':
        alg = 'trust_region'
    elif alg == 'L-BFGS':
        alg = 'lbfgs'

    pv = main_app.pv
    if pv == '1':
        pv = True
    else:
        pv = False

    descrip = main_app.descrip
    file = main_app.file
    dir = main_app.dir

    txt = main_app.txt
    if txt == '1':
        txt = True
    else:
        txt = False

    pdf = main_app.pdf
    if pdf == '1':
        pdf = True
    else:
        pdf = False

    info.virtual_echo(highs=lb_ppm,
                      lows=rb_ppm,
                      highs_n=lnb_ppm,
                      lows_n=rnb_ppm)

    info.matrix_pencil(trim=mpm_points)

    info.nonlinear_programming(trim=nlp_points,
                               maxit=maxit,
                               method=alg,
                               phase_variance=pv)

    if txt:
        info.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True)
    if pdf:
        info.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True, format='pdf')

    root = tk.Tk()
    res_app = ResultGUI(root, info)
    root.mainloop()
