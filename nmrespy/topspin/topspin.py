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
    def __init__(self, master, finfo, pinfo):

        self.master = master
        self.master.configure(background='white')
        self.master.title('NMR-ESPY')

        # spectra from pdata and raw FID
        self.p_spec = pinfo.get_data(pdata_key='1r')
        self.f_spec = np.real(fftshift(fft(finfo.get_data())))

        # basic info
        self.shifts_p = pinfo.get_shifts(unit='ppm')[0]
        self.shifts_f = finfo.get_shifts(unit='ppm')[0]
        self.sw_p = finfo.get_sw(unit='ppm')[0]
        self.off_p = finfo.get_offset(unit='ppm')[0]
        self.n_f = self.f_spec.shape[0]
        self.n_p = self.p_spec.shape[0]

        # left and right bounds (idx and ppm)
        self.lb = int(np.floor(7 * self.n_p / 16))
        self.rb = int(np.floor(9 * self.n_p / 16))
        self.lnb = int(np.floor(1 * self.n_p / 16))
        self.rnb = int(np.floor(2 * self.n_p / 16))

        self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                         self.n_p, direction='idx->ppm')

        self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                         self.n_p, direction='idx->ppm')

        self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                          self.n_p, direction='idx->ppm')

        self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                          self.n_p, direction='idx->ppm')

        # get GUI images
        self.espypath = os.path.dirname(nmrespy.__file__)
        self.nmrespy_image = Image.open(os.path.join(self.espypath,
                                        'topspin/images/nmrespy_full.png'))
        self.regsel_image = Image.open(os.path.join(self.espypath,
                                       'topspin/images/region_selection.png'))
        self.adset_image = Image.open(os.path.join(self.espypath,
                                      'topspin/images/advanced_settings.png'))

        # ======
        # FRAMES
        # ======

        self.pad = 10

        # leftframe -> spectrum plot and region sliders
        self.leftframe = ttk.Frame(self.master)

        self.leftframe.grid(column=0,
                            row=0,
                            sticky='nsew')

        self.rightframe = ttk.Frame(self.master)

        self.rightframe.grid(column=1,
                             row=0,
                             sticky='nsew')

        self.plotframe = ttk.Frame(self.leftframe)

        self.plotframe.grid(column=0,
                            row=0,
                            padx=(self.pad, 0),
                            pady=(self.pad, 0),
                            sticky='nsew')

        self.scaleframe = ttk.Frame(self.leftframe)

        self.scaleframe.grid(column=0,
                             row=1,
                             padx=(self.pad, 0),
                             pady=(self.pad, self.pad),
                             sticky='ew')

        self.logoframe = ttk.Frame(self.rightframe)

        self.logoframe.grid(column=0,
                            row=0,
                            padx=(self.pad, self.pad),
                            pady=(self.pad, 0),
                            sticky='ew')

        self.adsetframe = ttk.Frame(self.rightframe)

        self.adsetframe.grid(column=0,
                             row=1,
                             padx=(self.pad, self.pad),
                             pady=(self.pad, 0),
                             sticky='new')

        self.buttonframe = ttk.Frame(self.rightframe)

        self.buttonframe.grid(column=0,
                              row=2,
                              padx=(self.pad, self.pad),
                              pady=(self.pad, 0),
                              sticky='se')

        self.contactframe = ttk.Frame(self.rightframe)

        self.contactframe.grid(column=0,
                               row=3,
                               padx=(self.pad, self.pad),
                               pady=(self.pad, self.pad),
                               sticky='sw')

        # make frames expandable
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)


        self.leftframe.grid_columnconfigure(0, weight=1)
        self.leftframe.grid_rowconfigure(0, weight=1)

        self.rightframe.grid_rowconfigure(1, weight=1)

        self.plotframe.grid_columnconfigure(0, weight=1)
        self.plotframe.grid_rowconfigure(0, weight=1)
        self.logoframe.grid_columnconfigure(0, weight=1)
        self.logoframe.grid_rowconfigure(0, weight=1)


        # =============
        # NMR-ESPY LOGO
        # =============

        scale = 0.08
        [w, h] = self.nmrespy_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.nmrespy_image = self.nmrespy_image.resize((new_w, new_h),
                                                       Image.ANTIALIAS)
        self.nmrespy_img = ImageTk.PhotoImage(self.nmrespy_image)

        self.nmrespy_logo = ttk.Label(self.logoframe,
                                      image=self.nmrespy_img)

        self.nmrespy_logo.pack()

        # =============
        # SPECTRUM PLOT
        # =============

        # create plot
        self.fig = Figure(figsize=(6,3.5), dpi=170)
        self.ax = self.fig.add_subplot(111)
        self.specplot = self.ax.plot(self.shifts_p,
                                     self.p_spec,
                                     color='k',
                                     lw=0.6)[0]

        self.xlim = (self.shifts_p[0], self.shifts_p[-1])
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
        self.filtregion = Rectangle((self.rb_ppm, -2*self.ylim_init[1]),
                                     self.lb_ppm - self.rb_ppm,
                                     4*self.ylim_init[1],
                                     facecolor='#7fd47f')

        self.ax.add_patch(self.filtregion)

        # highlight the noise region
        self.noiseregion = Rectangle((self.rnb_ppm, -2*self.ylim_init[1]),
                                      self.lnb_ppm - self.rnb_ppm,
                                      4*self.ylim_init[1],
                                      facecolor='#66b3ff')

        self.ax.add_patch(self.noiseregion)

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


        # ================
        # REGION SELECTION
        # ================

        # scale 'Region Selction' title image and place into app
        scale = 0.42
        [w, h] = self.regsel_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.regsel_image = self.regsel_image.resize((new_w, new_h),
                                                     Image.ANTIALIAS)
        self.regsel_img = ImageTk.PhotoImage(self.regsel_image)
        self.regsel_logo = ttk.Label(self.scaleframe,
                                    image=self.regsel_img)

        self.regsel_logo.grid(row=1,
                              column=0,
                              columnspan=3,
                              sticky='nsw')

        # Left bound
        self.lb_title = ttk.Label(self.scaleframe,
                                 text='left bound')

        self.lb_title.grid(row=2,
                           column=0,
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        self.lb_slide = ttk.Scale(self.scaleframe,
                                  from_=1,
                                  to=self.n_p,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_lb)

        self.lb_slide.set(self.lb)

        self.lb_slide.grid(row=2,
                           column=1,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsew')


        self.lb_label = ttk.Label(self.scaleframe,
                                  text=f'{self.lb_ppm:.3f}')

        self.lb_label.grid(row=2,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # right bound
        self.rb_title = ttk.Label(self.scaleframe,
                                 text='right bound')

        self.rb_title.grid(row=3,
                           column=0,
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        self.rb_slide = ttk.Scale(self.scaleframe,
                                 from_=1,
                                 to=self.n_p,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_rb)

        self.rb_slide.set(self.rb)

        self.rb_slide.grid(row=3,
                           column=1,
                           padx=(self.pad/2, 0),
                           pady=(self.pad/2, 0),
                           sticky='nsew')


        self.rb_label = ttk.Label(self.scaleframe,
                                 text=f'{self.rb_ppm:.3f}')

        self.rb_label.grid(row=3,
                           column=2,
                           padx=(self.pad/2, self.pad/2),
                           pady=(self.pad/2, 0),
                           sticky='nsw')

        # left noise bound
        self.lnb_title = ttk.Label(self.scaleframe,
                                  text='left noise bound')

        self.lnb_title.grid(row=4,
                            column=0,
                            pady=(self.pad/2, 0),
                            sticky='nsw')

        self.lnb_slide = ttk.Scale(self.scaleframe,
                                   from_=1,
                                   to=self.n_p,
                                   orient=tk.HORIZONTAL,
                                   command=self.update_lnb)

        self.lnb_slide.set(self.lnb)

        self.lnb_slide.grid(row=4,
                            column=1,
                            padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0),
                            sticky='nsew')


        self.lnb_label = ttk.Label(self.scaleframe,
                                   text=f'{self.lnb_ppm:.3f}')

        self.lnb_label.grid(row=4,
                            column=2,
                            padx=(self.pad/2, self.pad/2),
                            pady=(self.pad/2, 0),
                            sticky='nsw')

        # right noise bound
        self.rnb_title = ttk.Label(self.scaleframe,
                                  text='right noise bound')

        self.rnb_title.grid(row=5,
                            column=0,
                            pady=(self.pad/2, 0),
                            sticky='nsw')

        self.rnb_slide = ttk.Scale(self.scaleframe,
                                  from_=1,
                                  to=self.n_p,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_rnb)

        self.rnb_slide.set(self.rnb)

        self.rnb_slide.grid(row=5,
                            column=1,
                            padx=(self.pad/2, 0),
                            pady=(self.pad/2, 0),
                            sticky='nsew')


        self.rnb_label = tk.Label(self.scaleframe,
                                  text=f'{self.rnb_ppm:.3f}')

        self.rnb_label.grid(row=5,
                            column=2,
                            padx=(self.pad/2, self.pad/2),
                            pady=(self.pad/2, 0),
                            sticky='nsw')

        self.scaleframe.columnconfigure(1, weight=1)

        # =================
        # ADVANCED SETTINGS
        # =================

        scale = 0.42
        [w, h] = self.adset_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.adset_image = self.adset_image.resize((new_w, new_h),
                                                   Image.ANTIALIAS)
        self.adset_img = ImageTk.PhotoImage(self.adset_image)

        self.adset_logo = ttk.Label(self.adsetframe,
                                    image=self.adset_img)

        self.adset_logo.grid(row=0,
                             column=0,
                             columnspan=3,
                             padx=(self.pad/2, 0),
                             pady=(self.pad, 0),
                             sticky='w')

        # number of points to consider in ITMPM
        self.mpm_label = ttk.Label(self.adsetframe,
                                   text='Points for MPM:')

        self.mpm_label.grid(row=1,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='nsw')

        self.n_mpm = ttk.Entry(self.adsetframe,
                               width=12)

        if self.n_p <= 4096:
            self.n_mpm.insert(0, str(self.n_p))
        else:
            self.n_mpm.insert(0, 4096)

        self.n_mpm.grid(row=1,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        val = int(np.floor(self.n_p/2))

        self.mpm_max_label = ttk.Label(self.adsetframe,
                                      text=f'Max. value: {val}')

        self.mpm_max_label.grid(row=2,
                                column=1,
                                columnspan=2,
                                padx=(self.pad/2, 0),
                                pady=(self.pad/2, 0),
                                sticky='nw')

        # number of points to consider in NLP
        self.nlp_label = ttk.Label(self.adsetframe,
                                  text='Points for NLP:')

        self.nlp_label.grid(row=3,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.n_nlp = ttk.Entry(self.adsetframe,
                                 width=12)

        if self.n_p <= 8192:
            self.n_nlp.insert(0, str(self.n_p))
        else:
            self.n_nlp.insert(0, 8192)

        self.n_nlp.grid(row=3,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        self.nlp_max_label = ttk.Label(self.adsetframe,
                                      text=f'Max. value: {val}')

        self.nlp_max_label.grid(row=4,
                                column=1,
                                columnspan=2,
                                padx=(self.pad/2, 0),
                                pady=(self.pad/2, 0),
                                sticky='nw')

        # maximum number of NLP iterations
        self.maxit_label = ttk.Label(self.adsetframe,
                                     text='Max. Iterations:')

        self.maxit_label.grid(row=5,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.maxiter = ttk.Entry(self.adsetframe,
                                 width=12)

        self.maxiter.insert(0, str(100))

        self.maxiter.grid(row=5,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        # NLP algorithm
        self.alg_label = ttk.Label(self.adsetframe,
                                   text='NLP Method:')

        self.alg_label.grid(row=6,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.algorithm = tk.StringVar(self.adsetframe)
        self.algorithm.set('Trust Region')

        self.algoptions = ttk.OptionMenu(self.adsetframe,
                                         self.algorithm,
                                         'Trust Region',
                                         'L-BFGS')

        self.algoptions.grid(row=6,
                             column=1,
                             columnspan=2,
                             padx=(self.pad/2, 0),
                             pady=(self.pad, 0))

        self.phase_label = ttk.Label(self.adsetframe,
                                     text='Inc. Phase Variance:')

        self.phase_label.grid(row=7,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.phase_var = tk.StringVar()
        self.phase_var.set('1')
        self.phase_box = ttk.Checkbutton(self.adsetframe,
                                         variable=self.phase_var)

        self.phase_box.grid(row=7,
                            column=1,
                            columnspan=2,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.descrip_label = ttk.Label(self.adsetframe,
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

        self.fname_label = ttk.Label(self.adsetframe,
                                     text='Filename:')

        self.fname_label.grid(row=9,
                              column=0,
                              padx=(self.pad/2, 0),
                              pady=(self.pad, 0),
                              sticky='w')

        self.fname = ttk.Entry(self.adsetframe,
                               width=16)

        self.fname.insert(0, 'NMREsPy_result')

        self.fname.grid(row=9,
                        column=1,
                        columnspan=2,
                        padx=(self.pad/2, 0),
                        pady=(self.pad, 0),
                        sticky='w')

        self.dir_label = ttk.Label(self.adsetframe,
                                   text='Directory:')

        self.dir_label.grid(row=10,
                            column=0,
                            padx=(self.pad/2, 0),
                            pady=(self.pad, 0),
                            sticky='w')

        self.dir = tk.StringVar()

        self.dir_bar = ttk.Entry(self.adsetframe,
                                 width=16)

        self.dir_bar.grid(row=10,
                          column=1,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.dir_button = ttk.Button(self.adsetframe,
                                     command=self.browse,
                                     width=1)

        self.dir_button.grid(row=10,
                             padx=(self.pad/2, 0),
                             pady=(self.pad, 0),
                             column=2)

        self.txtfile_label = ttk.Label(self.adsetframe,
                                       text='Save Textfile:')

        self.txtfile_label.grid(row=11,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='w')

        self.txtfile = tk.StringVar()
        self.txtfile.set('1')

        self.txt_box = ttk.Checkbutton(self.adsetframe,
                                       variable=self.txtfile)

        self.txt_box.grid(row=11,
                          column=1,
                          columnspan=2,
                          padx=(self.pad/2, 0),
                          pady=(self.pad, 0),
                          sticky='w')

        self.pdffile_label = ttk.Label(self.adsetframe,
                                       text='Save PDF:')

        self.pdffile_label.grid(row=12,
                                column=0,
                                padx=(self.pad/2, 0),
                                pady=(self.pad, 0),
                                sticky='w')

        self.pdffile = tk.StringVar()
        self.pdffile.set('0')

        self.pdf_box = ttk.Checkbutton(self.adsetframe,
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

        self.cancel_button = ttk.Button(self.buttonframe,
                                        text='Cancel',
                                        command=self.cancel,
                                        width=6)

        self.cancel_button.grid(row=0,
                                column=0)

        self.help_button = ttk.Button(self.buttonframe,
                                      text='Help',
                                      command=self.load_help,
                                      width=6)

        self.help_button.grid(row=0,
                              column=1,
                              padx=(self.pad, 0))

        self.save_button = ttk.Button(self.buttonframe,
                                      text='Save',
                                      command=self.save,
                                      width=6)

        self.save_button.grid(row=0,
                              column=2,
                              padx=(self.pad, 0))

        self.feedback = ttk.Label(self.contactframe,
                                  text='For queries/feedback, contact')

        self.feedback.grid(row=0,
                           column=0,
                           sticky='w')

        self.email = ttk.Label(self.contactframe,
                              text='simon.hulse@chem.ox.ac.uk',
                              font='Courier')

        self.email.grid(row=1,
                        column=0,
                        sticky='w')


    def update_lb(self, lb):
        lb = int(lb.split('.')[0])
        if lb < self.rb:
            self.lb = lb
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n_p, direction='idx->ppm')
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.lb_label.config(text=f'{self.lb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lb = self.rb - 1
            self.lb_ppm = _misc.conv_ppm_idx(self.lb, self.sw_p, self.off_p,
                                             self.n_p, direction='idx->ppm')
            self.lb_slide.set(self.lb_ppm)
            self.canvas.draw_idle()

    def update_rb(self, rb):
        rb = int(rb.split('.')[0])
        if rb > self.lb:
            self.rb = rb
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n_p, direction='idx->ppm')
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.rb_label.config(text=f'{self.rb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rb = self.lb + 1
            self.rb_ppm = _misc.conv_ppm_idx(self.rb, self.sw_p, self.off_p,
                                             self.n_p, direction='idx->ppm')
            self.rb_slide.set(self.rb_ppm)
            self.canvas.draw_idle()

    def update_lnb(self, lnb):
        lnb = int(lnb.split('.')[0])
        if lnb < self.rnb:
            self.lnb = lnb
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                              self.n_p, direction='idx->ppm')
            self.noiseregion.set_bounds(self.rnb_ppm,
                                        -2*self.ylim_init[1],
                                        self.lnb_ppm - self.rnb_ppm,
                                        4*self.ylim_init[1])
            self.lnb_label.config(text=f'{self.lnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lnb = self.rnb - 1
            self.lnb_ppm = _misc.conv_ppm_idx(self.lnb, self.sw_p, self.off_p,
                                              self.n_p, direction='idx->ppm')
            self.lnb_slide.set(self.lnb_ppm)
            self.canvas.draw_idle()

    def update_rnb(self, rnb):
        rnb = int(rnb.split('.')[0])
        if rnb > self.lnb:
            self.rnb = rnb
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                              self.n_p, direction='idx->ppm')
            self.noiseregion.set_bounds(self.rnb_ppm,
                                        -2*self.ylim_init[1],
                                        self.lnb_ppm - self.rnb_ppm,
                                        4*self.ylim_init[1])
            self.rnb_label.config(text=f'{self.rnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rnb = self.lnb + 1
            self.rnb_ppm = _misc.conv_ppm_idx(self.rnb, self.sw_p, self.off_p,
                                              self.n_p, direction='idx->ppm')
            self.rnb_slide.set(self.rnb_ppm)
            self.canvas.draw_idle()

    def browse(self):
        self.dir = filedialog.askdirectory()
        self.dir_bar.insert(0, self.dir)

    def untick_csv(self):
        self.csvfile.set('0')

    def load_help(self):
        import webbrowser
        path = os.path.join(self.path, 'topspin/docs/help.pdf')
        webbrowser.open_new(path)

    def save(self):
        self.lb = int(np.floor(self.lb))
        self.rb = int(np.ceil(self.rb))
        self.lnb = int(np.floor(self.lnb))
        self.rnb = int(np.ceil(self.rnb))
        self.mpm_points = int(self.n_mpm.get())
        self.nlp_points = int(self.n_nlp.get())
        self.maxiter = int(self.maxiter.get())
        self.alg = self.algorithm.get()
        self.pv = self.phase_var.get()
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

    fidinfo = load.import_bruker_fid(fidpath, ask_convdta=False)
    pdatainfo = load.import_bruker_pdata(pdatapath)

    root = tk.Tk()
    app = NMREsPyGUI(root, fidinfo, pdatainfo)
    root.mainloop()

    lb_ppm = app.lb_ppm,
    rb_ppm = app.rb_ppm,
    lnb_ppm = app.lnb_ppm,
    rnb_ppm = app.rnb_ppm,
    mpm_points = app.mpm_points,
    nlp_points = app.nlp_points,
    maxit = app.maxiter
    alg = app.alg

    if alg == 'Trust Region':
        alg = 'trust_region'
    elif alg == 'L-BFGS':
        alg = 'lbfgs'

    pv = app.pv
    if pv == '1':
        pv = True
    else:
        pv = False

    descrip = app.descrip
    file = app.file
    dir = app.dir

    txt = app.txt
    if txt == '1':
        txt = True
    else:
        txt = False

    pdf = app.pdf
    if pdf == '1':
        pdf = True
    else:
        pdf = False

    pdatainfo.virtual_echo(highs=lb_ppm,
                           lows=rb_ppm,
                           highs_n=lnb_ppm,
                           lows_n=rnb_ppm)

    pdatainfo.matrix_pencil(trim=mpm_points)

    pdatainfo.nonlinear_programming(trim=nlp_points,
                                    maxit=maxit,
                                    method=alg,
                                    phase_variance=pv)

    if txt:
        pdatainfo.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True)
    if pdf:
        pdatainfo.write_result(descrip=descrip, fname=file, dir=dir,
                               force_overwrite=True, format='pdf')

    pdatainfo.plot_result()


    io.save_info(info, fname=file, dir=dir)
    """
    if txt:
        descr = "NMR-ESPY - MPM result"
        io.write_para(x0, sw_hz, offset_hz, sfo, mpm_points, descr,
                      filename=file+'_mpm', dir=dir, order='freq')

        descr = "NMR-ESPY - NLP result"
        io.write_para(x, sw_hz, offset_hz, sfo, nlp_points, descr,
                      filename=file+'_nlp', dir=dir, order='freq')

    if pdf:
        descr = "NMR-ESPY result"
        io.write_para(x, sw_hz, offset_hz, sfo, nlp_points, descr,
                      filename=fname, order='freq', format='pdf')

    if csv:
        descr = "NMR-ESPY - NLP result"
        io.write_para(x, sw_hz, offset_hz, sfo, nlp_points, descr,
                      filename='nlp_result', order='freq', format='csv')
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.plot(info['shifts'], info['1r'], color='k')
    M = info['x'].shape[0]
    cols = cm.viridis(np.linspace(0, 1, M))
    for m, c in zip(range(M), cols):
        i = copy(info)
        i['p'] = i['x'][m]
        i = core.make_fid(i, parakey='p')
        s = fftshift(fft(i['synthetic_signal']))
        ax.plot(info['shifts'], np.real(s), color=c)
    ax.set_xlim(info['left_bound_ppm'], info['right_bound_ppm'])
    plt.show()
    exit()

    """
    # unpack parameters
    lb = params[0]
    rb = params[1]
    lnb = params[2]
    rnb = params[3]

    mpm_trunc = ad_params[0]
    nlp_trunc = ad_params[1]
    mit = ad_params[2]
    alg = ad_params[3]
    pc = ad_params[4]
    txt = ad_params[5]
    pdf = ad_params[6]
    csv = ad_params[7]
    fn = ad_params[8]
    """
    if alg == 'Trust Region':
        alg = 'trust-region'
    elif alg == 'L-BFGS':
        alg = 'lbfgs'
    else:
        print('HUH!?')
        exit()

    for param in [pc, txt, pdf, csv]:
        if param == 1:
            param = True
        else:
            param = False

    # checks to ensure bounds make sense
    if lb is not None and rb is not None:
        if lb <= rb:
            errmsg = 'The left bound (%s ppm) should be larger than the' %str(lb) \
                     + ' right bound (%s ppm).' %str(rb)
            root = tk.Tk()
            app = warnGUI(root, espypath, errmsg)
            root.mainloop()
            exit()

        if lb > dict['shifts'][0]:
            ls = round(dict['shifts'][0], 4)
            rs = round(dict['shifts'][-1], 4)
            errmsg = 'The left bound (%s ppm) and right bound' %str(lb) \
                     + '  (%s ppm) shoudld lie within the range' %str(rb) \
                     + ' %s to %s ppm.' %(ls, rs)
            root = tk.Tk()
            app = warnGUI(root, espypath, errmsg)
            root.mainloop()
            exit()

        if rb < dict['shifts'][-1]:
            ls = round(dict['shifts'][0], 4)
            rs = round(dict['shifts'][-1], 4)
            errmsg = 'The left bound (%s ppm) and right bound' %str(lb) \
                     + '  (%s ppm) shoudld lie within the range' %str(rb) \
                     + ' %s to %s ppm.' %(ls, rs)
            root = tk.Tk()
            app = warnGUI(root, espypath, errmsg)
            root.mainloop()
            exit()

    if lnb is not None and rnb is not None:
        if lnb <= rnb:
            errmsg = 'The left noise bound (%s ppm) should be larger' %str(lnb) \
                     + ' than the right noise bound (%s ppm).' %str(rnb)
            root = tk.Tk()
            app = warnGUI(root, espypath, errmsg)
            root.mainloop()
            exit()

        if lnb > dict['shifts'][0]:
            ls = round(dict['shifts'][0], 4)
            rs = round(dict['shifts'][-1], 4)
            errmsg = 'The left noise bound (%s ppm) and right noise' %str(lnb) \
                     + ' bound (%s ppm) should lie within the range' %str(rnb) \
                     + ' %s to %s ppm.' %(ls, rs)
            root = tk.Tk()
            app = warnGUI(root, espypath, errmsg)
            root.mainloop()
            exit()

        if rnb < dict['shifts'][-1]:
            ls = round(dict['shifts'][0], 4)
            rs = round(dict['shifts'][-1], 4)
            errmsg = 'The left noise bound (%s ppm) and right noise' %str(lnb) \
                     + ' bound (%s ppm) should lie within the range' %str(rnb) \
                     + ' %s to %s ppm.' %(ls, rs)
            root = tk.Tk()
            app = warnGUI(root, espypath, errmsg)
            root.mainloop()
            exit()

    if mpm_trunc == None:
        pass

    elif mpm_trunc > si:
        errmsg = 'The number of points specified for consideration in the' \
                 + ' Matrix Pencil Method (%s) is larger ' %str(mpm_trunc) \
                 + ' than the number of signal points (%s).' %str(si)
        root = tk.Tk()
        app = warnGUI(root, espypath, errmsg)
        root.mainloop()
        exit()

    if nlp_trunc == None:
        pass

    elif nlp_trunc > si:
        errmsg = 'The number of points specified for consideration durring' \
                 + ' Nonlinear Programming (%s) is larger ' %str(nlp_trunc) \
                 + ' than the number of signal points (%s).' %str(si)
        root = tk.Tk()
        app = warnGUI(root, espypath, errmsg)
        root.mainloop()
        exit()

    # derive the virtual echo and analyse
    if lb is None and rb is None and lnb is None and rnb is None:
        ve = ne.virtual_echo_1d(pdata_path, bound='None', noise_bound='None')

    elif lnb is None and rnb is None:
        ve = ne.virtual_echo_1d(pdata_path, bound=(lb, rb), noise_bound=None)

    else:
        ve = ne.virtual_echo_1d(pdata_path, bound=(lb, rb), noise_bound=(lnb, rnb))

    I_in = 0
    x0 = ne.matrix_pencil(ve[:mpm_trunc], 0, dict['sw_hz'])

    x = ne.nonlinear_programming(ve[:nlp_trunc], dict['sw_hz'], x0,
                                 method=alg, maxit=mit, phase_correct=pc)

    fid = ne.make_fid(x, dict['sw_hz'], N=dict['si'])

    twoup = os.path.dirname(os.path.dirname(pdata_path))
    # path of new directory
    for i in range(100):
        res_path = '%s00%s' %(twoup,  str(i))
        if os.path.exists(res_path):
            continue
        else:
            break

    shutil.copytree(src=twoup, dst=res_path)
    descr = 'Estimation of signal parameters using NMR-EsPy'

    if txt:
        io.write_para(x, descr, dir=res_path)
    if pdf:
        io.write_para(x, descr, dir=res_path, format='pdf')

    #========================================================
    # TODO
    # if csv:
    #     io.write_para(x, descr, dir=res_path, format='csv')
    #========================================================

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from numpy.fft import fftshift, fft

    l = int(np.ceil((((dict['offset_ppm'] + (dict['sw_ppm'] / 2) - lb) / dict['sw_ppm']) * dict['si'])))
    r = int(np.floor((((dict['offset_ppm'] + (dict['sw_ppm'] / 2) - rb) / dict['sw_ppm']) * dict['si'])))

    fig, ax = plt.subplots()
    ax.plot(dict['shifts'], np.real(dict['spec']), color='k')
    cols = cm.viridis(np.linspace(0, 1, x.shape[0]))
    for i, c in zip(np.arange(x.shape[0]), cols):
        f = ne.make_fid(x[i], dict['sw_hz'], N=dict['si'])
        ax.plot(dict['shifts'], np.real(fftshift(fft(f))), color=c)

    ax.invert_xaxis()
    ax.set_xlim(lb, rb)
    ax.set_ylim(0, np.amax(dict['spec'][l:r]) * 1.1)

    plt.show()

    root = tk.Tk()
    app = completeGUI(root, res_path)
    root.mainloop()
    exit()



    # os.remove(f'{res_path}/fid')
    # pdata_num = pdata_path.split('/')[-1]
    # os.remove(f'{res_path}/pdata/{pdata_num}')
    #
    # exit()
    #
    # ############################################
    # # 3) Set-up directory for estimated spectrum
    #
    # # directory number
    # dir_num = path.split('/')[-1]
    #
    # # parent directory
    # par_path = path[:-(len(dir_num) + 1)]
    #
    # # determine path of new directory
    # for i in range(100):
    #     npath = '%s/%s00%s' %(par_path, dir_num, str(i))
    #
    #     # check if the path already exists
    #     if os.path.exists('%s' %npath):
    #         continue
    #
    #     else:
    #         break
    #
    # # copy original data directory to new directory
    # shutil.copytree(src=path, dst=npath)
    #
    # # delete files that will be overwritten
    # os.remove('%s/fid' %npath)
    # os.remove('%s/acqus' %npath)
    # os.remove('%s/acqu' %npath)
    # os.remove('%s/pulseprogram' %npath)
    #
    # ##################################################################
    # # 4) Construct estimated FID using TD and SW specified by the user
    #
    # fid_est = ne.make_fid_1d(x, sw_h, td_user/2)
    #
    # ##################
    # # 5) Write results
    #
    # print('\033[92mSpectral Estimation complete\033[0m')
    # print()
    #
    # print('Estimated FID can be found in the following directory:')
    # print('%s/fid' %npath)
    # print()
    #
    # descr_txt = 'Parameters determined by application of spectral estimation\n\nData path: %s/fid\n' %path
    #
    # path_string = ('%s/fid' %path).replace('_', r'\_')
    #
    # descr_tex = r'Parameters determined by application of spectral estimation on' \
    #             + r' data in path:\\' + r'\texttt{' + path_string + '}'
    #
    # # write x to textfile and latex pdf
    # misc.write_para(x, descr_txt, filename='PARA', dir=npath, format='textfile')
    # misc.write_para(x, descr_tex, filename='PARA', dir=npath, format='latex')
    #
    # f = open('%s/pdata/%s/title' %(npath, ppath.split('/')[-1]), 'a')
    #
    # f.write('\n')
    # f.write('SPECTRUM ESTIMATED USING NMR-EsPy')
    #
    # f.close()
    #
    # ng.bruker.write(dir=npath, dic=dict, data=fid_est)
