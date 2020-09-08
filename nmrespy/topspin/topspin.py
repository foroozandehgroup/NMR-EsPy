#!/usr/bin/env python3

import os
import shutil
from copy import copy
from decimal import Decimal
import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import tkinter as tk
from tkinter import font
from tkinter import filedialog
from PIL import ImageTk, Image
from pdf2image import convert_from_path
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import nmrespy
from nmrespy import core
from nmrespy import io


class mainGUI:
    def __init__(self, master, info):
        matplotlib.use("TkAgg")
        rcParams.update({'figure.autolayout': True})
        self.master = master
        self.spec = info['1r'] + 1j * info['1i']
        self.shifts = info['shifts']
        self.sw = info['sw_ppm']
        self.off = info['offset_ppm']
        self.N = self.spec.shape[0]
        self.lb = int(np.floor(7 * self.N / 16))
        self.rb = int(np.floor(9 * self.N / 16))
        self.lnb = int(np.floor(1 * self.N / 16))
        self.rnb = int(np.floor(2 * self.N / 16))
        self.lb_ppm = core.index_to_ppm(self.lb, self.sw, self.off, self.N)
        self.rb_ppm = core.index_to_ppm(self.rb, self.sw, self.off, self.N)
        self.lnb_ppm = core.index_to_ppm(self.lnb, self.sw, self.off, self.N)
        self.rnb_ppm = core.index_to_ppm(self.rnb, self.sw, self.off, self.N)
        self.path = os.path.dirname(nmrespy.__file__)
        self.nmrespy_image = Image.open(os.path.join(self.path,
                                        'topspin/images/nmrespy_logo.png'))
        self.regsel_image = Image.open(os.path.join(self.path,
                                       'topspin/images/region_selection.png'))
        self.adset_image = Image.open(os.path.join(self.path,
                                      'topspin/images/advanced_settings.png'))
        self.master.configure(background='white')
        self.master.title('NMR-ESPY')

        # ======
        # FRAMES
        # ======

        # leftframe -> spectrum plot and region sliders
        self.leftframe = tk.Frame(self.master,
                                  bg='white',
                                  highlightthickness=2,
                                  highlightbackground='black')

        self.leftframe.grid(column=0,
                            row=0,
                            rowspan=3,
                            padx=(10,0),
                            pady=(10,10),
                            sticky='nsew')

        # titleframe -> for NMR-ESPY logo
        self.titleframe = tk.Frame(self.master,
                                   bg='white')

        self.titleframe.grid(column=1,
                             row=0,
                             padx=(10,10),
                             pady=(10,0),
                             sticky='nsew')

        # rightframe -> for advanced settings
        self.rightframe = tk.Frame(self.master,
                                   bg='white',
                                   highlightthickness=2,
                                   highlightbackground='black')

        self.rightframe.grid(column=1,
                             row=1,
                             padx=(10,10),
                             pady=(0, 10),
                             sticky='nsew')

        # buttonframe -> for cancel, help and save buttons
        self.buttonframe = tk.Frame(self.master,
                                    bg='white')

        self.buttonframe.grid(column=1,
                              row=2,
                              padx=(10,10),
                              pady=(0,10),
                              sticky='nsew')

        # make frames expandable
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # =============
        # NMR-ESPY LOGO
        # =============

        scale = 0.12
        [w, h] = self.nmrespy_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.nmrespy_image = self.nmrespy_image.resize((new_w, new_h),
                                                       Image.ANTIALIAS)
        self.nmrespy_img = ImageTk.PhotoImage(self.nmrespy_image)

        self.nmrespy_logo = tk.Label(self.titleframe,
                                     image=self.nmrespy_img,
                                     bg='white')

        self.nmrespy_logo.grid(row=0,
                               column=0,
                               columnspan=2,
                               padx=(10,0),
                               pady=(10,10),
                               sticky='nsew')

        # =============
        # SPECTRUM PLOT
        # =============

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
        self.limit_history = []
        self.limit_history.append([self.xlim_init, self.ylim_init])

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
        self.ax.tick_params(axis='both', which='major', labelsize=6)
        self.ax.set_yticks([])
        self.ax.set_facecolor('#e0e0e0')
        self.ax.spines['top'].set_color('#ffffff')
        self.ax.spines['bottom'].set_color('#ffffff')
        self.ax.spines['left'].set_color('#ffffff')
        self.ax.spines['right'].set_color('#ffffff')

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.leftframe)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, columnspan=3, sticky='nsew')

        # function for when mouse is clicked inside plot (grey region)
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
                if len(self.limit_history) == 1:
                    pass

                # if limit adjustment has been made, go back...
                else:
                    # if double click, go back to original
                    if event.dblclick:
                        self.limit_history = [self.limit_history[0]]
                        self.xlim = self.limit_history[0][0]
                        self.ylim = self.limit_history[0][1]
                        self.ax.set_xlim(self.xlim)
                        self.ax.set_ylim(self.ylim)
                        self.canvas.draw_idle()

                    # if single click, go back to previous view
                    else:
                        del self.limit_history[-1]
                        self.xlim = self.limit_history[-1][0]
                        self.ylim = self.limit_history[-1][1]
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
                        self.limit_history.append([self.xlim, self.ylim])
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
        self.regsel_logo = tk.Label(self.leftframe,
                                    image=self.regsel_img,
                                    bg='white')

        self.regsel_logo.grid(row=1,
                              column=0,
                              columnspan=3,
                              padx=(10,0),
                              sticky='nsw')

        # Left bound
        self.lb_title = tk.Label(self.leftframe,
                                 text='left bound',
                                 bg='white')

        self.lb_title.grid(row=2,
                           column=0,
                           padx=(10,5),
                           pady=(10,2),
                           sticky='nsw')

        self.lb_slide = tk.Scale(self.leftframe,
                                 from_=1,
                                 to=self.N,
                                 resolution=0.00001,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_lb,
                                 sliderlength=10,
                                 showvalue=0,
                                 highlightthickness=0,
                                 troughcolor='#7fd47f',
                                 activebackground='black',
                                 bg='white')

        self.lb_slide.set(self.lb)

        self.lb_slide.grid(row=2,
                           column=1,
                           pady=(10,2),
                           sticky='nsew')


        self.lb_label = tk.Label(self.leftframe,
                                 bg='white',
                                 text=f'{self.lb_ppm:.3f}')

        self.lb_label.grid(row=2,
                           column=2,
                           padx=(5,20),
                           pady=(10,2),
                           sticky='nsw')

        # right bound
        self.rb_title = tk.Label(self.leftframe,
                                 text='right bound',
                                 bg='white')

        self.rb_title.grid(row=3,
                           column=0,
                           padx=(10,5),
                           pady=(2,2),
                           sticky='nsw')

        self.rb_slide = tk.Scale(self.leftframe,
                                 from_=1,
                                 to=self.N,
                                 resolution=0.00001,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_rb,
                                 sliderlength=10,
                                 showvalue=0,
                                 highlightthickness=0,
                                 troughcolor='#7fd47f',
                                 activebackground='black',
                                 bg='white')

        self.rb_slide.set(self.rb)

        self.rb_slide.grid(row=3,
                           column=1,
                           pady=(2,2),
                           sticky='nsew')


        self.rb_label = tk.Label(self.leftframe,
                                 bg='white',
                                 text=f'{self.rb_ppm:.3f}')

        self.rb_label.grid(row=3,
                           column=2,
                           padx=(5,20),
                           pady=(2,2),
                           sticky='nsw')

        # left noise bound
        self.lnb_title = tk.Label(self.leftframe,
                                  text='left noise bound',
                                  bg='white')

        self.lnb_title.grid(row=4,
                            column=0,
                            padx=(10,5),
                            pady=(2,2),
                            sticky='nsw')

        self.lnb_slide = tk.Scale(self.leftframe,
                                  from_=1,
                                  to=self.N,
                                  resolution=0.00001,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_lnb,
                                  sliderlength=10,
                                  showvalue=0,
                                  highlightthickness=0,
                                  troughcolor='#66b3ff',
                                  activebackground='black',
                                  bg='white')

        self.lnb_slide.set(self.lnb)

        self.lnb_slide.grid(row=4,
                            column=1,
                            pady=(2,2),
                            sticky='nsew')


        self.lnb_label = tk.Label(self.leftframe,
                                  bg='white',
                                  text=f'{self.lnb_ppm:.3f}')

        self.lnb_label.grid(row=4,
                            column=2,
                            padx=(5,20),
                            pady=(2,2),
                            sticky='nsw')

        # right noise bound
        self.rnb_title = tk.Label(self.leftframe,
                                  text='right noise bound',
                                  bg='white')

        self.rnb_title.grid(row=5,
                            column=0,
                            padx=(10,5),
                            pady=(2,10),
                            sticky='nsw')

        self.rnb_slide = tk.Scale(self.leftframe,
                                  from_=1,
                                  to=self.N,
                                  resolution=0.00001,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_rnb,
                                  sliderlength=10,
                                  showvalue=0,
                                  highlightthickness=0,
                                  troughcolor='#66b3ff',
                                  activebackground='black',
                                  bg='white')

        self.rnb_slide.set(self.rnb)

        self.rnb_slide.grid(row=5,
                            column=1,
                            pady=(2,10),
                            sticky='nsew')


        self.rnb_label = tk.Label(self.leftframe,
                                  bg='white',
                                  text=f'{self.rnb_ppm:.3f}')

        self.rnb_label.grid(row=5,
                            column=2,
                            padx=(5,20),
                            pady=(2,10),
                            sticky='nsw')

        self.leftframe.grid_rowconfigure(0, weight=1)
        self.leftframe.grid_columnconfigure(1, weight=1)

        # =================
        # ADVANCED SETTINGS
        # =================

        scale = 0.44
        [w, h] = self.adset_image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.adset_image = self.adset_image.resize((new_w, new_h),
                                                   Image.ANTIALIAS)
        self.adset_img = ImageTk.PhotoImage(self.adset_image)

        self.adset_logo = tk.Label(self.rightframe,
                                   image=self.adset_img,
                                   bg='white')

        self.adset_logo.grid(row=1,
                             column=0,
                             columnspan=3,
                             padx=(10,0),
                             pady=(10,10),
                             sticky=tk.W)

        # number of points to consider in ITMPM
        self.mpm_label = tk.Label(self.rightframe,
                                  text='Points for ITMPM:',
                                  bg='white')

        self.mpm_label.grid(row=2,
                            column=0,
                            padx=(10,5),
                            sticky='NSW')

        self.N_mpm = tk.Entry(self.rightframe,
                              width=12,
                              bg='white',
                              highlightthickness=0)

        if self.N <= 4096:
            self.N_mpm.insert(0, str(self.N))
        else:
            self.N_mpm.insert(0, 4096)

        self.N_mpm.grid(row=2,
                        column=1,
                        columnspan=2,
                        padx=(0,10),
                        sticky='NSW')

        val = int(np.floor(self.N/2))

        self.mpm_max_label = tk.Label(self.rightframe,
                                      bg='white',
                                      text='Max. value: {}'.format(val))

        self.mpm_max_label.grid(row=3,
                                column=1,
                                columnspan=2,
                                sticky=tk.W)

        # number of points to consider in NLP
        self.nlp_label = tk.Label(self.rightframe,
                                  text='Points for NLP:',
                                  bg='white')

        self.nlp_label.grid(row=4,
                            column=0,
                            padx=(10,5),
                            pady=(15,0),
                            sticky='NSW')

        self.N_nlp = tk.Entry(self.rightframe,
                              width=12,
                              bg='white',
                              highlightthickness=0)

        if self.N <= 8192:
            self.N_nlp.insert(0, str(self.N))
        else:
            self.N_nlp.insert(0, 8192)

        self.N_nlp.grid(row=4,
                        column=1,
                        columnspan=2,
                        padx=(0,10),
                        pady=(15,0),
                        sticky='NSW')

        self.nlp_max_label = tk.Label(self.rightframe,
                                      bg='white',
                                      text='Max. value: {}'.format(val))

        self.nlp_max_label.grid(row=5,
                                column=1,
                                columnspan=2,
                                sticky='NSW')

        # maximum number of NLP iterations
        self.maxit_label = tk.Label(self.rightframe,
                                    text='Max. Iterations:',
                                    bg='white')

        self.maxit_label.grid(row=6,
                              column=0,
                              padx=(10,5),
                              pady=(15,0),
                              sticky='NSW')

        self.maxiter = tk.Entry(self.rightframe,
                                width=12,
                                bg='white',
                                highlightthickness=0)

        self.maxiter.insert(0, str(200))

        self.maxiter.grid(row=6,
                          column=1,
                          columnspan=2,
                          padx=(0,10),
                          pady=(15,0),
                          sticky='NSW')

        # NLP algorithm
        self.alg_label = tk.Label(self.rightframe,
                                  text='NLP Method:',
                                  bg='white')

        self.alg_label.grid(row=7,
                            column=0,
                            padx=(10,5),
                            pady=(15,0),
                            sticky='NSW')

        self.algorithm = tk.StringVar(self.rightframe)
        self.algorithm.set('Trust Region')
        self.algoptions = tk.OptionMenu(self.rightframe,
                                        self.algorithm,
                                        'Trust Region',
                                        'L-BFGS')
        self.algoptions['highlightthickness'] = 0
        self.algoptions['bg'] = 'white'

        self.algoptions.grid(row=7,
                             column=1,
                             columnspan=2,
                             padx=(0,10),
                             pady=(15,0))

        self.phase_label = tk.Label(self.rightframe,
                                    text='Inc. Phase Variance:',
                                    bg='white')

        self.phase_label.grid(row=8,
                              column=0,
                              padx=(10,5),
                              pady=(15,0),
                              sticky='NSW')

        self.phase_var = tk.StringVar()
        self.phase_var.set('1')
        self.phase_box = tk.Checkbutton(self.rightframe,
                                        variable=self.phase_var,
                                        fg='black',
                                        bg='white',
                                        highlightthickness=0)

        self.phase_box.grid(row=8,
                            column=1,
                            columnspan=2,
                            padx=(10,5),
                            pady=(15,0),
                            sticky='NSW')

        self.fname_label = tk.Label(self.rightframe,
                                    text='Filename:',
                                    bg='white')

        self.fname_label.grid(row=9,
                              column=0,
                              padx=(10,5),
                              pady=(15,0),
                              sticky='NSW')

        self.fname = tk.Entry(self.rightframe,
                              width=16,
                              highlightthickness=0)

        self.fname.insert(0, 'nmrespy_result')

        self.fname.grid(row=9,
                        column=1,
                        columnspan=2,
                        padx=(10,5),
                        pady=(15,0),
                        sticky='NSW')

        self.dir_label = tk.Label(self.rightframe,
                                  text='Directory:',
                                  bg='white')

        self.dir_label.grid(row=10,
                            column=0,
                            padx=(10,5),
                            pady=(15,0),
                            sticky='NSW')

        self.dir = tk.StringVar()

        self.dir_bar = tk.Entry(self.rightframe,
                                width=16,
                                highlightthickness=0)

        self.dir_bar.grid(row=10,
                          column=1,
                          padx=(10,0),
                          pady=(15,0),
                          sticky='NSW')

        self.dir_button = tk.Button(self.rightframe,
                                    command=self.browse,
                                    width=1,
                                    height=1)

        self.dir_button.grid(row=10,
                             column=2,
                             padx=(10,5),
                             pady=(15,0))

        self.txtfile_label = tk.Label(self.rightframe,
                                      text='Save Textfile:',
                                      bg='white')

        self.txtfile_label.grid(row=11,
                                column=0,
                                padx=(10,5),
                                pady=(15,0),
                                sticky='NSW')

        self.txtfile = tk.StringVar()
        self.txtfile.set('1')

        self.txt_box = tk.Checkbutton(self.rightframe,
                                      variable=self.txtfile,
                                      fg='black',
                                      bg='white',
                                      highlightthickness=0)

        self.txt_box.grid(row=11,
                          column=1,
                          columnspan=2,
                          padx=(10,5),
                          pady=(15,0),
                          sticky='NSW')

        self.pdffile_label = tk.Label(self.rightframe,
                                      text='Save PDF:',
                                      bg='white')

        self.pdffile_label.grid(row=12,
                                column=0,
                                padx=(10,5),
                                pady=(15,0),
                                sticky='NSW')

        self.pdffile = tk.StringVar()
        self.pdffile.set('0')

        self.pdf_box = tk.Checkbutton(self.rightframe,
                                      variable=self.pdffile,
                                      fg='black',
                                      bg='white',
                                      highlightthickness=0)

        self.pdf_box.grid(row=12,
                          column=1,
                          columnspan=2,
                          padx=(10,5),
                          pady=(15,0),
                          sticky='NSW')

        self.csvfile_label = tk.Label(self.rightframe,
                                      text='Save CSV:',
                                      bg='white')

        self.csvfile_label.grid(row=13,
                                column=0,
                                padx=(10,5),
                                pady=(15,10),
                                sticky='NSW')

        self.csvfile = tk.StringVar()
        self.csvfile.set('0')

        self.csv_box = tk.Checkbutton(self.rightframe,
                                      variable=self.csvfile,
                                      fg='black',
                                      bg='white',
                                      highlightthickness=0,
                                      command=self.untick_csv)

        self.csv_box.grid(row=13,
                          column=1,
                          columnspan=2,
                          padx=(10,5),
                          pady=(15,10),
                          sticky='NSW')

        for i in range(1, 13):
            self.rightframe.grid_rowconfigure(i, weight=1)

        # ======================
        # SAVE/HELP/QUIT BUTTONS
        # ======================

        self.cancel_button = tk.Button(self.buttonframe,
                                       text='Cancel',
                                       font='bold',
                                       bg='#ff6667',
                                       command=self.cancel,
                                       width=6)

        self.cancel_button.grid(row=0,
                                column=0,
                                padx=(5,0),
                                pady=(5,5))

        self.help_button = tk.Button(self.buttonframe,
                                     text='Help',
                                     font='bold',
                                     bg='#ffb266',
                                     command=self.load_help,
                                     width=6)

        self.help_button.grid(row=0,
                              column=1,
                              padx=(5,0),
                              pady=(5,5))

        self.save_button = tk.Button(self.buttonframe,
                                     text='Save',
                                     font='bold',
                                     bg='#7fd47f',
                                     command=self.save,
                                     width=6)

        self.save_button.grid(row=0,
                              column=2,
                              padx=(5,5),
                              pady=(5,5))

        self.feedback = tk.Label(self.buttonframe,
                                 text='For queries/feedback, contact',
                                 bg = 'white')

        self.feedback.grid(row=1,
                           column=0,
                           columnspan=3,
                           padx=(5,5),
                           sticky='SW')

        self.email = tk.Label(self.buttonframe,
                              text='simon.hulse@chem.ox.ac.uk',
                              font='Courier',
                              bg='white')

        self.email.grid(row=2,
                        column=0,
                        columnspan=3,
                        padx=(5,5),
                        sticky='SW')


        self.buttonframe.grid_columnconfigure(0, weight=1)
        self.buttonframe.grid_columnconfigure(1, weight=1)
        self.buttonframe.grid_columnconfigure(2, weight=1)
        self.buttonframe.grid_rowconfigure(0, weight=1)

    def update_lb(self, lb):
        lb = float(lb)
        if lb < self.rb:
            self.lb = lb
            self.lb_ppm = core.index_to_ppm(self.lb, self.sw, self.off, self.N)
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.lb_label.config(text=f'{self.lb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lb = self.rb - 1
            self.lb_ppm = core.index_to_ppm(self.lb, self.sw, self.off, self.N)
            self.lb_slide.set(self.lb_ppm)
            self.canvas.draw_idle()

    def update_rb(self, rb):
        rb = float(rb)
        if rb > self.lb:
            self.rb = rb
            self.rb_ppm = core.index_to_ppm(self.rb, self.sw, self.off, self.N)
            self.filtregion.set_bounds(self.rb_ppm,
                                       -2*self.ylim_init[1],
                                       self.lb_ppm - self.rb_ppm,
                                       4*self.ylim_init[1])
            self.rb_label.config(text=f'{self.rb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rb = self.lb + 1
            self.rb_ppm = core.index_to_ppm(self.rb, self.sw, self.off, self.N)
            self.rb_slide.set(self.rb_ppm)
            self.canvas.draw_idle()

    def update_lnb(self, lnb):
        lnb = float(lnb)
        if lnb < self.rnb:
            self.lnb = lnb
            self.lnb_ppm = core.index_to_ppm(self.lnb, self.sw, self.off, self.N)
            self.noiseregion.set_bounds(self.rnb_ppm,
                                        -2*self.ylim_init[1],
                                        self.lnb_ppm - self.rnb_ppm,
                                        4*self.ylim_init[1])
            self.lnb_label.config(text=f'{self.lnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.lnb = self.rnb - 1
            self.lnb_ppm = core.index_to_ppm(self.lnb, self.sw, self.off, self.N)
            self.lnb_slide.set(self.lnb_ppm)
            self.canvas.draw_idle()

    def update_rnb(self, rnb):
        rnb = float(rnb)
        if rnb > self.lnb:
            self.rnb = rnb
            self.rnb_ppm = core.index_to_ppm(self.rnb, self.sw, self.off, self.N)
            self.noiseregion.set_bounds(self.rnb_ppm,
                                        -2*self.ylim_init[1],
                                        self.lnb_ppm - self.rnb_ppm,
                                        4*self.ylim_init[1])
            self.rnb_label.config(text=f'{self.rnb_ppm:.3f}')
            self.canvas.draw_idle()
        else:
            self.rnb = self.lnb + 1
            self.rnb_ppm = core.index_to_ppm(self.rnb, self.sw, self.off, self.N)
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
        self.mpm_points = int(self.N_mpm.get())
        self.nlp_points = int(self.N_nlp.get())
        self.maxiter = int(self.maxiter.get())
        self.alg = self.algorithm.get()
        self.pv = self.phase_var.get()
        self.file = self.fname.get()
        self.dir = self.dir_bar.get()
        self.txt = self.txtfile.get()
        self.pdf = self.pdffile.get()
        self.csv = self.csvfile.get()
        self.master.destroy()

    def cancel(self):
        self.master.destroy()
        print("NMR-ESPY Cancelled :'(")
        exit()


class helpGUI:

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.scroll_y = tk.Scrollbar(self.frame,
                                     orient=tk.VERTICAL)

        self.pdf = tk.Text(self.frame,
                           yscrollcommand=self.scroll_y.set,
                           bg='grey')

        self.scroll_y.pack(side=tk.RIGHT,
                           fill=tk.Y)

        self.scroll_y.config(command=self.pdf.yview)

        self.pdf.pack(fill=tk.BOTH, expand=1)

        path = os.path.join(os.path.dirname(nmrespy.__file__),
                            'topspin/docs/help.pdf')

        self.pages = convert_from_path(path, size=(900,1.414*900))
        self.photos = []
        for i in range(len(self.pages)):
            self.photos.append(ImageTk.PhotoImage(self.pages[i]))
        for photo in self.photos:
            self.pdf.image_create(tk.END, image=photo)
            self.pdf.insert(tk.END, '\n\n')
        self.frame.pack(fill=tk.BOTH, expand=1)



class warnGUI:

    def __init__(self, master, errmsg):

        self.master = master
        self.frame = tk.Frame(self.master, width=400, height=130)

        self.sign = os.path.join(os.path.dirname(nmrespy.__file__),
                            'branding/nmrespy_logo/warning.png')

        scale = 0.4
        [w, h] = self.sign.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.sign = self.sign.resize((new_w, new_h), Image.ANTIALIAS)

        self.signimg = ImageTk.PhotoImage(self.sign)
        self.warnsign = tk.Label(self.master, image=self.signimg)
        self.warnsign.place(x=20, y=20)

        self.msg = tk.Message(self.master, text=errmsg, width=250)
        self.msg.place(x=110, y=20)

        self.close = tk.Button(self.master, text='Close', bg ='#fa867e',
                          command=self.quitWarn, width=12, height=1)
        self.close.place(x=280, y=90)

        self.frame.pack()

    def quitWarn(self):
        root.quit()

class completeGUI:

    def __init__(self, master, path):
        self.master = master
        self.monofont = font.Font(root=self.master, family='Courier')
        self.frame = tk.Frame(self.master, width=500, height=100)
        self.text1 = tk.Label(self.frame, text='Estimation Complete!')
        self.text1.place(x=20, y=20)
        self.text2 = tk.Label(self.frame, text='The result can be found in:')
        self.text2.place(x=20, y=40)
        self.text3 = tk.Label(self.frame, text=f'{path}', font=self.monofont)
        self.text3.place(x=20, y=60)
        self.frame.pack()

#######  ACTUAL ALGORITHM STARTS HERE #######
# path to nmrespy directory
nmrespy_path = os.path.dirname(nmrespy.__file__)

# extract path information
f = open('%s/topspin/tmp/info.txt' %nmrespy_path, 'r')
from_topspin = f.readlines()[0].split(' ')
f.close()
# os.remove('%s/topspin/tmp/info1d.txt' %nmrespy_path)

# import dictionary of spectral info
pdata_path = from_topspin[1]
info = io.import_bruker_pdata(pdata_path)

root = tk.Tk()
app = mainGUI(root, info)
root.mainloop()

lb = app.lb
rb = app.rb
lb_ppm = app.lb_ppm
rb_ppm = app.rb_ppm
lnb = app.lnb
rnb = app.rnb
lnb_ppm = app.lnb_ppm
rnb_ppm = app.rnb_ppm
mpm_points = app.mpm_points
nlp_points = app.nlp_points
maxit = app.maxiter
alg = app.alg
if alg == 'Trust Region':
    alg = 'trust-region'
elif alg == 'L-BFGS':
    alg = 'lbfgs'

pv = app.pv
if pv == '1':
    pv = True
else:
    pv = False
file = app.file
dir = app.dir
# TODO: Add Text widget to input description to result file
# descr = app.descr

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

csv = app.csv
if txt == '1':
    csv = True
else:
    csv = False

# check that at least one file type is ticked
if not txt and not pdf and not csv:
    errmsg = 'You have not specified any file type to save the result to!'
    root = tk.Tk()
    app = warnGUI(root, errmsg)
    root.mainloop()
    exit()

# constructing the phased, filtered FID
info = core.virtual_echo(info, (lb_ppm, rb_ppm), (lnb_ppm, rnb_ppm))
info = core.matrix_pencil(info, slice=mpm_points)
info = core.nonlinear_programming(info, slice=nlp_points, method=alg,
                                  phase_correct=pv, maxit=maxit)


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
        app = warnGUI(root, nmrespy_path, errmsg)
        root.mainloop()
        exit()

    if lb > dict['shifts'][0]:
        ls = round(dict['shifts'][0], 4)
        rs = round(dict['shifts'][-1], 4)
        errmsg = 'The left bound (%s ppm) and right bound' %str(lb) \
                 + '  (%s ppm) shoudld lie within the range' %str(rb) \
                 + ' %s to %s ppm.' %(ls, rs)
        root = tk.Tk()
        app = warnGUI(root, nmrespy_path, errmsg)
        root.mainloop()
        exit()

    if rb < dict['shifts'][-1]:
        ls = round(dict['shifts'][0], 4)
        rs = round(dict['shifts'][-1], 4)
        errmsg = 'The left bound (%s ppm) and right bound' %str(lb) \
                 + '  (%s ppm) shoudld lie within the range' %str(rb) \
                 + ' %s to %s ppm.' %(ls, rs)
        root = tk.Tk()
        app = warnGUI(root, nmrespy_path, errmsg)
        root.mainloop()
        exit()

if lnb is not None and rnb is not None:
    if lnb <= rnb:
        errmsg = 'The left noise bound (%s ppm) should be larger' %str(lnb) \
                 + ' than the right noise bound (%s ppm).' %str(rnb)
        root = tk.Tk()
        app = warnGUI(root, nmrespy_path, errmsg)
        root.mainloop()
        exit()

    if lnb > dict['shifts'][0]:
        ls = round(dict['shifts'][0], 4)
        rs = round(dict['shifts'][-1], 4)
        errmsg = 'The left noise bound (%s ppm) and right noise' %str(lnb) \
                 + ' bound (%s ppm) should lie within the range' %str(rnb) \
                 + ' %s to %s ppm.' %(ls, rs)
        root = tk.Tk()
        app = warnGUI(root, nmrespy_path, errmsg)
        root.mainloop()
        exit()

    if rnb < dict['shifts'][-1]:
        ls = round(dict['shifts'][0], 4)
        rs = round(dict['shifts'][-1], 4)
        errmsg = 'The left noise bound (%s ppm) and right noise' %str(lnb) \
                 + ' bound (%s ppm) should lie within the range' %str(rnb) \
                 + ' %s to %s ppm.' %(ls, rs)
        root = tk.Tk()
        app = warnGUI(root, nmrespy_path, errmsg)
        root.mainloop()
        exit()

if mpm_trunc == None:
    pass

elif mpm_trunc > si:
    errmsg = 'The number of points specified for consideration in the' \
             + ' Matrix Pencil Method (%s) is larger ' %str(mpm_trunc) \
             + ' than the number of signal points (%s).' %str(si)
    root = tk.Tk()
    app = warnGUI(root, nmrespy_path, errmsg)
    root.mainloop()
    exit()

if nlp_trunc == None:
    pass

elif nlp_trunc > si:
    errmsg = 'The number of points specified for consideration durring' \
             + ' Nonlinear Programming (%s) is larger ' %str(nlp_trunc) \
             + ' than the number of signal points (%s).' %str(si)
    root = tk.Tk()
    app = warnGUI(root, nmrespy_path, errmsg)
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
