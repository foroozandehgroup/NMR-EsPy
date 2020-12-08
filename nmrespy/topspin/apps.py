#!/usr/bin/env python3

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import nmrespy
import nmrespy.load as load
import nmrespy._misc as _misc
import nmrespy._plot as _plot

# useful paths to various directories
NMRESPYDIR = os.path.dirname(nmrespy.__file__)
TOPSPINDIR = os.path.join(NMRESPYDIR, 'topspin')
IMAGESDIR = os.path.join(TOPSPINDIR, 'images')

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

    image = Image.open(path)
    [w, h] = image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)


class CustomFrame(tk.Frame):
    """A Tkinter frame with a white background. Prevents me from having
    to write (and you read) self['bg'] = 'white' about 1000000 times"""

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self['bg'] = 'white'


class CustomToplevel(tk.Toplevel):
    """A Tkinter toplevel with a white background. Prevents me from having
    to write (and you read) self['bg'] = 'white' about 1000000 times. By
    default, it is also unable to be resized, and its title is 'NMR-EsPy'"""

    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self['bg'] = 'white'
        self.title('NMR-EsPy')
        self.resizable(False, False)


class WarnFrame(CustomToplevel):
    """A window in case the user does something silly."""

    def __init__(self, parent, msg):
        CustomToplevel.__init__(self, parent)
        self.title('NMR-EsPy - Error')

        # warning image
        self.img = get_PhotoImage(os.path.join(IMAGESDIR, 'warning.png'), 0.08)
        self.warn_sign = tk.Label(self, image=self.img, bg='white')
        self.warn_sign.grid(row=0, column=0, padx=(10,0), pady=10)

        # add text explaining the issue
        text = tk.Label(self, text=msg, wraplength=400, bg='white')
        text.grid(row=0, column=1, padx=10, pady=10)

        # close button
        close_button = tk.Button(self, text='Close', width=8, bg='#ff9894',
                                 highlightbackground='black',
                                 command=self.destroy)
        close_button.grid(row=1, column=1, padx=10, pady=(0,10))


class CustomNavigationToolbar(NavigationToolbar2Tk):
    """Tweak default matplotlib navigation bar to exclude subplot-config
    and save buttons. Also dialogues as cursor goes over plot, and bar
    is set to be white"""

    def __init__(self, canvas, parent):

        # slice toolitems (this gets rid of the unwanted buttons)
        self.toolitems = self.toolitems[:6]

        NavigationToolbar2Tk.__init__(
            self, canvas, parent, pack_toolbar=False
        )

        # make everything white
        self['bg'] = 'white'
        self._message_label['bg'] = 'white'
        for button in self.winfo_children():
            button['bg'] = 'white'

    def set_message(self, msg):
        pass


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


class DataType(tk.Toplevel):
    """GUI for asking user whether they want to analyse the raw FID or
    pdata"""

    def __init__(self, parent):

        CustomToplevel.__init__(self, parent)

        # parent is the controler in this case (NMREsPyApp)
        self.ctrl = parent

        # frame for the NMR-EsPy logo
        self.logoframe = tk.Frame(self, bg='white')
        self.logoframe.grid(row=0, column=0, rowspan=2)

        # frame for option boxes and descriptors
        self.mainframe = tk.Frame(self, bg='white')
        self.mainframe.grid(row=0, column=1)

        # frame for save/cancel buttons
        self.buttonframe = tk.Frame(self, bg='white')
        self.buttonframe.grid(row=1, column=1, sticky='e')

        # open info file. Gives paths to fid file and pdata directory
        with open(os.path.join(TOPSPINDIR, 'info.txt'), 'r') as fh:
            self.fidpath, self.pdatapath = fh.read().split(' ')

        # nmrespy logo
        self.img = get_PhotoImage(
            os.path.join(IMAGESDIR, 'nmrespy_full.png'), 0.08
        )
        self.logo = tk.Label(self.logoframe, image=self.img, bg='white')
        self.logo.grid(row=0, column=0, padx=10, pady=10)

        message = tk.Label(
            self.mainframe, text='Which data would you like to analyse?',
            font=('Helvetica', '12', 'bold'), bg='white'
        )
        message.grid(
            column=0, row=0, columnspan=2, padx=10, pady=(10,0)
        )

        pdata_label = tk.Label(
            self.mainframe, text='Processed Data', bg='white'
        )
        pdata_label.grid(
            column=0, row=1, padx=(10,0), pady=(10,0), sticky='w'
        )

        pdatapath = tk.Label(
            self.mainframe, text=f'{self.pdatapath}/1r', font='Courier',
            bg='white'
        )
        pdatapath.grid(column=0, row=2, padx=(10, 0), sticky='w')

        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = tk.Checkbutton(
            self.mainframe, variable=self.pdata, command=self.click_pdata,
            bg='white', highlightthickness=0, bd=0
        )
        self.pdata_box.grid(
            column=1, row=1, rowspan=2, padx=10, sticky='nsw'
        )

        fid_label = tk.Label(self.mainframe, text='Raw FID', bg='white')
        fid_label.grid(
            column=0, row=3, padx=(10, 0), pady=(10, 0), sticky='w'
        )

        fidpath = tk.Label(
            self.mainframe, text=f'{self.fidpath}/fid', font='Courier',
            bg='white'
        )
        fidpath.grid(column=0, row=4, padx=(10, 0), sticky='w')

        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = tk.Checkbutton(
            self.mainframe, variable=self.fid, command=self.click_fid,
            bg='white', highlightthickness=0, bd=0
        )
        self.fid_box.grid(
            column=1, row=3, rowspan=2, padx=10, sticky='nsw'
        )

        self.confirmbutton = tk.Button(
            self.buttonframe, text='Confirm', command=self.confirm, width=8,
            bg='#9eda88', highlightbackground='black'
        )
        self.confirmbutton.grid(
            column=1, row=0, padx=(5, 10), pady=10, sticky='e'
        )

        self.cancelbutton = tk.Button(
            self.buttonframe, text='Cancel', command=self.ctrl.destroy,
            width=8, bg='#ff9894', highlightbackground='black'
        )
        self.cancelbutton.grid(column=0, row=0, pady=10, sticky='e')

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
        self.ctrl.deiconify()
        self.destroy()


class NMREsPyApp(tk.Tk):
    """App for using NMR-EsPy."""

    # This is the controller
    # if you see self.ctrl throughout other classes in this file, it refers
    # to this class

    def __init__(self):
        tk.Tk.__init__(self)
        self.title('NMR-EsPy - Setup Calculation')
        self.protocol('WM_DELETE_WINDOW', self.destroy)

        # open window to ask user for data type (fid or pdata)
        # acquires dtype attribute
        self.withdraw()
        DataType(parent=self)

        # main container: everything goes into here
        container = tk.Frame(self, bg='white')
        # make resizeable in both dimensions
        container.pack(side='top', fill='both', expand=True)

        # make first row and first column of container resizeable.
        # this will be:
        # Plotframe (x and y)
        # Tabframe (x)
        # Logoframe (x)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # generate self.info: NMREsPyBruker class instance
        # FID data - transform to frequency domain
        if self.dtype == 'fid':
            self.info = load.import_bruker_fid(self.path, ask_convdta=False)
            self.spec = np.flip(fftshift(fft(self.info.get_data())))

        # pdata - combine real and imaginary components
        else:
            self.info = load.import_bruker_pdata(self.path)
            self.spec = self.info.get_data(pdata_key='1r') \
                        + 1j * self.info.get_data(pdata_key='1i')

        # unpack useful parameters into attributes
        # as 1D data, these will be one-element tuples
        self.shifts = self.info.get_shifts(unit='ppm')[0] # shifts for plot
        self.sw_p = self.info.get_sw(unit='ppm')[0] # sweep width (ppm)
        self.off_p = self.info.get_offset(unit='ppm')[0] # transmitter offset
        self.n = self.spec.shape[0] # number of points

        # TODO For consitency witht he rest of my code, it's probably
        # a good idea to use 2-tuples for the regions, rather than left
        # and right bonunds

        # initialise region of interest and noise region boundaries
        # values in array indices
        self.lb = int(np.floor(7 * self.n / 16)) # left bound
        self.rb = int(np.floor(9 * self.n / 16)) # right bound
        self.lnb = int(np.floor(1 * self.n / 16)) # left noise bound
        self.rnb = int(np.floor(2 * self.n / 16)) # right noise bound
        # number of points the region comprises
        self.region_size = self.rb - self.lb 

        # phase correction parameters
        self.pivot = int(np.floor(self.n / 2)) # location of pivot
        self.p0 = 0. # zero-order phase
        self.p1 = 0. # first-order phase

        # TODO I think that _misc.conv_ppm_idx is used only here.
        # Could include it as part of the apps module instead

        # convert boundaries and pivot from array indices to ppm
        # forms attributes called:
        # lb_ppm, rb_ppm, lnb_ppm, rnb_ppm, pivot_ppm
        for s in ['lb', 'rb', 'lnb', 'rnb', 'pivot']:
            self.__dict__[f'{s}_ppm'] = _misc.conv_ppm_idx(
                    self.__dict__[s], self.sw_p, self.off_p, self.n,
                    direction='idx->ppm',
            )

        # plot spectrum
        # construct figure of appropriate size and dpi
        self.fig = Figure(figsize=(6,3.5), dpi=170)
        # add axes to figure
        self.ax = self.fig.add_subplot(111)
        # plot spectrum
        self.specplot = self.ax.plot(
            self.shifts, np.real(self.spec), color='k', lw=0.6
        )[0]

        # set x-limits as edges of spectral window
        self.xlim = (self.shifts[0], self.shifts[-1])
        self.ax.set_xlim(self.xlim)

        # Get current y-limit. Will reset y-limits to this value after the
        # very tall noiseregion and filtregion rectangles have been added to
        # the plot
        self.ylim_init = self.ax.get_ylim()

        # highlight the spectral region to be filtered (green)
        # Rectangle's first 3 args: bottom left coords, width, height
        self.filtregion = Rectangle(
            (self.rb_ppm, -20*self.ylim_init[1]),
            self.lb_ppm - self.rb_ppm,
            40*self.ylim_init[1],
            facecolor='#7fd47f'
        )
        self.ax.add_patch(self.filtregion)

        # highlight the noise region (blue)
        self.noiseregion = Rectangle(
            (self.rnb_ppm, -20*self.ylim_init[1]),
            self.lnb_ppm - self.rnb_ppm,
            40*self.ylim_init[1],
            facecolor='#66b3ff'
        )
        self.ax.add_patch(self.noiseregion)

        # plot pivot line
        # alpha set to 0 to make invisible initially
        x = [self.pivot_ppm, self.pivot_ppm]
        y = [-20*self.ylim_init[1], 20*self.ylim_init[1]]
        self.pivotplot = self.ax.plot(x, y, color='r', alpha=0, lw=1)[0]

        # reset y limit
        self.ax.set_ylim(self.ylim_init)

        # aesthetic tweaks to plot
        self.ax.tick_params(axis='x', which='major', labelsize=6)
        self.ax.locator_params(axis='x', nbins=10)
        self.ax.set_yticks([])

        for direction in ('top', 'bottom', 'left', 'right'):
            self.ax.spines[direction].set_color('k')

        self.nucleus = self.info.get_nucleus()[0] # nucleus identifier
        self.ax.set_xlabel(_plot._generate_xlabel(self.nucleus), fontsize=8)

        # prevent user panning/zooming beyond spectral window
        Restrictor(self.ax, x=lambda x: x<= self.xlim[0])
        Restrictor(self.ax, x=lambda x: x>= self.xlim[1])

        # dict to store frames that will go inside container
        self.frames = {}

        # append all frames to the window

        # frame with plot. N.B. this frame is resizable (row 0, col 0)
        self.frames['PlotFrame'] = PlotFrame(parent=container, ctrl=self)
        self.frames['PlotFrame'].grid(
            row=0, column=0, columnspan=2, sticky='nsew'
        )

        # frame with customaisation tabs (region selection, phase data,
        # advanced estimation settings)
        self.frames['TabFrame'] = TabFrame(parent=container, ctrl=self)
        self.frames['TabFrame'].grid(
            row=1, column=0, columnspan=2, sticky='ew'
        )

        # frame with cancel/help/run buttons. Also has some contact info
        self.frames['SetupButtonFrame'] = SetupButtonFrame(
            parent=container, ctrl=self
        )
        self.frames['SetupButtonFrame'].grid(row=2, column=1, sticky='s')

        # frame with NMR-EsPy logo and MF group logo
        self.frames['LogoFrame'] = LogoFrame(
            parent=container, ctrl=self, scale=0.06
        )
        self.frames['LogoFrame'].grid(
            row=2, column=0, padx=10, pady=10, sticky='w'
        )


class PlotFrame(tk.Frame):
    """Contains a plot, along with navigation toolbar"""

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        # make figure canvas expandable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # place figure into canvas
        self.canvas = FigureCanvasTkAgg(self.ctrl.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        # construct navigation toolbar
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.toolbar.grid(column=0, row=1, sticky='e')


class TabFrame(tk.Frame):
    """Contains a notebook for region selection, phase correction, and
    advanced settings"""

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'
        # make column container scales adjustable
        self.columnconfigure(0, weight=1)

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

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky='ew', padx=10, pady=(0,10))
        # # whenever tab clicked, change plot so that either region selection
        # # rectangles are visible, or phase pivot, depending on tab selected
        self.notebook.bind('<<NotebookTabChanged>>',
                           lambda event: self.switch_tab())

        # dictionary of notebook frames
        self.nbframes = {}

        for F, title in zip((RegionFrame, PhaseFrame, AdvancedSettingsFrame),
                            ('Region Selection', 'Phase Correction',
                             'Advanced Settings')):

            frame = F(parent=self.notebook, ctrl=self.ctrl)
            self.notebook.add(frame, text=title, sticky='ew')
            self.nbframes[F.__name__] = frame


    def switch_tab(self):
        """Adjusts the appearence of the plot when a new tab is selected.
        Hides/reveals region rectangles and pivot plot as required. Toggles
        alpha between 1 and 0"""

        # detemine the active tab (0 = region selection, 1 = phase correction)
        tab = self.notebook.index(self.notebook.select())
        # set alpha values for region rectangles and pivot plot
        if tab in [0, 2]:
            # region selection tab and advanced settings tab
            filt = 1
            noise = 1
            pivot = 0

        else:
            # phase correction tab
            filt = 0
            noise = 0
            pivot = 1

        self.ctrl.filtregion.set_alpha(filt)
        self.ctrl.noiseregion.set_alpha(noise)
        self.ctrl.pivotplot.set_alpha(pivot)

        # draw updated figure
        self.ctrl.frames['PlotFrame'].canvas.draw_idle()


class RegionFrame(tk.Frame):
    """Frame inside SetupApp notebook - for altering region boundaries"""

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        # make scales expandable
        self.columnconfigure(1, weight=1)

        for row, s in enumerate(('lb', 'rb', 'lnb', 'rnb')):
            # construct text strings for scale titles
            text = ''
            for letter in s:
                if letter == 'l':
                    text += 'left '
                elif letter == 'r':
                    text += 'right '
                elif letter == 'n':
                    text += 'noise '
                else:
                    text += 'bound'

            # scale titles
            self.__dict__[f'{s}_title'] = title = \
                tk.Label(self, text=text, bg='white')

            # determine troughcolor of scale (0, 1: green; 2, 3: blue)
            if row < 2:
                tc = '#cbedcb'
            else:
                tc = '#cde6ff'

            # scale atts are called: lb_scale, rb_scale, lnb_scale, rnb_scale
            self.__dict__[f'{s}_scale'] = scale = \
                tk.Scale(
                    self, from_=0, to=self.ctrl.n - 1, orient=tk.HORIZONTAL,
                    showvalue=0, bg='white', sliderlength=15, bd=0,
                    highlightthickness=0, troughcolor=tc,
                    command=lambda value, s=s: self.ud_scale(value, s)
                )

            scale.set(self.ctrl.__dict__[s])

            # bound variable atts are called: lb_var, rb_var, lnb_var, rnb_var
            self.__dict__[f'{s}_var'] = var = tk.StringVar()
             # access initial value of bound (ppm)
            ppm = self.ctrl.__dict__[f'{s}_ppm']
            var.set(f'{ppm:.3f}')

            # entry atts are called: lb_entry, rb_entry, lnb_entry, rnb_entry
            self.__dict__[f'{s}_entry'] = entry = \
                tk.Entry(
                    self, textvariable=var, width=6, highlightthickness=0
                )

            entry.bind('<Return>', (lambda event, s=s: self.ud_entry(s)))

            # pad below AND above if bottom widget (row 3)...
            if row == 3:
                title.grid(row=row, column=0, padx=(10,0), pady=10, sticky='w')
                scale.grid(row=row, column=1, padx=(10,0), pady=10, sticky='ew')
                entry.grid(row=row, column=2, padx=10, pady=10, sticky='w')
            # ...otherwise, only pad above
            else:
                title.grid(row=row, column=0, padx=(10,0), pady=(10,0),
                           sticky='w')
                scale.grid(row=row, column=1, padx=(10,0), pady=(10,0),
                           sticky='ew')
                entry.grid(row=row, column=2, padx=10, pady=(10,0), sticky='w')


    def ud_entry(self, s):
        """Update the GUI after the user presses <Enter> whilst in an entry
        widget"""

        try:
            # check input can be interpreted as a float, and
            # obtain the array index corresponding to the ppm value
            index = _misc.conv_ppm_idx(float(self.__dict__[f'{s}_var'].get()),
                                       self.ctrl.sw_p,
                                       self.ctrl.off_p,
                                       self.ctrl.n,
                                       direction='ppm->idx')

            # update the relevant parameters
            self.ud_bound(index, s)

        except:
            # input cannot be interpreted as a float
            # reset the value in the entry to the current bound value
            reset = self.ctrl.__dict__[f'{s}_ppm']
            self.__dict__[f'{s}_var'].set(f'{reset:.3f}')


    def ud_scale(self, index, s):
        """Update the GUI after the user changes the slider on a scale
        widget"""

        # scale value is set by a StringVar, so convert to integer
        index = int(index)
        # update the relevant parameters
        self.ud_bound(index, s)


    def ud_bound(self, index, s):
        """Given an update index, and the identity of the bound to change,
        adjust relavent region parameters, and update the GUI."""

        # determine if we are considering a left or right bound
        if s[0] == 'l':
            twist = index
            left = index
            right = self.ctrl.__dict__[f'r{s[1:]}']
        else:
            left = self.ctrl.__dict__[f'l{s[1:]}']
            twist = index
            right = index

        if left < right: # all good, update bound attribute
            self.ctrl.__dict__[s] = twist

        else: # all not good, ensure left < right is True
            if twist == left:
                self.ctrl.__dict__[s] = right - 1
            else:
                self.ctrl.__dict__[s] = left + 1

        # convert index to ppm
        ppm = _misc.conv_ppm_idx(self.ctrl.__dict__[s],
                                 self.ctrl.sw_p,
                                 self.ctrl.off_p,
                                 self.ctrl.n,
                                 direction='idx->ppm')

        self.ctrl.__dict__[f'{s}_ppm'] = ppm
        self.__dict__[f'{s}_var'].set(f'{ppm:.3f}')


        # check if (lb, rb) or (lnb, rnb)
        # change the bound of the relevant rectangle
        if s[1] == 'b':
            # changing lb or rb
            self.ctrl.filtregion.set_bounds(
                self.ctrl.rb_ppm,
                -20 * self.ctrl.ylim_init[1],
                self.ctrl.lb_ppm - self.ctrl.rb_ppm,
                40 * self.ctrl.ylim_init[1],
            )

        else:
            # changing lnb or rnb
            self.ctrl.noiseregion.set_bounds(
                self.ctrl.rnb_ppm,
                -20 * self.ctrl.ylim_init[1],
                self.ctrl.lnb_ppm - self.ctrl.rnb_ppm,
                40 * self.ctrl.ylim_init[1],
            )

        # update plot
        self.__dict__[f'{s}_scale'].set(self.ctrl.__dict__[s])
        self.ctrl.frames['PlotFrame'].canvas.draw_idle()


class PhaseFrame(tk.Frame):
    """Frame inside SetupApp notebook - for phase correction of data"""

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        # make scales expandable
        self.columnconfigure(1, weight=1)

        for row, s in enumerate(('pivot', 'p0', 'p1')):
            # scale titles
            self.__dict__[f'{s}_title'] = title = \
                tk.Label(self, text=s, bg='white')

            # pivot scale
            if row == 0:
                tc = '#ffb0b0'
                from_ = 0
                to = self.ctrl.n - 1
                resolution = 1
            # p0 and p1 scales
            else:
                tc = '#e0e0e0'
                from_ = -np.pi
                to = np.pi
                resolution = 0.0001

                # p1: set between -10π and 10π rad
                if row == 2:
                    from_ *= 10
                    to *= 10

            self.__dict__[f'{s}_scale'] = scale = \
                tk.Scale(self,
                         troughcolor = tc,
                         from_ = from_,
                         to = to,
                         resolution=resolution,
                         orient = tk.HORIZONTAL,
                         bg = 'white',
                         sliderlength = 15,
                         bd = 0,
                         highlightthickness = 0,
                         relief = 'flat',
                         showvalue = 0,
                         command = lambda value, s=s: self.ud_scale(value, s))

            scale.set(self.ctrl.__dict__[s])

            self.__dict__[f'{s}_var'] = var = tk.StringVar()
            if s == 'pivot':
                value = self.ctrl.__dict__[f'{s}_ppm']
            else:
                value = self.ctrl.__dict__[s]
            var.set(f'{value:.3f}')

            self.__dict__[f'{s}_entry'] = entry = \
                tk.Entry(self,
                         textvariable=var,
                         width=6,
                         highlightthickness=0)

            entry.bind('<Return>', (lambda event, s=s: self.ud_entry(s)))


            if row == 2:
                title.grid(row=row, column=0, padx=(10,0), pady=10, sticky='w')
                scale.grid(row=row, column=1, padx=(10,0), pady=10, sticky='ew')
                entry.grid(row=row, column=2, padx=10, pady=10, sticky='w')
            else:
                title.grid(row=row, column=0, padx=(10,0), pady=(10,0),
                           sticky='w')
                scale.grid(row=row, column=1, padx=(10,0), pady=(10,0),
                           sticky='ew')
                entry.grid(row=row, column=2, padx=10, pady=(10,0),
                           sticky='w')


    def ud_scale(self, value, s):

        """Update the GUI after the user changes the slider on a scale
        widget"""
        if s == 'pivot':
            self.ud_pivot(int(value))
        else:
            # p0 and p1 scales
            self.ud_p0_p1(float(value), s)


    def ud_entry(self, s):
        """Update the GUI after the user changes and entry widget"""

        if s == 'pivot':
            try:
                # check input can be converted as a float
                self.ud_pivot(
                    _misc.conv_ppm_idx(
                        float(self.pivot_var.get()),
                        self.ctrl.sw_p,
                        self.ctrl.off_p,
                        self.ctrl.n,
                        direction='ppm->idx'
                    )
                )

            except:
                # invalid input: reset value
                self.pivot_var.set(self.ctrl.pivot_ppm)

        else:
            var = self.__dict__[f'{s}_var'].get()
            try:
                # check input can be converted to a float
                self.ud_p0_p1(float(var), s)
            except:
                # determine if user has given a value as a multiple of pi
                if var[-2:] == 'pi':
                    try:
                        self.ud_p0_p1(float(var[:-2]) * np.pi, s)
                    except:
                        # invalid input: reset value
                        self.__dict__[f'{s}_var'].set(
                            f'{self.ctrl.__dict__[s]:.3f}'
                        )


    def ud_pivot(self, index):
        """Deals with a change to either the pivot scale or entry widget"""

        # check index is in the suitable range (check for entry widget)
        if self.pivot_scale['from'] <= index <= self.pivot_scale['to']:
            self.ctrl.pivot = index
            self.ctrl.pivot_ppm = \
                _misc.conv_ppm_idx(index,
                                   self.ctrl.sw_p,
                                   self.ctrl.off_p,
                                   self.ctrl.n,
                                   direction='idx->ppm')

            self.pivot_var.set(f'{self.ctrl.pivot_ppm:.3f}')

            x = [self.ctrl.pivot_ppm, self.ctrl.pivot_ppm]
            self.ctrl.pivotplot.set_xdata(x)
            self.ud_phase()
            self.ctrl.frames['PlotFrame'].canvas.draw_idle()

        else:
            self.pivot_var.set(f'{self.ctrl.pivot_ppm:.3f}')


    def ud_p0_p1(self, phase, s):
        """Deals with a change to either the p0/p1 scale or entry widget"""

        # check angle is in the suitable range (check for entry widget)
        # floor and ceil the lower and upper bounds to give a bit of leighway
        # (get bugs at extremes of scale if this rounding isn't included)

        low = self.decifloor(self.__dict__[f'{s}_scale']['from'], 3)
        high = self.deciceil(self.__dict__[f'{s}_scale']['to'], 3)

        if  low <= phase <= high:
            pass
        else:
            # angle outside range: wrap it!
            phase = (phase + np.pi) % (2 * np.pi) - np.pi

        self.__dict__[f'{s}_var'].set(f'{phase:.3f}')
        self.ctrl.__dict__[s] = phase
        self.__dict__[f'{s}_scale'].set(phase)
        self.ud_phase()
        self.ctrl.frames['PlotFrame'].canvas.draw_idle()


    def ud_phase(self):
        """Phase the spectral data and update the figure plot's y data"""

        pivot = self.ctrl.pivot
        p0 = self.ctrl.p0
        p1 = self.ctrl.p1
        n = self.ctrl.n

        # apply phase correcion to original spectral data
        spec = self.ctrl.spec * np.exp(
            1j * (p0 + (p1 * np.arange(-pivot, -pivot + n, 1) / n))
        )
        
        # update y-data of spectrum plot
        self.ctrl.specplot.set_ydata(np.real(spec))

    def deciceil(self, value, precision):
        """round a number up to a certain number of deicmal places"""
        return np.round(value + 0.5 * 10**(-precision), precision)


    def decifloor(self, value, precision):
        """round a number down to a certain number of deicmal places"""
        return np.round(value - 0.5 * 10**(-precision), precision)


class AdvancedSettingsFrame(tk.Frame):
    """Frame inside SetupApp notebook - for customising details about the
    optimisation routine"""

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        # create multiple frames (one for each row)
        # don't need to conform to single grid for whole frame:
        # more organic layout
        self.rows = {}

        for i in range(6):
            frame = tk.Frame(self, bg='white')
            if i == 4:
                frame.grid(row=i, column=0, sticky='w', padx=10, pady=10)
            else:
                frame.grid(row=i, column=0, sticky='w', padx=10, pady=(10,0))
            self.rows[f'{i+1}'] = frame

        # determine default number of points for MPM and NLP
        self.mpm_points_var = tk.StringVar()
        self.nlp_points_var = tk.StringVar()

        for var, n in zip((self.mpm_points_var, self.nlp_points_var),
                          (4096, 8192)):
            if self.ctrl.n >= n:
                var.set(str(n))
            else:
                var.set(str(self.ctrl.n))

        # --- ROW 1 ---
        mpm_title = tk.Label(
            self.rows['1'], text='Matrix Pencil', bg='white',
            font=('Helvetica', 12, 'bold')
        )
        mpm_title.grid(row=0, column=0)

        # --- ROW 2 ---
        datapoint_label = tk.Label(
            self.rows['2'], text='Datapoints to consider:', bg='white'
        )
        datapoint_label.grid(row=0, column=0)

        self.mpm_points_entry = tk.Entry(
            self.rows['2'], width=8, highlightthickness=0,
            textvariable=self.mpm_points_var
        )
        self.mpm_points_entry.grid(row=0, column=1, padx=(10,0))

        oscillator_label = tk.Label(
            self.rows['2'], text='Number of oscillators:', bg='white'
        )
        oscillator_label.grid(row=0, column=2, padx=(20,0))

        self.oscillator_var = tk.StringVar()
        self.oscillator_var.set('')

        self.oscillator_entry = tk.Entry(
            self.rows['2'], width=8, highlightthickness=0, bg='white',
            textvariable=self.oscillator_var, state='disabled'
        )
        self.oscillator_entry.grid(row=0, column=3, padx=(10,0))

        use_mdl_label = tk.Label(self.rows['2'], text='Use MDL:', bg='white')
        use_mdl_label.grid(row=0, column=4, padx=(10,0))

        self.use_mdl = tk.StringVar()
        self.use_mdl.set('1')

        self.mdl_checkbutton = tk.Checkbutton(
            self.rows['2'], variable=self.use_mdl, bg='white',
            highlightthickness=0, bd=0, command=self.ud_mdl_button
        )
        self.mdl_checkbutton.grid(row=0, column=5)


        # --- ROW 3 ---
        nlp_title = tk.Label(
            self.rows['3'], text='Nonlinear Programming', bg='white',
            font=('Helvetica', 12, 'bold')
        )
        nlp_title.grid(row=0, column=0)

        # --- ROW 4 ---
        # construct a 2 x 6 grid to arrange the various widgets
        self.row4_frames = {}
        for r in range(2):
            for c in range(6):
                self.row4_frames[f'{r},{c}'] = tk.Frame(
                    self.rows['4'], bg='white'
                )
                self.row4_frames[f'{r},{c}'].grid(row=r, column=c, sticky='w')

        datapoint_label = tk.Label(
            self.row4_frames['0,0'], text='Datapoints to consider:', bg='white'
        )
        datapoint_label.grid(row=0, column=0)

        self.nlp_points_entry = tk.Entry(
            self.row4_frames['0,1'], width=8, highlightthickness=0,
            textvariable=self.nlp_points_var
        )
        self.nlp_points_entry.grid(row=0, column=0, padx=(10,0))

        self.nlp_algorithm = tk.StringVar()
        self.nlp_algorithm.set('Trust Region')

        nlp_method_label = tk.Label(
            self.row4_frames['0,2'], text='NLP algorithm:', bg='white'
        )
        nlp_method_label.grid(row=0, column=0, padx=(20,0))

        options = ['Trust Region', 'L-BFGS']
        self.algorithm_menu = tk.OptionMenu(
            self.row4_frames['0,3'], self.nlp_algorithm, *options
        )
        self.algorithm_menu['bg'] = 'white'
        self.algorithm_menu['borderwidth'] = 0
        self.algorithm_menu['width'] = 10
        self.algorithm_menu['menu']['bg'] = 'white'

        self.nlp_algorithm.trace('w', self.ud_max_iterations)

        self.algorithm_menu.grid(row=0, column=0, padx=(10,0))


        self.max_iterations = tk.StringVar()
        self.max_iterations.set('100')

        max_iterations_label = tk.Label(
            self.row4_frames['0,4'], text='Maximum iterations:', bg='white'
        )
        max_iterations_label.grid(row=0, column=0, padx=(20,0))

        self.max_iterations_entry = tk.Entry(
            self.row4_frames['0,5'], width=8, highlightthickness=0, bg='white',
            textvariable=self.max_iterations
        )
        self.max_iterations_entry.grid(row=0, column=0, padx=(10,0))

        phase_variance_label = tk.Label(
            self.row4_frames['1,0'], text='Optimise phase variance:',
            bg='white'
        )
        phase_variance_label.grid(row=0, column=0, pady=(10,0))

        self.phase_variance = tk.StringVar()
        self.phase_variance.set('1')

        self.phase_var_checkbutton = tk.Checkbutton(
            self.row4_frames['1,1'], variable=self.phase_variance, bg='white',
            highlightthickness=0, bd=0
        )
        self.phase_var_checkbutton.grid(row=0, column=0, pady=(10,0))

        # amplitude/frequency thresholds
        amplitude_thold_label = tk.Label(
            self.row4_frames['1,2'], text='Amplitude threshold:', bg='white')
        amplitude_thold_label.grid(row=0, column=0, padx=(20,0), pady=(10,0))

        self.amplitude_thold = tk.StringVar()
        self.amplitude_thold.set('0.001')

        self.amplitude_thold_entry = tk.Entry(
            self.row4_frames['1,3'], width=8, highlightthickness=0, bg='white',
            textvariable=self.amplitude_thold, state='disabled'
        )
        self.amplitude_thold_entry.grid(row=0, column=0, padx=(10,0), pady=(10,0))

        self.use_amp_thold = tk.StringVar()
        self.use_amp_thold.set('0')

        self.amplitude_thold_checkbutton = tk.Checkbutton(
            self.row4_frames['1,3'], variable=self.use_amp_thold, bg='white',
            highlightthickness=0, bd=0, command=self.ud_amp_thold_button
        )
        self.amplitude_thold_checkbutton.grid(row=0, column=1, pady=(10,0))

        frequency_thold_label = tk.Label(
            self.row4_frames['1,4'], text='Frequency threshold:', bg='white')
        frequency_thold_label.grid(row=0, column=0, padx=(20,0), pady=(10,0))

        self.frequency_thold = tk.StringVar()
        self.frequency_thold.set(f'{1 / self.ctrl.sw_p:.4f}')

        self.frequency_thold_entry = tk.Entry(
            self.row4_frames['1,5'], width=8, highlightthickness=0, bg='white',
            textvariable=self.frequency_thold
        )
        self.frequency_thold_entry.grid(row=0, column=0, padx=(10,0), pady=(10,0))

        self.use_freq_thold = tk.StringVar()
        self.use_freq_thold.set('1')

        self.frequency_thold_checkbutton = tk.Checkbutton(
            self.row4_frames['1,5'], variable=self.use_freq_thold, bg='white',
            highlightthickness=0, bd=0, command=self.ud_freq_thold_button
        )
        self.frequency_thold_checkbutton.grid(row=0, column=1, pady=(10,0))


    def ud_mdl_button(self):
        """For when the user clicks on the checkbutton relating to use the
        MDL"""

        value = int(self.use_mdl.get()) # 0 or 1
        if value:
            self.oscillator_entry['state'] = 'disabled'
            self.oscillator_var.set('')
        else:
            self.oscillator_entry['state'] = 'normal'


    def ud_amp_thold_button(self):
        """For when the user clicks on the checkbutton relating to whether
        or not to impose an amplitude threshold"""

        value = int(self.use_amp_thold.get()) # 0 or 1
        if value:
            self.amplitude_thold_entry['state'] = 'normal'
        else:
            self.amplitude_thold_entry['state'] = 'disabled'


    def ud_freq_thold_button(self):
        """For when the user clicks on the checkbutton relating to whether
        or not to impose a frequency threshold"""

        value = int(self.use_freq_thold.get()) # 0 or 1
        if value:
            self.frequency_thold_entry['state'] = 'normal'
        else:
            self.frequency_thold_entry['state'] = 'disabled'


    def ud_max_iterations(self, *args):
        """Called when user changes the NLP algorithm. Sets the default
        number of maximum iterations for the given method"""

        method = self.nlp_algorithm.get()
        if method == 'Trust Region':
            self.max_iterations.set('100')
        elif method == 'L-BFGS':
            self.max_iterations.set('500')


class ResultFrame(tk.Toplevel):
    """App for dealing with result of estimation. Enables user to tweak
    oscillators (and re-run the optimisation with the updated parameter array),
    customise the final plot, and save results."""

    def __init__(self, parent, ctrl):
        tk.Toplevel.__init__(self)
        self.title('NMR-EsPy - Result')
        self.protocol('WM_DELETE_WINDOW', self.destroy)

        # main container: everything goes inside here
        container = tk.Frame(self, bg='white')
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # instance of nmrespy.core.NMREsPyBruker

        # plot result
        self.fig, self.ax, self.lines, self.labels = self.ctrl.info.plot_result()
        # edit figure resolution and size
        self.fig.set_dpi(170)
        self.fig.set_size_inches(6, 3.5)

        # restrict x-axis to spectral window of interest
        xlim = self.ax.get_xlim()
        Restrictor(self.ax, x=lambda x: x<= xlim[0]) # restrict left
        Restrictor(self.ax, x=lambda x: x>= xlim[1]) # restrict right

        # contain all frames inside dictionary (makes it easy to acquire
        # attributes whilst in different classes)
        self.frames = {}

        # append all frames to the window
        self.frames['PlotFrame'] = PlotFrame(parent=container, ctrl=self.ctrl)
        self.frames['PlotFrame'].grid(
            row=0, column=0, columnspan=2, sticky='nsew'
        )

        self.frames['LogoFrame'] = LogoFrame(
            parent=container, ctrl=self.ctrl, scale=0.06
        )
        self.frames['LogoFrame'].grid(
            row=1, column=0, padx=10, pady=10, sticky='w'
        )

        self.frames['ResultButtonFrame'] = ResultButtonFrame(
            parent=container, ctrl=self.ctrl
        )
        self.frames['ResultButtonFrame'].grid(row=1, column=1, sticky='s')



class RerunSettingsFrame(AdvancedSettingsFrame):

    def __init__(self, parent, ctrl):
        AdvancedSettingsFrame.__init__(self, parent, ctrl)
        self.ctrl = ctrl
        self.rows['0'].grid_forget()
        self.rows['1'].grid_forget()


class RootButtonFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        # TODO this will need to be changed once the docs go online
        docpath = '/home/simon/Documents/DPhil/projects/' \
                  + 'p0-FID_signal_processing/code/Python_Scripts/' \
                  + 'NMR-EsPy/docs/_build/html/topspin_gui.html'

        self.cancel_button = tk.Button(
            self, text='Cancel', width=8, bg='#ff9894',
            highlightbackground='black', command=self.ctrl.destroy
        )
        self.cancel_button.grid(
            row=1, column=0, padx=(10,0), pady=(10,0), sticky='e'
        )

        self.help_button = tk.Button(
            self, text='Help', width=8, bg='#ffb861',
            highlightbackground='black',
            command=lambda: webbrowser.open_new(docpath)
        )
        self.help_button.grid(
            row=1, column=1, padx=(10,0), pady=(10,0), sticky='e'
        )

        # command varies - will need to be defined from the class that
        # inherits from this
        self.save_button = tk.Button(
            self, text='Run', width=8, bg='#9eda88',
            highlightbackground='black'
        )
        self.save_button.grid(
            row=1, column=2, padx=10, pady=(10,0), sticky='e'
        )

        contact_info_1 = tk.Label(
            self, text='For queries/feedback, contact', bg='white'
        )
        contact_info_1.grid(
            row=2, column=0, columnspan=3, padx=10, pady=(10,0), sticky='w'
        )

        email = 'simon.hulse@chem.ox.ac.uk'
        contact_info_2 = tk.Label(
            self, text=email, bg='white', font='Courier', fg='blue',
            cursor='hand1'
        )
        contact_info_2.bind(
            '<Button-1>', lambda e: webbrowser.open_new(f'mailto:{email}')
        )

        contact_info_2.grid(
            row=3, column=0, columnspan=3, padx=10, pady=(0,10), sticky='w'
        )


class SetupButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, parent, ctrl):
        RootButtonFrame.__init__(self, parent, ctrl)
        self.ctrl = ctrl
        self.save_button['command'] = self.run

    def run(self):
        """Set up the estimation routine"""

        # get parameters
        spec = self.ctrl.specplot.get_ydata()

        region = (self.lb, self.rb)
        noise_region = (self.lnb, self.rnb)

        pivot = self.ctrl.pivot
        p0 = self.ctrl.p0
        p1 = self.ctrl.p1

        adsetframe = self.ctrl.frames['TabFrame'].nbframes['AdvancedSettingsFrame']

        mpm_points = adsetframe.mpm_points_var.get()
        nlp_points = adsetframe.nlp_points_var.get()
        try:
            mpm_points = (int(mpm_points),)
        except:
            msg = 'The number of points for the MPM is not valid' \
                  + f' (\'{mpm_points}\' could not be converted to an integer)'
            WarnFrame(self.ctrl, msg)
            return
        if mpm_points[0] > self.ctrl.n:
            msg = 'The number of points for the MPM is too large' \
                  + f' (it should be less than or equal to {self.ctrl.n})'
            WarnFrame(self.ctrl, msg)
            return

        try:
            nlp_points = (int(nlp_points),)
        except:
            msg = 'The number of points for nonlinear programming is not' \
                  + f' valid (\'{nlp_points}\' could not be converted to an' \
                  + ' integer)'
            WarnFrame(self.ctrl, msg)
            return
        if nlp_points[0] > self.ctrl.n:
            msg = 'The number of points for nonlinear programming is too' \
                  + f' large (it should be less than or equal to {self.ctrl.n})'
            WarnFrame(self.ctrl, msg)
            return

        if int(adsetframe.use_mdl.get()):
            M_in = 0
        else:
            M_in = adsetframe.oscillator_var.get()
            try:
                M_in = int(M_in)
            except:
                 msg = f'The number of oscillators for the MPM (\'{M_in}\')' \
                       + ' could not be interpreted as an integer.'
                 WarnFrame(self.ctrl, msg)
                 return

        algorithm = adsetframe.nlp_algorithm.get()
        if algorithm == 'Trust Region':
            algorithm = 'trust_region'
        elif algorithm == 'L-BFGS':
            algorithm = 'lbfgs'

        max_iterations = adsetframe.max_iterations.get()
        try:
            max_iterations = int(max_iterations)
        except:
            msg = 'The number of maximum iterations for NLP' \
                  + f'(\'{max_iterations}\') could not be interpreted as an' \
                  + ' integer.'
            WarnFrame(self.ctrl, msg)
            return

        phase_variance = bool(int(adsetframe.phase_variance.get()))

        use_amp_thold = int(adsetframe.use_amp_thold.get())

        if use_amp_thold:
            amplitude_thold = adsetframe.amplitude_thold.get()
            if amplitude_thold == '':
                amplitude_thold = None
            else:
                try:
                    amplitude_thold = float(amplitude_thold)
                except:
                    msg = f'The amplitude threshold (\'{amplitude_thold}\')' \
                          + ' could not be interpreted as a float.'
                    WarnFrame(self.ctrl, msg)
                    return

        else:
            amplitude_thold = None

        use_freq_thold = int(adsetframe.use_freq_thold.get())

        if use_freq_thold:
            frequency_thold = adsetframe.frequency_thold.get()
            if frequency_thold == '':
                frequency_thold = None
            else:
                try:
                    frequency_thold = float(frequency_thold)
                except:
                    msg = f'The frequency threshold (\'{frequency_thold}\')' \
                          + ' could not be interpreted as a float.'
                    WarnFrame(self.ctrl, msg)
                    return

        else:
            frequency_thold = None

        self.ctrl.destroy()

        self.ctrl.info.frequency_filter(
            region=region, nose_region=noise_region, p0=p0, p1=p1,
            cut=True,
        )

        self.ctrl.info.matrix_pencil(trim=mpm_points, M_in=M_in)

        self.ctrl.info.nonlinear_programming(
            trim=nlp_points, maxit=max_iterations, method=algorithm,
            phase_variance=phase_variance, amp_thold=amplitude_thold,
            freq_thold=frequency_thold
        )


        self.ctrl.info.pickle_save(
            fname='tmp.pkl', dir=TOPSPINDIR, force_overwrite=True
        )

        ResultFrame()
        # subprocess.Popen(
        #     [sys.executable, os.path.join(TOPSPINDIR, 'result.py')]
        # )


class ResultButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, parent, ctrl):
        RootButtonFrame.__init__(self, parent, ctrl)
        self.parent = parent
        self.ctrl = ctrl

        self.save_button['command'] = self.save_options
        self.save_button['text'] = 'Save'

        self.edit_parameter_button = tk.Button(
            self, text='Edit Parameter Estimate', width=20, bg='#7fccf3',
            highlightbackground='black', command=self.edit_parameters
        )
        self.edit_parameter_button.grid(
            row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=(10,0)
        )

    def edit_parameters(self):
        EditParams(parent=self, ctrl=self.ctrl)

    def save_options(self):
        SaveFrame(parent=self, ctrl=self.ctrl)


class LogoFrame(tk.Frame):
    """Contains the NMR-EsPy logo (who doesn't like a bit of publicity)"""

    def __init__(self, parent, ctrl, scale=0.08):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        self.nmrespy_img = get_PhotoImage(
            os.path.join(IMAGESDIR, 'nmrespy_full.png'), scale
        )
        self.nmrespy_logo = tk.Label(
            self, image=self.nmrespy_img, bg='white', cursor='hand1'
        )

        # TODO this will need to be changed once the docs go online
        docpath = '/home/simon/Documents/DPhil/projects/' \
                  + 'p0-FID_signal_processing/code/Python_Scripts/' \
                  + 'NMR-EsPy/docs/_build/html/index.html'

        self.nmrespy_logo.bind(
            '<Button-1>', lambda e: webbrowser.open_new(docpath)
        )
        self.nmrespy_logo.grid(row=0, column=0)

        self.mfgroup_img = get_PhotoImage(
            os.path.join(IMAGESDIR, 'mf_logo.png'), scale*10
        )
        self.mfgroup_logo = tk.Label(
            self, image=self.mfgroup_img, bg='white', cursor='hand1'
        )

        mfgrouppath = 'http://foroozandeh.chem.ox.ac.uk/home'

        self.mfgroup_logo.bind(
            '<Button-1>', lambda e: webbrowser.open_new(mfgrouppath)
        )
        self.mfgroup_logo.grid(row=0, column=1, padx=(40,0))


class EditParams(tk.Toplevel):
    """Window allowing user to edit the estimation result."""

    def __init__(self, parent, ctrl):

        tk.Toplevel.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'
        self.resizable(False, False)
        self.grab_set()

        # frame to contain the table of parameters
        self.table = tk.Frame(self, bg='white')
        self.table.grid(row=0, column=0)
        # generate table inside frame
        self.construct_table()
        # frame to contain various buttons: mrege, split, manual edit, close
        self.buttonframe = tk.Frame(self, bg='white')
        self.buttonframe.grid(row=1, column=0, sticky='e')


        # Re-run nonlinear programming
        self.rerunbutton = tk.Button(
            self.buttonframe, text='Re-run NLP',
            highlightbackground='black', state='active', command=self.rerun
        )
        self.rerunbutton.grid(column=0, row=0, rowspan=2, pady=10, stick='ns')

        # add oscillator(s)
        self.addbutton = tk.Button(
            self.buttonframe, text='Add Oscillator(s)',
            highlightbackground='black', state='active', command=self.add
        )
        self.addbutton.grid(
            row=0, column=1, sticky='ew', padx=(10,0), pady=(10,0)
        )

        # add oscillator(s)
        self.removebutton = tk.Button(
            self.buttonframe, text='Remove Oscillator(s)',
            highlightbackground='black', state='disabled', command=self.remove
        )
        self.removebutton.grid(
            row=0, column=2, sticky='ew', padx=(10,0), pady=(10,0)
        )

        # manually edit parameters associated with oscillator
        self.manualbutton = tk.Button(
            self.buttonframe, text='Edit Manually',
            highlightbackground='black', state='disabled',
            command=self.manual_edit
        )
        self.manualbutton.grid(
            row=0, column=3, sticky='ew', padx=10, pady=(10,0)
        )

        # split selected oscillator
        self.splitbutton = tk.Button(
            self.buttonframe, text='Split Oscillator',
            highlightbackground='black', state='disabled', command=self.split
        )
        self.splitbutton.grid(
            row=1, column=1, sticky='ew', padx=(10,0), pady=(10,10)
        )

        # merge selected oscillators
        self.mergebutton = tk.Button(
            self.buttonframe, text='Merge Oscillators',
            highlightbackground='black', state='disabled', command=self.merge
        )
        self.mergebutton.grid(
            row=1, column=2, sticky='ew', padx=(10,0), pady=(10,10)
        )

        # close window
        self.closebutton = tk.Button(
            self.buttonframe, text='Close', highlightbackground='black',
            command=self.destroy
        )
        self.closebutton.grid(
            row=1, column=3, sticky='ew', padx=10, pady=(10,10)
        )


    def construct_table(self, reconstruct=False):

        if reconstruct:
            for widget in self.table.winfo_children():
                widget.destroy()

        # column titles
        column = 0
        for title in ('#', 'Amplitude', 'Phase', 'Frequency', 'Damping'):
            label = tk.Label(self.table, bg='white', text=title)

            if column == 0:
                label.grid(
                    row=0, column=column, ipadx=10, pady=(10,0), sticky='w'
                )
            elif column == 4:
                label.grid(
                    row=0, column=column, padx=(5,10), pady=(10,0), sticky='w'
                )
            else:
                label.grid(
                    row=0, column=column, padx=(5,0), pady=(10,0), sticky='w'
                )

            column += 1

        # store oscillator labels, entry widgets, and string variables
        self.table_labels = [] # tk.Label instances (M elements)
        self.table_entries = [] # tk.Entry instances (M x 4 elements)
        self.table_variables = [] # tk.StringVar instances (M x 4 elements)

        for i, oscillator in enumerate(self.ctrl.info.get_theta()):

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
            self.removebutton['state'] = 'disabled'
            self.splitbutton['state'] = 'disabled'
            self.mergebutton['state'] = 'disabled'
            self.manualbutton['state'] = 'disabled'

        # activate split and manual edit buttons
        # deactivate merge button (can't merge one oscillator...)
        elif activated_number == 1:
            self.removebutton['state'] = 'normal'
            self.splitbutton['state'] = 'normal'
            self.mergebutton['state'] = 'disabled'
            self.manualbutton['state'] = 'normal'

        # activate merge button
        # deactivate split and manual edit buttons (ambiguous with multiple
        # oscillators selected)
        else: # activated_number > 1
            self.removebutton['state'] = 'normal'
            self.splitbutton['state'] = 'disabled'
            self.mergebutton['state'] = 'normal'
            self.manualbutton['state'] = 'disabled'


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
        ud_plot(self.ctrl)

        # destroy and reconstruct the data table to match the new theta
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


class AddFrame(tk.Toplevel):
    """Window for adding oscillators.
    Opened after calling :py:meth:`EditParams.add`."""

    def __init__(self, parent, ctrl):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.ctrl = ctrl


        self['bg'] = 'white'
        self.resizable(False, False)
        self.grab_set()

        self.tableframe = tk.Frame(self, bg='white')
        self.tableframe.grid(row=0, column=0)

        self.entries = []
        self.vars = []

        entry_row = []
        var_row = []

        column = 0
        for title in ('Amplitude', 'Phase', 'Frequency', 'Damping'):
            label = tk.Label(self.tableframe, bg='white', text=title)

            var = tk.StringVar()
            var.set('')

            entry = tk.Entry(
                self.tableframe, bg='white', width=12, textvariable=var,
                highlightthickness=0
            )

            if column == 0:
                label.grid(
                    row=0, column=column, padx=(10,5), pady=(10,0), sticky='w'
                )
                entry.grid(row=1, column=column, padx=(10,5), pady=(5,0))
            elif column == 3:
                label.grid(
                    row=0, column=column, padx=(0,10), pady=(10,0), sticky='w'
                )
                entry.grid(row=1, column=column, padx=(0,10), pady=(5,0))
            else:
                label.grid(
                    row=0, column=column, padx=(0,5), pady=(10,0), sticky='w'
                )
                entry.grid(row=1, column=column, padx=(0,5), pady=(5,0))

            var_row.append(var)
            entry_row.append(entry)

            column += 1

        self.entries.append(entry_row)
        self.vars.append(var_row)

        self.buttonframe = tk.Frame(self, bg='white')
        self.buttonframe.grid(row=1, column=0, sticky='e')

        self.add_button = tk.Button(
            self.buttonframe, text='Add', width=3,
            highlightbackground='black', command=self.add_row,
        )
        self.add_button.grid(row=0, column=0, padx=(0,10), pady=(10,10))

        self.cancel_button = tk.Button(
            self.buttonframe, text='Cancel', width=8, bg='#ff9894',
            highlightbackground='black', command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=1, padx=(0,10), pady=(10,10))

        self.save_button = tk.Button(
            self.buttonframe, text='Save', width=8, bg='#9eda88',
            highlightbackground='black', command=self.save,
        )
        self.save_button.grid(row=0, column=2, padx=(0,10), pady=(10,10))

    def add_row(self):

        entry_row = []
        var_row = []

        row = len(self.entries) + 1

        for column in range(4):
            var = tk.StringVar()
            var.set('')

            entry = tk.Entry(
                self.tableframe, bg='white', width=12, textvariable=var,
                highlightthickness=0
            )

            if column == 0:
                entry.grid(row=row, column=column, padx=(10,5), pady=(5,0))
            elif column == 3:
                entry.grid(row=row, column=column, padx=(0,10), pady=(5,0))
            else:
                entry.grid(row=row, column=column, padx=(0,5), pady=(5,0))

            var_row.append(var)
            entry_row.append(entry)

        self.entries.append(entry_row)
        self.vars.append(var_row)


    def save(self):

        oscillators = []
        try:
            for var_row in self.vars:
                oscillator_row = []
                for var in var_row:

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


def ud_plot(ctrl):
    """Reconstructs the result plot after a change to the oscillators is
    made

    Parameters
    ----------

    ctrl : nmrespy.topspin.ResultApp
        controller
    """

    # get new lines and labels
    # also obtaining ax to acquire new y-limits
    _, ax, lines, labels = ctrl.info.plot_result()

    # wipe lines and text instances from the axis
    ctrl.ax.lines = []
    ctrl.ax.texts = []
    # wipe lines and text labels from the ctrl
    ctrl.lines = {}
    ctrl.labels = {}

    # plot data line onto axis
    ctrl.lines['data'] = ctrl.ax.plot(
        lines['data'].get_xdata(), lines['data'].get_ydata(),
        color=lines['data'].get_color(), lw=lines['data'].get_lw()
    )

    # plot oscillator lines and add oscillator text labels
    # append these to the lines and labels attributes of the ctrl
    lines_and_labels = zip(list(lines.values())[1:], list(labels.values()))
    for i, (line, label) in enumerate(lines_and_labels):

        key = f'osc{i+1}'
        # plot oscillator
        ctrl.lines[key] = ctrl.ax.plot(
            line.get_xdata(), line.get_ydata(), color=line.get_color(),
            lw=line.get_lw()
        )

        # add oscillator label
        x, y = label.get_position()
        ctrl.labels[key] = ctrl.ax.text(x, y, label.get_text())

    # update y-limits to fit new lines
    ctrl.ax.set_xlim(ax.get_xlim())
    ctrl.ax.set_ylim(ax.get_ylim())
    # draw the new plot!
    ctrl.frames['PlotFrame'].canvas.draw_idle()



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
            self, text='GeneralFrame', font=('Helvetica', 20, 'bold'), bg='white'
        ).pack(padx=30, pady=30)


class AxesFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        tk.Label(
            self, text='AxesFrame', font=('Helvetica', 20, 'bold'), bg='white'
        ).pack(padx=30, pady=30)


class LinesFrame(tk.Frame):

    def __init__(self, parent, ctrl):
        tk.Frame.__init__(self, parent)
        self.ctrl = ctrl
        self['bg'] = 'white'

        items = ['data']
        items += [f'osc{i+1}' for i in range(self.ctrl.info.theta.shape[0])]

        tk.Label(
            self, text='Color:', font=('Helvetica', 12, 'bold'), bg='white'
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
                lwframe, text='Linewidth:', font=('Helvetica', 12, 'bold'),
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
            self, text='LabelsFrame', font=('Helvetica', 20, 'bold'), bg='white'
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
                    troughcolor=tc, length=500, highlightthickness=0,
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
