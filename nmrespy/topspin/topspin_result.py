from copy import deepcopy
from itertools import cycle
import os
import random

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np

import nmrespy
from nmrespy import load


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
        self.ax = ax
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



class Main(tk.Frame):
    def __init__(self, master, info):
        tk.Frame.__init__(self, master)
        self.grid(row=0, column=0, sticky='nsew')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self['bg'] = 'white'

        self.info = info
        self.fig, self.ax, self.lines, self.labels = self.info.plot_result()
        self.fig.set_dpi(170)
        self.fig.set_size_inches(6, 3.5)

        self.xlim = self.ax.get_xlim()
        restrict1 = Restrictor(self.ax, x=lambda x: x<= self.xlim[0])
        restrict2 = Restrictor(self.ax, x=lambda x: x>= self.xlim[-1])


        # self.rightframe contains everything other than the plot
        self.rightframe = tk.Frame(self, bg='white')
        self.rightframe.grid(column=1, row=0, sticky='nsew')
        self.rightframe.rowconfigure(2, weight=1)

        # --- RESULT PLOT ------------------------------------------------------
        self.plotframe = PlotFrame(self, self.fig)
        self.logoframe = LogoFrame(self.rightframe)
        self.editframe = EditFrame(self.rightframe, self.lines, self.labels, self.plotframe.canvas)
        self.saveframe = SaveFrame(self.rightframe)
        self.buttonframe = ButtonFrame(self.rightframe, self.saveframe, self.info)
        self.contactframe = ContactFrame(self.rightframe)

        self.logoframe.columnconfigure(0, weight=1)





class PlotFrame(tk.Frame):
    def __init__(self, master, fig):
        tk.Frame.__init__(self, master)

        self.fig = fig

        self.config(bg='white')
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


class EditFrame(tk.Frame):
    def __init__(self, master, lines, labels, figcanvas):
        tk.Frame.__init__(self, master)
        self.master = master
        self.lines = lines
        self.labels = labels
        self.figcanvas = figcanvas

        self.config(bg='white')
        self.grid(row=1, column=0, sticky='ew')
        self.columnconfigure(0, weight=1)

        self.editoptions = tk.Label(self, text='Edit Options', bg='white',
                                    font=('Helvetica', 14))
        self.edit_params = tk.Button(self, text='Edit Parameters',
                                     command=print('TODO'), width=12,
                                     bg='#e0e0e0', highlightbackground='black')
        self.edit_lines = tk.Button(self, text='Edit Lines',
                                    command=self.config_lines, width=12,
                                    bg='#e0e0e0', highlightbackground='black')
        self.edit_labels = tk.Button(self, text='Edit Labels',
                                     command=print('TODO'), width=12,
                                     bg='#e0e0e0', highlightbackground='black')

        self.editoptions.grid(row=0, column=0, sticky='w')
        self.edit_params.grid(row=1, column=0)
        self.edit_lines.grid(row=2, column=0)
        self.edit_labels.grid(row=3, column=0)


    def config_lines(self):
        cfg = ConfigLines(self, self.lines, self.figcanvas)
        # print(cfg.val)


class SaveFrame(tk.Frame):
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

        path = os.path.dirname(nmrespy.__file__)
        image = Image.open(os.path.join(path, 'topspin/images/folder_icon.png'))
        scale = 0.02
        [w, h] = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image)

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


class ConfigLines(tk.Toplevel):
    def __init__(self, master, lines, figcanvas):
        tk.Toplevel.__init__(self, master)

        self.lines = lines
        self.figcanvas = figcanvas

        self.config(bg='white')

        opts = ['All Oscillators']
        keys = self.lines.keys()
        for key in keys:
            key = key.replace('osc', 'Oscillator ').replace('data', 'Data')
            opts.append(key)

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
        for opt in opts:
            self.list.insert(tk.END, opt)

        self.list.bind('<Double-Button-1>', (lambda entry: self.showlineeditor()))

        self.list.activate(0)
        self.list.selection_set(0)

        self.list.grid(row=0, column=0, sticky='new')

        self.close_but = tk.Button(self.buttonframe, text='Close', width=8,
                                   bg='#ff9894', highlightbackground='black',
                                   command=self.close)
        self.close_but.grid(row=0, column=0, sticky='w')

        self.showlineeditor()


    def showlineeditor(self):

        index = self.list.curselection()
        name = self.list.get(index)

        for widget in self.customframe.winfo_children():
            widget.destroy()

        if 'Oscillator ' in name:
            number = int(name.strip('Oscillator '))
            self.osceditframe = OscEdit(self.customframe, self.lines, number,
                                        self.figcanvas)

        elif name == 'Data':
            self.osceditframe = OscEdit(self.customframe, self.lines, 0,
                                        self.figcanvas)

        elif name == 'All Oscillators':
            self.osceditframe = OscMultiEdit(self.customframe, self.lines,
                                             self.figcanvas)

    def close(self):
        self.destroy()


class OscEdit(tk.Frame):
    def __init__(self, master, lines, number, figcanvas):
        tk.Frame.__init__(self, master)

        self.grid(row=0, column=0, sticky='nsew')

        self.lines = lines
        self.figcanvas = figcanvas

        if number is 0:
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
        self.lwframe.grid(row=1, column=0, sticky='ew')
        self.lsframe.grid(row=2, column=0, sticky='ew')

        # tweak scale and entry commands to dynamically change plot
        scales = [self.colorframe.r_scale, self.colorframe.g_scale,
                  self.colorframe.b_scale]
        commands = [lambda r: self.colorframe.ud_r_sc(r, dynamic=True,
                                     line=self.line, figcanvas=self.figcanvas),
                    lambda g: self.colorframe.ud_g_sc(g, dynamic=True,
                                     line=self.line, figcanvas=self.figcanvas),
                    lambda b: self.colorframe.ud_b_sc(b, dynamic=True,
                                     line=self.line, figcanvas=self.figcanvas)]
        for scale, command in zip(scales, commands):
            scale['command'] = command











        # --- FRAME 3: Linewidth ----------------------------------------------
        self.lw_ttl = tk.Label(self.lwframe, text='Linewidth', bg='white',
                               font=('Helvetica', 12))

        self.lw_scale = tk.Scale(self.lwframe, from_=0, to=5, orient=tk.HORIZONTAL,
                                 showvalue=0, bg='white', sliderlength=15, bd=0,
                                 troughcolor=f'#808080', length=200,
                                 highlightthickness=0, command=self.ud_lw_sc,
                                 resolution=0.001)
        self.lw_scale.set(self.lw)

        self.lw_ent = tk.Entry(self.lwframe, bg='white', text=f'{self.lw:.3f}',
                               width=6, highlightthickness=0)

        self.lw_ent.bind('<Return>', (lambda event: self.ud_lw_ent()))

        self.lw_ttl.grid(row=1, column=0, sticky='w', columnspan=2)
        self.lw_scale.grid(row=2, column=0, sticky='ew')
        self.lw_ent.grid(row=2, column=1, sticky='w')

        # --- FRAME 4: Linestyle Optionmenu -----------------------------------
        self.ls_ttl = tk.Label(self.lsframe, text='Linestyle', bg='white',
                               font=('Helvetica', 12))

        self.ls_options = ('solid', 'dotted', 'dashed', 'dashdot')
        self.ls_str = tk.StringVar()
        self.ls_str.set(self.ls_options[0])
        self.ls_optmenu = tk.OptionMenu(self.lsframe, self.ls_str, *self.ls_options,
                                        command=self.ud_ls)

        self.ls_ttl.grid(row=0, column=0, sticky='w')
        self.ls_optmenu.grid(row=0, column=1, sticky='w')

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


class OscMultiEdit(tk.Frame):
    def __init__(self, master, lines, figcanvas):
        tk.Frame.__init__(self, master)

        self.grid(row=0, column=0, sticky='nsew')

        self.lines = lines
        self.figcanvas = figcanvas

        self.colors = []
        for key in [k for k in self.lines.keys() if k is not 'data']:
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

        self.colorframe.grid(row=0, column=0, sticky='ew')
        self.lwframe.grid(row=1, column=0, sticky='ew')
        self.lsframe.grid(row=2, column=0, sticky='ew')

        # --- FRAME 1: Color List  --------------------------------------------
        self.color_ttl = tk.Label(self.colorframe, text='Color List', bg='white',
                                  font=('Helvetica', 12))
        self.colorlist = tk.Listbox(self.colorframe)
        for i, color in enumerate(self.colors):
            self.colorlist.insert(tk.END, color)
            self.colorlist.itemconfig(i, foreground=color)

        self.add_but = tk.Button(self.colorframe, text='Add', width=8,
                                 bg='#9eda88', highlightbackground='black',
                                 command=self.add_color)
        self.rm_but = tk.Button(self.colorframe, text='Remove', width=8,
                                 bg='#ff9894', highlightbackground='black',
                                 command=self.rm_color)

        self.color_ttl.grid(row=0, column=0, columnspan=3, sticky='w')
        self.colorlist.grid(row=1, column=0, rowspan=3)
        self.add_but.grid(row=0, column=1, sticky='w')
        self.rm_but.grid(row=1, column=1, sticky='w')

        # --- FRAME 2: Linewidth ----------------------------------------------
        self.lw_ttl = tk.Label(self.lwframe, text='Linewidth', bg='white',
                               font=('Helvetica', 12))

        self.lw_scale = tk.Scale(self.lwframe, from_=0, to=5, orient=tk.HORIZONTAL,
                                 showvalue=0, bg='white', sliderlength=15, bd=0,
                                 troughcolor=f'#808080', length=200,
                                 highlightthickness=0, command=self.ud_lw_sc,
                                 resolution=0.001)
        self.lw_scale.set(self.lw)

        self.lw_ent = tk.Entry(self.lwframe, bg='white', text=f'{self.lw:.3f}',
                               width=6, highlightthickness=0)

        self.lw_ent.bind('<Return>', (lambda event: self.ud_lw_ent()))

        self.lw_ttl.grid(row=0, column=0, sticky='w', columnspan=2)
        self.lw_scale.grid(row=1, column=0, sticky='ew')
        self.lw_ent.grid(row=1, column=1, sticky='w')

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

        # generate random hex color
        r = lambda: random.randint(0,255)
        color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
        self.colorpicker = ColorPicker(self.colorwindow, init_color=color)
        self.buttonframe = tk.Frame(self.colorwindow, bg='white')

        self.colorpicker.grid(row=0, column=0, sticky='ew')
        self.buttonframe.grid(row=1, column=0, sticky='ew')

        self.save_but = tk.Button(self.buttonframe, text='Confirm', width=8,
                                  bg='#9eda88', highlightbackground='black',
                                  command=self.confirm)

        self.cancel_but = tk.Button(self.buttonframe, text='Cancel', width=8,
                                  bg='#ff9894', highlightbackground='black',
                                  command=self.cancel)

        self.cancel_but.grid(row=2, column=0, sticky='e')
        self.save_but.grid(row=2, column=1, sticky='e')

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
        for key in [l for l in self.lines if l is not 'data']:
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
        for key in [l for l in self.lines if l is not 'data']:
            self.lines[key].set_lw(self.lw)
        self.figcanvas.draw_idle()

    def ud_linestyles(self, ls):
        self.ls = ls
        for key in [l for l in self.lines if l is not 'data']:
            self.lines[key].set_ls(self.ls)
        self.figcanvas.draw_idle()

    def close(self):
        self.master.destroy()


class ColorPicker(tk.Frame):
    def __init__(self, master, init_color):
        tk.Frame.__init__(self, master)

        self.color = init_color
        self.r = self.color[1:3] # red hex
        self.g = self.color[3:5] # green hex
        self.b = self.color[5:] # blue hex

        self.topframe = tk.Frame(self, bg='white')
        self.botframe = tk.Frame(self, bg='white')
        self.topframe.grid(row=0, column=0, sticky='ew')
        self.botframe.grid(row=1, column=0, sticky='ew')

        # --- FRAME 1: Color scales and entries  ------------------------------
        self.color_ttl = tk.Label(self.topframe, text='Color', bg='white',
                                  font=('Helvetica', 12))
        self.r_lab = tk.Label(self.topframe, text='R', fg='#ff0000', bg='white')
        self.g_lab = tk.Label(self.topframe, text='G', fg='#008000', bg='white')
        self.b_lab = tk.Label(self.topframe, text='B', fg='#0000ff', bg='white')

        self.r_scale = tk.Scale(self.topframe, from_=0, to=255,
                                orient=tk.HORIZONTAL, showvalue=0, bg='white',
                                sliderlength=15, bd=0,
                                troughcolor=f'#{self.r}0000', length=200,
                                highlightthickness=0,
                                command=lambda r: self.ud_r_sc(r))

        self.g_scale = tk.Scale(self.topframe, from_=0, to=255,
                                orient=tk.HORIZONTAL, showvalue=0, bg='white',
                                sliderlength=15, bd=0,
                                troughcolor=f'#00{self.g}00', length=200,
                                highlightthickness=0, command=self.ud_g_sc)

        self.b_scale = tk.Scale(self.topframe, from_=0, to=255,
                                orient=tk.HORIZONTAL, showvalue=0, bg='white',
                                sliderlength=15, bd=0,
                                troughcolor=f'#0000{self.b}', length=200,
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

        self.swatch = tk.Canvas(self.topframe, width=40, height=40)
        self.rect = self.swatch.create_rectangle(0, 0, 40, 40, fill=self.color)

        self.color_ttl.grid(row=0, column=0, columnspan=4, sticky='w')
        self.r_lab.grid(row=1, column=0, sticky='w')
        self.g_lab.grid(row=2, column=0, sticky='w')
        self.b_lab.grid(row=3, column=0, sticky='w')
        self.r_scale.grid(row=1, column=1, sticky='ew')
        self.g_scale.grid(row=2, column=1, sticky='ew')
        self.b_scale.grid(row=3, column=1, sticky='ew')
        self.r_ent.grid(row=1, column=2, sticky='w')
        self.g_ent.grid(row=2, column=2, sticky='w')
        self.b_ent.grid(row=3, column=2, sticky='w')
        self.swatch.grid(row=1, rowspan=3, column=3)

        # --- FRAME 2: Matplotlib color entry ---------------------------------
        self.mplcol_lab = tk.Label(self.botframe, text='Matplotlib color:',
                                   bg='white')
        self.mplcol_ent = tk.Entry(self.botframe, bg='white', width=20,
                                   highlightthickness=0)
        self.mplcol_ent.bind('<Return>', (lambda event: self.ud_mplcol_ent()))

        self.mplcol_lab.grid(row=0, column=0, sticky='w')
        self.mplcol_ent.grid(row=0, column=1, sticky='w')


    def ud_r_sc(self, r, dynamic=False, line=None, figcanvas=None):
        self.r = '{:02x}'.format(int(r))
        self.color = '#' + self.r + self.g + self.b
        self.r_scale['troughcolor'] = f'#{self.r}0000'
        self.r_ent.delete(0, tk.END)
        self.r_ent.insert(0, self.r.upper())
        self.swatch.itemconfig(self.rect, fill=self.color)

        if dynamic:
            line.set_color(self.color)
            figcanvas.draw_idle()


    def ud_g_sc(self, g, dynamic=False, line=None, figcanvas=None):
        self.g = '{:02x}'.format(int(g))
        self.color = '#' + self.r + self.g + self.b
        self.g_scale['troughcolor'] = f'#00{self.g}00'
        self.g_ent.delete(0, tk.END)
        self.g_ent.insert(0, self.g.upper())
        self.swatch.itemconfig(self.rect, fill=self.color)

        if dynamic:
            line.set_color(self.color)
            figcanvas.draw_idle()

    def ud_b_sc(self, b, dynamic=False, line=None, figcanvas=None):
        self.b = '{:02x}'.format(int(b))
        self.color = '#' + self.r + self.g + self.b
        self.b_scale['troughcolor'] = f'#0000{self.b}'
        self.b_ent.delete(0, tk.END)
        self.b_ent.insert(0, self.b.upper())
        self.swatch.itemconfig(self.rect, fill=self.color)

        if dynamic:
            line.set_color(self.color)
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







def main():
    info = load.pickle_load('result', dir=os.path.expanduser('~'))
    root = tk.Tk()
    root.title('NMR-EsPy - Result')
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    window = Main(root, info)
    root.mainloop()



if __name__ == '__main__':
    main()
