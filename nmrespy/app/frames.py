# frames.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 11:21:46 GMT

import webbrowser

import tkinter as tk

import nmrespy._paths_and_links as pl
import nmrespy.app.config as cf
import nmrespy.app.custom_widgets as wd

# TODO for animation
# from matplotlib.animation import FuncAnimation
# from matplotlib.backends import backend_tkagg
# import matplotlib.pyplot as plt
# import numpy as np


class LogoFrame(wd.MyFrame):
    """Contains the NMR-EsPy and MF groups logos"""

    def __init__(self, master, logos="both", scale=0.6):

        super().__init__(master)

        column = 0
        padx = 0

        if logos in ["both", "nmrespy"]:
            # add NMR-EsPy logo
            self.nmrespy_img = cf.get_PhotoImage(
                pl.NMRESPYLOGOPATH, scale / 2.3
            )
            self.nmrespy_logo = wd.MyLabel(
                self, image=self.nmrespy_img, cursor="hand1"
            )
            # provide link to NMR-EsPy docs
            self.nmrespy_logo.bind(
                "<Button-1>", lambda e: webbrowser.open_new(pl.GITHUBLINK),
            )
            self.nmrespy_logo.grid(row=0, column=column)

            column += 1
            padx = (40, 0)

        if logos in ["both", "mfgroup"]:
            # add MF group logo
            self.mfgroup_img = cf.get_PhotoImage(pl.MFLOGOPATH, scale)
            self.mfgroup_logo = wd.MyLabel(
                self, image=self.mfgroup_img, cursor="hand1",
            )
            # provide link to MF group website
            self.mfgroup_logo.bind(
                "<Button-1>", lambda e: webbrowser.open_new(pl.MFGROUPLINK)
            )
            self.mfgroup_logo.grid(row=0, column=column, padx=padx)


class WarnWindow(wd.MyToplevel):
    """A window in case the user does something silly."""

    def __init__(self, parent, msg):
        super().__init__(parent)
        self.title("NMR-EsPy - Warning")

        # warning image
        self.img = cf.get_PhotoImage(cf.WARNINGPATH, 0.08)
        self.warn_sign = wd.MyLabel(self, image=self.img)
        self.warn_sign.grid(row=0, column=0, padx=(10, 0), pady=10)

        # add text explaining the issue
        text = wd.MyLabel(self, text=msg, wraplength=400)
        text.grid(row=0, column=1, padx=10, pady=10)

        # close button
        close_button = wd.MyButton(
            self,
            text="Close",
            bg="#ff9894",
            command=self.destroy,
        )
        close_button.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10))


class DataType(wd.MyToplevel):
    """GUI for asking user whether they want to analyse the raw FID or
    pdata

    Parameters
    ----------
    parent : tk.Tk

    paths : dict
        Dictionary with two entries:

        * `'pdata'` - Path to processed data
        * `'fid'`` - Path to raw FID file
    """

    def __init__(self, ctrl, paths):
        self.ctrl = ctrl
        self.paths = paths
        super().__init__(self.ctrl)

        # --- Configure frames -------------------------------------------
        # Frame for the NMR-EsPy logo
        self.logo_frame = LogoFrame(self, logos="nmrespy", scale=0.8)
        # Frame containing path labels and checkboxes
        self.main_frame = wd.MyFrame(self)
        # Frame containing confirm/cancel buttons
        self.button_frame = wd.MyFrame(self)
        # Arrange frames
        self.logo_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.main_frame.grid(row=0, column=1, padx=(0, 10), pady=(10, 0))
        self.button_frame.grid(
            row=1,
            column=1,
            padx=(0, 10),
            pady=(0, 10),
            sticky="e",
        )

        # --- Frame heading ----------------------------------------------
        msg = wd.MyLabel(
            self.main_frame,
            text="Which data would you like to analyse?",
            font=(cf.MAINFONT, "12", "bold"),
        )
        msg.grid(column=0, row=0, columnspan=2, padx=10, pady=(10, 0))

        # --- Processd data checkbutton and labels -----------------------
        pdata_label = wd.MyLabel(self.main_frame, text="Processed Data")
        pdata_label.grid(column=0, row=1, padx=(10, 0), pady=(10, 0), sticky="w")

        pdatapath = wd.MyLabel(
            self.main_frame,
            text=f"{str(self.paths['pdata'])}/1r",
            font=("Courier", 11),
        )
        pdatapath.grid(column=0, row=2, padx=(10, 0), sticky="w")

        # `self.pdata` can be 0 and 1. Specifies whether to use pdata or not
        # This is directly dependent on `self.fid`. When one is `1`, the
        # other is `0`.
        self.pdata = tk.IntVar()
        self.pdata.set(1)
        self.pdata_box = wd.MyCheckbutton(
            self.main_frame,
            variable=self.pdata,
            command=self.click_pdata,
        )
        self.pdata_box.grid(column=1, row=1, rowspan=2, padx=(10, 0), sticky="nsw")

        # --- FID checkbutton and labels ---------------------------------
        fid_label = wd.MyLabel(self.main_frame, text="Raw FID")
        fid_label.grid(
            column=0,
            row=3,
            padx=(10, 0),
            pady=(10, 0),
            sticky="w",
        )

        fidpath = wd.MyLabel(
            self.main_frame,
            text=f"{str(self.paths['fid'])}/fid",
            font=("Courier", 11),
        )
        fidpath.grid(column=0, row=4, padx=(10, 0), sticky="w")

        # Initially have set to `0`, i.e. pdata is set to the default.
        self.fid = tk.IntVar()
        self.fid.set(0)
        self.fid_box = wd.MyCheckbutton(
            self.main_frame,
            variable=self.fid,
            command=self.click_fid,
        )
        self.fid_box.grid(column=1, row=3, rowspan=2, padx=(10, 0), sticky="nsw")

        # --- Confirm and Cancel buttons ---------------------------------
        self.confirmbutton = wd.MyButton(
            self.button_frame,
            text="Confirm",
            command=self.confirm,
            bg=cf.BUTTONGREEN,
        )
        self.confirmbutton.grid(
            column=1,
            row=0,
            padx=(5, 0),
            pady=(10, 0),
            sticky="e",
        )

        self.cancelbutton = wd.MyButton(
            self.button_frame,
            text="Cancel",
            command=self.ctrl.destroy,
            bg=cf.BUTTONRED,
        )
        self.cancelbutton.grid(column=0, row=0, pady=(10, 0), sticky="e")
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
            self.path = self.paths["fid"]
        else:
            self.path = self.paths["pdata"]
        self.destroy()


class RootButtonFrame(wd.MyFrame):
    def __init__(self, master, cancel_msg="Are you sure you want to close NMR-EsPy?"):
        super().__init__(master)

        self.cancel_msg = cancel_msg

        self.cancel_button = wd.MyButton(
            self, text="Cancel", bg=cf.BUTTONRED, command=self.cancel
        )
        self.cancel_button.grid(
            row=1,
            column=0,
            padx=(10, 0),
            pady=(10, 0),
            sticky="e",
        )

        self.help_button = wd.MyButton(
            self,
            text="Help",
            bg=cf.BUTTONORANGE,
            command=lambda: webbrowser.open_new(pl.DOCSLINK),
        )
        self.help_button.grid(row=1, column=1, padx=(10, 0), pady=(10, 0), sticky="e")

        # Command varies - will need to be defined from the class that
        # inherits from this
        # For example, see SetupButtonFrame
        self.green_button = wd.MyButton(self, bg=cf.BUTTONGREEN)
        self.green_button.grid(
            row=1,
            column=2,
            padx=10,
            pady=(10, 0),
            sticky="e",
        )

        contact_info_1 = wd.MyLabel(
            self,
            text="For queries/feedback, contact",
        )
        contact_info_1.grid(
            row=2,
            column=0,
            columnspan=3,
            padx=10,
            pady=(10, 0),
            sticky="w",
        )

        contact_info_2 = wd.MyLabel(
            self,
            text="simon.hulse@chem.ox.ac.uk",
            font="Courier",
            fg="blue",
            cursor="hand1",
        )
        contact_info_2.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new(pl.MAILTOLINK),
        )

        contact_info_2.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=10,
            pady=(0, 10),
            sticky="w",
        )

    def cancel(self):

        check = ConfirmWindow(
            parent=self,
            msg=self.cancel_msg,
            yes_text="Yes",
            no_text="No",
        )
        self.wait_window(check)

        if check.conf:
            # Destroy NMREsPyApp
            self.master.destroy()


class ConfirmWindow(wd.MyToplevel):
    """A window to double-check the user wants to do something."""

    def __init__(self, parent, msg, yes_text="Confirm", no_text="Cancel"):
        super().__init__(parent)

        self.conf = False

        # add text explaining the issue
        text = wd.MyLabel(self, text=msg, wraplength=400)
        text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # close button
        cancel_button = wd.MyButton(
            self,
            text=no_text,
            bg=cf.BUTTONRED,
            command=self.cancel,
        )
        cancel_button.grid(row=1, column=0, padx=10, pady=(0, 10))

        confirm_button = wd.MyButton(
            self,
            text=yes_text,
            bg=cf.BUTTONGREEN,
            command=self.confirm,
        )
        confirm_button.grid(row=1, column=1, padx=(0, 10), pady=(0, 10))

    def cancel(self):
        self.conf = False
        self.destroy()

    def confirm(self):
        self.conf = True
        self.destroy()


# # TODO: fix
# class WaitingWindow(MyToplevel):
#     """A window with an animation that appears while the estimation routine
#     is running."""
#
#     def __init__(self, master):
#         super().__init__(master)
#
#         # Create a simple FID for the animation
#         para = np.array([[1, 0, 2, 1.5]])
#         n = 256
#         sw = [50.]
#         self.fid = np.real(sig.make_fid(para, [n], sw)[0])
#
#         # Create a figure
#         self.fig = plt.figure(figsize=(3, 2))
#         self.ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])
#         self.ax.set_xlim(-10, n + 10)
#         pad = 0.05 * np.amax(self.fid)
#         self.ax.set_ylim(np.amin(self.fid) - pad, np.amax(self.fid) + pad)
#
#         self.fig.patch.set_facecolor(BGCOLOR)
#         self.ax.set_facecolor(BGCOLOR)
#
#         for pos in ('top', 'bottom', 'right', 'left'):
#             self.ax.spines[pos].set_visible(False)
#
#         self.ax.set_xticks([])
#         self.ax.set_yticks([])
#
#         def hex_color():
#             """Generates a random hex colour"""
#             def r():
#                 return random.randint(0, 255)
#             return f'#{r():02x}{r():02x}{r():02x}'
#
#         self.line, = self.ax.plot(
#             [], [], color=hex_color(), lw=5, solid_capstyle='round',
#         )
#
#         def animate(i):
#             x = self.line.get_xdata()
#             y = self.line.get_ydata()
#             s = x.size
#
#             if s == n:
#                 self.line.set_xdata(np.array([]))
#                 self.line.set_ydata(np.array([]))
#                 self.line.set_color(hex_color())
#             else:
#                 self.line.set_xdata(np.hstack((x, np.arange(s, s + 4))))
#                 self.line.set_ydata(np.hstack((y, self.fid[s:s + 4])))
#
#             return self.line
#
#         ani = FuncAnimation(self.fig, animate, interval=20)
#
#         self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, master=self)
#         self.canvas.draw()
#         self.canvas.get_tk_widget().grid(
#             column=0, row=0, padx=50, pady=(50, 20),
#         )
#
#         self.label = MyLabel(
#             self, text='Estimating...', font=('Helvetica', 14, 'bold')
#         ).grid(row=1, padx=50, pady=(0, 30))
