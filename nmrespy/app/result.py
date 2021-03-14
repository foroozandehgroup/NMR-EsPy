import pathlib
from tkinter import filedialog

from matplotlib.backends import backend_tkagg

from .config import *
from .custom_widgets import *
from .frames import *


class Result(MyToplevel):
    def __init__(self, master):

        self.estimator = master.estimator
        # Generate figure of result
        self.result_plot = self.estimator.plot_result()
        self.result_plot.fig.set_size_inches(6, 3.5)
        self.result_plot.fig.set_dpi(170)

        # Prevent panning outside the selected region
        xlim = self.result_plot.ax.get_xlim()
        Restrictor(self.result_plot.ax, x=lambda x: x<= xlim[0])
        Restrictor(self.result_plot.ax, x=lambda x: x>= xlim[1])

        # --- Construction of the result GUI ------------------------------
        super().__init__(master)

        self.resizable(True, True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Canvas for figure
        self.canvas = backend_tkagg.FigureCanvasTkAgg(
            self.result_plot.fig, master=self,
        )
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            column=0, row=0, columnspan=2, padx=10, pady=10, sticky='nsew',
        )

        # Frame containing the navigation toolbar and advanced settings
        # button
        self.toolbar_frame = MyFrame(self)
        self.toolbar_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar = MyNavigationToolbar(
            self.canvas, parent=self.toolbar_frame,
        )
        self.toolbar.grid(row=0, column=0, sticky='w', padx=(10,0), pady=(0,5))

        # Frame with NMR-EsPy an MF group logos
        self.logo_frame = LogoFrame(self, scale=0.72)
        self.logo_frame.grid(row=2, column=0, sticky='w', padx=10, pady=10)

        # Frame with cancel/help/run/advanced settings buttons
        self.button_frame = ResultButtonFrame(self)
        self.button_frame.grid(
            row=1, column=1, rowspan=2, sticky='se', padx=10, pady=10,
        )


class ResultButtonFrame(RootButtonFrame):
    """Button frame for SetupApp. Buttons for quitting, loading help,
    and running NMR-EsPy"""

    def __init__(self, master):
        super().__init__(master)
        self.green_button['command'] = self.save_options
        self.green_button['text'] = 'Save'

        self.edit_parameter_button = MyButton(
            self, text='Edit Parameter Estimate', command=self.edit_parameters,
        )
        self.edit_parameter_button.grid(
            row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=(10,0)
        )

    def edit_parameters(self):
        print('TODO')
        # EditParams(parent=self, ctrl=self.master)

    def save_options(self):
        SaveFrame(self.master)


class SaveFrame(MyToplevel):
    """Toplevel for choosing how to save estimation result"""

    def __init__(self, master):
        super().__init__(master)

        # self.grab_set()

        # main contents of the toplevel
        self.file_frame = MyFrame(self)
        self.file_frame.grid(row=0, column=0)
        # buttons at the bottom of the frame
        self.button_frame = MyFrame(self)
        self.button_frame.grid(row=1, column=0, sticky='e')

        # --- RESULT FIGURE ---
        MyLabel(self.file_frame, text='Save Figure').grid(
            row=0, column=0, padx=(10,0), pady=(10,0), sticky='w'
        )
        # specifier for whether or not to save a figure
        self.figure_var = tk.IntVar()
        self.figure_var.set(1)
        # checkbutton to choose whether to save a figure
        self.figure_check = MyCheckbutton(
            self.file_frame, variable=self.figure_var,
            command=(lambda: self.check('figure')),
        )
        self.figure_check.grid(row=0, column=1, padx=(2,0), pady=(10,0))
        # open the figure customiser
        self.figure_button = MyButton(
            self.file_frame, text='Customise Figure',
            command=self.customise_figure,
        )
        self.figure_button.grid(
            row=0, column=2, columnspan=3, padx=10, pady=(10,0), sticky='ew',
        )

        # --- OTHER FILES: PDF, TEXT, PICKLE ---
        titles = ('Save textfile:', 'Save PDF:', 'Save CSV:', 'Pickle result:')
        self.fmts = ('txt', 'pdf', 'csv')
        for i, (title, tag) in enumerate(zip(titles, self.fmts+('pkl',))):

            MyLabel(self.file_frame, text=title).grid(
                row=i+1, column=0, padx=(10,0), pady=(10,0), sticky='w',
            )

            # variable which dictates whether to save the filetype
            self.__dict__[f'{tag}_var'] = save_var = tk.IntVar()
            save_var.set(1)

            self.__dict__[f'{tag}_check'] = check = \
            MyCheckbutton(
                self.file_frame, variable=save_var,
                command=(lambda tag=tag: self.check(tag))
            )
            check.grid(row=i+1, column=1, padx=(2,0), pady=(10,0))

            MyLabel(self.file_frame, text='Filename:').grid(
                row=i+1, column=2, padx=(15,0), pady=(10,0), sticky='w'
            )

            self.__dict__[f'{tag}_name'] = fname_var = tk.StringVar()
            fname_var.set('nmrespy_result')

            self.__dict__[f'{tag}_entry'] = entry = \
            MyEntry(
                self.file_frame, textvariable=fname_var, width=20,
            )
            entry.grid(row=i+1, column=3, padx=(5,0), pady=(10,0))

            MyLabel(self.file_frame, text=f".{tag}").grid(
                row=i+1, column=4, padx=(0,10), pady=(10,0), sticky='w',
            )

        # --- DESCRIPTION FOR TEXTFILE AND PDF ---
        MyLabel(self.file_frame, text='Description:').grid(
            row=5, column=0, padx=(10,0), pady=(10,0), sticky='nw',
        )

        self.descr_box = tk.Text(self.file_frame, width=40, height=3)
        self.descr_box.grid(
            row=5, column=1, columnspan=4, padx=10, pady=(10,0), sticky='ew',
        )

        # --- DIRECTORY TO SAVE FILES TO ---
        MyLabel(self.file_frame, text='Directory:').grid(
            row=6, column=0, padx=(10,0), pady=(10,0), sticky='w',
        )

        self.dir_var = tk.StringVar()
        self.dir_var.set(str(pathlib.Path.home()))

        self.dir_entry = tk.Entry(
            self.file_frame, textvariable=self.dir_var, width=25,
            highlightthickness=0
        )
        self.dir_entry.grid(
            row=6, column=1, columnspan=3, padx=(10,0), pady=(10,0), sticky='ew'
        )

        self.img = get_PhotoImage(IMAGESPATH / 'folder_icon.png', scale=0.02)

        self.dir_button = MyButton(
            self.file_frame, command=self.browse, image=self.img, width=40
        )
        self.dir_button.grid(row=6, column=4, padx=(5,5), pady=(10,0))

        # cancel button - returns usere to result toplevel
        self.cancel_button = MyButton(
            self.button_frame, text='Cancel', bg=BUTTONRED,
            command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=0, pady=10)

        # save button - determines what file types to save and generates them
        self.save_button = MyButton(
            self.button_frame, text='Save', width=8, bg=BUTTONGREEN,
            command=self.save
        )
        self.save_button.grid(row=0, column=1, padx=10, pady=10)


    def check(self, tag):
        """Deals with when user clicks a checkbutton"""
        var = self.__dict__[f'{tag}_var'].get()
        state = 'normal' if var else 'disabled'

        if tag == 'figure':
            self.figure_button['state'] = state
        else:
            self.__dict__[f'{tag}_entry']['state'] = state

    def browse(self):
        """Directory selection using tkinter's filedialog"""
        self.dir_var.set(filedialog.askdirectory(initialdir=self.dir_var.get()))

    def save(self):
        # Check directory is valid
        dir = self.dir_var.get()
        descr = self.descr_box.get('1.0', 'end-1c')

        for fmt in self.fmts:
            # Check text, PDF and CSV files
            if self.__dict__[f'{fmt}_var'].get():
                self.master.estimator.write_result(
                    path=f"{dir}/{self.__dict__[f'{fmt}_name'].get()}",
                    description=descr, fmt=fmt, force_overwrite=True,
                )

        # Check pickle
        if self.pkl_var.get():
            self.master.estimator.to_pickle(
                path=f"{dir}/{self.pkl_name.get()}", force_overwrite=True
            )


        self.master.destroy()

    def customise_figure(self):
        print('TODO')
        # CustomiseFigureFrame(self, self.master)
