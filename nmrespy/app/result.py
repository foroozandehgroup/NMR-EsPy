import pathlib
import subprocess
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

        self.grab_set()


        # --- Result figure ----------------------------------------------
        self.fig_frame = MyFrame(self)
        self.fig_frame.grid(row=0, column=0, pady=(10,0), padx=10, sticky='w')

        MyLabel(
            self.fig_frame, text='Result Figure',
            font=('Helvetica', 12, 'bold'),
        ).grid(row=0, column=0, columnspan=2, sticky='w')
        MyLabel(
            self.fig_frame, text='Save Figure:',
        ).grid(
            row=1, column=0, sticky='w', pady=(10,0),
        )
        MyLabel(self.fig_frame, text='Format:').grid(
            row=2, column=0, sticky='w', pady=(10,0),
        )
        MyLabel(self.fig_frame, text='Filename:').grid(
            row=3, column=0, sticky='w', pady=(10,0),
        )
        MyLabel(self.fig_frame, text='dpi:').grid(
            row=4, column=0, sticky='w', pady=(10,0),
        )
        MyLabel(self.fig_frame, text='Size (cm):').grid(
            row=5, column=0, sticky='w', pady=(10,0),
        )

        self.save_fig = tk.IntVar()
        self.save_fig.set(1)
        self.save_fig_checkbutton = MyCheckbutton(
            self.fig_frame, variable=self.save_fig, command=self.ud_save_fig,
        )
        self.save_fig_checkbutton.grid(row=1, column=1, sticky='w', pady=(10,0))

        self.fig_fmt = tk.StringVar()
        self.fig_fmt.set('pdf')
        self.fig_fmt.trace('w', self.ud_fig_fmt)

        options = ('eps', 'jpg', 'pdf', 'png', 'ps', 'svg')
        self.fig_fmt_optionmenu = tk.OptionMenu(
            self.fig_frame, self.fig_fmt, *options
        )
        self.fig_fmt_optionmenu['bg'] = BGCOLOR
        self.fig_fmt_optionmenu['width'] = 5
        self.fig_fmt_optionmenu['highlightbackground'] = 'black'
        self.fig_fmt_optionmenu['highlightthickness'] = 1
        self.fig_fmt_optionmenu['menu']['bg'] = BGCOLOR
        self.fig_fmt_optionmenu['menu']['activebackground'] = ACTIVETABCOLOR
        self.fig_fmt_optionmenu['menu']['activeforeground'] = 'white'
        self.fig_fmt_optionmenu.grid(
            row=2, column=1, sticky='w', pady=(10,0),
        )

        self.fig_name_frame = MyFrame(self.fig_frame)
        self.fig_name_frame.grid(row=3, column=1, sticky='w', pady=(10,0))
        self.fig_name = tk.StringVar()
        self.fig_name.set('nmrespy_figure')
        self.fig_name_entry = MyEntry(
            self.fig_name_frame, textvariable=self.fig_name, width=18,
            return_command=self.ud_file_name, return_args=(self.fig_name,),
        )
        self.fig_name_entry.grid(column=0, row=0)

        self.fig_fmt_label = MyLabel(self.fig_name_frame)
        self.ud_fig_fmt()
        self.fig_fmt_label.grid(column=1, row=0, padx=(2,0), pady=(5,0))

        self.fig_dpi = value_var_dict(300, '300')
        self.fig_dpi_entry = MyEntry(
            self.fig_frame, textvariable=self.fig_dpi['var'], width=6,
            return_command=self.ud_fig_dpi, return_args=(),
        )
        self.fig_dpi_entry.grid(row=4, column=1, sticky='w', pady=(10,0))

        self.fig_width = value_var_dict(15, '15')
        self.fig_height = value_var_dict(10, '10')

        self.fig_size_frame = MyFrame(self.fig_frame)
        self.fig_size_frame.grid(row=5, column=1, sticky='w', pady=(10,0))

        MyLabel(self.fig_size_frame, text='w:').grid(column=0, row=0)
        MyLabel(self.fig_size_frame, text='h:').grid(column=2, row=0)

        for i, (dim, value) in enumerate(zip(('width', 'height'), (15, 10))):

            self.__dict__[f"fig_{dim}"] = value_var_dict(value, str(value))
            self.__dict__[f"fig_{dim}_entry"] = MyEntry(
                self.fig_size_frame,
                textvariable=self.__dict__[f"fig_{dim}"]["var"],
                return_command=self.ud_fig_size,
                return_args=(dim,)
            )

            padx, column = ((2, 5), 1) if i == 0 else ((2, 0), 3)
            self.__dict__[f"fig_{dim}_entry"].grid(
                column=column, row=0, padx=padx,
            )

        # --- Other result files -----------------------------------------
        self.file_frame = MyFrame(self)
        self.file_frame.grid(row=1, column=0, padx=10, sticky='w')

        MyLabel(
            self.file_frame, text='Result Files',
            font=('Helvetica', 12, 'bold')
        ).grid(row=0, column=0, pady=(20,0), columnspan=4, sticky='w')

        MyLabel(
            self.file_frame, text='Format:'
        ).grid(row=1, column=0, pady=(10,0), columnspan=2, sticky='w')
        MyLabel(
            self.file_frame, text='Filename:'
        ).grid(
            row=1, column=2, columnspan=2, padx=(20,0), pady=(10,0), sticky='w',
        )

        titles = ('Text:', 'PDF:', 'CSV:')
        self.fmts = ('txt', 'pdf', 'csv')
        for i, (title, tag) in enumerate(zip(titles, self.fmts)):


            MyLabel(self.file_frame, text=title).grid(
                row=i+2, column=0, pady=(10,0), sticky='w',
            )

            # Dictates whether to save the file format
            self.__dict__[f'save_{tag}'] = save_var = tk.IntVar()
            save_var.set(1)

            self.__dict__[f'{tag}_check'] = check = \
            MyCheckbutton(
                self.file_frame, variable=save_var,
                command=lambda tag=tag: self.ud_save_file(tag)
            )
            check.grid(row=i+2, column=1, padx=(2,0), pady=(10,0), sticky='w')

            self.__dict__[f'name_{tag}'] = fname_var = tk.StringVar()
            fname_var.set('nmrespy_result')

            self.__dict__[f'{tag}_entry'] = entry = \
            MyEntry(
                self.file_frame, textvariable=fname_var, width=18,
                return_command=self.ud_file_name, return_args=(fname_var,),
            )
            entry.grid(row=i+2, column=2, padx=(20,0), pady=(10,0), sticky='w')

            self.__dict__[f'{tag}_ext'] = ext = MyLabel(
                self.file_frame, text=f".{tag}",
            )
            ext.grid(row=i+2, column=3, pady=(15,0), sticky='w')

        # Check if `pdflatex` exists.
        # If not, disable PDF option.
        check_latex = subprocess.run(
            ['pdflatex -v'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        )

        if check_latex.returncode != 0:
            self.save_pdf.set(0)
            self.pdf_check['state'] = 'disabled'
            self.pdf_entry['state'] = 'disabled'
            self.name_pdf.set('')
            self.pdf_ext['fg'] = '#808080'

        MyLabel(self.file_frame, text='Description:').grid(
            row=5, column=0, columnspan=4, pady=(10,0), sticky='w',
        )

        self.descr_box = MyText(self.file_frame, width=30, height=3)
        self.descr_box.grid(
            row=6, column=0, columnspan=4, pady=(10,0), sticky='ew',
        )

        # --- Pickle Estimator -------------------------------------------
        self.pickle_frame = MyFrame(self)
        self.pickle_frame.grid(row=2, column=0, padx=10, sticky='w')

        MyLabel(
            self.pickle_frame, text='Estimator',
            font=('Helvetica', 12, 'bold')
        ).grid(row=0, column=0, pady=(20,0), columnspan=4, sticky='w')

        MyLabel(
            self.pickle_frame, text='Save Estimator:',
        ).grid(
            row=1, column=0, sticky='w', pady=(10,0),
        )
        MyLabel(
            self.pickle_frame, text='Filename:',
        ).grid(
            row=2, column=0, sticky='w', pady=(10,0),
        )

        self.pickle_estimator = tk.IntVar()
        self.pickle_estimator.set(1)
        self.pickle_estimator_checkbutton = MyCheckbutton(
            self.pickle_frame, variable=self.pickle_estimator,
            command=self.ud_pickle_estimator,
        )
        self.pickle_estimator_checkbutton.grid(
            row=1, column=1, sticky='w', pady=(10,0),
        )

        self.pickle_name_frame = MyFrame(self.pickle_frame)
        self.pickle_name_frame.grid(row=2, column=1, sticky='w', pady=(10,0))
        self.pickle_name = tk.StringVar()
        self.pickle_name.set('estimator')
        self.pickle_name_entry = MyEntry(
            self.pickle_name_frame, textvariable=self.pickle_name, width=18,
            return_command=self.ud_file_name, return_args=(self.pickle_name,),
        )
        self.pickle_name_entry.grid(column=0, row=0)

        self.pickle_ext_label = MyLabel(
            self.pickle_name_frame, text='.pkl'
        )
        self.pickle_ext_label.grid(column=1, row=0, padx=(2,0), pady=(5,0))

        # --- Directory selection ----------------------------------------
        self.dir_frame = MyFrame(self)
        self.dir_frame.grid(row=3, column=0, padx=10, sticky='w')

        MyLabel(
            self.dir_frame, text='Directory',
            font=('Helvetica', 12, 'bold')
        ).grid(row=0, column=0, pady=(20,0), columnspan=2, sticky='w')

        self.dir_name = tk.StringVar()
        path = pathlib.Path.home()
        self.dir_name = value_var_dict(path, str(path))

        self.dir_entry = MyEntry(
            self.dir_frame, textvariable=self.dir_name['var'], width=30,
            return_command=self.ud_dir, return_args=(),
        )
        self.dir_entry.grid(
            row=1, column=0, pady=(10,0), sticky='w'
        )

        self.img = get_PhotoImage(IMAGESPATH / 'folder_icon.png', scale=0.02)

        self.dir_button = MyButton(
            self.dir_frame, command=self.browse, image=self.img, width=32,
            bg=BGCOLOR,
        )
        self.dir_button.grid(row=1, column=1, padx=(5,0), pady=(10,0))

        # --- Save/cancel buttons ----------------------------------------
        # buttons at the bottom of the frame
        self.button_frame = MyFrame(self)
        self.button_frame.grid(
            row=4, column=0, padx=10, pady=(0,10), sticky='e'
        )
        # cancel button - returns usere to result toplevel
        self.cancel_button = MyButton(
            self.button_frame, text='Cancel', bg=BUTTONRED,
            command=self.destroy,
        )
        self.cancel_button.grid(row=0, column=0, pady=(10,0))

        # save button - determines what file types to save and generates them
        self.save_button = MyButton(
            self.button_frame, text='Save', width=8, bg=BUTTONGREEN,
            command=self.save
        )
        self.save_button.grid(
            row=0, column=1, padx=(10,0), pady=(10,0),
        )

    # --- Save window methods --------------------------------------------
    # Figure settings
    def ud_save_fig(self):
        state = 'normal' if self.save_fig.get() else 'disabled'
        widgets = [
            self.fig_fmt_optionmenu,
            self.fig_name_entry,
            self.fig_dpi_entry,
            self.fig_width_entry,
            self.fig_height_entry,
        ]

        for widget in widgets:
            widget['state'] = state

        self.fig_fmt_label['fg'] = '#000000' if state == 'normal' else '#808080'

    def ud_fig_fmt(self, *args):
        self.fig_fmt_label['text'] = f".{self.fig_fmt.get()}"

    def ud_fig_dpi(self, *args):
        try:
            # Try to convert dpi text variable as an int. Both ensures
            # the user-given value can be converted to a numerical value
            # and removes any decimal places, if given. Reset as a string
            # afterwards.
            dpi = int(self.fig_dpi['var'].get())
            if not dpi > 0:
                raise
            self.fig_dpi['var'].set(str(dpi))
            self.fig_dpi['value'] = dpi

        except:
            # Failed to convert to int, reset to previous value
            self.fig_dpi['var'].set(str(self.fig_dpi['value']))

    def ud_fig_size(self, dim):
        try:
            length = float(self.__dict__[f"fig_{dim}"]["var"].get())
            if not length > 0:
                raise

            if check_int(length):
                # If length is an integer, remove decimal places.
                length = int(length)

            self.__dict__[f"fig_{dim}"]["value"] = length
            self.__dict__[f"fig_{dim}"]["var"].set(str(length))

        except:
            # Failed to convert to float, reset to previous value
            pass

        self.__dict__[f"fig_{dim}"]["var"].set(
            str(self.__dict__[f"fig_{dim}"]["value"])
        )

    # Result file settings
    def ud_save_file(self, tag):
        state = 'normal' if self.__dict__[f'save_{tag}'].get() else 'disabled'
        self.__dict__[f'{tag}_entry']['state'] = state
        self.__dict__[f'{tag}_ext']['fg'] = \
            '#000000' if state == 'normal' else '#808080'

    def ud_file_name(self, var):
        name = var.get()
        var.set("".join(x for x in name if x.isalnum() or x in " _-"))

    # Pickle estimator
    def ud_pickle_estimator(self):
        state = 'normal' if self.pickle_estimator.get() else 'disabled'
        self.pickle_name_entry['state'] = state
        self.pickle_ext_label['fg'] = \
            '#000000' if state == 'normal' else '#808080'

    # Save directory
    def browse(self):
        """Directory selection using tkinter's filedialog"""
        name = filedialog.askdirectory(initialdir=self.dir_name['value'])
        # If user clicks close cross, an empty tuple is returned
        if name:
            self.dir_name['value'] = pathlib.Path(name).resolve()
            self.dir_name['var'].set(str(self.dir_name['value']))

    def ud_dir(self):
        path = pathlib.Path(self.dir_name['var'].get()).resolve()
        if path.is_dir():
            self.dir_name['value'] = path
            self.dir_name['var'].set(path)
        else:
            self.dir_name['var'].set(str(self.dir_name['value']))


    def save(self):
        if not check_invalid_entries(self):
            msg = "Some parameters have not been validated."
            WarnFrame(self, msg=msg)
            return

        # Directory
        dir = self.dir_name['value']

        # Figure
        if self.save_fig.get():
            # Generate figure path
            fig_fmt = self.fig_fmt.get()
            dpi = self.fig_dpi['value']
            fig_name = self.fig_name.get()
            fig_path = dir / f"{fig_name}.{fig_fmt}"

            # Convert size from cm -> inches
            fig_size = (
                self.fig_width['value'] / 2.54,
                self.fig_height['value'] / 2.54,
            )
            fig = self.master.result_plot.fig
            fig.set_size_inches(*fig_size)
            fig.savefig(fig_path, dpi=dpi)

        # Result files
        for fmt in ('txt', 'pdf', 'csv'):
            if self.__dict__[f'save_{fmt}'].get():
                name = self.__dict__[f'name_{fmt}'].get()
                path = str(dir / name)
                description = self.descr_box.get('1.0', 'end-1c')
                if description == '':
                    description = None

                self.master.estimator.write_result(
                    path=path, description=description, fmt=fmt,
                    force_overwrite=True,
                )

        if self.pickle_estimator.get():
            name = self.pickle_name.get()
            path = str(dir / name)
            self.master.estimator.to_pickle(
                path=path, force_overwrite=True
            )

        self.master.destroy()

    def customise_figure(self):
        print('TODO')
        # CustomiseFigureFrame(self, self.master)