# write.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Writing estimation results to .txt, .pdf and .csv files"""

import csv
import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import scipy.linalg as slinalg

from nmrespy import *
from nmrespy._errors import *
from nmrespy._misc import ArgumentChecker, PathManager, significant_figures


def write_result(
    parameters, errors=None, path='./nmrespy_result', sfo=None, integrals=None,
    description=None, info_headings=None, info=None, sig_figs=5,
    sci_lims=(-2, 3), fmt='txt', force_overwrite=False, pdflatex_exe=None,
    fprint=True,
):
    """Writes an estimation result to a .txt, .pdf or .csv file.

    Parameters
    ----------
    parameters : numpy.ndarray
        The estimated parameter array.

    errors : numpy.ndarray or None, default: None
        The errors associated with the parameters. If not `None`, the shape
        of `errors` should match that of `parameters`.

    path : str, default: './nmrespy_result'
        The path to save the file to. DO NOT INCLUDE A FILE EXTENSION.
        This will be added based on the value of `fmt`. For example, if you
        set `path` to `.../result.txt` and `fmt` is `'txt'`, the resulting
        file will be `.../result.txt.txt`

    sfo : [float], [float, float] or None:
        The transmitter offset in each dimension. This is required to
        express frequencies in ppm as well as Hz. If `None`, frequencies
        will only be expressed in Hz.

    integrals : list or None
        Peak integrals of each oscillator. Note that
        ``parameters.shape[0] == len(integrals)`` should be satisfied.

    fmt : 'txt', 'pdf' or 'csv', default: 'txt'
        File format. See notes for details on system requirements for PDF
        generation.

    force_overwrite : bool. default: False
        Defines behaviour if ``f'{path}.{fmt}'`` already exists:

        * If `force_overwrite` is set to `False`, the user will be prompted
          if they are happy overwriting the current file.
        * If `force_overwrite` is set to `True`, the current file will be
          overwritten without prompt.

    description : str or None, default: None
        A descriptive statement.

    info_headings : list or None, default: None
        Headings for experiment information. Could include items like
        `'Sweep width (Hz)'`, `'Transmitter offset (Hz)'`, etc.
        N.B. All the elements in `info_headings` should be strings!

    info : list or None, default: None
        Information that corresponds to each heading in `info_headings`.
        N.B. All the elements in `info` should be strings!

    sig_figs : int or None default: 5
        The number of significant figures to give to parameter values. If
        `None`, the full value will be used.

    sci_lims : (int, int) or None, default: (-2, 3)
        Given a value ``(-x, y)``, for positive `x` and `y`, any parameter `p`
        value which satisfies ``p < 10 ** -x`` or ``p >= 10 ** y`` will be
        expressed in scientific notation, rather than explicit notation.
        If `None`, all values will be expressed explicitely.

    pdflatex_exe : str or None, default: None
        The path to the system's ``pdflatex`` executable.

        .. note::

           You are unlikely to need to set this manually. It is primarily
           present to specify the path to ``pdflatex.exe`` on Windows when
           the NMR-EsPy GUI has been loaded from TopSpin.

    fprint : bool, default: True
        Specifies whether or not to print information to the terminal.

    Raises
    ------
    LaTeXFailedError
        With `fmt` set to `'pdf'`, this will be raised if an error
        was encountered in trying to run ``pdflatex``.

    Notes
    -----
    To generate PDF result files, it is necessary to have a LaTeX installation
    set up on your system.
    For a simple to set up implementation that is supported on all
    major operating systems, consider
    `TexLive <https://www.tug.org/texlive/>`_. To ensure that
    you have a functioning LaTeX installation, open a command
    prompt/terminal and type:

    .. code::

       $ pdflatex -version
       pdfTeX 3.14159265-2.6-1.40.20 (TeX Live 2019/Debian)
       kpathsea version 6.3.1
       Copyright 2019 Han The Thanh (pdfTeX) et al.
       There is NO warranty.  Redistribution of this software is
       covered by the terms of both the pdfTeX copyright and
       the Lesser GNU General Public License.
       For more information about these matters, see the file
       named COPYING and the pdfTeX source.
       Primary author of pdfTeX: Han The Thanh (pdfTeX) et al.
       Compiled with libpng 1.6.37; using libpng 1.6.37
       Compiled with zlib 1.2.11; using zlib 1.2.11
       Compiled with xpdf version 4.01

    The following is a full list of packages that your LaTeX installation
    will need to successfully compile the generated .tex file:

    * `amsmath <https://ctan.org/pkg/amsmath?lang=en>`_
    * `array <https://ctan.org/pkg/array?lang=en>`_
    * `booktabs <https://ctan.org/pkg/booktabs?lang=en>`_
    * `cmbright <https://ctan.org/pkg/cmbright>`_
    * `geometry <https://ctan.org/pkg/geometry>`_
    * `hyperref <https://ctan.org/pkg/hyperref?lang=en>`_
    * `longtable <https://ctan.org/pkg/longtable>`_
    * `siunitx <https://ctan.org/pkg/siunitx?lang=en>`_
    * `tcolorbox <https://ctan.org/pkg/tcolorbox?lang=en>`_
    * `varwidth <https://www.ctan.org/pkg/varwidth>`_
    * `xcolor <https://ctan.org/pkg/xcolor?lang=en>`_

    If you wish to check the packages are available, use ``kpsewhich``:

    .. code::

        $ kpsewhich booktabs.sty
        /usr/share/texlive/texmf-dist/tex/latex/booktabs/booktabs.sty

    If a pathname appears, the package is installed to that path.
    """

    # --- Check validity of arguments ------------------------------------
    try:
        dim = int(parameters.shape[1] / 2) - 1
    except Exception:
        raise TypeError(
            f'{cols.R}parameters should be a numoy array{cols.END}'
        )

    components = [
        (parameters, 'parameters', 'parameter'),
        (path, 'path', 'str'),
        (fmt, 'fmt', 'file_fmt'),
        (force_overwrite, 'force_overwrite', 'bool'),
        (fprint, 'fprint', 'bool'),
    ]

    if errors is not None:
        components.append((errors, 'errors', 'parameter'))
    if sfo is not None:
        components.append((sfo, 'sfo', 'float_list'))
    if description is not None:
        components.append((description, 'description', 'str'))
    if info_headings is not None and info is not None:
        components.append((info_headings, 'info_headings', 'list'))
        components.append((info, 'info', 'list'))
    elif info_headings is None and info is None:
        pass
    else:
        raise ValueError(
            f'{cols.R}info and info_headings should either both be lists'
            f' of the same length, or both be None.{cols.END}'
        )
    if sig_figs is not None:
        components.append((sig_figs, 'sig_figs', 'positive_int'))
    if sci_lims is not None:
        components.append((sci_lims, 'sci_lims', 'pos_neg_tuple'))
    if integrals is not None:
        components.append((integrals, 'integrals', 'list'))

    ArgumentChecker(components, dim)
    # Should be same number of integrals as number of oscillators
    if isinstance(integrals, list) and len(integrals) != parameters.shape[0]:
        raise ValueError(
            f'{cols.R}integrals should have the same number of elements as'
            f' parameters.shape[0]{cols.END}'
        )
    # info and info_headings should be the same length if lists
    if isinstance(info, list) and len(info) != len(info_headings):
        raise ValueError(
            f'{cols.R}info and info_headings should be the same'
            f' length{cols.END}'
        )
    # parameters and errors should be the same shape, if errors is not None
    if isinstance(errors, np.ndarray) and errors.shape != parameters.shape:
        raise ValueError(
            f'{cols.R}`parameters` and `errors` should be the same'
            f' shape{cols.END}'
        )

    # Get full path
    path = Path(path).resolve()
    # Append extension to file path
    path = path.parent / (path.name + f'.{fmt}')
    # Check path is valid (check directory exists, ask user if they are happy
    # overwriting if file already exists).
    pathres = PathManager(path.name, path.parent).check_file(force_overwrite)
    # Valid path, we are good to proceed
    if pathres == 0:
        pass
    # Overwrite denied by the user. Exit the program
    elif pathres == 1:
        exit()
    # pathres == 2: Directory specified doesn't exist
    else:
        raise ValueError(
            f'{cols.R}The directory implied by path does not exist{cols.END}'
        )

    # Checking complete...

    # --- Construct nested list of components for parameter table --------
    param_titles, param_table = \
        _construct_paramtable(
            parameters, errors, integrals, sfo, sig_figs, sci_lims, fmt,
        )

    # --- Write to the specified file type -------------------------------
    if fmt == 'txt':
        _write_txt(
            path, description, info_headings, info, param_titles, param_table,
            fprint,
        )
    elif fmt == 'pdf':
        _write_pdf(
            path, description, info_headings, info, param_titles, param_table,
            pdflatex_exe, fprint,
        )
    elif fmt == 'csv':
        _write_csv(
            path, description, info_headings, info, param_titles, param_table,
            fprint,
        )


def _write_txt(
    path, description, info_headings, info, param_titles, param_table, fprint,
):
    """
    Writes parameter estimate to a textfile.

    Parameters
    -----------
    path : pathlib.Path
        File path

    description : str or None, default: None
        A descriptive statement.

    info_headings : list or None, default: None
        Headings for experiment information.

    info : list or None, default: None
        Information that corresponds to each heading in `info_headings`.

    param_titles : list
        Titles for parameter array table.

    param_table : list
        Array of contents to append to the result table.

    fprint: bool
        Specifies whether or not to print output to terminal.
    """

    # --- Write header ---------------------------------------------------
    # Time and date
    msg = f'{_timestamp()}\n'
    # User-provided description
    if description is not None:
        msg += f'\nDescription:\n{description}\n'

    # Table of experiment information
    if info is not None:
        msg += '\nExperiment Information:\n'
        msg += _txt_tabular([info_headings, info]) + '\n'

    # --- Add parameter table --------------------------------------------
    # Table of oscillator parameters
    # N.B. list(map(list, zip(*param_table))) effectively transposes the
    # list, which is given row-by-row. _txt_tabular takes a nested list
    # of columns as arguments.
    msg += _txt_tabular(
        list(map(list, zip(*param_table))), titles=param_titles,
        separator='│',
    )

    # --- Write footer ---------------------------------------------------
    msg += ("\nEstimation performed using NMR-EsPy\n"
            "Author: Simon Hulse ~ simon.hulse@chem.ox.ac.uk\n"
            "If used in any publications, please cite:\n"
            "<no papers yet...>\n"
            "For more information, visit the GitHub repo:\n")

    msg += f'{GITHUBLINK}'
    # Save message to textfile
    with open(path, 'w', encoding='utf-8') as file:
        file.write(msg)

    if fprint:
        print(f'{cols.G}Saved result to {path}{cols.END}')


def _write_pdf(
    path, description, info_headings, info, param_titles, param_table,
    pdflatex_exe, fprint,
):
    """Writes parameter estimate to a PDF using ``pdflatex``.

    Parameters
    -----------
    path : pathlib.Path
        File path

    description : str or None, default: None
        A descriptive statement.

    info_headings : list or None, default: None
        Headings for experiment information.

    info : list or None, default: None
        Information that corresponds to each heading in `info_headings`.

    param_titles : list
        Titles for parameter array table.

    param_table : list
        Array of contents to append to the result table.

    fprint: bool
        Specifies whether or not to print output to terminal.
    """
    # Open text of template .tex file which will be amended
    with open(NMRESPYPATH / 'config/latex_template.txt', 'r') as fh:
        txt = fh.read()

    # Add image paths and weblinks to TeX document
    # If on Windows, have to replace paths of the form:
    # C:\a\b\c
    # to:
    # C:/a/b/c
    patterns = (
        '<MFLOGOPATH>',
        '<NMRESPYLOGOPATH>',
        '<DOCSLINK>',
        '<MFGROUPLINK>',
        '<BOOKICONPATH>',
        '<GITHUBLINK>',
        '<GITHUBLOGOPATH>',
        '<MAILTOLINK>',
        '<EMAILICONPATH>',
    )

    paths = (
        MFLOGOPATH,
        NMRESPYLOGOPATH,
        DOCSLINK,
        MFGROUPLINK,
        BOOKICONPATH,
        GITHUBLINK,
        GITHUBLOGOPATH,
        MAILTOLINK,
        EMAILICONPATH,
    )

    for pattern, path_ in zip(patterns, paths):
        txt = txt.replace(pattern, str(path_).replace('\\', '/'))

    # Include a timestamp
    txt = txt.replace('<TIMESTAMP>', _timestamp().replace('\n', '\\\\'))

    # --- Description ----------------------------------------------------
    if description is None:
        # No description given, remove relavent section of .tex file
        txt = txt.replace(
            '% user provided description\n\\subsection*{Description}\n'
            '<DESCRIPTION>',
            '',
        )

    else:
        txt = txt.replace('<DESCRIPTION>', description)

    # --- Experiment Info ------------------------------------------------
    if info is None:
        # No info given, remove relavent section of .tex file
        txt = txt.replace(
            '\n% experiment parameters\n'
            '\\subsection*{Experiment Information}\n'
            '\\hspace{-6pt}\n'
            '\\begin{tabular}{ll}\n<INFOTABLE>\n'
            '\\end{tabular}\n',
            '',
        )

    else:
        # Construct 2-column tabular of experiment info headings and values
        rows = list(list(row) for row in zip(info_headings, info))
        info_table = _latex_tabular(rows)
        txt = txt.replace('<INFOTABLE>', info_table)

    # --- Parameter Table ------------------------------------------------
    # Determine number of columns required
    txt = txt.replace('<COLUMNS>', len(param_titles) * 'c')
    # Construct parameter title and table body
    txt = txt.replace('<PARAMTITLES>', _latex_tabular([param_titles]))
    txt = txt.replace('<PARAMTABLE>', _latex_tabular(param_table))

    # Incude plus-minus symbol. For denoting errors.
    txt = txt.replace("±", "$\\pm$ ")

    # TODO support for including result figure
    txt = txt.replace(
        '% figure of result\n\\begin{center}\n'
        '\\includegraphics{<FIGURE_PATH>}\n\\end{center}\n',
        '',
    )

    # --- Generate PDF using pdflatex ------------------------------------
    # Create required file paths:
    # .tex and .pdf paths with temporary directory (this is where the files
    # will be initially created)
    # .tex and .pdf files with desired directory (files will be moved from
    # temporary directory to desired directory once pdflatex is run).
    tex_tmp_path = Path(tempfile.gettempdir()) / path.with_suffix('.tex').name
    pdf_tmp_path = Path(tempfile.gettempdir()) / path.name
    tex_final_path = path.with_suffix('.tex')
    pdf_final_path = path

    # Write contents to cwd tex file
    with open(tex_tmp_path, 'w', encoding='utf-8') as fh:
        fh.write(txt)

    try:
        if pdflatex_exe is None:
            pdflatex_exe = "pdflatex"
        # -halt-on-error flag is vital. If any error arises in running
        # pdflatex, the program would get stuck
        subprocess.run(
            [pdflatex_exe,
             '-halt-on-error',
             f'-output-directory={tex_tmp_path.parent}',
             tex_tmp_path],
            stdout=subprocess.DEVNULL,
            check=True,
        )

        # Move pdf and tex files from temp directory to desired directory
        shutil.move(tex_tmp_path, tex_final_path)
        shutil.move(pdf_tmp_path, pdf_final_path)

    except subprocess.CalledProcessError:
        # pdflatex came across an error
        shutil.move(tex_tmp_path, tex_final_path)
        raise LaTeXFailedError(tex_final_path)

    except FileNotFoundError:
        # Most probably, pdflatex does not exist
        raise LaTeXFailedError(tex_final_path)

    # Remove other LaTeX files
    os.remove(tex_tmp_path.with_suffix('.out'))
    os.remove(tex_tmp_path.with_suffix('.aux'))
    os.remove(tex_tmp_path.with_suffix('.log'))

    # # TODO: remove figure file if it exists
    # try:
    #     os.remove(figure_path)
    # except UnboundLocalError:
    #     pass

    # Print success message
    if fprint:
        print(f'{cols.G}Result successfuly output to:\n'
              f'{pdf_final_path}\n'
              'If you wish to customise the document, the TeX file can'
              ' be found at:\n'
              f'{tex_final_path}{cols.END}')


def _write_csv(
    path, description, info_headings, info, param_titles, param_table,
    fprint,
):
    """Writes parameter estimate to a CSV.

    Parameters
    -----------
    path : pathlib.Path
        File path

    description : str or None, default: None
        A descriptive statement.

    info_headings : list or None, default: None
        Headings for experiment information.

    info : list or None, default: None
        Information that corresponds to each heading in `info_headings`.

    param_titles : list
        Titles for parameter array table.

    param_table : list
        Array of contents to append to the result table.

    fprint: bool
        Specifies whether or not to print output to terminal.
    """

    with open(path, 'w', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        # Timestamp
        writer.writerow([_timestamp().replace('\n', ' ')])
        writer.writerow([])
        # Description
        if description is not None:
            writer.writerow(['Description:', description])
            writer.writerow([])
        # Experiment info
        if info is not None:
            writer.writerow(['Experiment Info:'])
            for row in zip(info_headings, info):
                writer.writerow(row)
            writer.writerow([])
        # Parameter table
        writer.writerow(['Result:'])
        writer.writerow(param_titles)
        for row in param_table:
            writer.writerow(row)

    if fprint:
        print(f'{cols.G}Saved result to {path}{cols.END}')


def _map_to_latex_titles(titles):
    """Given a list of titles produced by :py:func:`_construct_paramtable`,
    Generate equivalent titles for LaTeX.

    Uses a simple literal matching approach.
    """

    latex_titles = []

    for title in titles:
        if title == 'Osc.':
            latex_titles.append('$m$')
        elif title == 'Amp.':
            latex_titles.append('$a_m$')
        elif title == 'Phase (rad)':
            latex_titles.append('$\\phi_m\\ (\\text{rad})$')
        elif title == 'Freq. (Hz)':
            latex_titles.append('$f_m\\ (\\text{Hz})$')
        elif title == 'Freq. (ppm)':
            latex_titles.append('$f_m\\ (\\text{ppm})$')
        elif title == 'Freq. 1 (Hz)':
            latex_titles.append('$f_{1,m}\\ (\\text{Hz})$')
        elif title == 'Freq. 1 (ppm)':
            latex_titles.append('$f_{1,m}\\ (\\text{ppm})$')
        elif title == 'Freq. 2 (Hz)':
            latex_titles.append('$f_{2,m}\\ (\\text{Hz})$')
        elif title == 'Freq. 2 (ppm)':
            latex_titles.append('$f_{2,m}\\ (\\text{ppm})$')
        elif title == 'Damp. (s⁻¹)':
            latex_titles.append('$\\eta_m\\ (\\text{s}^{-1})$')
        elif title == 'Damp. 1 (s⁻¹)':
            latex_titles.append('$\\eta_{1,m}\\ (\\text{s}^{-1})$')
        elif title == 'Damp. 2 (s⁻¹)':
            latex_titles.append('$\\eta_{2,m}\\ (\\text{s}^{-1})$')
        elif title == 'Integ.':
            latex_titles.append('$\\int$')
        elif title == 'Norm. Integ.':
            latex_titles.append(
                '$\\nicefrac{\\int}{\\left\\lVert\\int\\right\\rVert}$'
            )

    return latex_titles


def _construct_paramtable(parameters, errors, integrals, sfo, sig_figs,
                          sci_lims, fmt):
    """
    Creates a nested list of values to input to parameter table, with
    desired formatting.

    Parameters
    -----------
    parameters : numpy.ndarray
        Parameter array.

    errors : numpy.ndarray or None
        Parameter errors.

    integrals : list
        Oscillator peak integrals.

    sfo : [float], or [float, float] or None
        Transmitter offset frequency (MHz) in each dimension.

    sig_figs : int or None
        Desired nuber of significant figures.

    sci_lims : (int, int) or None
        Bounds defining thresholds for using scientific notation.

    Returns
    --------
    titles : list
        Titles of parameter table

    table : list
        Values of parameter table.
    """

    M = parameters.shape[0]
    dim = int(parameters.shape[1] / 2) - 1

    # --- Create titles for parameter table ------------------------------
    titles = ['Osc.', 'Amp.', 'Phase (rad)']
    if dim == 1:
        titles += ['Freq. (Hz)']
        if sfo is not None:
            titles += ['Freq. (ppm)']
        titles += ['Damp. (s⁻¹)']

    else:
        titles += ['Freq. 1 (Hz)', 'Freq. 2 (Hz)']
        if sfo is not None:
            titles += ['Freq. 1 (ppm)', 'Freq. 2 (ppm)']
        titles += ['Damp. 1 (s⁻¹)', 'Damp. 2 (s⁻¹)']

    table = []
    # Norm of integrals
    if integrals is not None:
        titles += ['Integ.', 'Norm. Integ.']
        norm_integrals = list(np.array(integrals) / slinalg.norm(integrals))

    if fmt == 'pdf':
        titles = _map_to_latex_titles(titles)

    # --- Generate string representations of parameter values ------------

    # Shorthand function
    def _mystr(value):
        return _strval(value, sig_figs, sci_lims, fmt)

    # Construct rows of table (each row corresponds to one oscillator)
    for m in range(M):
        row = []
        # Oscillator number
        row.append(f'{m+1}')
        # Amplitude
        row.append(_mystr(parameters[m, 0]))
        # Phase
        row.append(_mystr(parameters[m, 1]))
        # Frequencies
        for i in range(2, 2 + dim):
            # Hz
            row.append(_mystr(parameters[m, i]))
            # ppm (if sfo provided)
            if sfo is not None:
                row.append(_mystr(parameters[m, i] / sfo[i - 2]))
        # Damping
        for i in range(2 + dim, 2 + 2 * dim):
            row.append(_mystr(parameters[m, i]))
        if integrals is not None:
            # Integrals
            row.append(_mystr(integrals[m]))
            # Normalised integrals
            row.append(_mystr(norm_integrals[m]))
        # Add row to table
        table.append(row)

        if errors is not None:
            # first element is blank entry for oscillator column
            err_row = ['']
            # Amplitude
            err_row.append("±" + _mystr(errors[m, 0]))
            # Phase
            err_row.append("±" + _mystr(errors[m, 1]))
            # Frequencies
            for i in range(2, 2 + dim):
                # Hz
                err_row.append("±" + _mystr(errors[m, i]))
                # ppm (if sfo provided)
                if sfo is not None:
                    err_row.append("±" + _mystr(errors[m, i] / sfo[i - 2]))
            # Damping
            for i in range(2 + dim, 2 + 2 * dim):
                err_row.append("±" + _mystr(errors[m, i]))

            # TODO: Integral errors
            if integrals is not None:
                err_row = err_row + 2 * ['-']
            table.append(err_row)

    return titles, table


def _timestamp():
    """Constructs a string with time/date information.

    Returns
    -------
    timestamp : str
        Of the form:

        .. code::

           hh:mm:ss
           dd-mm-yy
    """
    now = datetime.datetime.now()
    d = now.strftime('%d')  # Day
    m = now.strftime('%m')  # Month
    y = now.strftime('%Y')  # Year
    t = now.strftime('%X')  # Time (hh:mm:ss)
    return f'{t}\n{d}-{m}-{y}'


def _strval(value, sig_figs, sci_lims, fmt):
    """Convert float to formatted string.

    Parameters
    ----------
    value - float
        Value to convert.

    sig_figs - int or None
        Number of significant figures.

    sci_lims - (int, int) or None
        Bounds defining thresholds for using scientific notation.

    Returns
    -------
    strval - str
        Formatted value.
    """

    # Set to correct number of significant figures
    if isinstance(sig_figs, int):
        value = significant_figures(value, sig_figs)
    # Determine whether to express the value in scientific notation or not
    return _scientific_notation(value, sci_lims, fmt)


def _scientific_notation(value, sci_lims, fmt):
    """Converts value to scientific notation

    Parameters
    ----------
    value : float
        Value to process

    sci_lims : (int, int)
        See description in :py:func:`write_result`

    fmt : 'txt', 'pdf', or 'csv'
        File format.
    """
    # If user speicifed to never user scientific notation, or the value
    # does not have a sufficiently high exponent, or the file type is a csv,
    # simply return the value unedited, as a string.
    if (sci_lims is None or
            (abs(value) < 10 ** sci_lims[1] and
             abs(value) >= 10 ** (sci_lims[0])) or
            fmt == 'csv'):
        return str(value)

    # Convert to scientific notation
    # Regex is used to remove any trailing zeros, ie:
    # 12345000e8 -> 12345e8
    value = re.sub(
        r'0+e', 'e', f'{value:e}'.replace('+0', '').replace('-0', '-'),
    )

    if fmt == 'txt':
        return value
    # For LaTeX, convert to \num{value}
    # This is a siunitx macro which takes a value of the form
    # 1234e5 and converts to 1234x10⁵
    elif fmt == 'pdf':
        return f'\\num{{{value}}}'


def _txt_tabular(columns, titles=None, separator=''):
    """Tabularises a list of lists, with the option of including titles.
    Used in textfile outputs.

    Parameters
    ----------
    columns : list
        A list of lists, with each sublist representing the columns of the
        table. Each list must be of the same length.

    titles : None or list, default: None
        Titles for the table. If desired, `titles` should be of the same
        length as all of the sublists in `columns`.

    separator : str, default: ''
        Column separator. By default, an empty string is used. (See first
        example above).

    Returns
    -------
    table : str
        A string with the contents of `titles` (opt.) and `columns`
        tabularised.

    Examples
    --------
    A simple example with `titles` specified and the default `separator`:

    .. code:: python3

       >>> from nmrespy.write import _txt_tabular
       >>> columns = [['A1', 'B1'], ['A2', 'B2'], ['A3', 'B3']]
       >>> titles = ['title 1', 'title 2', 'title 3']
       >>> print(_txt_tabular(columns, titles=titles))
       title 1  title 2  title 3
       ────────┼────────┼────────
       A1       A2       A3
       B1       B2       B3

    You may want to set `separator` to something like ``'│'``
    in order for a nicer layout when you have titles:

    .. code:: python3

       >>> # Same as before...
       >>> print(_txt_tabular(columns, titles=titles, separator='│'))
       title 1 │title 2 │title 3
       ────────┼────────┼────────
       A1      │A2      │A3
       B1      │B2      │B3
    """
    # If titles are given, append to the top of each column
    if titles:
        for i, (title, column) in enumerate(zip(titles, columns)):
            columns[i] = [title] + column
    # --- Determine width of each column ---------------------------------
    pads = []
    for column in columns:
        # For each column find the longest string, and set its length
        # as the width
        pads.append(max(len(str(element)) for element in column))

    # --- Construct table ------------------------------------------------
    table = ''
    # Iterate of rows (transpose array)
    for i, row in enumerate(zip(*columns)):
        # Iterate over each adjacent pair of elements in row
        for j, (pad, e1, e2) in enumerate(zip(pads, row, row[1:])):
            # Determine amount of padding between pair
            p = pad - len(e1) + 1
            # First element -> don't want any padding before it.
            # All other elements are padded from the left.
            if j == 0:
                # Case for first pairing in row
                table += f"{e1}{p*' '}{separator}{e2}"
            else:
                table += f"{p*' '}{separator}{e2}"
        # Add newline character at end of row
        table = f'{table}\n'

        # At end of the first row, check if titles were given
        # If so, add a horizontal line underneath to separate the titles
        # from the other contents
        if titles and i == 0:
            for k, pad in enumerate(pads):
                p = pad + 1
                # Add a bar that looks like this: '────────┼'
                table += f"{p*'─'}┼"
            # Once the bar has been completed, remove the trailing '┼'
            # and add a newline
            table = f'{table[:-1]}\n'

    return table


def _latex_tabular(rows):
    """Creates a string of text that denotes a tabular entity in LaTeX

    Parameters
    ----------
    rows : list
        Nested list, with each sublist containing elements of a single row
        of the table.

    Returns
    -------
    table : str
        LaTeX-formated table

    Example
    -------
    .. code:: python3

       >>> from nmrespy.write import _latex_tabular
       >>> rows = [['A1', 'A2', 'A3'], ['B1', 'B2', 'B3']]
       >>> print(_latex_tabular(rows))
       A1 & A2 & A3 \\\\
       B1 & B2 & B3 \\\\
    """
    table = ''
    for row in rows:
        table += ' & '.join([e for e in row]) + ' \\\\\n'
    return table
