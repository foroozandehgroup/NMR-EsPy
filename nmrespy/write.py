# write.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Writing estimation results to .txt, .pdf and .csv files"""

import csv
import datetime
import os
from pathlib import Path
import re
from shutil import copyfile
import subprocess

import numpy as np
import scipy.linalg as slinalg

from nmrespy import *
from nmrespy._misc import ArgumentChecker, PathManager
from ._errors import *

def write_result(
    parameters, path='./nmrespy_result', sfo=None, integrals=None,
    description=None, info_headings=None, info=None, sig_figs=5,
    sci_lims=(-2,3), fmt='txt', force_overwrite=False,
):
    """Writes an estimation result to a .txt, .pdf or .csv file.

    Parameters
    ----------
    parameters : numpy.ndarray
        The estimated parameter array.

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

    .. code:: bash

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
    * `xcolor <https://ctan.org/pkg/xcolor?lang=en>`_

    Most of these are pretty ubiquitous and are likely to be installed
    even with lightweight LaTeX installations. If you wish to check the
    packages are available, use ``kpsewhich``:

    .. code:: bash

        $ kpsewhich booktabs.sty
        /usr/share/texlive/texmf-dist/tex/latex/booktabs/booktabs.sty

    If a pathname appears, the package is installed to that path.
    """

    # --- Check validity of arguments ------------------------------------
    try:
        dim = int(parameters.shape[1] / 2) - 1
    except:
        raise TypeError(f'{cols.R}parameters should be a numoy array{cols.END}')

    components = [
        (parameters, 'parameters', 'parameter'),
        (path, 'path', 'str'),
        (fmt, 'fmt', 'file_fmt'),
        (force_overwrite, 'force_overwrite', 'bool'),
    ]

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

    if isinstance(integrals, list) and len(integrals) != parameters.shape[0]:
        raise ValueError(
            f'{cols.R}integrals should have the same number of elements as'
            f' parameters.shape[0]{cols.END}'
        )
    if isinstance(info, list) and len(info) != len(info_headings):
        raise ValueError(
            f'{cols.R}info and info_headings should be the same'
            f' length{cols.END}'
        )

    path = Path(path).resolve()
    path_result = PathManager(path.name, path.parent).check_file(force_overwrite)
    # Valid path, we are good to proceed
    if path_result == 0:
        pass
    # Overwrite denied by the user. Exit the program
    elif path_result == 1:
        exit()
    # path_result = 2: Directory specified doesn't exist
    else:
        raise ValueError(
            f'{cols.R}The directory implied by path does not exist{cols.END}'
        )

    # Append extension to file path
    path = path.parent / (path.name + f'.{fmt}')

    # Checking complete...

    # --- Construct nested list of components for parameter table --------
    param_titles, param_table = \
        _construct_paramtable(
            parameters, integrals, sfo, sig_figs, sci_lims, fmt,
        )

    # --- Write to the specified file type -------------------------------
    if fmt == 'txt':
        _write_txt(
            path, description, info_headings, info, param_titles, param_table,
        )
    elif fmt == 'pdf':
        _write_pdf(
            path, description, info_headings, info, param_titles, param_table,
        )
    # TODO
    elif fmt == 'csv':
        _write_csv(
            path, description, info_headings, info, param_titles, param_table,
        )

def _write_txt(
    path, description, info_headings, info, param_titles, param_table
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
        Headings for experiment information. Could include items like
        `'Sweep width (Hz)'`, `Transmitter offset (Hz)`, etc.

    info : list or None, default: None
        Information that corresponds to each heading in `info_headings`.

    param_titles : list
        Titles for parameter array table.

    param_table : list
        Array of contents to append to the result table.
    """

    # --- Write header ---------------------------------------------------
    # Time and date
    msg = f'{_timestamp()}\n'
    # User-provided description
    if description is not None:
        msg += f'\nDescription:\n{description}\n\nExperiment Information:\n'
    else:
        msg += '\nExperiment Information:\n'

    # Table of experiment information
    if info is not None:
        msg += _txt_tabular([info_headings, info]) + '\n'
    # Table of oscillator parameters
    msg += _txt_tabular(
        list(map(list, zip(*param_table))), titles=param_titles,
        separator=' │',
    )

    # Blurb at bottom of file
    msg += \
    """\nEstimation performed using NMR-EsPy
Author: Simon Hulse ~ simon.hulse@chem.ox.ac.uk
If used in any publications, please cite:
<no papers yet...>
For more information, visit the GitHub repo:
"""
    msg += f'\n{GITHUBPATH}'
    # save message to textfile
    with open(path, 'w') as file:
        file.write(msg)

    print(f'{cols.G}Saved result to {path}{cols.END}')


def _write_pdf(
    path, description, info_headings, info, param_titles, param_table,
):
    """
    Writes result of NMR-EsPy to a pdf.
    """

    with open(Path(NMRESPYPATH) / 'config/latex_template.txt', 'r') as fh:
        txt = fh.read()

    txt = txt.replace('<MFLOGOPATH>', MFLOGOPATH)
    txt = txt.replace('<NMRESPYLOGOPATH>', NMRESPYLOGOPATH)
    txt = txt.replace('<TIMESTAMP>', _timestamp().replace('\n', '\\\\'))

    if description is None:
        txt = txt.replace(
            '% user provided description\n\\subsection*{Description}\n'
            '<DESCRIPTION>',
            '',
        )

    else:
        txt = txt.replace('<DESCRIPTION>', description.replace('_', '\\_'))

    print(repr(txt))
    if info is None:
        txt = txt.replace(
            '\n% experiment parameters\n\\subsection*{Experiment Information}\n'
            '\\hspace{-6pt}\n\\begin{tabular}{ll}\n<INFOTABLE>\n'
            '\\end{tabular}\n',
            '',
        )

    else:
        rows = list(list(row) for row in zip(info_headings, info))
        info_table = _latex_tabular(rows)
        txt = txt.replace('<INFOTABLE>', info_table)

    txt = txt.replace('<COLUMNS>', len(param_titles) * 'c')
    txt = txt.replace('<PARAMTITLES>', _latex_tabular([param_titles]))
    txt = txt.replace('<PARAMTABLE>', _latex_tabular(param_table))

    # TODO support for including result figure
    txt = txt.replace(
        '% figure of result\n\\begin{center}\n'
        '\\includegraphics{<FIGURE_PATH>}\n\\end{center}\n',
        '',
    )

    txt = txt.replace('<GITHUBPATH>', GITHUBPATH)
    txt = txt.replace('<GITHUBLOGOPATH>', GITHUBLOGOPATH)
    txt = txt.replace('<MAILTOPATH>', MAILTOPATH)
    txt = txt.replace('<EMAILICONPATH>', EMAILICONPATH)

    # Convert from name.pdf -> name.tex
    tex_cwd_path = Path().cwd() / path.with_suffix('.tex').name
    pdf_cwd_path = Path().cwd() / path.name
    tex_final_path = path.with_suffix('.tex')
    pdf_final_path = path

    with open(tex_cwd_path, 'w') as fh:
        fh.write(txt)

    try:
        # -halt-on-error flag is vital. If any error arises in running
        # pdflatex, the program would get stuck
        run_latex = subprocess.run(
            ['pdflatex', '-halt-on-error', tex_cwd_path],
            stdout=subprocess.DEVNULL,
            check=True,
        )

        # rename pdf and tex files
        os.rename(tex_cwd_path, tex_final_path)
        os.rename(pdf_cwd_path, pdf_final_path)

    except subprocess.CalledProcessError:
        # pdflatex came across an error (or pdflatex doesn't exist)
        os.rename(tex_cwd_path, tex_final_path)
        raise LaTeXFailedError(tex_final_path)

    # Remove other LaTeX files
    os.remove(tex_final_path.with_suffix('.out'))
    os.remove(tex_final_path.with_suffix('.aux'))
    os.remove(tex_final_path.with_suffix('.log'))

    # # TODO: remove figure file if it exists
    # try:
    #     os.remove(figure_path)
    # except UnboundLocalError:
    #     pass

    # print success message
    msg = print(
        f'{cols.G}Result successfuly output to:\n' \
        f'{pdf_final_path}\n'
        f'If you wish to customise the document, the TeX file can'
        f'be found at:\n'
        f'{tex_final_path}{cols.END}'
    )

def _write_csv(path, description, info_headings, info, param_titles, param_table):

    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow([_timestamp().replace('\n', ' ')])
        writer.writerow([])
        if description is not None:
            writer.writerow(['Description:', description])
            writer.writerow([])
        if info is not None:
            writer.writerow(['Experiment Info:'])
            for row in zip(info_headings, info):
                writer.writerow(row)
            writer.writerow([])
        writer.writerow(['Result:'])
        writer.writerow(param_titles)
        for row in param_table:
            writer.writerow(row)



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
            latex_titles.append('$\\nicefrac{\\int}{\\left\\lVert\\int\\right\\rVert}$')

    return latex_titles


def _construct_paramtable(parameters, integrals, sfo, sig_figs, sci_lims, fmt):
    """
    Creates a nested list of values to input to results file table.

    Parameters
    -----------
    parameters : numpy.ndarray
        Parameter array.

    integrals : list
        Oscillator peak integrals.

    sfo : [float], or [float, float] or None
        Transmitter offset frequency (MHz) in each dimension.

    sig_figs : int or None
        Desired nuber of significant figures.

    sci_lims : (int, int) or None
        Bounds defining threshold for using scientific notation.

    Returns
    --------
    table : list
        Values for result file table.
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
        for i in range(2, 2+dim):
            # Hz
            row.append(_mystr(parameters[m, i]))
            # ppm (if sfo provided)
            if sfo is not None:
                row.append(_mystr(parameters[m, i] / sfo[i-2]))
        # Damping
        for i in range(2+dim, 2+2*dim):
            row.append(_mystr(parameters[m, i]))
        if integrals is not None:
            # Integrals
            row.append(_mystr(integrals[m]))
            # Normalised integrals
            row.append(_mystr(norm_integrals[m]))
        # Add row to table
        table.append(row)

    return titles, table


def _timestamp():
    """Constructs a string with time/date information."""
    now = datetime.datetime.now()
    d = now.strftime('%d') # day
    m = now.strftime('%m') # month
    y = now.strftime('%Y') # year
    t = now.strftime('%X') # time (hh:mm:ss)
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
        Specifies range of values to be formatted normmaly, and which
        to be formatted using scientific notation.

    Returns
    -------
    strval - str
        Formatted value.
    """

    # Set to correct number of significant figures
    if isinstance(sig_figs, int):
        value = round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)
    # If value of form 123456.0, convert to 123456
    if value.is_integer():
        value = int(value)
    # Determine whether to express the value in scientific notation or not
    return _scientific_notation(value, sci_lims, fmt)

def _scientific_notation(value, sci_lims, fmt):
    """fmt should be 'txt' or 'pdf'
    """
    # If user speicifed to never user scientific notation, or the value
    # does not have a sufficiently high exponent, or the file type is a csv,
    # simply return the value unedited, as a string.
    if sci_lims is None or \
    abs(value) < 10 ** sci_lims[1] and abs(value) >= 10 ** (sci_lims[0]) or \
    fmt == 'csv':
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


def _txt_tabular(columns, titles=None, separator=' '):
    """Tabularises a list of lists, with the option of including titles.

    Parameters
    ----------
    columns : list
        A list of lists, representing the columns of the table. Each list
        must be of the same length.

    titles : None or list, default: None
        Titles for the table. If desired, the ``titles`` should be of the same
        length as all of the lists in ``columns``.

    separator : str, default: ' '
        Column separator

    Returns
    -------
    msg : str
        A string with the contents of ``columns`` tabularised.
    """

    if titles:
        for i,(title, column) in enumerate(zip(titles, columns)):
            columns[i] = [title] + column

    pads = []
    print(columns)
    for column in columns:
        print(column)
        pads.append(max(len(str(element)) for element in column))

    msg = ''
    for i, row in enumerate(zip(*columns)):
        for j, (pad, e1, e2) in enumerate(zip(pads, row, row[1:])):
            p = pad - len(e1)
            if j == 0:
                msg += f"{e1}{p*' '}{separator}{e2}"
            else:
                msg += f"{p*' '}{separator}{e2}"
        if titles and i == 0:
            for i, pad in enumerate(pads):
                if i == 0:
                    msg += f"\n{(pad+1)*'─'}┼"
                else:
                    msg += f"{(pad+1)*'─'}┼"
            msg = msg[:-1]
        msg += '\n'

    return msg

def _latex_tabular(rows):
    msg = ''
    for row in rows:
        msg += ' & '.join([e for e in row]) + ' \\\\\n'
    return msg
