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
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import numpy.linalg as nlinalg

from nmrespy import (RED, ORA, GRE, END, ExpInfo, USE_COLORAMA, NMRESPYPATH,
                     NMRESPYLOGOPATH, MFLOGOPATH, DOCSLINK, GITHUBLINK,
                     GITHUBLOGOPATH, MFGROUPLINK, BOOKICONPATH, MAILTOLINK,
                     EMAILICONPATH)
from nmrespy import _errors, _misc, sig

if USE_COLORAMA:
    import colorama
    colorama.init()


# TODO
def _raise_error(exception, msg, kill_on_error):
    if kill_on_error:
        raise exception(msg)
    else:
        return None


def _append_suffix(path: Path, fmt: str) -> Path:
    if not path.suffix == f'.{fmt}':
        path = path.with_suffix(f'.{fmt}')
    return path


def _configure_save_path(
    path: str, fmt: str, force_overwrite: bool
) -> Union[Path, None]:
    """Determine the path to save the file to.

    Parameters
    ----------
    path
        Path given by the user, to be processed.

    fmt
        File format specified. Will be one of ``'txt'``, ``'pdf'``, ``'csv'``.

    force_overwrite
        Whether or not to ask the user if they are happy overwriting the
        file if it already exists.

    Returns
    -------
    path: Union[pathlib.Path, None]
        The path to the result file, or ``None``, if the path given already
        exists, and the user has specified not to overwrite.

    Raises
    ------
    ValueError
        If the directory implied by ``path`` does not exist.
    """
    path = _append_suffix(Path(path).resolve(), fmt)
    if path.is_file():
        response = _ask_overwrite(path, force_overwrite)
        if not response:
            print(f'{RED}Overwrite of file {path} denied. File will not be '
                  f'overwritten.{END}')
            return None
        return path

    if not path.parent.is_dir():
        msg = (f'{RED}The directory specified by `path` does not '
               f'exist:\n{path.parent}{END}')
        raise _errors.ValueError(msg)

    return path


def _ask_overwrite(path: Path, force: bool) -> bool:
    """Determine whether the user is happy to overwrite an existing file."""
    if force:
        return True
    prompt = (
        f'{ORA}The file {str(path)} already exists. Overwrite?\n'
        f'Enter [y] or [n]:{END}'
    )
    return _misc.get_yes_no(prompt)


def write_result(
    expinfo: ExpInfo, params: np.ndarray, *,
    errors: Union[np.ndarray, None] = None, path: str = './nmrespy_result',
    fmt: str = 'txt', description: Union[str, None] = None,
    info_headings: Union[Iterable[str], None] = None,
    info: Union[Iterable[str], None] = None, sig_figs: Union[int, None] = 5,
    sci_lims: Union[Tuple[int, int], None] = (-2, 3),
    force_overwrite: bool = False, pdflatex_exe: Union[str, None] = None,
    fprint: bool = True, kill_on_error: bool = False
) -> None:
    """Writes an estimation result to a .txt, .pdf or .csv file.

    Parameters
    ----------
    expinfo
        Information of experiment.

    params
        The estimated parameter array.

    errors
        The errors associated with the params. If not ``None``, the shape
        of ``errors`` must match that of ``params``.

    path
        The path to save the file to. Note that if the appropriate file format
        suffix is not provided to the path, it will be appended.

    fmt
        File format. Should be one of ``'txt'``, ``'pdf'``, ``'csv'``. See
        notes for details on system requirements for PDF generation.

    description
        A descriptive statement.

    info_headings
        Headings for experiment information. Could include items like
        `'Sweep width (Hz)'`, `'Transmitter offset (Hz)'`, etc.

    info
        Information that corresponds to each heading in ``info_headings``.

    sig_figs
        The number of significant figures to give to parameter values. If
        ``None``, the full value will be used.

    sci_lims
        Given a value ``(x, y)``, for ``x < 0`` and ``y > 0``, any parameter
        ``p`` value which satisfies ``p < 10 ** x`` or ``p >= 10 ** y`` will
        be expressed in scientific notation, rather than explicit notation.
        If ``None``, all values will be expressed explicitely.

    force_overwrite
        Defines behaviour if the path specified already exists:

        * If set to ``False``, the user will be asked if they are happy
          overwriting the current file.
        * If set to ``True``, the current file will be overwritten without
          prompt.

    pdflatex_exe
        The path to the system's ``pdflatex`` executable.

        .. note::

           You are unlikely to need to set this manually. It is primarily
           present to specify the path to ``pdflatex.exe`` on Windows when
           the NMR-EsPy GUI has been loaded from TopSpin.

    fprint
        Specifies whether or not to print information to the terminal.

    kill_on_error
        Specifies whether to raise an error if a fault is determined.

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
       --snip--
       Compiled with zlib 1.2.11; using zlib 1.2.11
       Compiled with xpdf version 4.01

    If you see something similar to the above output, you should be all good.

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

    .. code:: bash

        $ kpsewhich booktabs.sty
        /usr/share/texlive/texmf-dist/tex/latex/booktabs/booktabs.sty

    If a pathname appears, the package is installed to that path.
    """
    if not isinstance(expinfo, ExpInfo):
        raise TypeError(f'{RED}Check `expinfo` is valid.{END}')
        dim = expinfo.unpack('dim')

    try:
        if dim != int(params.shape[1] / 2) - 1:
            raise ValueError(
                f'{RED}The dimension of `expinfo` does not agree with the '
                'parameter array. `expinfo.dim == (params.shape[1] / 2) '
                f'- 1` is not satified.{END}'
            )
    except AttributeError:
        # params.shape raised an attribute error
        raise TypeError(
            f'{RED}`params` should be a numpy array{END}'
        )
    if dim >= 3:
        raise errors.MoreThanTwoDimError()

    checker = _misc.ArgumentChecker(dim=dim)
    checker.stage(
        (params, 'params', 'parameter'),
        (errors, 'errors', 'parameter', True),
        (path, 'path', 'str'),
        (fmt, 'fmt', 'file_fmt'),
        (force_overwrite, 'force_overwrite', 'bool'),
        (fprint, 'fprint', 'bool'),
        (description, 'description', 'str', True),
        (info_headings, 'info_headings', 'list', True),
        (info, 'info', 'list', True),
        (sig_figs, 'sig_figs', 'positive_int', True),
        (sci_lims, 'sci_lims', 'pos_neg_tuple', True),
    )
    checker.check()

    if len(list(filter(lambda x: x is None, [info_headings, info]))) == 1:
        raise ValueError(
            f'{RED}`info` and `info_headings` should either both be lists'
            f' of the same length, or both be None.{END}'
        )
    # info and info_headings should be the same length if lists
    if isinstance(info, list) and len(info) != len(info_headings):
        raise ValueError(
            f'{RED}`info` and `info_headings` should be the same'
            f' length{END}'
        )
    # params and errors should be the same shape, if errors is not None
    if isinstance(errors, np.ndarray) and errors.shape != params.shape:
        raise ValueError(
            f'{RED}`params` and `errors` should be the same'
            f' shape{END}'
        )

    path = _configure_save_path(path, fmt, force_overwrite)
    if not path:
        if kill_on_error:
            print(f'{ORA}Skipping call to `write_result`...{END}')
            return None
        else:
            print(f'{RED}Exiting program...{END}')

    # Short-hand function for value formatting
    def fmtval(value):
        return _format_value(value, sig_figs, sci_lims, fmt)

    param_table = _construct_paramtable(params, errors, fmtval)

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
    path, description, info_headings, info, table, fprint,
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
    msg += _txt_tabular(table, titles=True)

    # --- Write footer ---------------------------------------------------
    msg += ("\nEstimation performed using NMR-EsPy\n"
            "Author: Simon Hulse ~ simon.hulse@chem.ox.ac.uk\n"
            "If used in any publications, please cite:\n"
            "<no papers yet...>\n"
            "For more information, visit the GitHub repo:\n")

    msg += GITHUBLINK
    # Save message to textfile
    with open(path, 'w', encoding='utf-8') as file:
        file.write(msg)

    if fprint:
        print(f'{GRE}Saved result to {path}{END}')


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
        '<TIMESTAMP>'
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

    # TODO
    # Put all LatEx compilation stuff in separate function
    # compile_status = _compile_latex_pdf()

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
        raise _errors.LaTeXFailedError(tex_final_path)

    except FileNotFoundError:
        # Most probably, pdflatex does not exist
        raise _errors.LaTeXFailedError(tex_final_path)

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
        print(f'{GRE}Result successfuly output to:\n'
              f'{pdf_final_path}\n'
              'If you wish to customise the document, the TeX file can'
              ' be found at:\n'
              f'{tex_final_path}{END}')


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
        print(f'{GRE}Saved result to {path}{END}')


def _map_to_latex_titles(titles: List[str]) -> List[str]:
    """Map title names to equivalents for LaTeX.

    See also :py:func:`_make_titles`.
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
        else:
            raise ValueError(f'{RED}BUG!!! Unrecognised argument in'
                             f'_map_to_latex_titles{END}')

    return latex_titles


def _make_titles(expinfo: ExpInfo, fmt: str) -> List[str]:
    """Create titles for the parameter table."""
    dim, sfo = expinfo.unpack('dim', 'sfo')
    inc_ppm = sfo is not None
    titles = ['Osc.', 'Amp.', 'Phase (rad)']
    if dim == 1:
        titles += ['Freq. (Hz)']
        if inc_ppm:
            titles += ['Freq. (ppm)']
        titles += ['Damp. (s⁻¹)']

    else:
        titles += ['Freq. 1 (Hz)', 'Freq. 2 (Hz)']
        if inc_ppm:
            titles += ['Freq. 1 (ppm)', 'Freq. 2 (ppm)']
        titles += ['Damp. 1 (s⁻¹)', 'Damp. 2 (s⁻¹)']

    titles += ['Integ.', 'Norm. Integ.']
    return titles if fmt != 'pdf' else _map_to_latex_titles(titles)


def _construct_paramtable(
    params: np.ndarray, errors: Union[np.ndarray, None],
    expinfo: ExpInfo, fmt: str, fmtval: Callable[[float], str]
) -> List[List[str]]:
    """Make a nested list of values for parameter table.

    Parameters
    -----------
    params
        Parameter array.

    errors
        Parameter errors.

    expinfo
        Experiment informtion.

    fmt
        File format. Will be one of ``'txt'``, ``'pdf'``, ``'csv'``.

    fmtval
        Callable which converts a ``float`` to a formatted ``str``. This
        basically calls :py:func:`_format_string`, with a pre-determined
        set of arguments.

    Returns
    --------
    table: List[List[str]]
        Parameter table. Note that the first row of the table (``table[0]``)
        contains the titles form each column.
    """
    titles = _make_titles(expinfo, fmt)
    paramtable = _make_parameter_table(params, expinfo)
    paramtable = _format_parameter_table(paramtable, fmtval)
    if isinstance(errors, np.ndarray):
        errortable = _make_error_table(errors, expinfo)
        errortable = _format_error_table(errortable, fmtval)
        # Interleave parameter and error rows
        table = [val for pair in zip(paramtable, errortable) for val in pair]

    else:
        table = paramtable
    return [titles] + table


def _make_parameter_table(params: np.ndarray, expinfo: ExpInfo) -> np.ndarray:
    dim, sfo = expinfo.unpack('dim', 'sfo')
    inc_ppm = sfo is not None
    m = params.shape[0]
    integrals = _compute_integrals(expinfo, params)
    integral_norm = nlinalg.norm(integrals)
    if inc_ppm:
        table = np.zeros((m, 5 + 3 * dim))
    else:
        table = np.zeros((m, 5 + 2 * dim))

    table[:, 0] = np.arange(1, m + 1)          # Oscillator labels
    table[:, 1:3 + dim] = params[:, :2 + dim]  # Amplitude, phase, freq (Hz)

    # Freq (ppm)
    if inc_ppm:
        table[:, 3 + dim: 3 + 2 * dim] = params[:, 2: 2 + dim] / np.array(sfo)

    table[:, -2 - dim: -2] = params[:, 2 + dim:]  # Damping
    table[:, -2] = integrals                      # Integrals
    table[:, -1] = integrals / integral_norm      # Normalised integrals

    return table


def _make_error_table(errors: np.ndarray, expinfo: ExpInfo) -> np.ndarray:
    dim, sfo = expinfo.unpack('dim', 'sfo')
    inc_ppm = sfo is not None
    m = errors.shape[0]
    if inc_ppm:
        table = np.zeros((m, 5 + 3 * dim))
    else:
        table = np.zeros((m, 5 + 2 * dim))

    table[:, 0] = np.full((m,), np.nan)         # Oscillator labels (blank)
    table[:, 1:3 + dim] = errors[:, :2 + dim]  # Amplitude, phase, freq (Hz)

    # Freq (ppm)
    if inc_ppm:
        table[:, 3 + dim: 3 + 2 * dim] = errors[:, 2: 2 + dim] / np.array(sfo)

    table[:, -2 - dim: -2] = errors[:, 2 + dim:]  # Damping
    table[:, -2:] = np.full((m, 2), np.nan)       # Integrals (blank)

    return table


def _format_parameter_table(
    paramtable: np.ndarray, fmtval: Callable[[float], str]
) -> List[str]:
    return [[fmtval(x) for x in row] for row in paramtable]


def _format_error_table(
    errortable: np.ndarray, fmtval: Callable[[float], str]
) -> List[str]:
    return [[f'±{fmtval(x)}' if not np.isnan(x) else '-' for x in row]
            for row in errortable]


def _compute_integrals(expinfo: ExpInfo, params: np.ndarray) -> np.ndarray:
    return np.array([sig.oscillator_integral(osc, expinfo) for osc in params])


def _timestamp() -> str:
    """Constructs a string with time/date information.

    Returns
    -------
    timestamp: str
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


def _format_value(
    value: float, sig_figs: Union[int, None],
    sci_lims: Union[Tuple[int, int], None], fmt: str
) -> str:
    """Convert float to formatted string.

    Parameters
    ----------
    value
        Value to convert.

    sig_figs
        Number of significant figures.

    sci_lims
        Bounds defining thresholds for using scientific notation.

    fmt
        Desired file format.

    Returns
    -------
    strval: str
        Formatted value.
    """
    if isinstance(sig_figs, int):
        value = _significant_figures(value, sig_figs)

    if (sci_lims is None) or (value == 0) or (fmt == 'csv'):
        return str(value)

    # Determine the value of the exponent to check whether the value should
    # be expressed in scientific or normal notation.
    exp_search = re.search(r'e(\+|-)(\d+)', f'{value:e}')
    exp_sign = exp_search.group(1)
    exp_mag = int(exp_search.group(2))

    if (exp_sign == '+' and exp_mag < sci_lims[1] or
            exp_sign == '-' and exp_mag < -sci_lims[0]):
        return str(value)

    value = _scientific_notation(value)
    return value if fmt == 'txt' else f"\\num{{{value}}}"


def _significant_figures(value: float, s: int) -> Union[int, float]:
    """Round a value to a certain number of significant figures.

    Parameters
    ----------
    value
        Value to round.

    s
        Significant figures.

    Returns
    -------
    rounded_value: Union[int, float]
        Value rounded to ``s`` significant figures. If the resulting value
        is an integer, it will be converted from ``float`` to ``int``.
    """
    if value == 0:
        return 0

    value = round(value, s - int(np.floor(np.log10(abs(value)))) - 1)
    # If value of form 123456.0, convert to 123456
    if float(value).is_integer():
        value = int(value)

    return value


def _scientific_notation(value: float) -> str:
    """Convert ``value`` to a string with scientific notation.

    Parameters
    ----------
    value
        Value to process.

    Returns
    -------
    sci_value
        String denoting ``value`` in scientific notation.
    """
    return re.sub(r'\.?0+e(\+|-)0?', r'e\1', f'{value:e}')


def _txt_tabular(rows: List[List[str]], titles: bool = False) -> str:
    """Tabularise a list of lists.

    Parameters
    ----------
    rows
        A list of lists, with each sublist representing the rows of the
        table. Each sublist must be of the same length.

    titles
        If ``True``, the first entry in ``rows`` will be treated as the
        titles of the table, and be separated from the other rows by a bar.

    Returns
    -------
    table
        Tabularised content of ``rows``.
    """
    pads = []
    for column in zip(*rows):
        # For each column find the longest string, and set its length
        # as the width
        pads.append(max(len(str(element)) for element in column))

    separator = '│' if titles else ' '

    table = ''
    for i, row in enumerate(rows):
        # Iterate over each adjacent pair of elements in row
        for j, (pad, e1, e2) in enumerate(zip(pads, row, row[1:])):
            # Amount of padding between pair
            p = pad - len(e1) + 1
            # First element -> don't want any padding before it.
            # All other elements are padded from the left.
            if j == 0:
                table += f"{e1}{p * ' '}{separator}{e2}"
            else:
                table += f"{p * ' '}{separator}{e2}"
        table += '\n'

        # Add a horizontal line underneath the first row to separate the
        # titles from the other contents
        if titles and i == 0:
            for k, pad in enumerate(pads):
                p = pad + 1
                # Add a bar that looks like this: '────────┼'
                table += f"{p * '─'}┼"
            # Remove the trailing '┼' and add a newline
            table = table[:-1] + '\n'

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
