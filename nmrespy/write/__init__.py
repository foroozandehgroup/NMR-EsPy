# write.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Writing estimation results to .txt, .pdf and .csv files"""

from collections import deque
from pathlib import Path
import re
from typing import Callable, List, Tuple, Union

import numpy as np
import numpy.linalg as nlinalg

from nmrespy import RED, ORA, END, ExpInfo, USE_COLORAMA, _misc, sig
from . import textfile, pdffile, csvfile
if USE_COLORAMA:
    import colorama
    colorama.init()


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
        raise ValueError(msg)

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
    expinfo: ExpInfo, params: np.ndarray,
    errors: Union[np.ndarray, None] = None, *, path: str = './nmrespy_result',
    fmt: str = 'txt', description: Union[str, None] = None,
    sig_figs: Union[int, None] = 5,
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
        (sig_figs, 'sig_figs', 'positive_int', True),
        (sci_lims, 'sci_lims', 'pos_neg_tuple', True),
    )
    checker.check()

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

    param_table = _construct_paramtable(params, errors, expinfo, fmt, fmtval)
    info_table = _construct_infotable(expinfo)

    if fmt == 'txt':
        textfile.write(path, param_table, info_table, description, fprint)
    elif fmt == 'pdf':
        pdffile.write(
            path, param_table, info_table, description, pdflatex_exe, fprint,
        )
    elif fmt == 'csv':
        csvfile.write(path, param_table, info_table, description, fprint)


def _construct_infotable(expinfo: ExpInfo) -> List[List[str]]:
    dim, sw, offset, sfo, nuclei = \
        expinfo.unpack('dim', 'sw', 'offset', 'sfo', 'nuclei')
    titles = ['Parameter'] + [f'F{i}' for i in range(1, dim + 1)]
    names = deque(['Sweep width (Hz)', 'Transmitter offset (Hz)'])
    values = deque([[str(x) for x in param] for param in (sw, offset)])
    if sfo:
        names.appendleft('Transmitter frequency (MHz)')
        names.insert(2, 'Sweep width (ppm)')
        names.append('Transmitter offset (ppm)')
        values.appendleft([str(x) for x in sfo])
        values.insert(2, [str(x / y) for x, y in zip(sw, sfo)])
        values.append([str(x / y) for x, y in zip(offset, sfo)])
    if nuclei:
        names.appendleft('Nucleus')
        values.appendleft([x for x in nuclei])
    infotable = [[name] + value for name, value in zip(names, values)]
    return [titles] + infotable


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
        elif title == 'Integral':
            latex_titles.append('$\\int$')
        elif title == 'Norm. Integral':
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

    titles += ['Integral', 'Norm. Integral']
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
