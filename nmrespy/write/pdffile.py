import datetime
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Dict, List, Union
from nmrespy import (GRE, END, NMRESPYPATH, MFLOGOPATH, NMRESPYLOGOPATH,
                     DOCSLINK, MFGROUPLINK, BOOKICONPATH, GITHUBLINK,
                     GITHUBLOGOPATH, MAILTOLINK, EMAILICONPATH, USE_COLORAMA,
                     _errors, plot)
if USE_COLORAMA:
    import colorama
    colorama.init()

TMPDIR = pathlib.Path(tempfile.gettempdir())

def write(
    path: pathlib.Path, param_table: List[List[str]],
    info_table: Union[List[List[str]], None], description: Union[str, None],
    pdflatex_exe: Union[str, None], fprint: bool,
    figure: Union[plot.NmrespyPlot, None]
):
    """Writes parameter estimate to a PDF using ``pdflatex``.

    Parameters
    -----------
    path : pathlib.Path
        File path

    param_table
        Table of estimation result parameters.

    info_table
        Table of experiment information.

    description
        A descriptive statement.

    pdflatex_exe
        Path to ``pdflatex`` executable.

    fprint
        Specifies whether or not to print output to terminal.

    figure
        If an instance of :py:class:`~nmrespy.plot.NmrespyPlot`, and ``fmt``
        is set to ``'pdf'``. The  figure will be included in the result file.

    **figure_kwargs
        Keyword arguments for the :py:func:`~nmrespy.plot.plot_result`
        function.
    """
    try:
        text = _read_template()
    except Exception as e:
        raise e

    text = _append_links_and_paths(text)
    text = _append_timestamp(text)
    text = _append_description(text, description)
    text = _append_info_table(text, info_table)
    text = _append_param_table(text, param_table)
    text = _append_figure(text, figure)

    texpaths = _get_texpaths(path)
    with open(texpaths['tmp']['tex'], 'w', encoding='utf-8') as fh:
        fh.write(text)

    _compile_tex(texpaths, pdflatex_exe)
    _cleanup(texpaths)

    if fprint:
        print(f'{GRE}Result successfuly output to:\n'
              f"{texpaths['final']['pdf']}\n"
              'If you wish to customise the document, the TeX file can'
              ' be found at:\n'
              f"{texpaths['final']['tex']}{END}")


def _read_template() -> str:
    """Extract LaTeX template text."""
    with open(NMRESPYPATH / 'config/latex_template.txt', 'r') as fh:
        text = fh.read()
    return text


def _append_links_and_paths(template: str) -> str:
    """Inputs web links and image paths into the text."""
    # Add image paths and weblinks to TeX document
    stuff = {
        '<MFLOGOPATH>': MFLOGOPATH,
        '<NMRESPYLOGOPATH>': NMRESPYLOGOPATH,
        '<DOCSLINK>': DOCSLINK,
        '<MFGROUPLINK>': MFGROUPLINK,
        '<BOOKICONPATH>': BOOKICONPATH,
        '<GITHUBLINK>': GITHUBLINK,
        '<GITHUBLOGOPATH>': GITHUBLOGOPATH,
        '<MAILTOLINK>': MAILTOLINK,
        '<EMAILICONPATH>': EMAILICONPATH,
    }

    for before, after in stuff.items():
        # On Windows, have to replace paths C:\a\b\c -> C:/a/b/c
        template = template.replace(before, str(after).replace('\\', '/'))

    return template


def _append_timestamp(text: str) -> str:
    return text.replace('<TIMESTAMP>', _timestamp())


def _append_description(text: str, description: Union[str, None]) -> str:
    """Append description to text."""
    if description is None:
        return text.replace(
            '% user provided description\n\\subsection*{Description}\n'
            '<DESCRIPTION>',
            '',
        )
    else:
        return text.replace('<DESCRIPTION>', description)


def _append_info_table(text: str, info_table: List[List[str]]) -> str:
    return text.replace('<INFOTABLE>', _latex_longtable(info_table))


def _append_param_table(text: str, param_table: List[List[str]]) -> str:
    param_table = [[e.replace('±', '$\\pm$') for e in row]
                   for row in param_table]
    return text.replace('<PARAMTABLE>', _latex_longtable(param_table))


def _latex_longtable(rows: List[List[str]]) -> str:
    """Creates a string of text for a LaTeX longtable.

    Parameters
    ----------
    rows
        The contents of the table. The first element is assumed to be a list
        of titles.

    Returns
    -------
    longtable: str
        LaTeX longtable
    """
    column_specifier = ' '.join(['c' for _ in range(len(rows[0]))])
    table = f"\\begin{{longtable}}[l]{{{column_specifier}}}\n\\toprule\n"
    for i, row in enumerate(rows):
        table += ' & '.join([e for e in row])
        if i == 0:
            table += '\n\\\\\\midrule\n'
        else:
            table += '\\\\\n'
    table += '\\bottomrule\n\\end{longtable}'
    return table


def _append_figure(text: str, figure: Union[plot.NmrespyPlot, None]):
    if isinstance(figure, plot.NmrespyPlot):
        path = TMPDIR / 'figure.pdf'
        figure.fig.savefig(path, dpi=600, format='pdf')
        text.replace(
            '<RESULTFIGURE>',
            '% figure of result\n\\begin{center}\n'
            f'\\includegraphics{{{path}}}\n\\end{{center}}'
        )
    else:
        text.replace('\n<RESULTFIGURE>', '')

    return text


def _timestamp() -> str:
    """Construct a string with time and date information.

    Returns
    -------
    timestamp
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
    return f'{t}\\\\{d}-{m}-{y}'


def _get_texpaths(path: pathlib.Path) -> Dict[str, Dict[str, pathlib.Path]]:
    return {
        'tmp': {
            suffix: TMPDIR / path.with_suffix(f'.{suffix}').name
            for suffix in ('tex', 'pdf', 'out', 'aux', 'log')
        },
        'final': {
            'tex': path.with_suffix('.tex'),
            'pdf': path
        }
    }


def _compile_tex(
    texpaths: Dict[str, Dict[str, pathlib.Path]], pdflatex_exe: str
) -> None:
    if pdflatex_exe is None:
        pdflatex_exe = "pdflatex"

    to_compile = texpaths['tmp']['tex']
    try:
        subprocess.run(
            [
                pdflatex_exe,
                '-halt-on-error',
                f'-output-directory={to_compile.parent}',
                to_compile
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )

    except Exception:
        shutil.move(texpaths['tmp']['tex'], texpaths['final']['tex'])
        raise _errors.LaTeXFailedError(texpaths['final']['tex'])


def _cleanup(texpaths: Dict[str, Dict[str, pathlib.Path]]) -> None:
    for f in texpaths['tmp'].values():
        os.remove(f)
        figurepath = TMPDIR / 'figure.pdf'
    if figurepath.is_file():
        os.remove(figurepath)
