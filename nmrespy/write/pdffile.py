# pdffile.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 12:20:16 GMT

import datetime
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Union

from nmrespy._colors import GRE, END, USE_COLORAMA
import nmrespy._paths_and_links as pl

if USE_COLORAMA:
    import colorama
    colorama.init()

from nmrespy._errors import LaTeXFailedError
from nmrespy.plot import NmrespyPlot

if USE_COLORAMA:
    import colorama

    colorama.init()

TMPDIR = pathlib.Path(tempfile.gettempdir())


def header() -> str:
    now = datetime.datetime.now()
    return(
        "\\documentclass[8pt]{article}\n"
        "\\usepackage[a4paper,landscape,margin=1in]{geometry}\n"
        "\\usepackage{cmbright}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{nicefrac}\n"
        "\\usepackage{siunitx}\n"
        "\\usepackage{array}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage{xcolor}\n"
        "\\definecolor{urlblue}{HTML}{0000ee}\n"
        "\\usepackage{hyperref}\n"
        "\\hypersetup{%\n"
        "colorlinks = true,%\n"
        "urlcolor   = urlblue,%\n"
        "}\n"
        "\\usepackage{tcolorbox}\n"
        "\\usepackage{varwidth}\n"
        "\\setlength\\parindent{0pt}\n"
        "\\pagenumbering{gobble}\n"
        "\\begin{document}\n"
        "\\begin{figure}[!ht]\n"
        "% MF group logo\n"
        "\\begin{minipage}[b][2.5cm][c]{.72\\textwidth}\n"
        f"\\href{{{pl.MFGROUPLINK}}}%\n"
        f"{{\\includegraphics[scale=1.8]{{{pl.MFLOGOPATH}}}}}\n"
        "\\end{minipage}\n"
        "% NMR-EsPy logo\n"
        "\\begin{minipage}[b][2.5cm][c]{.27\\textwidth}\n"
        f"\\href{{{pl.DOCSLINK}}}%\n"
        f"{{\\includegraphics[scale=0.5]{{{pl.NMRESPYLOGOPATH}}}}}\n"
        "\\end{minipage}\n"
        "\\end{figure}\n"
        f"\\texttt{{{now.strftime('%X')} "
        f"{now.strftime('%d')}-{now.strftime('%m')}-{now.strftime('%y')}}}\n"
    )


def experiment_info(table: List[List[str]]) -> str:
    return titled_table("Experiment Information", table)


def parameter_table(table: List[List[str]]) -> str:
    return titled_table("Estimation Result", table)


def titled_table(title: str, table: List[List[str]]) -> str:
    text = "\\section{{{title}}}\n"
    text += f"{tabular(table, titles=True)}"
    return text


def tabular(rows: List[List[str]], titles: bool = False) -> str:
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
    column_specifier = " ".join(["c" for _ in range(len(rows[0]))])
    table = f"\\begin{{longtable}}[l]{{{column_specifier}}}\n\\toprule\n"
    for i, row in enumerate(rows):
        table += " & ".join([texify(e) for e in row])
        if titles and i == 0:
            table += "\\\\\n\\midrule\n"
        else:
            table += "\\\\\n"
    table += "\\bottomrule\n\\end{longtable}"
    return table


def texify(entry: str) -> str:
    print(entry)
    print(re.search(r"(₀|₁|₂|₃|₄|₅|₆|₇|₈|₉)", entry))
    if entry == "a":
        return "$a$"
    elif entry == "ϕ (rad)":
        return "$\\phi$ (rad)"
    elif bool(re.search(r"(₀|₁|₂|₃|₄|₅|₆|₇|₈|₉)", entry)):
        entry = u''.join(
            dict(
                zip(
                    u"₀₁₂₃₄₅₆₇₈₉",
                    ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]
                )
            ).get(c, c)
            for c in entry
        )
        entry = f"${entry}$"
    if "η" in entry:
        entry = entry.replace("η", "$\\eta$").replace("s⁻¹", "s$^{-1}$")

    try:
        float(entry)
        return f"$\\num{{{entry}}}$"
    except ValueError:
        return entry


def footer() -> str:
    refs = [
        f"\\item {paper['citation']}\\\\ \\href{{{paper['doi']}}}{{{paper['doi']}}}"
        for paper in pl.PAPERS.values()
    ]
    quotes = [[i for i, c in enumerate(ref) if c == "\""] for ref in refs]
    refs = "\n".join(
        [
            f"{ref[:quote[0]]}"
            f"\\textit{{{ref[quote[0] : quote[1] + 1]}}}"
            f"{ref[quote[1] + 1:]}"
            for quote, ref in zip(quotes, refs)
        ]
    )

    return(
        "\\small\n"
        "\\begin{tcolorbox}[hbox]\n"
        "\\begin{varwidth}{12cm}\n"
        "Estimation performed using \\textsc{NMR-EsPy}.\\\\\n"
        "Author: Simon Hulse\\\\\n"
        "For more information:\\\\[5pt]\n"
        f"{{\\raisebox{{-4pt}}{{\\includegraphics[scale=0.029]{{"
        f"{pl.BOOKICONPATH}}}}}}}\\hspace{{1em}}\\href{{{pl.DOCSLINK}}}"
        f"{{\\texttt{{{pl.DOCSLINK}}}}}\\\\[5pt]\n"
        f"{{\\raisebox{{-4pt}}{{\\includegraphics[scale=0.12]{{"
        f"{pl.GITHUBLOGOPATH}}}}}}}\\hspace{{1em}}\\href{{{pl.GITHUBLINK}}}"
        f"{{\\texttt{{{pl.GITHUBLINK}}}}}\\\\[5pt]\n"
        f"{{\\raisebox{{-3pt}}{{\\includegraphics[scale=0.015]{{"
        f"{pl.EMAILICONPATH}}}}}}}\\hspace{{1em}}\\href{{{pl.MAILTOLINK}}}"
        "{\\texttt{simon.hulse@chem.ox.ac.uk}}\\\\[5pt]\n"
        "If used in a publication, please cite:\\\\\n"
        "\\begin{itemize}[leftmargin=*, nosep, label={}]\n"
        f"{refs}\n"
        "\\end{itemize}\n"
        "\\end{varwidth}\n"
        "\\end{tcolorbox}\n"
        "\\end{document}"
    )


def write(
    path: pathlib.Path,
    param_table: List[List[str]],
    info_table: Union[List[List[str]], None],
    description: Union[str, None],
    pdflatex_exe: Union[str, None],
    fprint: bool,
    figure: Union[NmrespyPlot, None],
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

    _write_pdf(path, text, pdflatex_exe, fprint)


def _write_pdf(
    path: pathlib.Path, text: str, pdflatex_exe: Optional[str], fprint: bool
) -> None:
    texpaths = _get_texpaths(path)
    with open(texpaths["tmp"]["tex"], "w", encoding="utf-8") as fh:
        fh.write(text)

    _compile_tex(texpaths, pdflatex_exe)
    _cleanup(texpaths)

    if fprint:
        print(
            f"{GRE}Result successfuly output to:\n"
            f"{texpaths['final']['pdf']}\n"
            "If you wish to customise the document, the TeX file can"
            " be found at:\n"
            f"{texpaths['final']['tex']}{END}"
        )


def _read_template() -> str:
    """Extract LaTeX template text."""
    with open(pl.NMRESPYPATH / "config/latex_template.txt", "r") as fh:
        text = fh.read()
    return text


def _append_links_and_paths(template: str) -> str:
    """Inputs web links and image paths into the text."""
    # Add image paths and weblinks to TeX document
    stuff = {
        "<MFLOGOPATH>": pl.MFLOGOPATH,
        "<NMRESPYLOGOPATH>": pl.NMRESPYLOGOPATH,
        "<DOCSLINK>": pl.DOCSLINK,
        "<MFGROUPLINK>": pl.MFGROUPLINK,
        "<BOOKICONPATH>": pl.BOOKICONPATH,
        "<GITHUBLINK>": pl.GITHUBLINK,
        "<GITHUBLOGOPATH>": pl.GITHUBLOGOPATH,
        "<MAILTOLINK>": pl.MAILTOLINK,
        "<EMAILICONPATH>": pl.EMAILICONPATH,
    }

    for before, after in stuff.items():
        # On Windows, have to replace paths C:\a\b\c -> C:/a/b/c
        template = template.replace(before, str(after).replace("\\", "/"))

    return template


def _append_timestamp(text: str) -> str:
    return text.replace("<TIMESTAMP>", _timestamp())


def _append_description(text: str, description: Union[str, None]) -> str:
    """Append description to text."""
    if description is None:
        return text.replace(
            "% user provided description\n\\subsection*{Description}\n" "<DESCRIPTION>",
            "",
        )
    else:
        return text.replace("<DESCRIPTION>", description)


def _append_info_table(text: str, info_table: List[List[str]]) -> str:
    return text.replace("<INFOTABLE>", _latex_longtable(info_table))


def _append_param_table(text: str, param_table: List[List[str]]) -> str:
    param_table = [[e.replace("±", "$\\pm$") for e in row] for row in param_table]
    return text.replace("<PARAMTABLE>", _latex_longtable(param_table))


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
    column_specifier = " ".join(["c" for _ in range(len(rows[0]))])
    table = f"\\begin{{longtable}}[l]{{{column_specifier}}}\n\\toprule\n"
    for i, row in enumerate(rows):
        table += " & ".join([e for e in row])
        if i == 0:
            table += "\n\\\\\\midrule\n"
        else:
            table += "\\\\\n"
    table += "\\bottomrule\n\\end{longtable}"
    return table


def _append_figure(text: str, figure: Union[NmrespyPlot, None]):
    if isinstance(figure, NmrespyPlot):
        path = TMPDIR / "figure.pdf"
        figure.fig.savefig(path, dpi=600, format="pdf")
        text = text.replace(
            "<RESULTFIGURE>",
            "% figure of result\n\\begin{center}\n"
            f"\\includegraphics{{{path}}}\n\\end{{center}}",
        )
    else:
        text = text.replace("<RESULTFIGURE>", "")

    return text


def timestamp() -> str:
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
    d = now.strftime("%d")  # Day
    m = now.strftime("%m")  # Month
    y = now.strftime("%Y")  # Year
    t = now.strftime("%X")  # Time (hh:mm:ss)
    return f"{t}\\\\{d}-{m}-{y}"


def _get_texpaths(path: pathlib.Path) -> Dict[str, Dict[str, pathlib.Path]]:
    return {
        "tmp": {
            suffix: TMPDIR / path.with_suffix(f".{suffix}").name
            for suffix in ("tex", "pdf", "out", "aux", "log")
        },
        "final": {"tex": path.with_suffix(".tex"), "pdf": path},
    }


def _compile_tex(
    texpaths: Dict[str, Dict[str, pathlib.Path]], pdflatex_exe: str
) -> None:
    if pdflatex_exe is None:
        pdflatex_exe = "pdflatex"

    src = texpaths["tmp"]["tex"]
    dst = texpaths["final"]["tex"]
    try:
        subprocess.run(
            [pdflatex_exe, "-halt-on-error", f"-output-directory={src.parent}", src],
            stdout=subprocess.DEVNULL,
            check=True,
        )

    except Exception or subprocess.SubprocessError:
        shutil.move(src, dst)
        raise LaTeXFailedError(dst)


def _cleanup(texpaths: Dict[str, Dict[str, pathlib.Path]]) -> None:
    shutil.copy(texpaths["tmp"]["tex"], texpaths["final"]["tex"])
    if texpaths["tmp"]["pdf"].is_file():
        shutil.copy(texpaths["tmp"]["pdf"], texpaths["final"]["pdf"])

    files = [p / TMPDIR / "figure.pdf" for p in texpaths["tmp"].values()]
    for f in filter(lambda f: f.is_file(), files):
        os.remove(f)
