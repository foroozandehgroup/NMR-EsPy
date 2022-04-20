# pdffile.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 06 Apr 2022 11:44:40 BST

import datetime
import re
from typing import List

import nmrespy._paths_and_links as pl


def header() -> str:
    now = datetime.datetime.now()
    return(
        "\\documentclass[8pt]{article}\n"
        "\\usepackage[a4paper,landscape,margin=1in]{geometry}\n"
        "\\usepackage{cmbright}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{amsmath}\n"
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


def titled_table(title: str, table: List[List[str]]) -> str:
    text = f"\\section*{{{title}}}\n"
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
    if entry == "a":
        return "$a$"

    if entry == "ϕ (°)":
        return "$\\phi$ ($^{\\circ}$)"

    # Frequency and damping factor labels: f₁, η₂ etc.
    search_freq_damp = re.search(r"^(f|η)(₁|₂|₃)?", entry)
    if search_freq_damp:
        entry = u''.join(
            dict(
                zip(
                    u"₁₂₃",
                    ["_1", "_2", "_3"],
                )
            ).get(c, c)
            for c in entry
        )
        entry = f"${entry[:3]}${entry[3:]}"
        if "η" in entry:
            entry = entry.replace("η", "\\eta").replace("s⁻¹", "s$^{-1}$")

    if entry == "∫":
        return "$\\int$"

    # Check if number
    number_regex = (
        r"(-?\d+(?:\.\d+)?(?:e(?:\+|-)\d+)?)"
        r"( ± (-?\d+(?:\.\d+)?(?:e(?:\+|-)\d+)?))?"
    )
    number_match = re.fullmatch(number_regex, entry)
    if number_match is None:
        return entry

    elif "±" in entry:
        return "\\begin{tabular}[c]{@{}c@{}}$" + " \\\\ $\\pm".join(
            [
                f"\\num{{{match.group(0)}}}$" for match in
                re.finditer(r"(-?\d+(?:\.\d+)?(?:e(?:\+|-)\d+)?)", entry)
            ]
        ) + "\\end{tabular}"

    else:
        return f"$\\num{{{entry}}}$"


def footer() -> str:
    refs = [
        f"\\item {paper['citation']}\\\\ \\href{{{paper['doi']}}}{{{paper['doi']}}}"
        for paper in pl.PAPERS.values()
    ]
    quotes = [[i for i, c in enumerate(ref) if c == "\""] for ref in refs]
    refs = "\n".join(
        [
            f"{ref[:quote[0]]}"
            f"\\textit{{``{ref[quote[0] + 1 : quote[1] + 1]}}}"
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
