# textfile.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 05 Apr 2022 15:54:08 BST

import datetime
from textwrap import TextWrapper
from typing import List

import nmrespy._paths_and_links as pl


def header() -> str:
    now = datetime.datetime.now()
    return (
        f"{now.strftime('%X')} "
        f"{now.strftime('%d')}-{now.strftime('%m')}-{now.strftime('%y')}"
    )


def underline(text: str) -> str:
    return text + f"\n{len(text) * '─'}\n"


def experiment_info(table: List[List[str]]) -> str:
    return titled_table("Experiment Information", table)


def titled_table(title: str, table: List[List[str]]) -> str:
    text = underline(title)
    text += f"\n{tabular(table, titles=True)}"
    return text


def footer() -> str:
    refs = "\n\n".join(
        [
            TextWrapper(width=80).fill(paper['citation']) +
            f"\n({paper['doi']})"
           for paper in pl.PAPERS.values()
        ]
    )
    return boxify(
        "Estimation performed using NMR-EsPy\n"
        "Author: Simon Hulse (simon.hulse@chem.ox.ac.uk)\n\n"
        f"If used in any publications, please cite:\n{refs}\n\n"
        f"For more information, visit the GitHub repo:\n{pl.GITHUBLINK}\n",
        1,
    )


def boxify(text: str, pad: int) -> str:
    old_lines = text.split("\n")[:-1]
    longest = max([len(line) for line in old_lines])

    new_lines = [f"┌{(longest + 2 * pad) * '─'}┐"]
    for line in old_lines:
        padding = pad * " "
        space = (longest - len(line)) * " "
        new_lines.append(f"│{padding}{line}{space}{padding}│")

    new_lines.append(f"└{(longest + 2 * pad) * '─'}┘")
    return "\n".join(new_lines)


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
    pads = []
    for column in zip(*rows):
        # For each column find the longest string, and set its length
        # as the width
        pads.append(max(len(element) for element in column))

    separator = "│" if titles else " "

    table = ""
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
        table += "\n"

        # Add a horizontal line underneath the first row to separate the
        # titles from the other contents
        if titles and i == 0:
            for k, pad in enumerate(pads):
                p = pad + 1
                # Add a bar that looks like this: '────────┼'
                table += f"{p * '─'}┼"
            # Remove the trailing '┼' and add a newline
            table = table[:-1] + "\n"

    return table
