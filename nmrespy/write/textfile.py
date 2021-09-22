import datetime
import pathlib
from typing import List, Union
from nmrespy import GRE, END, GITHUBLINK


def write(
    path: pathlib.Path, param_table: List[List[str]],
    info_table: Union[List[List[str]], None], description: Union[str, None],
    fprint: bool,
) -> None:
    """Write a result textfile.

    Parameters
    -----------
    path
        File path.

    param_table
        Table of estimation result parameters.

    info_table
        Table of experiment information.

    description
        A descriptive statement.

    fprint
        Specifies whether or not to print output to terminal.
    """
    text = _create_contents(param_table, info_table, description)

    try:
        _write_file(path, text)
    except Exception as e:
        raise e

    if fprint:
        print(f'{GRE}Saved result to\n{path}{END}')


def _write_file(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path``.

    Parameters
    ----------
    path
        Path to file.

    text
        Text to insert.
    """
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)


def _create_contents(
    param_table: List[List[str]], info_table: Union[List[List[str]], None],
    description: Union[str, None]
) -> str:
    """Create text to be inserted into a file.

    See :py:func:`_write_txt` for a description of the parameters.

    Returns
    -------
    contents: str
        Text to insert into file.
    """
    msg = f'{_timestamp()}\n\n'
    if description:
        msg += f'Description:\n{description}\n\n'
    if info_table:
        msg += (f'Experiment Information:\n'
                f'{_txt_tabular(info_table, titles=True)}\n\n')
    msg += f'{_txt_tabular(param_table, titles=True)}\n\n'
    msg += _footer()

    return msg


def _footer() -> str:
    """Descriptive text for end of file."""
    return ("Estimation performed using NMR-EsPy\n"
            "Author: Simon Hulse ~ simon.hulse@chem.ox.ac.uk\n"
            "If used in any publications, please cite:\n"
            "<no papers yet...>\n"
            f"For more information, visit the GitHub repo:\n{GITHUBLINK}\n")


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
    return f'{t}\n{d}-{m}-{y}'
