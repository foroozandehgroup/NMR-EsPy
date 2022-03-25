# csvfile.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 12:19:51 GMT

import csv
import datetime
import pathlib
from typing import List, Union
from nmrespy._colors import GRE, END, USE_COLORAMA

if USE_COLORAMA:
    import colorama
    colorama.init()


def write(
    path: pathlib.Path,
    param_table: List[List[str]],
    info_table: List[List[str]],
    description: Union[str, None],
    fprint: bool,
) -> None:
    """Writes parameter estimate to a CSV.

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
    with open(path, "w", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([_timestamp().replace("\n", " ")])
        writer.writerow([])
        if description:
            writer.writerow(["Description:", description])
            writer.writerow([])
        writer.writerow(["Experiment Info:"])
        for row in info_table:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["Result:"])
        for row in param_table:
            writer.writerow(row)

    if fprint:
        print(f"{GRE}Saved result to {path}{END}")


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
    d = now.strftime("%d")  # Day
    m = now.strftime("%m")  # Month
    y = now.strftime("%Y")  # Year
    t = now.strftime("%X")  # Time (hh:mm:ss)
    return f"{t}\n{d}-{m}-{y}"
