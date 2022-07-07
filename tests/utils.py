# utils.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 06 Jul 2022 14:29:29 BST

from pathlib import Path
import platform
import re
import subprocess
from typing import Iterable

import numpy as np

__logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1


def sigfigs(arr: np.ndarray, x: int) -> np.ndarray:
    arrsgn = np.sign(arr)
    absarr = arrsgn * arr
    mantissas, binaryExponents = np.frexp(absarr)

    decimalExponents = __logBase10of2 * binaryExponents
    omags = np.floor(decimalExponents)

    mantissas *= 10.0 ** (decimalExponents - omags)

    if type(mantissas) is float or isinstance(mantissas, np.floating):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0

    else:
        fixmsk = mantissas < 1.0,
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0

    result = arrsgn * np.around(mantissas, decimals=x - 1) * 10.0 ** omags
    return result


def aequal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, rtol=0, atol=1e-8)


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    return all([abs(xa - xb) <= 1e-8 if xa is not None and xb is not None
                else (xa is None and xb is None)
                for xa, xb in zip(a, b)])


def close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return np.allclose(a, b, rtol=0, atol=tol)


def view_files(to_view: Iterable[Path], view_content: bool) -> None:
    if view_content:
        for path in to_view:
            if path.suffix in [".txt", ".log"]:
                prog = "vi"
            elif path.suffix == ".pdf":
                prog = "evince"
            subprocess.run([prog, str(path)])


def latex_exists():
    if platform.system() == "Windows":
        cmd = "where"
    else:
        cmd = "which"

    return subprocess.run(
        [cmd, "pdflatex"], stdout=subprocess.DEVNULL,
    ).returncode == 0


def extract_params_from_txt(path) -> np.ndarray:
    with open(path, "r") as fh:
        lines = fh.readlines()
    rows = filter(lambda line: re.match(r"(\d|\s){5}│", line), lines)
    params = []
    errors = []
    for row in rows:
        row = row[6:]
        entries = re.findall(
            r"-?\d+(?:\.\d+)?(?:e-\d+)? ± \d+(?:\.\d+)?(?:e-\d+)?",
            row,
        )
        nparams = len(entries)

        if nparams == 5:
            # dim = 1 and ppm is present
            entries.pop(3)
        if nparams == 8:
            # dim = 2 and ppm is present
            entries.pop(3)
            entries.pop(5)
        params.append([float(x.split(" ± ")[0]) for x in entries])
        errors.append([float(x.split(" ± ")[1]) for x in entries])

    params = np.array(params)
    errors = np.array(errors)
    params[:, 1] /= (180 / np.pi)
    errors[:, 1] /= (180 / np.pi)

    order = np.argsort(params[:, 2])
    params = params[order]
    errors = errors[order]

    return params, errors
