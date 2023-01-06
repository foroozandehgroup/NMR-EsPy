# __main__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 06 Jan 2023 00:19:20 GMT

"""Run when the user calls the nmrespy module from a command line.

Provides access to the following functionality:

* Load the NMR-EsPy GUI to run an estimation
  (``python -m nmrespy -e <DATAPATH>``)
* Load the NMR-EsPy GUI to inspect an estimation result
  (``python -m nmrespy -r <RESULTPATH>``)
* Install to NMR-EsPy GUI loader into TopSpin
  (``python -i nmrespy -r <RESULTPATH>``)
"""

import argparse
import pathlib

from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._install_to_topspin import main as install_to_topspin
from nmrespy import Estimator1D, Estimator2DJ

if USE_COLORAMA:
    import colorama

    colorama.init()
from .app.main import NMREsPyApp


def fmt_path(path: str) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        NMR-EsPy GUI-related commands.
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--setup1d",
        help="Loads the nmrespy GUI to set up an estimation for 1D data."
    )

    group.add_argument(
        "--setup2dj",
        help=(
            "Loads the nmrespy GUI to set up an estimation for data acquired by a "
            "2DJ experiemnt."
        ),
    )

    group.add_argument(
        "--result",
        help="Loads the nmrespy GUI to view an estimation result."
    )

    group.add_argument(
        "-i",
        "--install-to-topspin",
        dest="install_to_topspin",
        action="store_true",
        help="Install the nmrespy GUI to TopSpin",
    )

    # Internal flag used to indicate that the app was loaded from inside
    # TopSpin. When this occurs, it is necessary to ascertain whether the
    # user would like to analyse data from the fid file or pdata file.
    parser.add_argument(
        "-t",
        "--topspin",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # Internal flag used to specify the path to the pdflatex executable.
    # This is necessary for Windows users that would like to generate
    # PDF result figures when loading the app inside TopSpin:
    # When calling commands using subprocess.Popen() inside TopSpin, the
    # PATH does not match the "usual" PATH, and `pdflatex` is not
    # recgonised.
    parser.add_argument(
        "-p",
        "--pdflatex",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    if args.pdflatex == "None":
        args.pdflatex = None

    if args.install_to_topspin:
        install_to_topspin()
        exit()

    res = False
    if args.setup1d is not None:
        path = fmt_path(args.setup1d)
        dtype = "1D"
        # Try to generate the estimator to ensure the path is valid
        Estimator1D.new_bruker(path)

    elif args.setup2dj is not None:
        path = fmt_path(args.setup2dj)
        dtype = "2DJ"
        # Try to generate the estimator to ensure the path is valid
        Estimator2DJ.new_bruker(path)

    elif args.result is not None:
        path = fmt_path(args.result)
        dtype = None
        if not path.is_file():
            raise ValueError(
                f"{RED}\nThe path you have specified doesn't exist:\n{path}" f"{END}"
            )
        res = True

    gui = NMREsPyApp(
        path=path,
        res=res,
        dtype=dtype,
        topspin=args.topspin,
        pdflatex=args.pdflatex,
    )
    gui.mainloop()
