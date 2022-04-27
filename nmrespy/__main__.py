# __main__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 26 Apr 2022 12:14:41 BST

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

if USE_COLORAMA:
    import colorama

    colorama.init()
from .app import NMREsPyApp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        NMR-EsPy GUI-related commands.
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-e",
        "--estimate",
        help="Loads the nmrespy GUI to set up an estimation. Argument should "
        "be a path to a Bruker data directory.",
    )
    group.add_argument(
        "-r",
        "--result",
        help="Loads the nmrespy GUI to view an estimation result. Argument "
        "should be a path to a pickled estimator instance.",
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

    if args.install_to_topspin:
        install_to_topspin()
        exit()

    if args.estimate is not None:
        path = pathlib.Path(args.estimate).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                f"{RED}\nThe path you have specified doesn't exist:\n{path}" f"{END}"
            )
        res = False

    else:
        path = pathlib.Path(args.result).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"{RED}\nThe path you have specified doesn't exist:\n{path}" f"{END}"
            )
        res = True

    app = NMREsPyApp(
        path=path,
        res=res,
        topspin=args.topspin,
        pdflatex=args.pdflatex,
    )
    app.mainloop()
