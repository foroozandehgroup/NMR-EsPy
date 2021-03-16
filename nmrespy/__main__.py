import argparse
import pathlib
import sys

from nmrespy import *
from nmrespy.load import load_bruker
from nmrespy._install_to_topspin import main as install_to_topspin
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
from .app import NMREsPyApp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="""
        NMR-EsPy GUI-related commands.
        """)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-e', '--estimate',
        help="Loads the nmrespy GUI to set up an estimation. Argument should "
             "be a path to a Bruker data directory.",
    )
    group.add_argument(
        '-r', '--result',
        help="Loads the nmrespy GUI to view an estimation result. Argument "
             "should be a path to a pickled estimator instance.",
    )
    group.add_argument(
        '-i', '--install-to-topspin', dest='install_to_topspin',
        action='store_true', help="Install the nmrespy GUI to TopSpin"
    )
    parser.add_argument(
        '-t', '--topspin', action='store_true', help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.install_to_topspin:
        install_to_topspin()
        exit()

    if args.estimate is not None:
        path = pathlib.Path(args.estimate).resolve()
        if not path.is_dir():
            raise ValueError(
                f"{cols.R}\nThe path you have specified doesn't exist:\n{path}"
                f"{cols.END}"
            )
        res = False

    else:
        path = pathlib.Path(args.result).resolve()
        if not path.is_file():
            raise ValueError(
                f"{cols.R}\nThe path you have specified doesn't exist:\n{path}"
                f"{cols.END}"
            )
        res = True

    app = NMREsPyApp(path=path, topspin=args.topspin, res=res)
    app.mainloop()
