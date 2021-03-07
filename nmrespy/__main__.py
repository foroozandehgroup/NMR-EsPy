import argparse
import pathlib
import sys

from nmrespy import *
from nmrespy.load import load_bruker
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
from .app import NMREsPyApp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', action='store', required=False)
    parser.add_argument('-t', '--topspin', action='store_true')
    args = parser.parse_args()

    if args.path is None:
        path = inupt("Please specify a path to Bruker data: ")
    else:
        path = args.path

    path = pathlib.Path(path).resolve()

    if not path.is_dir():
        raise ValueError(
            f"{cols.R}\nThe path you have specified doesn't exist:\n{path}"
            f"{cols.END}"
        )

    app = NMREsPyApp(path=path, topspin=args.topspin)
    app.mainloop()
