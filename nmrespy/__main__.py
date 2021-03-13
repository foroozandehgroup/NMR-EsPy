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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--estimate')
    group.add_argument('-r', '--result')
    parser.add_argument('-t', '--topspin', action='store_true')

    args = parser.parse_args()

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
