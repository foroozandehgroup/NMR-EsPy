import pathlib
import sys

from nmrespy import *
from ._cols import *
from .app import NMREsPyApp

if __name__ == '__main__':
    if len(sys.args) == 1:
        path = inupt("Please specify a path to Bruker data")
    elif len(sys.args) == 2:
        path = sys.args[1]
    else:
        raise TypeError(
            f"{cols.R}Toom many arguments provided{cols.END}"
        )

    path = pathlib.Path(path).resolve()
    print(path)


    app = NMREsPyApp()
    app.mainloop()
