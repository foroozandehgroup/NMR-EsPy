# _files.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 05 Apr 2022 14:29:27 BST

from pathlib import Path
import pickle
from typing import Any, Optional, Union

from nmrespy._colors import END, GRE, USE_COLORAMA
from nmrespy._misc import get_yes_no

if USE_COLORAMA:
    import colorama
    colorama.init()


def append_suffix(path: Path, suffix: Optional[str]) -> Path:
    if suffix is None:
        return path
    if not path.suffix == f".{suffix}":
        path = path.with_suffix(f".{suffix}")
    return path


def configure_path(path: Union[str, Path], suffix: str) -> Path:
    return append_suffix(Path(path).expanduser(), suffix)


def save_file(content: Any, path: Path, binary: bool = False, fprint: bool = True):
    mode = "wb" if binary else "w"
    enc = None if binary else "utf-8"
    with open(path, mode, encoding=enc) as fh:
        if binary:
            pickle.dump(content, fh, pickle.HIGHEST_PROTOCOL)
        else:
            fh.write(content)

    if fprint:
        print(f"{GRE}Saved file {path}.{END}")


def open_file(path, binary: bool = False) -> Any:
    mode = "rb" if binary else "r"
    enc = None if binary else "utf-8"
    with open(path, mode, encoding=enc) as fh:
        if binary:
            return pickle.load(fh)
        else:
            return fh.read()


def check_saveable_path(
    obj: Any,
    suffix: Optional[str],
    force_overwrite: bool,
) -> Optional[str]:
    if isinstance(obj, (Path, str)):
        path = configure_path(obj, suffix)
    else:
        return "Should be a pathlib.Path object or a str specifying a path."

    directory = path.parent
    if not directory.is_dir():
        return f"The parent directory {directory} doesn't exist."

    if not force_overwrite and path.is_file():
        response = get_yes_no(f"{path} already exists. Overwrite?")
        if not response:
            return "Overwrite not permitted."


def check_existent_path(obj: Any, suffix: Optional[str] = None) -> Optional[str]:
    if isinstance(obj, (Path, str)):
        path = configure_path(obj, suffix)
    else:
        return "Should be a pathlib.Path object or a str specifying a path."

    if not path.is_file():
        return f"Path {path} does not exist."


def check_existent_dir(obj: Any) -> Optional[str]:
    if isinstance(obj, (Path, str)):
        path = Path(obj).expanduser()
    else:
        return "Should be a pathlib.Path object or a str specifying a path."

    if not path.is_dir():
        return f"Path {path} is not a directory."
