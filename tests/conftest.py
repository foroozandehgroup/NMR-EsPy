# conftest.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 07 Jun 2022 18:07:27 BST

import builtins
import io
import os
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import pytest


def patch_open(open_func: callable, files: Iterable[Path]) -> callable:
    def open_patched(
        path, mode="r", buffering=-1, encoding=None, errors=None, newline=None,
        closefd=True, opener=None
    ):
        path = Path(path).resolve()
        if "w" in mode and not path.is_file():
            files.append(path)
            if path.suffix == ".tex":
                files.append(path.with_suffix(".pdf"))
        return open_func(
            path, mode=mode, buffering=buffering, encoding=encoding,
            errors=errors, newline=newline, closefd=closefd, opener=opener,
        )
    return open_patched


@pytest.fixture
def cleanup_files(monkeypatch):
    files = []
    monkeypatch.setattr(builtins, "open", patch_open(builtins.open, files))
    monkeypatch.setattr(io, "open", patch_open(io.open, files))
    yield
    for file in files:
        os.remove(file)
