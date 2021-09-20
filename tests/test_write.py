from pathlib import Path
import pytest
import subprocess
from context import nmrespy
from nmrespy import write as nwrite
from nmrespy import _cols as cols

USER_INPUT = True
FILE = Path(__file__).resolve().parent / 'file.pdf'


def test_configure_save_path():
    assert nwrite._configure_save_path('file', 'pdf', True) == (True, FILE)

    # Make file `file.pdf` and assert that same result is returned when
    # `force_overwrite` is `True`
    subprocess.run(['touch', 'file.pdf'])
    assert nwrite._configure_save_path('file', 'pdf', True) == \
        (True, FILE)

    if USER_INPUT:
        # Set `force_overwrite` to False and ensure the user is prompted
        print(f"\n{cols.R}PLEASE PRESS y{cols.END}")
        assert nwrite._configure_save_path('file', 'pdf', False) == \
            (True, FILE)

        print(f"{cols.R}PLEASE PRESS n{cols.END}")
        assert nwrite._configure_save_path('file', 'pdf', False) == \
            (False,
             f'{cols.R}Overwrite of file {str(FILE)} denied. Result '
             f'file will not be written.{cols.END}')

    subprocess.run(['rm', 'file.pdf'])

    # Non-existent directory
    invalid_path = FILE.parent / 'extra_dir' / 'file'
    assert nwrite._configure_save_path(invalid_path, 'pdf', False) == \
        (False,
         f'{cols.R}The directory specified by `path` does not exist:\n'
         f'{invalid_path.parent}{cols.END}')


def test_raise_error():
    msg = 'Value is not valid.'
    with pytest.raises(ValueError) as exc_info:
        nwrite.raise_error(ValueError, msg, True)
    assert str(exc_info.value) == msg

    assert nwrite.raise_error(ValueError, msg, False) is None
