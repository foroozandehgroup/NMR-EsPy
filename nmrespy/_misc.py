# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 30 Mar 2023 11:49:03 BST

"""Various miscellaneous functions/classes for internal nmrespy use."""

from collections.abc import Callable
import functools
import re
from typing import Any, Dict, Iterable, Optional

import numpy as np

from nmrespy._colors import RED, ORA, GRE, END, USE_COLORAMA
from nmrespy._sanity import sanity_check, funcs as sfuncs

if USE_COLORAMA:
    import colorama
    colorama.init()


def copydoc(fromfunc, sep="\n"):
    """Decorator: copy the docstring of `fromfunc`."""
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ is None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func
    return _decorator


def get_yes_no(prompt: str) -> bool:
    """Ask user to input 'yes' or 'no'.

    User should provide one of the following: ``'y'``, ``'Y'``, ``'n'``,
    ``'N'``. If an invalid response is given, the user is asked again.
    """
    print(f"{prompt}\nEnter [y] or [n]")
    while True:
        response = input().lower()
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            print(f"{RED}Invalid input. Please enter [y] or [n]:{END}")


def boxed_text(text: str, color: Optional[str] = None) -> str:
    if color is None:
        start_color = ""
        end_color = ""
    else:
        start_color = color
        end_color = END

    return (
        f"{start_color}┌{len(text) * '─'}┐\n"
        f"│{text}│\n"
        f"└{len(text) * '─'}┘{end_color}"
    )


def start_end_wrapper(start_text: str, end_text: str) -> Callable:
    """Print a message prior to and after a method call."""

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def inner(*args, **kwargs) -> Any:
            print(boxed_text(start_text, ORA))
            result = f(*args, **kwargs)
            print(boxed_text(end_text, GRE))

            return result
        return inner
    return decorator


def latex_nucleus(nucleus: str) -> str:
    r"""Create a isotope symbol string for processing by LaTeX.

    Parameters
    ----------
    nucleus
        Of the form ``f'{mass}{sym}'``, where ``mass`` is the nuceleus'
        mass number and ``sym`` is its chemical symbol. I.e. for
        lead-207, `nucleus` should be ``'207Pb'``.

    Returns
    -------
    latex_nucleus
        Of the form ``f'$^{mass}${sym}'`` i.e. given ``'207Pb'``, the
        return value would be ``'$^{207}$Pb'``

    Raises
    ------
    ValueError
        If `nucleus` does not match the regex ``r'^\d+[a-zA-Z]+$'``
    """
    sanity_check(("nucleus", nucleus, sfuncs.check_nucleus))
    mass = re.search(r"\d+", nucleus).group()
    sym = re.search(r"[a-zA-Z]+", nucleus).group()
    return f"$^{{{mass}}}${sym}"


def proc_kwargs_dict(
    kwargs: Optional[Dict],
    default: Optional[Dict] = None,
    to_pop: Optional[Iterable[str]] = None,
) -> Dict:
    if default is None:
        default = {}

    if kwargs is None:
        kwargs = default
    else:
        if to_pop is not None:
            for s in to_pop:
                kwargs.pop(s, None)
        for k, v in default.items():
            kwargs.setdefault(k, v)

    return kwargs


def wrap_phases(phases: np.ndarray) -> np.ndarray:
    return (phases + np.pi) % (2 * np.pi) - np.pi
