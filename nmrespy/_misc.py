# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 17:19:10 GMT

"""Various miscellaneous functions/classes for internal nmrespy use."""

from collections.abc import Callable
import functools
import re
from typing import Any

from nmrespy._colors import RED, GRE, END, USE_COLORAMA
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


def start_end_wrapper(start_text: str, end_text: str) -> Callable:
    """Print a message prior to and after a method call."""

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def inner(*args, **kwargs) -> Any:

            inst = args[0]
            if inst.fprint is False:
                return f(*args, **kwargs)

            print(
                f"{GRE}{len(start_text) * '='}\n"
                f"{start_text}\n"
                f"{len(start_text) * '='}{END}"
            )

            result = f(*args, **kwargs)

            print(
                f"{GRE}{len(end_text) * '='}\n"
                f"{end_text}\n"
                f"{len(end_text) * '='}{END}"
            )

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
