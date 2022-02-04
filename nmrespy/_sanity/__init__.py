# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 04 Feb 2022 16:14:28 GMT

from typing import Any, Iterable
from nmrespy import RED, END, USE_COLORAMA

if USE_COLORAMA:
    import colorama
    colorama.init()


class CheckItem:
    """Object which implements a sanity check."""
    def __init__(
        self, name: str, obj: Any, func: callable, funcargs: Iterable[Any] = (),
        none_allowed: bool = False,
    ) -> None:
        self.__dict__.update(locals())
        self.name = name
        if none_allowed and obj is None:
            self.msg = None
        else:
            self.msg = func(obj, *funcargs)


def sanity_check(func_name: str, *param_sets: Iterable[Iterable[Any]]) -> None:
    """Handles checking of inputs.

    Parameters
    ----------
    func_name
        The name of the function/class that is being called, within which sanity
        checking is taking place.

    param_sets
        Iterable of information regarding the objects to check:

        * ``name``: The name of the argument, as it appears in the function
          signature.
        * ``obj``: The object provided by the user as the argument.
        * ``func``: Callable which will be used to check the validity of the
          argument.
        * ``funcargs``: Iterable of any additional arguments that are required
          for checking.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If an argument does not pass it's sanity check.
    """
    for param_set in param_sets:
        check_item = CheckItem(*param_set)
        if isinstance(check_item.msg, str):
            errmsg = (
                f"{RED}{func_name}:\n"
                f"`{check_item.name}` is invalid:\n"
                f"{check_item.msg}{END}."
            )
            raise TypeError(errmsg)
