# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 08 Mar 2022 17:54:57 GMT

import inspect
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


def sanity_check(*param_sets: Iterable[Iterable[Any]]) -> None:
    """Handles checking of inputs.

    Parameters
    ----------
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
    func_name = inspect.stack()[1][3]
    for param_set in param_sets:
        check_item = CheckItem(*param_set)
        if isinstance(check_item.msg, str):
            errmsg = (
                f"{RED}{func_name}:\n"
                f"`{check_item.name}` is invalid:\n"
                f"{check_item.msg}{END}."
            )
            raise TypeError(errmsg)
