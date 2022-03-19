# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 18 Mar 2022 12:59:15 GMT

import inspect
from typing import Any, Iterable, Optional
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
    funcname = get_name(inspect.currentframe())
    for param_set in param_sets:
        check_item = CheckItem(*param_set)
        if isinstance(check_item.msg, str):
            errmsg = (
                f"{RED}{funcname}:\n"
                f"`{check_item.name}` is invalid:\n"
                f"{check_item.msg}{END}."
            )
            raise TypeError(errmsg)


def get_name(frame: inspect.types.FrameType) -> Optional[str]:
    # https://stackoverflow.com/questions/2654113/how-to-get-the-callers-method-name-in-the-called-method
    funcname = inspect.getouterframes(frame, 2)[1][3]
    try:
        try:
            self_obj = frame.f_back.f_locals["self"]
        except KeyError:
            classname = None
        classname = type(self_obj).__name__
    finally:
        del frame
    return f"{classname}.{funcname}" if classname is not None else funcname
