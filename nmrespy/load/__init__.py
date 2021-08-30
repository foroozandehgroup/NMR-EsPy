from collections.abc import Iterable
from numbers import Number
from typing import Union
import nmrespy._cols as cols
from .bruker import load_bruker


class ExpInfo:
    """Stores general information about experiments.
    """
    def __init__(self,
        sw: Union[int, float, Iterable[Union[int, float]]],
        offset: Union[int, float, Iterable[Union[int, float]], None] = None,
        sfo: Union[int, float, Iterable[Union[int, float]], None] = None,
        nuclei: Union[str, Iterable[str], None] = None,
        dim: Union[int, None] = None,
        **kwargs
    ):
        # Be leinient with parameter specfiication.
        # Mopst of nmrespy expects parameters to be lists with either
        # floats or ints for rigour.

        # If dim is specified, will be strict with ensuring each
        # parameter has the correct number of values. If not, will
        # duplicate values to match correct dim.

        self.__dict__.update(locals())
        names = ['sw']
        instances = [Number]
        for name, inst in zip(('offset', 'sfo', 'nuclei'),
                              (Number, Number, str)):
            if self.__dict__[name] is not None:
                names.append(name)
                instances.append(inst)

        errmsg = ("f{cols.R}Unable to process input{cols.END}")

        for name, inst in zip(names, instances):
            value = self.__dict__[name]
            # If single value (not in list/tuple/etc.) is given, pack into
            # a list (values will be converted to tuples at the end)
            if isinstance(value, inst):
                if inst == Number:
                    # Convert numerical value to float
                    self.__dict__[name] = [float(value)]
                else:
                    # Case for nuclei, which should be a string
                    self.__dict__[name] = [value]

            elif isinstance(value, Iterable):
                if not all([isinstance(v, inst) for v in value]):
                    raise ValueError(errmsg)

                if inst == Number:
                    self.__dict__[name] = [float(v) for v in value]
                else:
                    self.__dict__[name] = list(value)

            else:
                raise ValueError(errmsg)

        if isinstance(dim, int):
            for name in names:
                diff = dim - len(self.__dict__[name])
                if diff == 0:
                    pass
                elif diff > 0:
                    self.__dict__[name] += diff * [self.__dict__[name][-1]]
                else:
                    raise ValueError(errmsg)

        else:
            lengths = [len(self.__dict__[name]) for name in names]
            # Check all lists are of the same length
            if len(set(lengths)) > 1:
                raise ValueError(errmsg)
            else:
                self.dim = lengths[0]

        if self.offset is None:
            self.offset = [0.] * self.dim

        for name in names:
            self.__dict__[name] = tuple(self.__dict__[name])
