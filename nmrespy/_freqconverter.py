# _freqconverter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Mar 2022 11:40:22 GMT

from typing import Iterable, Optional, Union

from nmrespy import ExpInfo
from nmrespy._sanity import sanity_check, funcs as sfuncs


class FrequencyConverter:
    """Handle frequency unit conversion."""

    def __init__(
        self,
        expinfo: ExpInfo,
        pts: Optional[Iterable[int]] = None,
    ) -> None:
        """Initialise the converter.

        Parameters
        ----------
        expinfo
            Information on experiemnt parameters.

        pts
            The points associated with the signal. If ``None``, and
            ``expinfo.default_pts`` is an iterable if ints, this will be used.
            If ``expinfo.default_pts`` is also ``None``, conversion to and from
            array indices will not be permitted (only between Hz and ppm).
        """
        sanity_check(("expinfo", expinfo, sfuncs.check_expinfo))
        sanity_check(("pts", pts, sfuncs.check_points, (expinfo.dim,), True))
        self.expinfo = expinfo

        if pts is None:
            if self.expinfo.default_pts is None:
                self.pts = None
            else:
                self.pts = self.expinfo.default_pts
        else:
            self.pts = pts

        self.ppm_valid = self.expinfo.sfo is not None
        self.idx_valid = self.pts is not None

    @property
    def dim(self) -> int:
        return self.expinfo.dim

    def convert(
        self,
        lst: Iterable[Optional[Union[Iterable[Union[float, int]], float, int]]],
        conversion: str,
    ) -> Iterable[Union[int, float]]:
        """Convert quantities contained within an iterable.

        Parameters
        ----------
        lst
            An iterable of numerical values, with the same length as
            ``len(self)``.

        conversion
            A string denoting the coversion to be applied. Its form should be
            ``f"{from}->{to}"``, where ``"from"`` and ``"to"`` are each one of the
            following:

            * ``"idx"``: array index
            * ``"hz"``: Hertz
            * ``"ppm"``: parts per million

        Returns
        -------
        converted_lst
            An iterable of the same structure as ``lst``, with converted values.
        """
        sanity_check(
            ("lst", lst, sfuncs.check_convertible_list, (self.dim,)),
            (
                "conversion", conversion, sfuncs.check_frequency_conversion,
                (self.ppm_valid,),
            )
        )

        if "idx" in conversion and not self.idx_valid:
            raise ValueError("Unable to convert to and from indices.")

        converted_lst = []
        for dim, elem in enumerate(lst):
            if elem is None:
                converted_lst.append(None)
            elif isinstance(elem, (tuple, list)):
                converted_lst.append(
                    type(elem)([
                        self._convert_value(x, dim, conversion) for x in elem
                    ])
                )
            else:
                # elem is a float/int...
                converted_lst.append(self._convert_value(elem, dim, conversion))

        return type(lst)(converted_lst)

    def _convert_value(
        self, value: Union[int, float], dim: int, conversion: str
    ) -> Union[int, float]:
        """Convert frequency."""
        pts = self.pts[dim]
        sw = self.expinfo.sw[dim]
        off = self.expinfo.offset[dim]
        if self.ppm_valid:
            sfo = self.expinfo.sfo[dim]

        if sfo is None and "ppm" in conversion:
            return value

        elif conversion == "idx->hz":
            return off + sw * (0.5 - (float(value) / (pts - 1)))

        elif conversion == "idx->ppm":
            return (off + sw * (0.5 - (float(value) / (pts - 1)))) / sfo

        elif conversion == "ppm->idx":
            return int(round(((pts - 1)) / (2 * sw) * (sw + 2 * (off - (value * sfo)))))

        elif conversion == "ppm->hz":
            return value * sfo

        elif conversion == "hz->idx":
            return int(round((pts - 1) / (2 * sw) * (sw + 2 * (off - value))))

        elif conversion == "hz->ppm":
            return value / sfo

        else:
            # conversion will be one of 'hz->hz', 'ppm->ppm', 'idx->idx'
            return value
