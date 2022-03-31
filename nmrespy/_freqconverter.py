# _freqconverter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 31 Mar 2022 13:20:10 BST

import re
from typing import Any, Iterable, Optional, Union

import numpy as np

from nmrespy._sanity import sanity_check
from nmrespy._sanity.funcs import isiter, isnum


def check_convertible_list(obj: Any, dim: int) -> Optional[str]:
    if not isiter(obj):
        return "Should be a tuple or list."
    if len(obj) != dim:
        return f"Should be of length {dim}."

    msg = (
        "Each element should be one of:\n"
        "A numerical type\n"
        "A tuple or list of numerical types\n"
        "A one-dimensional numpy array\n"
        "None"
    )

    for elem in obj:
        if isnum(elem) or elem is None:
            pass
        elif isiter(elem):
            if not all([isnum(x) for x in elem]):
                return msg
        elif isinstance(elem, np.ndarray) and elem.ndim == 1:
            pass
        else:
            return msg


def check_frequency_conversion(
    obj: Any,
    hz_ppm_valid: bool,
    hz_idx_valid: bool,
    ppm_idx_valid: bool,
) -> Optional[str]:
    pattern = r"^(idx|ppm|hz)->(idx|ppm|hz)$"
    if not bool(re.match(pattern, obj)):
        return (
            "Should be a str of the form \"{from}->{to}\", "
            "where {from} and {to} are each one of \"idx\", \"hz\", or \"ppm\""
        )
    if (("ppm" in obj) and ("hz" in obj)) and (not hz_ppm_valid):
        return "Conversion between Hz and ppm is not possible."
    elif (("hz" in obj) and ("idx" in obj)) and (not hz_idx_valid):
        return "Conversion between Hz and array indices is not possible."
    elif (("ppm" in obj) and ("idx" in obj)) and (not ppm_idx_valid):
        return "Conversion between ppm and array indices is not possible."


class FrequencyConverter:
    """Handle frequency unit conversion.

    Conversion between Hz and ppm requires the transmitter frequency (MHz).
    Conversion to array indices also requires the sweep width, transmitter offset,
    and number of signal points.
    """

    def __init__(
        self,
        sfo: Optional[Iterable[float]] = None,
        sw: Optional[Iterable[float]] = None,
        offset: Optional[Iterable[float]] = None,
        pts: Optional[Iterable[int]] = None,
    ) -> None:
        """Initialise the converter.

        Parameters
        ----------
        sfo
            The transmitter frequency (MHz). If ``None``, conversion to and
            from ppm will not be possible.

        sw
            The sweep width (spectral window) (Hz).
            If ``None``, conversion to and from array indices will not be
            possible.

        offset
            The transmitter offset (Hz).
            If ``None``, conversion to and from array indices will not be
            possible.

        pts
            The points associated with the signal.
            If ``None``, conversion to and from array indices will not be
            possible.
        """
        # TODO: sanity check
        self._sfo = sfo
        self._sw = sw
        self._offset = offset
        self._pts = pts
        self.hz_ppm_valid = self._sfo is not None
        self.hz_idx_valid = all(
            [x is not None for x in (self._sw, self._offset, self._pts)]
        )
        self.ppm_idx_valid = (
            self.hz_idx_valid and self.hz_ppm_valid and
            all([sfo is not None for sfo in self._sfo])
        )

    def __len__(self) -> int:
        if self.hz_ppm_valid:
            return len(self._sfo)
        elif self.hz_idx_valid:
            return len(self._sw)
        else:
            return None

    def convert(
        self,
        to_convert: Iterable[Optional[Union[Iterable[Union[float, int]], float, int]]],
        conversion: str,
    ) -> Iterable[Union[int, float]]:
        """Convert quantities contained within an iterable.

        Parameters
        ----------
        to_convert
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
        converted
            An iterable of the same structure as ``to_convert``, with converted values.
        """
        sanity_check(
            (
                "conversion", conversion, check_frequency_conversion,
                (self.hz_ppm_valid, self.hz_idx_valid, self.ppm_idx_valid),
            )
        )

        if self._duplicate_conversion(conversion):
            return to_convert

        sanity_check(
            ("to_convert", to_convert, check_convertible_list, (len(self),)),
        )

        converted = []
        for dim, elem in enumerate(to_convert):
            if elem is None:
                converted.append(None)
            elif isinstance(elem, (tuple, list)):
                converted.append(
                    type(elem)([
                        self._convert_value(x, dim, conversion) for x in elem
                    ])
                )
            else:
                # elem is a float/int...
                converted.append(self._convert_value(elem, dim, conversion))

        return type(to_convert)(converted)

    def _convert_value(
        self, value: Union[int, float], dim: int, conversion: str
    ) -> Union[int, float]:
        if "ppm" in conversion:
            sfo = self._sfo[dim]
            # By-pass for dimensions where sfo is not defined.
            # i.e. indirect dimension of 2DJ should always be in Hz.
            if sfo is None:
                return value
        if "idx" in conversion:
            sw = self._sw[dim]
            off = self._offset[dim]
            pts = self._pts[dim]

        if conversion == "idx->hz":
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

    @staticmethod
    def _duplicate_conversion(conversion: str) -> bool:
        return conversion in ["idx->idx", "ppm->ppm", "hz->hz"]
