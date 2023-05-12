# twodim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 12 May 2023 14:56:55 BST

from __future__ import annotations
from typing import Iterable, Optional, Tuple

import numpy as np

import nmrespy as ne
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)

from nmrespy.estimators import logger
from nmrespy.estimators import Estimator


class Estimator2D(Estimator):
    """Estimator class for 2D data comprising a pair of States signals."""

    default_mpm_trim = (64, 64)
    default_nlp_trim = (128, 128)
    default_max_iterations_exact_hessian = 40
    default_max_iterations_gn_hessian = 100

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        isotopes: Iterable[str],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: Tuple[float, float],
        field: float = 11.74,
        nuclei: Tuple[str, str] = ("13C", "1H"),
        snr: Optional[float] = 30.,
        lb: float = 6.91,
    ) -> Estimator2D:
        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int_list, (), {"length": 2, "min_value": 1}),
            ("sw", sw, sfuncs.check_float_list, (), {"length": 2, "must_be_positive": True}),  # noqa: E501
            ("offset", offset, sfuncs.check_float_list, (), {"length": 2}),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nuclei", nuclei, sfuncs.check_nucleus_list, (), {"length": 2}),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            ("lb", lb, sfuncs.check_float, (), {"greater_than_zero": True})
        )
        nspins = len(shifts)
        sanity_check(
            ("couplings", couplings, sfuncs.check_spinach_couplings, (nspins,), {}, True),  # noqa: E501
            ("isotopes", isotopes, sfuncs.check_nucleus_list, (), {"length": nspins}),
        )

        if couplings is None:
            couplings = []

        fid, sfo = cls._run_spinach(
            "hsqc_sim", shifts, couplings, isotopes, field, pts, sw, offset, nuclei,
            to_int=[4], to_double=[5, 6],
        )
        fid_pos = np.array(fid["pos"]).T
        fid_neg = np.array(fid["neg"]).T
        fid = np.zeros((2, *fid_pos.shape), dtype="complex128")
        fid[0] = fid_pos
        fid[1] = fid_neg
        sfo = tuple([x[0] for x in sfo])

        if snr is not None:
            fid = ne.sig.add_noise(fid, snr)

        fid = ne.sig.exp_apodisation(fid, lb, axes=(1, 2))

        expinfo = ne.ExpInfo(
            dim=2,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nuclei,
            default_pts=fid.shape[1:],
        )

        return cls(fid, expinfo)
