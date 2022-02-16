# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 15 Feb 2022 10:42:33 GMT

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from nmr_sims.experimental import Parameters
from nmr_sims.experiments.jres import jres
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo
from nmrespy.core import Estimator


class JresEstimator(Estimator):
    def __init__(
        self, data: np.ndarray, datapath: Optional[Path], expinfo: ExpInfo,
    ) -> None:
        super().__init__(data, datapath, expinfo)

    @classmethod
    def new_simulated(
        cls,
        spin_system: SpinSystem,
        points: Tuple[int, int],
        sweep_widths: Tuple[Union[str, float, int], Union[str, float, int]],
        offset: Union[str, float, int] = 0.0,
        channel: Union[str, Nucleus] = "1H",
        snr: Optional[float] = None,
    ) -> JresEstimator:
        """Generate an estimator with a simulated J-Resolved dataset.

        Parameters
        ----------

        spin_system
            Spin system specification for generating the simulated dataset.
        """
        # TODO sanity checking
        result = jres(spin_system, points, sweep_widths, [offset], [channel])
        data = result._fid["fid"]
        info = result._dim_info
        expinfo = ExpInfo(
            sw=[info[i]["sw"] for i in (0, 1)],
            offset=[info[i]["off"] for i in (0, 1)],
            sfo=1e-6 * info[0]["nuc"].gamma / (2 * np.pi) * result._field,
            nuclei=info[0]["nuc"].name,
            dim=2,
        )
        return cls(data, None, expinfo)


if __name__ == "__main__":
    spin_system = SpinSystem({
        1: {
            "shift": 5.5,
            "couplings": {
                2: 4.6,
                3: 4.6,
            }
        },
        2: {
            "shift": 8.2,
        },
        3: {
            "shift": 8.2,
        },
    })

    # Experiment parameters
    channel=["1H"],
    sweep_widths=["50Hz", "10ppm"],
    points=[64, 256],
    offset=["5ppm"],

    e = JresEstimator.new_simulated(spin_system, params)
