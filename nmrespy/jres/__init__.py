# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 15 Mar 2022 13:45:42 GMT

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from nmr_sims.experiments.jres import JresSimulation
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo, sig
from nmrespy.core import Estimator
from nmrespy.freqfilter import Region
from nmrespy._sanity import sanity_check
import nmrespy._sanity.funcs as sfuncs


class JresEstimator(Estimator):
    def __init__(
        self, data: np.ndarray, datapath: Optional[Path], expinfo: ExpInfo,
    ) -> None:
        super().__init__(data, datapath, expinfo)

    @classmethod
    def new_synthetic_from_simulation(
        cls,
        spin_system: SpinSystem,
        sweep_widths: Tuple[float, float],
        offset: float,
        pts: Tuple[int, int],
        channel: Union[str, Nucleus] = "1H",
        f2_unit: str = "ppm",
        snr: Optional[float] = 30.0,
    ) -> JresEstimator:
        """Generate an estimator with data derived from a J-resolved experiment
        simulation.

        Simulations are performed using the
        `nmr_sims.experiments.jres.JresEstimator
        <https://foroozandehgroup.github.io/nmr_sims/content/references/experiments/
        pa.html#nmr_sims.experiments.jres.JresEstimator>`_
        class.

        Parameters
        ----------
        spin_system
            Specification of the spin system to run simulations on.
            `See here <https://foroozandehgroup.github.io/nmr_sims/content/
            references/spin_system.html#nmr_sims.spin_system.SpinSystem.__init__>`_
            for more details.

        sweep_widths
            The sweep width in each dimension. The first element, corresponding to
            F1, should be in Hz. The second element, corresponding to F2, should have
            a value which matches ``f2_unit``.

        offset
            The transmitter offset. The value's unit should match with ``f2_unit``.

        pts
            The number of points sampled in each dimension.

        channel
            Nucleus targeted in the experiment simulation. ¹H is set as the default.
            `See here <https://foroozandehgroup.github.io/nmr_sims/content/
            references/nuclei.html>`__ for more information.

        f2_unit
            The unit that the sweep width and transmitter offset in F2 are given in.
            Should be either ``"ppm"`` (default) or ``"hz"``.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.
        """
        sanity_check(
            ("spin_system", spin_system, sfuncs.check_spin_system),
            ("sweep_widths", sweep_widths, sfuncs.check_positive_float_list, (2,)),
            ("offset", offset, sfuncs.check_float),
            ("pts", pts, sfuncs.check_points, (2,)),
            ("channel", channel, sfuncs.check_nucleus),
            ("f2_unit", f2_unit, sfuncs.check_one_of, ("hz", "ppm")),
            ("snr", snr, sfuncs.check_float, (), True),
        )
        sweep_widths[0] = f"{sweep_widths[0]}hz"
        sweep_widths[1] = f"{sweep_widths[1]}{f2_unit}"
        offset = f"{offset}{f2_unit}"
        sim = JresSimulation(
            spin_system, pts, sweep_widths, offset, channel,
        )
        sim.simulate()
        _, data = sim.fid
        data = data.T
        if snr is not None:
            data += sig._make_noise(data, snr)

        expinfo = ExpInfo(
            sw=sim.sweep_widths,
            offset=[0, sim.offsets[0]],
            sfo=2 * sim.sfo,
            nuclei=2 * [sim.channels[0].name],
            dim=2,
        )
        return cls(data, None, expinfo)

    def _extract_region(
        self,
        region: Tuple[float, float],
        noise_region: Tuple[float, float],
        region_unit: str,
    ) -> Tuple[Region, Region]:
        if region_unit == "hz":
            region_check = sfuncs.check_jres_region_hz
        elif region_unit == "ppm":
            region_check = sfuncs.check_jres_region_ppm

        sanity_check(
            ("region", region, region_check, (self._expinfo,), True),
            ("noise_region", noise_region, region_check, (self._expinfo,), True),
        )

        sw1 = self._expinfo.sw[0]
        region = [
            (sw1 / 2, -sw1 / 2),
            tuple([x for x in region]),
        ]
        noise_region = [
            (sw1 / 2, -sw1 / 2),
            tuple([x for x in noise_region]),
        ]
        region[1] = self.converter.convert(region, f"{region_unit}->hz")[1]
        noise_region[1] = self.converter.convert(noise_region, f"{region_unit}->hz")[1]

        return region, noise_region

    def _make_spectrum(self) -> np.ndarray:
        ve = np.zeros(
            (self._data.shape[0], 2 * self._data.shape[1] - 1),
            dtype="complex",
        )
        for i, fid in enumerate(self._data):
            ve[i] = sig.make_virtual_echo([fid])

        return sig.ft(ve, axes=1)


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
    channel = "1H"
    sweep_widths = [50., 10.]
    points = [64, 256]
    offset = 5.

    estimator = JresEstimator.new_synthetic_from_simulation(
        spin_system, sweep_widths, offset, points, channel=channel, f2_unit="ppm",
    )
    print(dir(estimator))
    estimator.estimate([6.0, 5.0], [8.0, 7.5])
