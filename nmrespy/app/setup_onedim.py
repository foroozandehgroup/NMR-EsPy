# setup_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 18 Oct 2022 14:30:21 BST

import copy

import nmrespy as ne
from nmrespy.app.stup import Setup1DType
import nmrespy.app.config as cf


class Setup1D(Setup1DType):
    default_maxits = {
        "Exact Hessian": "100",
        "Gauss-Newton": "200",
        "L-BFGS": "500",
    }

    def __init__(self, ctrl):
        super().__init__(ctrl)

    def conv_1d(self, value, conversion):
        return self.estimator.convert([value], conversion)[0]

    def construct_1d_figure(self):
        super().construct_1d_figure(
            self.plot_frame,
            self.estimator.spectrum.real,
            cf.BGCOLOR,
        )

    def update_spectrum(self):
        data = copy.deepcopy(self.estimator.data)
        data = ne.sig.exp_apodisation(data, self.lb)
        data[0] *= 0.5
        self.spec_line.set_ydata(
            ne.sig.phase(
                ne.sig.ft(data),
                (self.p0["rad"],),
                (self.p1["rad"],),
                (self.pivot["idx"],),
            ).real
        )
        self.canvas_1d.draw_idle()
