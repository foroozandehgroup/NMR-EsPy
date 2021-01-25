# _filter.py
# frequecy filtration of NMR data using super_Gaussian functions
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# ==============================================
# SGH 25-1-21
# I have tagged various methods as follows:
#
# TODO: MAKE 2D COMPATIBLE
#
# These only support 1D data treatment currently
# ==============================================

import copy
import itertools

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import numpy.random as nrandom

import scipy.linalg as slinalg

class FrequencyFilter:

    def __init__(self, data, region, noise_region, p0, p1, cut, cut_ratio):

        self.data = data
        self.region = region
        self.noise_region = noise_region
        self.p0 = p0
        self.p1 = p1
        self.cut = cut
        self.cut_ratio = cut_ratio

        self._generate_spectrum()
        self._double_bounds()
        self._super_gaussian()
        self._sg_noise()
        self._filter_spectrum()
        self._generate_uncut_fid()

        if self.cut:
            self._cut_spectrum()

        self._generate_filtered_fid()


    def _generate_spectrum(self):
        """Zero fill, Fourier transform, flip, and phase data"""

        # TODO: MAKE 2D COMPATIBLE

        # zero fill data to double its size
        zf_data = np.hstack(
            (self.data, np.zeros(self.data.shape, dtype='complex'))
        )

        # fourier transform, flip, and phase data
        self.init_spectrum = np.real(self._phase(fftshift(fft(zf_data))[::-1]))


    def _double_bounds(self):
        """Double region and noise_region bound values. Required due to
        zero filling data to double its size"""

        db_region = []
        db_noise_region = []

        # double values of indices (as doubled size of signal)
        for reg_dim, nreg_dim in zip(self.region, self.noise_region):

            reg_sublst = []
            nreg_sublst = []

            for reg_bnd, nreg_bnd in zip(reg_dim, nreg_dim):
                reg_sublst.append(2 * reg_bnd)
                nreg_sublst.append(2 * nreg_bnd)

            db_region.append(reg_sublst)
            db_noise_region.append(nreg_sublst)

        self.region = db_region
        self.noise_region = db_noise_region


    def _phase(self, spectrum):
        """
        First order phase correction applied to Fourier transformed data.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Spectrum

        Returns
        -------
        spectrum_phased : numpy.ndarray
            Phased spectrum
        """

        # TODO: MAKE 2D COMPATIBLE

        n = spectrum.shape[0]
        p = np.exp(1j * self.p0) * np.exp(1j * self.p1 * np.arange(n) / n)
        return spectrum * p


    def _super_gaussian(self, p=40):
        """
        Generates a super-Gaussian for filtration of frequency-domian data.

        . math::
           g_n = \exp \left(-2^{p+1} \left(\frac{n - c}{b}\right)^p\right),
       		  \quad \forall n \in \{0, 1, \cdots, N-1\}

        Parameters
        ----------
        p : int, default: 40
            Power of the super-Gaussian.
        """

        # determine center and bandwidth of super gaussian
        self.center = []
        self.bw = []

        # loop over each dimension
        # determine center of region and bandwidth
        for bounds in self.region:
            self.center.append((bounds[0] + bounds[1]) / 2)
            self.bw.append(bounds[1] - bounds[0])

        # construct super gaussian
        for n, c, b in zip(self.init_spectrum.shape, self.center, self.bw):
            sg = np.exp(-2**(p+1) * ((np.arange(1, n+1) - c) / b)**p)
            try:
                self.super_gaussian = self.super_gaussian[..., None] * sg
            except AttributeError:
                self.super_gaussian = sg


    def _sg_noise(self):
        """Creates an array of noise whose variance at each point reflects
        the super-Gaussian amplitude"""

        # extract noise
        noise_slice = [np.s_[bnds[0]:bnds[1]] for bnds in self.noise_region]
        noise = self.init_spectrum[tuple(noise_slice)]

        # determine noise mean and variance
        mean = np.mean(noise)
        variance = np.var(noise)

        # construct synthetic noise
        n = self.super_gaussian.shape
        self.noise = np.zeros(n)

        # generate a nested loop covering all points in the array
        indices = []
        for n in self.super_gaussian.shape:
            indices.append(list(range(n)))

        for idx in itertools.product(*indices):
            point_var = variance * (1 - self.super_gaussian[idx])
            self.noise[idx] = nrandom.normal(0, np.sqrt(point_var))

        # correct for veritcal baseline shift
        # TODO consult Ali - should this be done?
        self.noise += mean * (1 - self.super_gaussian)

    def _filter_spectrum(self):

        self.filtered_spectrum = \
            self.init_spectrum * self.super_gaussian + self.noise


    def _cut_spectrum(self):

        cut_slice = []
        self.filtered_n = [] # number of points the cut signal is made of
        self.filtered_sw = [] # sweep width of the cut signal
        self.filtered_off = [] # offset of the cut signal

        # determine slice indices
        for n, c, b in zip(self.filtered_spectrum.shape, self.center, self.bw):

            # minimum and maximum array indices of cut signal
            min = int(np.floor(c - (b / 2 * self.cut_ratio)))
            max = int(np.ceil(c + (b / 2 * self.cut_ratio)))

            # ensure cut region remain within valid span of values (0 - N-1)
            if min < 0:
                min = 0
            if max > n:
                max = n

            cut_slice.append(np.s_[min:max])

            self.filtered_n.append(max - min)
            # sw and off will be converted to hz inside NMREsPyBruker
            self.filtered_sw.append(max - min)
            self.filtered_off.append((min + max) / 2)

        self.filtered_spectrum = self.filtered_spectrum[tuple(cut_slice)]


    def _generate_uncut_fid(self):

        # TODO: MAKE 2D COMPATIBLE

        self.uncut_fid = ifft(ifftshift(self.filtered_spectrum))
        half = tuple(np.s_[0:int(n // 2)] for n in self.uncut_fid.shape)
        self.uncut_fid = 2 * self.uncut_fid[half]
        self.uncut_norm = slinalg.norm(self.uncut_fid)


    def _generate_filtered_fid(self):

        # TODO: MAKE 2D COMPATIBLE
        #
        # Will have to check this ifft approach is correct for 2D data
        # I haven't tested it yet

        if self.cut:
            half = tuple(np.s_[0:int(n // 2)] for n in self.filtered_spectrum.shape)
            self.filtered_signal = ifft(ifftshift(self.filtered_spectrum))[half]
            self.cut_norm = slinalg.norm(self.filtered_signal)

            self.filtered_signal = \
                (self.uncut_norm / self.cut_norm) * self.filtered_signal

        else:
            self.filtered_signal = self.uncut_fid
