Theory Overview
===============

On this page, I provide a brief overview of the goal of NMR-EsPy and how
it attempts to achieve this. If you simply wish to use NMR-EsPy without
worrying about the deatils of how it works, feel free to skip this.

The signal produced as a result of a typical NMR experiment (FID) can be
thought of as a summation of a number of complex sinusoidal oscillators, in
the presence of experimental noise. For the general case of a signal with
:math:`D` dimensions, we expect the function form of the signal at any time
point from the start of acquisition onwards to be:

.. math::
   y(t_1, \\dots, t_D) = \\sum_{m=1}^{M}
   \\left\\lbrace a_m \\exp\\left(\\mathrm{i} \\phi_m\\right)
   \\prod_{d=1}^{D} \\exp\\left[\\left(2 \\pi \\mathrm{i} f_{d,m} -
   \\eta_{d,m}\\right)t_d\\right]\\right\\rbrace + w(t_1, \\dots, t_D),

where

* :math:`t_d` is the time cosidered in the :math:`d`\-th dimension
* :math:`M` is the number of oscillators
* :math:`a_m` is the amplitude of oscillator :math:`m`
* :math:`\phi_m` is the phase of oscillator :math:`m`
* :math:`f_{d,m}` is the frequency of oscillator :math:`m` in the
  :math:`d`\-th dimension
* :math:`\eta_{d,m}` is the damping factor of oscillator :math:`m` in the
  :math:`d`\-th dimension
* :math:`w(t_1, \dots, t_D)` is the contribution from experimental noise

Of course, NMR signals are digitised, and so the raw output of an experiment
will be a :math:`D`-dimensional array :math:`\boldsymbol{Y}` of a finite
shape:

.. math::
   \boldsymbol{Y} \in \mathbb{C}^{N_1 \times \dots \times N_D},

where :math:`N_d` is the number of points sampled in the :math:`d`\-th
dimension. Assuming that the signal is uniformly sampled in each dimesnion,
the time at which any point is sampled is given by

.. math::
   \tau_{n}^{(d)} = n_d \Delta t_d, \quad n_d \in \{0, 1, \dots, N_d - 1\},

where :math:`\Delta t_d` is the sampling rate (the time between successive
samples) in dimension :math:`d`. The discrete fid :math:`\boldsymbol{Y}`
therefore has elements `Y\left[n_1, \dots, n_D\right]` of the form

.. math::
  Y\left[n_1, \dots, n_D\right] = \sum_{m=1}^{M} \left\lbrace a_m
  \exp\left(\mathrm{i} \phi_m\right) \prod_{d=1}^{D} \exp\left[\left(2 \pi
  \mathrm{i} f_{d,m} - \eta_{d,m}\right) \tau_n^{(d)}\right]\right\rbrace
  + W(n_1, \dots, n_D)

Writing this in a more succinct notation:

.. math::
   \boldsymbol{Y} = \boldsymbol{X}\left(\boldsymbol{\theta}\right) +
   \boldsymbol{W}

where the vector :math:`\boldsymbol{\theta} \in \mathbb{R}^{(2+2D)M}` cotains
all the deterministic parameters which describe the signal

.. math::
   \boldsymbol{\theta} = \left[a_1, a_2, \dots, a_M, \phi_1, \dots, \phi_M,
   f_{1,1}, \dots, f_{1,M}, \dots, f_{D,1}, \dots, f_{D,M}, \eta_{1,1}, \dots
   \eta_{D,M}\right]^{\mathrm{T}}

The goal of NMR-EsPy is to estimate :math:`\boldsymbol{\theta}` for a given
signal. This is rather challenging, especially for signals with many resonances,
as one doesn't even know how many oscilltors are contained within the signal in
general (:math:`M`).

To achieve this, we apply `Newton's Method` to determine the optimal set of
parameters to describe the signal. Newton' Method is an iterative procedure,
which considers a particular function to be minimised. In the case of NMR-EsPy,
the function is the following:

.. math::
   \mathcal{F}(\boldsymbol{\theta}) = \lVert \tilde{\boldsymbol{Y}} -
   \boldsymbol{X}(\boldsymbol{\theta}) \rVert_2^2

where :math:`\tilde{\boldsymbol{Y}} = \boldsymbol{Y} / \lVert \boldsymbol{Y}
\rVert`. THere are numerous variants of Newton's method, but the general idea
is to approximate the neighbourhood about the current value of
:math:`\boldsymbol{\theta}` to be quadratic, and to determine a step of a
certain direction and size such that the value of
:math:`\mathcal{F}(\boldsymbol{\theta})` is reduced. This procedure is repeated
multiple times until the routine converges at a minimum.

Depending on the circumstances, it is often worthwhile to minimise the variance
of phases in the FID's estimate. NMR-EsPy also allows minimisation of the
following function to achieve this:

.. math::
   \mathcal{F}^{\phi}(\boldsymbol{\theta}) = \lVert \tilde{\boldsymbol{Y}} -
   \boldsymbol{X}(\boldsymbol{\theta}) \rVert_2^2 + \mathrm{Var}
   (\boldsymbol{\phi})
