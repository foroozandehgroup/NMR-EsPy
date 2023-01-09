Using ``Estimator1D``
=====================

The :py:class:`nmrespy.Estimator1D` class is provided for the consideration of 1D NMR data.

Generating an instance
----------------------

There are a few ways to create a new instance of the estimator depending on the source of the data

Bruker data
^^^^^^^^^^^

It is possible to load both raw FID data and processed spectral data from
Bruker using :py:meth:`~nmrespy.Estimator1D.new_bruker`. All that is needed is
the path to the dataset:

1. If you wish to import FID data, set the path as ``"<path_to_data>/<expno>/"``.
   There should be an ``fid`` file and an ``acqus`` file directly under this
   directory. The data in the ``fid`` file will be imported, and the artefact
   from digital filtering is removed by a first-order phase shift.

   .. note::

       If you import FID data, there is a high chance that you will need to
       phase the data, and apply baseline correction before proceeding to run
       estimation. Look at :py:meth:`~nmrespy.Estimator1D.phase_data` and
       :py:meth:`~nmrespy.Estimator1D.baseline_correction`, respectively.

   .. code:: pycon

       >>> import nmrespy as ne
       >>> estimator = ne.Estimator1D.new_bruker("/home/simon/nmr_data/andrographolide/1")
       >>> estimator.phase_data(p0=2.653, p1=-5.686, pivot=13596)
       >>> estimator.baseline_correction()

2. To import the processed data, set the path as
   ``"<path_to_data>/<expno>/pdata/<procno>"``. There should be a ``1r`` file
   and a ``procs`` file directly under this directory. The data in ``1r`` will
   be Inverse Fourier Transformed, and the resulting time-domain signal is sliced
   so that only the first half is retained.

   .. note::

       It can be more convienient to provide processed data, even though the
       data will be converted to the time-domain for estimation, as you can
       then rely on TopSpin's automated processing scripts to phase and
       baseline correct. However, **you should not apply any window function to
       the data.** The ony exception is exponential damping, though if you
       don't need it, it's best you don't use it.

   .. code::

       >>> import nmrespy as ne
       >>> estimator = ne.Estimator1D.new_bruker("home/simon/nmr_data/andrographolide/1/pdata/1")
       >>> # Note there is no need for extra data-processing steps

Simulated data from a set of oscillator parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create an estimator with synthetic data constructed from known
parameters using :py:meth:`~nmrespy.Estimator1D.new_synthetic_from_parameters`
The parameters must be provided as a 2D NumPy array with ``params.shape[1] ==
4``. Each row should contain an oscillator's amplitude, phase (rad), frequency
(Hz), and damping factor (s⁻¹).

.. code:: pycon

    >>> import nmrespy as ne
    >>> import numpy as np
    >>> # Using frequencies of 2,3-Dibromopropanoic acid
    >>> params = np.array([
    >>>     [1, 0, 1864.4, 5],
    >>>     [1, 0, 1855.8, 5],
    >>>     [1, 0, 1844.2, 5],
    >>>     [1, 0, 1835.6, 5],
    >>>     [1, 0, 1981.4, 5],
    >>>     [1, 0, 1961.2, 5],
    >>>     [1, 0, 1958.8, 5],
    >>>     [1, 0, 1938.6, 5],
    >>>     [1, 0, 2265.6, 5],
    >>>     [1, 0, 2257.0, 5],
    >>>     [1, 0, 2243.0, 5],
    >>>     [1, 0, 2234.4, 5],
    >>> ])
    >>> sfo = 500.
    >>> estimator = ne.Estimator1D.new_synthetic_from_parameters(
    >>>     params=params,
    >>>     pts=2048,
    >>>     sw=1. * sfo,  # 1ppm
    >>>     offset=4.1 * sfo,  # 4.1ppm
    >>>     sfo=sfo,
    >>>     snr=40.,
    >>> )

Simulated data from Spinach
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you have installed the :ref:`relevant requirements <SPINACH_INSTALL>`,
you can create an instance with data simulated using Spinach with
:py:meth:`~nmrespy.Estimator1D.new_spinach`. It is necessary to provide:

* A list of floats for the chemical shifts of each nucleus
* A list with 3-element tuples of the form ``(spin1, spin2, coupling)`` for
  the couplings (N.B. the spin indices start at ``1`` rather than ``0``).
* An int for the number of datapoints
* A float for the sweep width.
* (Optionally) a float for the transmitter offset (Hz).
* (Optionally) a float for the transmitter frequency (MHz).
* (Optionally) a str for the nucleus identity.
* (Optionally) a float for the signal's approximate SNR in dB.
* (Optionally) a float for the exponential damping factor.

Note this may take some time in order to start-up MATLAB and run the simulation.

.. code:: pycon

    >>> import nmrespy as ne
    >>> # 2,3-Dibromopropanoic acid
    >>> shifts = [3.7, 3.92, 4.5]
    >>> couplings = [(1, 2, -10.1), (1, 3, 4.3), (2, 3, 11.3)]
    >>> sfo = 500.
    >>> offset = 4.1 * sfo  # Hz
    >>> sw = 1. * sfo
    >>> estimator = ne.Estimator1D.new_spinach(
    >>>     shifts=shifts,
    >>>     couplings=couplings,
    >>>     pts=2048,
    >>>     sw=sw,
    >>>     offset=offset,
    >>>     sfo=sfo,
    >>> )

Viewing the dataset
-------------------

You can inspect the data associated with the estimator with
:py:meth:`~nmrespy.Estimator1D.view_data`

.. code::

    >>> estimator.view_data(freq_unit="ppm")

.. image:: ../media/estimator_1d_view_data.png
   :align: center

You can acquire the time-domain data with :py:meth:`~nmrespy.Estimator1D.data`,
and the corresponding spectrum with :py:meth:`~nmrespy.Estimator1D.spectrum`.
