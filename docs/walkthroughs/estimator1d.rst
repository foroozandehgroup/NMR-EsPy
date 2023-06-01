.. _ESTIMATOR1D:

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
    >>>     sw=1.2 * sfo,  # 1ppm
    >>>     offset=4.1 * sfo,  # 4.1ppm
    >>>     sfo=sfo,
    >>>     snr=40.,
    >>> )

.. note::

    For the rest of this tutorial, we will be using the estimator created in
    the above code snippet.

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
    >>> sw = 1.2 * sfo
    >>> estimator = ne.Estimator1D.new_spinach(
    >>>     shifts=shifts,
    >>>     couplings=couplings,
    >>>     pts=2048,
    >>>     sw=sw,
    >>>     offset=offset,
    >>>     sfo=sfo,
    >>> )

Viewing and accessing the dataset
---------------------------------

You can inspect the data associated with the estimator with
:py:meth:`~nmrespy.Estimator1D.view_data`

.. code:: pycon

    >>> estimator.view_data(freq_unit="ppm")

.. image:: ../media/estimator_1d_view_data.png
   :align: center

You can acquire the time-domain data with :py:meth:`~nmrespy.Estimator1D.data`,
the associated timepoints can be retrieved using
:py:meth:`~nmrespy.Estimator1D.get_timepoints`.  The spectral data is accessed
with :py:meth:`~nmrespy.Estimator1D.spectrum`, and the corresponding chemical
shifts  with :py:meth:`~nmrespy.Estimator1D.get_shifts`.

.. todo::

   Provide an example of using these methods

Estimating the dataset
----------------------

The generation of parameters estimates for the dataset is facilitated using the
:py:meth:`~nmrespy.Estimator1D.estimate` method. In most scenarios, your
dataset will possess too many oscillators for it to be feasible computationally
to estimate the entire signal at once. For this reason, NMR-EsPy generates
frequency-filtered "sub-FIDs" to break the problem down into more manageable
chunks. To create suitable sub-FIDs, it is important to select regions where
the bounds are placed at points that comprise the baseline. As well as this, a
region that comprises just the baseline must be indicated. In the figure below,
the red region would be inappropriate as it slices through signal. The green
region is acceptable, as the bounds are located on the baseline. Finally, the
grey region is a suitable noise region as it contains only baseline.

.. image:: ../media/good_bad_noise_regions.png
   :align: center

For our dataset, we will estimate three regions, comprising each multiplet
structure in the spectrum. A region should be given as a tuple of 2 floats,
specifying the left and right boundaries of the region of interest (the order
of these doesn't matter). By default, these are assumed to be given in Hz,
unless ``region_unit`` is set to ``"ppm"``.

.. code:: pycon

    >>> regions = [(4.6, 4.4), (4.02, 3.82), (3.8, 3.6)]
    >>> noise_region = (4.3, 4.25)
    >>> for region in regions:
    >>>     estimator.estimate(
    >>>         region=region, noise_region=noise_region, region_unit="ppm",
    >>>     )

Inspecting estimation results
-----------------------------

.. note::

    **Result Indices**

    Each time the :py:meth:`~nmrespy.Estimator1D.estimate` method is called, the
    result is appended to a list of all recorded results. For many methods that use
    estimation results, an argument called ``indices`` exists. This lets you specify
    the results you are interested in. By default all results will be used
    (``indices = None``). A subset of the results can be considered by including a
    list of integers. For example ``indices = [0, 2]`` would mean only the 1st
    and 3rd results acquired with the estimator are considered.

A NumPy array of the generated results can be acquired using
:py:meth:`~nmrespy.Estimator1D.get_params`. The corresponding errors associated
with each parameters are obtained with
:py:meth:`~nmrespy.Estimator1D.get_errors`.

.. code:: pycon

    >>> # All params, frequencies in Hz:
    >>> estimator.get_params()
    [[ 1.0018e+00  1.5921e-03  1.8356e+03  7.0187e+00]
     [ 1.0003e+00  2.4881e-03  1.8442e+03  6.9968e+00]
     [ 1.0024e+00  1.5817e-03  1.8558e+03  7.0281e+00]
     [ 1.0008e+00  9.1591e-04  1.8644e+03  7.0007e+00]
     [ 1.0022e+00  7.1936e-04  1.9386e+03  7.0109e+00]
     [ 9.9470e-01 -7.4609e-04  1.9588e+03  6.9866e+00]
     [ 1.0080e+00 -1.0112e-03  1.9612e+03  7.0448e+00]
     [ 1.0009e+00 -7.1398e-04  1.9814e+03  7.0131e+00]
     [ 1.0003e+00  1.1306e-03  2.2344e+03  7.0095e+00]
     [ 1.0011e+00  6.0150e-04  2.2430e+03  7.0011e+00]
     [ 9.9902e-01  2.8231e-04  2.2570e+03  6.9856e+00]
     [ 1.0004e+00 -1.8229e-03  2.2656e+03  7.0057e+00]]
    >>> # All errors, frequencies in Hz
    >>> estimator.get_errors()
    [[0.0013 0.0013 0.0019 0.0121]
     [0.0014 0.0014 0.002  0.0124]
     [0.0014 0.0014 0.002  0.0125]
     [0.0013 0.0013 0.0019 0.012 ]
     [0.0012 0.0012 0.0018 0.0114]
     [0.0036 0.0036 0.0034 0.0212]
     [0.0036 0.0036 0.0034 0.0213]
     [0.0012 0.0012 0.0018 0.0114]
     [0.0013 0.0013 0.0019 0.0116]
     [0.0013 0.0013 0.0019 0.0118]
     [0.0013 0.0013 0.0019 0.0118]
     [0.0013 0.0013 0.0018 0.0116]]
    >>> # Params for first region, frequencies in ppm
    >>> estimator.get_params(indices=[0], funit="ppm")
    [[ 1.0003e+00  1.1306e-03  4.4688e+00  7.0095e+00]
     [ 1.0011e+00  6.0150e-04  4.4860e+00  7.0011e+00]
     [ 9.9902e-01  2.8231e-04  4.5140e+00  6.9856e+00]
     [ 1.0004e+00 -1.8229e-03  4.5312e+00  7.0057e+00]]
    >>> # Params for second and third regions, split up
    >>> estimator.get_params(indices=[1, 2], merge=False, funit="ppm")
    [array([[ 1.0022e+00,  7.1936e-04,  3.8772e+00,  7.0109e+00],
           [ 9.9470e-01, -7.4609e-04,  3.9176e+00,  6.9866e+00],
           [ 1.0080e+00, -1.0112e-03,  3.9224e+00,  7.0448e+00],
           [ 1.0009e+00, -7.1398e-04,  3.9628e+00,  7.0131e+00]]), array([[1.0018e+00, 1.5921e-03, 3.6712e+00, 7.0187e+00],
           [1.0003e+00, 2.4881e-03, 3.6884e+00, 6.9968e+00],
           [1.0024e+00, 1.5817e-03, 3.7116e+00, 7.0281e+00],
           [1.0008e+00, 9.1591e-04, 3.7288e+00, 7.0007e+00]])]

Writing result tables
^^^^^^^^^^^^^^^^^^^^^

Tables of parameters can be saved to ``.txt`` and ``.pdf`` formats. using
:py:meth:`~nmrespy.Estimator1D.write_result`. For PDF generation, you will
need a working LaTeX installation. See the :ref:`installation instructions
<LATEX_INSTALL>`.

.. code::

    >>> for fmt in ("txt", "pdf"):
    >>>     estimator.write_result(
    >>>         path="tutorial_1d",
    >>>         fmt=fmt,
    >>>         description="Simulated 2,3-Dibromopropanoic acid signal.",
    >>>     )
    Saved file tutorial_1d.txt.
    Saved file tutorial_1d.tex.
    Saved file tutorial_1d.pdf.
    You can view and customise the corresponding TeX file at tutorial_1d.tex.

.. only:: html

    * :download:`tutorial_1d.txt <../downloads/tutorial_1d.txt>`: Text file.
    * :download:`tutorial_1d.pdf <../downloads/tutorial_1d.pdf>`: PDF file.
    * :download:`tutorial_1d.tex <../downloads/tutorial_1d.tex>`: TeX file used to generate the PDF

Creating result plots
^^^^^^^^^^^^^^^^^^^^^

Figures giving an overview of the estimation result can be generated using

.. code::

    >>> for (txt, indices) in zip(("complete", "index_1"), (None, [1])):
    >>>     fig, ax = estimator.plot_result(
    >>>         indices=indices,
    >>>         figure_size=(4.5, 3.),
    >>>         region_unit="ppm",
    >>>         axes_left=0.03,
    >>>         axes_right=0.97,
    >>>         axes_top=0.98,
    >>>         axes_bottom=0.09,
    >>>     )
    >>>     fig.savefig(f"tutorial_1d_{txt}_fig.pdf")

.. only:: html

    * :download:`tutorial_1d_complete_fig.pdf
      <../downloads/tutorial_1d_complete_fig.pdf>`:
      result for all regions considered
    * :download:`tutorial_1d_index_1_fig.pdf
      <../downloads/tutorial_1d_index_1_fig.pdf>`: result for 2nd estimated
      region only (index 1)

Saving the estimator
^^^^^^^^^^^^^^^^^^^^

The estimator object itself can be saved and reloaded for future use with the
:py:meth:`~nmrespy.Estimator1D.to_pickle` and
:py:meth:`~nmrespy.Estimator1D.from_pickle` methods, respectively:

.. code::

    >>> estimator.to_pickle("tutorial_1d")
    Saved file tutorial_1d.pkl.
    >>> # Load the estimator and save to the `estimator_cp` variable
    >>> estimator_cp = ne.Estimator1D.from_pickle("tutorial_1d")

.. only:: html

    * :download:`tutorial_1d.pkl <../downloads/tutorial_1d.pkl>`

Saving a logfile
^^^^^^^^^^^^^^^^

A logfile listing all the methods called on the estimator can be saved using
:py:meth:`~nmrespy.Estimator1D.save_log`:

.. code::

    >>> estimator.save_log("tutorial_1d")
    Saved file tutorial_1d.log.

.. only:: html

    * :download:`tutorial_1d.log <../downloads/tutorial_1d.log>`

Editing parameter results
-------------------------

**TODO**
