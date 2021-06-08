Example Walkthrough
===================

This page provides an outline of basic NMR-EsPy usage through writing
Python code. If you wish to use the graphical user interface instead,
:doc:`follow this link <gui/index>`.

As an illustration of the typical steps involved in using NMR-EsPy, we will
consider an example dataset that ships with TopSpin 4. Assuming you installed
TopSpin in the default path, this should be present in the path:

* Linux: ``/opt/topspin4.x.y/examdata/exam1d_1H/1/pdata/1``
* Windows: ``C:\Bruker\TopSpin4.x.y\examdata\exam1d_1H\1\pdata\1``

In what follows, as I am using TopSpin 4.0.8, I shall be replacing
``topspin4.x.y`` with ``topspin4.0.8``.

I recommend that you follow this walkthrough using a Python
interpreter to ensure everything runs smoothly on your system.

Generating an Estimator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get started, it is necessary to import the :py:class:`nmrespy.core.Estimator`
class. A new instance of this class is initialised using the static
:py:meth:`~nmrespy.core.Estimator.new_bruker` method:

.. code:: python3

   >>> from nmrespy.core import Estimator
   >>> # Specify the path to the directory containing 1r
   >>> path = "/opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1"
   >>> estimator = Estimator.new_bruker(path)
   >>> type(estimator)
   <class 'nmrespy.core.Estimator'>

.. note::

   If you are using Windows, you should also import and initialise ``colorama``
   to ensure that coloured output is possible. If you do not, you may see a
   fair amount of gobbledygook:

   .. code:: python3

      >>> import colorama
      >>> colorama.init()

Information about the estimator can be seen by printing it:

.. code:: python3

   >>> print(estimator)
   <nmrespy.core.Estimator at 0x7f50e8ad8fa0>
   source : bruker_pdata
   data : [ 8237241.76470947      +0.j          1834272.48552552-9941412.67849912j
    -7908307.89165751+1371281.69800517j ...
           0.              +0.j                0.              +0.j
           0.              +0.j        ]
   dim : 1
   n : [32768]
   path : /opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1
   sw : [5494.50549450549]
   offset : [2249.20599998768]
   sfo : [500.132249206]
   nuc : ['1H']
   fmt : <i4
   filter_info : None
   result : None
   errors : None


An interactive plot of the data, in the frequency domain, can be seen using the
:py:meth:`~nmrespy.core.Estimator.view_data` method:

.. code:: python3

   >>> estimator.view_data()

.. only:: html

  .. image:: media/walkthrough/figures/view_data.png
     :align: center
     :scale: 80%

.. only:: latex

  .. image:: media/walkthrough/figures/view_data.png
    :align: center
    :width: 450


Frequency Filtration
^^^^^^^^^^^^^^^^^^^^

For complex NMR signals, it is typically necessary to consider a subset of
the frequency space at any time, otherwise the computational burden would be
too large. To overcome this, it is possible to derive a time-domain signal
which has been generated via frequency-filtration.

In this example, I am going to focus on the spectral region between
5.54-5.42ppm. The region looks like this:

.. only:: html

  .. image:: media/walkthrough/figures/spectral_region.png
     :align: center

.. only:: latex

  .. image:: media/walkthrough/figures/spectral_region.png
    :align: center
    :width: 450

To generate a frequency-filtered signal from the imported data, the
:py:meth:`~nmrespy.core.Estimator.frequency_filter` method is used. All well as
specifying the region of interest, it is also necessary to provide a region
that appears to contain no signals (this is used to gain an insight into the
data's noise variance). In this example, I will set this region to be -0.15 to
-0.3ppm.

.. code:: python3

   >>> estimator.frequency_filter([[5.54, 5.42]], [[-0.15, -0.3]])

.. note::

  Be aware of the use of two sets of parentheses around the regions specified.
  This may seem odd, but a nested list is required to ensure compatibility
  with 2D data as well.

Estimating the Signal Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matrix Pencil Method
--------------------

Now that a frequency filtered signal has been generated, we can begin the
estimation routine. Before estimating the signal parameters using nonlinear
programming (NLP), an initial guess of the parameters is required. We can derive
this guess using :py:meth:`~nmrespy.core.Estimator.matrix_pencil`:

.. code:: python3

   >>> estimator.matrix_pencil()
   ============================
   MATRIX PENCIL METHOD STARTED
   ============================
   --> Pencil Parameter: 358
   --> Hankel data matrix constructed:
   Size:   718 x 359
   Memory: 3.9331MiB
   --> Performing Singular Value Decomposition...
   --> Determining number of oscillators...
       Number of oscillators will be estimated using MDL
       Number of oscillations: 12
   --> Determining signal poles...
   --> Determining complex amplitudes...
   --> Checking for oscillators with negative damping...
       None found
    =============================
    MATRIX PENCIL METHOD COMPLETE
    =============================
    Time elapsed: 0 mins, 0 secs, 388 msecs

The result of the estimation is stored within the ``result`` attribute,
which can be accessed using :py:meth:`~nmrespy.core.Estimator.get_result`.

Nonlinear Programming
---------------------

The ``result`` attribute is next subjected to a NLP routine using the
:py:meth:`~nmrespy.core.Estimator.nonlinear_programming` method. As the
frequency-filtered data was derived from well-phased spectral
data, the optional ``phase_variance`` argument is set to ``True``. The
optimisation routine will then ensure that the estimate's phases are similar to
each other (and hopefully very close to 0), and will often remove excessive
oscillators from the Matrix Pencil result (note that our initial guess in
this example contains 12 oscillators).

.. code:: python3

   >>> estimator.nonlinear_programming(phase_variance=True)
   =============================
   NONLINEAR PROGRAMMING STARTED
   =============================
   | niter |f evals|CG iter|  obj func   |tr radius |   opt    |  c viol  | penalty  |CG stop|
   |-------|-------|-------|-------------|----------|----------|----------|----------|-------|
   |   1   |   1   |   0   | +1.6287e-01 | 1.00e+00 | 9.30e-02 | 0.00e+00 | 1.00e+00 |   0   |
   |   2   |   2   |   1   | +9.0652e-02 | 7.00e+00 | 6.92e-01 | 0.00e+00 | 1.00e+00 |   2   |
   |   3   |   3   |   9   | +9.0652e-02 | 7.00e-01 | 6.92e-01 | 0.00e+00 | 1.00e+00 |   3   |


   --snip--

   |  100  |  100  |  966  | +6.4830e-04 | 1.27e-01 | 2.56e-03 | 0.00e+00 | 1.00e+00 |   2   |

   --snip--

   Negative amplitudes detected. These oscillators will be removed
   Updated number of oscillators: 9
   | niter |f evals|CG iter|  obj func   |tr radius |   opt    |  c viol  | penalty  |CG stop|
   |-------|-------|-------|-------------|----------|----------|----------|----------|-------|
   |   1   |   1   |   0   | +1.2497e-03 | 1.00e+00 | 1.08e-01 | 0.00e+00 | 1.00e+00 |   0   |

   --snip--

   |  100  |  100  | 2228  | +8.5451e-04 | 9.95e+00 | 2.47e-05 | 0.00e+00 | 1.00e+00 |   1   |

   --snip--

   ==============================
   NONLINEAR PROGRAMMING COMPLETE
   ==============================
   Time elapsed: 0 mins, 3 secs, 186 msecs

The ``result`` attribute has now been updated with the result obtained using
NLP. The routine also computes the errors associated with each parameter,
which can be accessed with :py:meth:`~nmrespy.core.Estimator.get_errors`.

Saving Results
^^^^^^^^^^^^^^

Writing Results to a Text File/PDF/CSV
--------------------------------------

The estimation result can be written to ``.txt``, ``.pdf`` and ``.csv``
formats, using the :py:meth:`~nmrespy.core.Estimator.write_result` method.

.. code:: python3

  >>> msg = "Example estimation result for NMR-EsPy docs."
  >>> for fmt in ["txt", "pdf", "csv"]:
  ...     estimator.write_result(path="example", description=msg, fmt=fmt)
  ...
  Saved result to /<pwd>/example.txt
  Result successfully output to:
  /<pwd>/example.pdf
  If you wish to customise the document, the TeX file can be found at:
  /<pwd>/example.tex
  Saved result to /<pwd>/example.csv

.. only:: html

   The files generated are as follows:

   * :download:`example.txt <media/walkthrough/example.txt>`
   * :download:`example.tex <media/walkthrough/example.tex>`
   * :download:`example.pdf <media/walkthrough/example.pdf>`
   * :download:`example.csv <media/walkthrough/example.csv>`


.. note::

   In order to generate PDF files, you will need a LaTeX installation on
   your system. See the documentation for the
   :py:func:`nmrespy.write.write_result` function for more details.

Generating Result Figures
-------------------------

To generate a figure of the result, you can use the
:py:meth:`~nmrespy.core.Estimator.plot_result` method, which utilises
`matplotlib <https://matplotlib.org/>`_. There is wide scope for customising
the plot, which is described in detail in
:doc:`Figure Customisation <misc/figure_customisation>`.
See `Summary`_ below for an example of some basic plot customisation.

.. code:: python3

   >>> plot = estimator.plot_result()
   >>> plot.fig.savefig("plot_example.png")

.. only:: html

   * :download:`example_plot.png <media/walkthrough/figures/plot_example.png>`

Pickling Estimator Instances
----------------------------

The estimator instance can be serialised, and saved to a byte stream using
Python's `pickle <https://docs.python.org/3/library/pickle.html>`_ module,
with :py:meth:`~nmrespy.core.Estimator.to_pickle`:

.. code::

   >>> estimator.to_pickle(path="pickle_example")
   Saved instance of Estimator to /<pwd>/pickle_example.pkl

The estimator can subsequently be recovered using
:py:meth:`~nmrespy.core.Estimator.from_pickle`:

.. code:: python3

   >>> estimator_cp = Estimator.from_pickle(path="pickle_example")
   >>> type(estimator_cp)
   <class 'nmrespy.core.Estimator'>

Saving a Logfile
----------------

A summary of the methods applied to the estimator can be saved using the
:py:meth:`~nmrespy.core.Estimator.save_logfile` method:

.. code:: python3

   >>> estimator.save_logfile(path="logfile_example")
   Log file successfully saved to /<pwd>/logfile_example.log

.. only:: html

   * :download:`logfile_example.log <media/walkthrough/logfile_example.log>`

Summary
^^^^^^^

A script which performs the entire procedure described above is as follows.
Note that further customisation has been applied to the plot to give it an
"aesthetic upgrade".

.. code:: python3

    from nmrespy.core import Estimator

    # Path to data. You'll need to change the 4.0.8 bit if you are using a
    # different TopSpin version.

    # --- UNIX users ---
    path = "/opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1"

    # --- Windows users ---
    # path = "C:/Bruker/TopSpin4.0.8/examdata/exam1d_1H/1/pdata/1"

    estimator = Estimator.new_bruker(path)

    # --- Frequency filter & estimate ---
    estimator.frequency_filter([[5.54, 5.42]], [[-0.15, -0.3]])
    estimator.matrix_pencil()
    estimator.nonlinear_programming(phase_variance=True)

    # --- Write result files ---
    msg = "Example estimation result for NMR-EsPy docs."
    for fmt in ["txt", "pdf", "csv"]:
        estimator.write_result(path="example", description=msg, fmt=fmt)

    # --- Plot result ---
    # Set oscillator colours using the viridis colourmap
    plot = estimator.plot_result(oscillator_colors='viridis')
    # Shift oscillator labels
    # Move the labels associated with oscillators 1, 2, 5, and 6
    # to the right and up.
    plot.displace_labels([1,2,5,6], (0.02, 0.01))
    # Move the labels associated with oscillators 3, 4, 7, and 8
    # to the left and up.
    plot.displace_labels([3,4,7,8], (-0.04, 0.01))
    # Move oscillators 9's label to the right
    plot.displace_labels([9], (0.02, 0.0))

    # Save figure as a PNG
    plot.fig.savefig("plot_example_edited.png")

    # Save the estimator to a binary file.
    estimator.to_pickle(path="pickle_example")

    # Save a logfile of method calls
    estimator.save_logfile(path="logfile_example")

.. only:: html

   * :download:`nmrespy_example.py <media/walkthrough/nmrespy_example.py>`
   * :download:`plot_example_edited.png <media/walkthrough/figures/plot_example_edited.png>`

More features are provided by the :py:class:`~nmrespy.core.Estimator` beyond
what is described on this page, but this gives an overview of the primary
functionality.
