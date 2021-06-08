Figure Customisation
====================

On this page, you can find out how to customise result figures to suit your
preferences. It will help to have some experience with
`matplotlib <https://matplotlib.org/stable/index.html>`_.

Editing result figures is the only major feature that the NMR-EsPy GUI does
not provide support for, so if you use the GUI and want to make changes to the
figure you'll have to do this manually with a Python script I'm afraid. I hope
to provide support for figure customisation within the GUI in a later version.

.. note::

  The next two sections are intended for GUI users. If you manually write
  scripts for the full estimation process, you can jump to
  `Generate the Estimation Figure`_

Getting Started
^^^^^^^^^^^^^^^

If you are using the GUI, when saving the result of an estimation routine, make
sure you "pickle" the estimator (see the section *Saving the Result* on
:doc:`this page <../gui/usage/result>`).

You will now have to create a Python script, which will perform the following:

* Load the estimator (an instance of the :py:class:`~nmrespy.core.Estimator` class).
* Generate a figure of the estimator using the
  :py:class:`~nmrespy.core.Estimator.plot_result` method.
* Customise the properties of the figure.
* Save the figure.

At the top of the script, place the following:

.. code:: python3

   #!/usr/bin/python3
   from nmrespy.core import Estimator

Load the Estimator
^^^^^^^^^^^^^^^^^^

To load the estimator, include the following line:

.. code:: python3

   estimator = Estimator.from_pickle("path/to/estimator")

where ``path\to\estimator`` is the full or relative path to the saved estimator
file. **DO NOT INCLUDE THE** ``.pkl`` **EXTENSION**.

.. _Generate the Estimation Figure:

Generate the Estimation Figure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate the estimator figure, add the following line:

.. code:: python3

   plot = estimator.plot_result()

Giving the :py:meth:`~nmrespy.core.Estimator.plot_result` method no arguments
will produce a figure which is identical to the one produced by using the GUI.
There are some things that you can customise by providing arguments to the
method. Notable things that can be tweaked are:

* The frequency unit of the spectrum (available options are ppm (default) or Hz).
* Whether or not to include a plot of the model (sum of individual oscillators)
  and/or a plot of the residual (difference between the data and the model).
* The colours of the data, residual, model, and individual oscillators.
* The vertical positioning of the residual and the model.
* Whether or not to include oscillator labels
* If you are familiar with matplotlib and the use of stylesheets, you can also
  specify a path to a stylesheet. One thing to note with stylesheets is that
  even if you have a colour cycle specified in the sheet, it will be overwritten,
  so you must still manually specify the desired oscillator colours if you want
  these to not be the default colours.

Have a look at the :py:func:`nmrespy.plot.plot_result` function for a
description of the acceptable arguments. Only the following arguments should
be used (the others are automatically determined internally by the
:py:meth:`nmrespy.core.Estimator.plot_result` method):

* `shifts_unit`
* `plot_residual`
* `plot_model`
* `residual_shift`
* `model_shift`
* `data_color`
* `oscillator_colors`
* `residual_color`
* `model_color`
* `labels`
* `stylesheet`

Further Customisation
^^^^^^^^^^^^^^^^^^^^^

If there are further features you would like to customise, this can be done
by editing ``plot``.

General Guidance
----------------

:py:meth:`nmrespy.core.Estimator.plot_result` produces an instance of the
:py:class:`nmrespy.plot.NmrespyPlot` class. This possesses four notable
attributes:

* `fig` (`matplotlib.figure.Figure <https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.figure.Figure.html>`_) the overall figure.
* `ax` (`matplotlib.axes._subplots.AxesSubplot <https://matplotlib.org/3.3.1/api/axes_api.html#the-axes-class>`_) the figure axes.
* `lines` (A dictionary of `matplotlib.lines.Line2D <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
  objects).
* `labels` (A dictionary of `matplotlib.text.Text <https://matplotlib.org/stable/api/text_api.html?highlight=text#matplotlib.text.Text>`_
  objects).

The `lines` dictionary will possess the following keys:

* `'data'`
* `'residual'` (if the residual was chosen to be plotted)
* `'model'` (if the model was chose to be plotted)
* For each oscillator, the corresponding line is given a numerical value as
  its key (i.e. keys will be ``1``, ``2``, to ``<M>`` where ``<M>`` is the
  number of oscillators in the result.)

As an example, to access the line object corresponding to oscillator 6, and
change its line-width to 2 px, you should use
``plot.lines[6].set_linewidth(2)``.

The `labels` dictionary will possess numerical keys from ``1`` to ``<M>``
(same as oscillator keys in the `lines` dictionary).

Specific Things
---------------

I have written some "convenience methods" to achieve certain things that
I anticipate users will frequently want to carry out.

* **Re-positioning oscillator labels**
  Often, the automatically-assigned positions of the numerical labels that are
  given to each oscillator can overlap with other items in the figure, which is
  not ideal. To re-position the oscillator labels, use the
  :py:meth:`~nmrespy.plot.NmrespyPlot.displace_labels` method:

  .. code:: python3

     # Load the estimation result and create the plot (see above)
     path = "/path/to/estimator"
     estimator = Estimator.from_pickle(path)
     plot = estimator.plot_result()

     # Shift the labels for oscillators 1, 2, 5 and 6
     # to the right and up
     plot.displace_labels([1,2,5,6], (0.02, 0.01))

     # Shift the labels for oscillators 3, 4, 7 and 8
     # to the left and up
     plot.displace_labels([3,4,7,8], (-0.05, 0.01))

  .. note::

     The size of displacement is given using the
     `axes co-ordinate system <https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html#axes-coordinates>`_

  The plot before and after shifting the label positions are as follows:

  .. only:: html

    .. image:: ../media/misc/displace.png
      :align: center
      :scale: 30%

  .. only:: latex

    .. image:: ../media/misc/displace.png
      :align: center
      :scale: 70%

* **Appending a result plot to a subplot**
  You may wish to set the result plot as a subplot amongst other plots that
  make up a figure. The :py:meth:`~nmrespy.plot.NmrespyPlot.transfer_to_axes`
  method enables this to be achieved very easily. Here is a simple example:

  .. code:: py

    from nmrespy.core import Estimator
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the estimation result and create the plot (see above)
    path = "estimator"
    estimator = Estimator.from_pickle(path)
    plot = estimator.plot_result()

    # Create a simple figure with two subplots (side-by-side)
    fig = plt.figure(figsize=(8, 4))
    ax0 = fig.add_axes([0.05, 0.15, 0.4, 0.8])
    ax1 = fig.add_axes([0.55, 0.15, 0.4, 0.8])

    # Add some random stuff to ax1
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax1.plot(x, y, color='k')
    ax1.plot(0.7 * x[60:90], 0.7 * y[60:90], color='k')
    ax1.plot(0.05 * x - 0.4, 0.05 * y + 0.4, color='k')
    ax1.plot(0.05 * x + 0.4, 0.05 * y + 0.4, color='k')

    # Transfer contents of the result plot to ax0
    plot.transfer_to_axes(ax0)

    # Add label to each axes
    # N.B. This has been done after the call to `transfer_to_axes`
    # If it had been done beforehand, the a) label would have been cleared.
    for txt, ax in zip(('a)', 'b)'), (ax0, ax1)):
        ax.text(0.02, 0.92, txt, fontsize=14, transform=ax.transAxes)

    plt.show()

  The resulting figure is as follows:

  .. only:: html

    .. image:: ../media/misc/subplot.png
      :align: center
      :scale: 30%

  .. only:: latex

    .. image:: ../media/misc/subplot.png
      :align: center
      :scale: 70%

If there are other features that you would like to see added to the
:py:meth:`~nmrespy.plot.NmrespyPlot` class to increase the ease
of generating figures, feel free to make a
`pull request <https://github.com/foroozandehgroup/NMR-EsPy/pulls>`_,
or `get in touch <mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query>`_.

Save the Figure
^^^^^^^^^^^^^^^

Once you have completed all the desired customisation, the ``fig`` object can be
saved using `matplotlib.savefig <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>`_:

.. code:: python3

   plot.fig.savefig("<path/to/figure>", ...)
