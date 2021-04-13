Figure Customisation
====================

Unfortunately, the GUI does not provide support for customising the result
figure beyond setting its size and dpi. I hope to provide support for more
customisation in a later version. The only way to customise the figure
is by writing a Python script which achieves this. This page outlines how this
can be done. It will help to have some experience with
`matplotlib <https://matplotlib.org/stable/index.html>`_
to perform customisations to your figure.

Getting Started
^^^^^^^^^^^^^^^

When saving the result of an estimation routine, make sure you "pickle" the
estimator (see the section *Saving the Result* on :doc:`this page <gui_usage>`).

You will now have to create a Python script, which will perform the following:

* Load the estimator
* Generate a figure of the estimator
* Customise the properties of the figure
* Save the figure

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

Generate the Estimator Figure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate the estimator figure, add the following line:

.. code:: python3

   plot = estimator.plot_result()

Giving the :py:meth:`~nmrespy.core.Estimator.plot_result` method no arguments
will produce a figure which is identical to the one produced by using the GUI.
There are some things that you can customise by providing arguments to the
method. Notable things that can be tweaked are:

* Whether or not to include a plot of the model (sum of individual oscillators)
  and/or a plot of the residual (difference between the data and the model).
* The colours of the data, residual, model, and individual oscillators.
* The vertical positioning of the residual and the model.
* Whether or not to include oscillator labels
* If you are familiar with matplotlib and the use of stylesheets, you can also
  specify a path to a stylesheet, enabling highly tunable customisation.

Have a look at the :py:func:`nmrespy.plot.plot_result` function for a
description of the acceptable arguments. Only the following arguments should
be used:

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

Customise the Figure
^^^^^^^^^^^^^^^^^^^^

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
* For each oscillator, the corresponding line is given a numerical value
  (i.e. keys will be ``1``, ``2``, to ``<M>`` where ``<M>`` is the number of
  oscillators in the result.)

i.e. to access the line object corresponding to oscillator 6, and change
its line-width to 2 px, you should use ``plot.lines[6].set_linewidth(2)``.

The `labels` dictionary will possess numerical keys from ``1`` to ``<M>``
(same as oscillator keys in the `lines` dictionary).

Specific Things
---------------

I have written some "convenience methods" to achieve certain things that
I anticipate users will frequently want to carry out.

* **Re-positioning oscillator labels**
  Often, the numerical labels that are given to each oscillator in the plot
  can overlap with other items in the figure, which is not ideal. To
  re-position the oscillator labels, use the
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

  Note that the size of displacement is given using the
  `axes co-ordinate system <https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html#axes-coordinates>`_

  The result in the above example is the following:

  .. image:: ../_static/gui/displace.png
     :align: center
     :scale: 30%

* **Appending a result plot to a subplot**
  You may wish to have the result plot as a subplot amongst other plots that
  make up a figure. The :py:meth:`~nmrespy.plot.NmrespyPlot.transfer_to_axes`
  method enables this to be achieved very easily. Here is a simple example:

  .. code:: python3

      from nmrespy.core import Estimator
      import numpy as np
      import matplotlib.pyplot as plt

      # Load the estimation result and create the plot (see above)
      path = "/path/to/estimator"
      estimator = Estimator.from_pickle(path)
      plot = estimator.plot_result()

      # Create a simple figure with two subplots
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

      plt.show()

  The resulting figure is as follows:

  .. image:: ../_static/gui/subplot.png
     :align: center
     :scale: 30%

If there are other features that you would like to see added to the
:py:meth:`~nmrespy.plot.NmrespyPlot` class to improve the convenience
of generating figures, feel free to make a
`pull request <https://github.com/foroozandehgroup/NMR-EsPy/pulls>`_,
or `get in contact <mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query>`_.
