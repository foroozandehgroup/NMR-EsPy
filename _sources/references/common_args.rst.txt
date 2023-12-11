Common Arguments
================

Listed here are descriptions of arguments which commonly appear in methods of the
various estimator classes.

.. _INDICES:

indices
-------

A list of the indices of results to consider. Index ``0`` corresponds to the
first result obtained using the estimator, ``1`` corresponds to the next, etc.
You can also you negative ints for backward-indexing. For example ``-1``
corresponds to the last result acquired. If ``None``, all results will be
considered.

Suppose you have called the estimator's ``estimate`` method 3 times:

.. code:: python3

    estimator.estimate(region=(5., 4.5), ...)
    estimator.estimate(region=(3., 2.5), ...)
    estimator.estimate(region=(1., 0.5), ...)

With ``indices=[0]``, only the result corresponding to the 5-4.5 region will be
considered. With ``indices=[1, 2]``, both the results corresponding to the 3-2.5
and 1-0.5 regions will be considered. With ``indices=[-1]``, only the
result corresponding to 1-0.5 will be considered.

.. _INDEX:

index
-----

An integer denoting the estimation result to consider. See :ref:`INDICES` for more
details.

.. _COLOR_CYCLE:

color cycle
-----------

The following is a complete list of options:

* If a `valid matplotlib colour
  <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ is
  given, all multiplets will be given this color.
* If a string corresponding to a `matplotlib colormap
  <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
  is given, the multiplets will be consecutively shaded by linear
  increments of this colormap.
* If an iterable object containing valid matplotlib colors is
  given, these colors will be cycled.
  For example, if ``oscillator_colors = ['r', 'g', 'b']``:

  + Multiplets 0, 3, 6, ... would be :red:`red (#FF0000)`
  + Multiplets 1, 4, 7, ... would be :green:`green (#008000)`
  + Multiplets 2, 5, 8, ... would be :blue:`blue (#0000FF)`

* If ``None``, the default colouring method will be applied, which
  involves cycling through the following colors:

  + :oscblue:`#1063E0`
  + :oscorange:`#EB9310`
  + :oscgreen:`#2BB539`
  + :oscred:`#D4200C`

.. _XAXIS_TICKS:

xaxis_ticks
-----------

Specifies custom x-axis ticks for each region, overwriting the default
ticks. Should be of the form: ``[(i, (a, b, ...)), (j, (c, d, ...)), ...]``
where ``i`` and ``j`` are ints indicating the region under consideration,
and ``a``-``d`` are floats indicating the tick values.
