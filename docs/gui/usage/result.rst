Estimation Result
=================

Once the routine is complete, a new window will load with the following
appearance:

.. only:: html

  .. image:: ../../media/gui/windows/result_window.png
    :align: center

.. only:: latex

  .. image:: ../../media/gui/windows/result_window.png
    :align: center
    :width: 400


Featured in the result plot are:

* The data selected (black).
* Individual peaks that comprise the estimation result
  (:oscblue:`m`\ :oscorange:`u`\ :oscgreen:`l`\ :oscred:`t`\
  :oscblue:`i`\ :oscorange:`-`\ :oscgreen:`c`\ :oscred:`o`\
  :oscblue:`l`\ :oscorange:`o`\ :oscgreen:`u`\ :oscred:`r`\
  :oscblue:`e`\ :oscorange:`d`).
  Each of these is given a numerical label.
* The residual between the data and the model (:grey:`grey`).

Tweaking and Re-optimising the result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There may be circumstances where you feel that the estimation result has not
succeeded in describing certain aspects of the data well. For example, it may
fit two very closely-separated resonances with a single oscillator.

The *Edit Parameter Estimate* button provides functionality to manually change
the estimation result. Upon making such changes, the optimiser should be re-run
to achieve a minimisation of the cost function of interest.

.. warning::

  This feature is present to fix "glaring" incorrect features of the estimation
  result. It is not intended for liberal "fudging" of the result.

A window with the following appearance will appear after you click
*Edit Parameter Estimate*:

.. only:: html

  .. image:: ../../media/gui/windows/edit_parameters_window_1.png
    :align: center

.. only:: latex

  .. image:: ../../media/gui/windows/edit_parameters_window_1.png
    :align: center
    :width: 350

Each oscillator is listed with its associated parameters. The numerical values
assigned to each oscillator match those in the result figure.

Initially, no oscillators are selected. When this is the case, two buttons
are active: *Add*, which allows you to add extra oscillators, and *Close*,
which closes the window.

To select and oscillator, left-click the associated numerical label. This will
highlight the oscillator. Here is an example after oscillator 3 is clicked:

.. only:: html

  .. image:: ../../media/gui/windows/edit_parameters_window_2.png
    :align: center

.. only:: latex

  .. image:: ../../media/gui/windows/edit_parameters_window_2.png
    :align: center
    :width: 350

To de-select the oscillator, simply left-click the numerical label again. Two
new buttons are activated when one oscillator is selected: *Remove*, which will
purge the selected oscillator from the result, and *Split*, which allows you
to create "child oscillators" with the same cumulative amplitude as the parent.

To select multiple oscillators at a single time, left-click on each oscillator
label whilst holding ``<Shift>``:

.. only:: html

  .. image:: ../../media/gui/windows/edit_parameters_window_3.png
    :align: center

.. only:: latex

  .. image:: ../../media/gui/windows/edit_parameters_window_3.png
    :align: center
    :width: 350

When more than one oscillator is selected, the *Merge* button is activated,
along with the Remove button.

The Add Button
--------------

The Add button allows you to add an oscillator with arbitrary parameters to
the estimation result. The main circumstance that this may be useful is when
there is a low-intensity oscillator in the data which the estimation routine
has failed to identify. Extensive use of this button is not advised.

.. note::

  The add button provides exactly the same functionality as
  :py:meth:`~nmrespy.core.Estimator.add_oscillators`.

Clicking on the Add button when no oscillators are selected will load the
following window:

.. only:: html

  .. image:: ../../media/gui/windows/add_window.png
    :align: center

.. only:: latex

  .. image:: ../../media/gui/windows/add_window.png
    :align: center
    :width: 350

You need to input the desired parameters that make up the oscillator to be
added. Each entry box needs to be validated by pressing ``<Return>`` after
inputting the desired value:

* Amplitudes must be positive.
* Phases may be any numerical value. The value you provide will be wrapped
  to be in the range :math:`\left(-\pi, \pi\right]`.
* Frequencies must be within the spectral range of the data considered.
* Damping factors must be positive.

If you wish to include more than one extra oscillator, click the *Add* button.
This will append an extra row to the table. When you have a parameter table
with all entry boxes validated (i.e. none of them are red), click *Confirm*
to append the changes to the result. If you want to quit the window without
making any changes, press *Cancel*.

The Remove Button
-----------------

If one or more oscillators are selected, the Remove button will purge these
from the result.

.. note::

  The add button provides exactly the same functionality as
  :py:meth:`~nmrespy.core.Estimator.remove_oscillators`.

The Merge Button
----------------

With more than one oscillator selected, the merge button will remove all
selected oscillators, and create a single oscillator with parameters that
reflect the selected oscillators. The new oscillator's amplitude will be
the sum of the selected oscillators, and the other parameters will be the
mean of the selected oscillators.

The main use of this is to merge a "superfluous" set of oscillators which
are modelling a single resonance in the data.

.. note::

  The add button provides exactly the same functionality as
  :py:meth:`~nmrespy.core.Estimator.merge_oscillators`.

The Split Button
----------------

With one oscillator selected, the Split button will purge the oscillator
and in its place create a series of "child" oscillators.

.. note::

  The add button provides exactly the same functionality as
  :py:meth:`~nmrespy.core.Estimator.split_oscillator`.

The following window loads when you click *Split*:

.. only:: html

  .. image:: ../../media/gui/windows/split_window.png
    :align: center

.. only:: latex

  .. image:: ../../media/gui/windows/split_window.png
    :align: center
    :width: 250

* The *Number of oscillators* box specifies how many child oscillators
  to generate.
* The *Frequency separation* box specifies how far apart adjacent child
  oscillators will be. You can choose the units to be in Hz or ppm. By
  default, it is set at 2Hz
* The *Amplitude ratio* box specifies the relative amplitudes of the
  oscillators. A valid input for this box takes the form of :math:`n`
  integers, each separated by a colon, where :math:`n` is the value in
  the *Number of oscillators* box. By default, this will be set to be
  :math:`n` 1s separated by colons, such that all child oscillators will
  have the same amplitude.

  .. note::

    If you are familiar with regular expressions, the value in the Amplitude
    ratio box should match :regexp:`^\d+(:\d+){n}$`.

To enact the splitting, click *Confirm*.

Re-running the Optimiser
------------------------

If you have made any changes to the estimation result, you will notice that
the bottom right button has changed from *Close* to *Re-run optimiser*. As well
as this, the *Reset* button to the left has been activated. NMR-EsPy does
not allow you to save an estimation result for which the last step of the
process was manual editing of the result. As such, if you wish to enact the
changes made, you have to re-run nonlinear programming with the current result
as the initial guess.

Undoing changes
---------------

If you decide that you want to undo all the changes made in the
`Edit Parameter Estimate` window, simply click the Reset button.

Saving the result
^^^^^^^^^^^^^^^^^

Clicking the *Save* button loads the following window:

.. only:: html

  .. image:: ../../media/gui/windows/save_window.png
    :align: center
    :scale: 50%

.. only:: latex

  .. image:: ../../media/gui/windows/save_window.png
    :align: center
    :width: 200

Result Figure
-------------

  This section is used for specifying whether to save a result figure, and
  for customising some simple figure settings.

  + `Save Figure` - Whether to save a figure or not.
  + `Format` - The figure's file format. Valid options are ``eps``, ``png``,
    ``pdf``, ``jpg``, ``ps`` and ``svg``.
  + `Filename` - The name of the file to save the Figure to.
  + `dpi` - Dots per inch.
  + `Size (cm)` - The width and height of the figure, in centimeters.

  .. note::
    The most up-voted answer to
    `this Stack Overflow question <https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size>`_ provides a good
    description of the relationship between figure size and dpi.

  .. note::
    Beyond specifying the dpi and size of the figure, the GUI does not provide
    any means of customising the appearance of the figure in this version.
    I intend to provide support of for in a future version.  At the moment,
    the only means of customising the figure is to do it by writing a Python
    script. I provide an outline of how you can achieve certain customisations
    :doc:`here <../../misc/figure_customisation>`

Result Files
------------

  Used for saving a table of result parameters to various file formats.
  For each of the valid formats (``txt``, ``pdf``, and ``csv``), the associated
  tick-boxes are used for specifying whether or not to generate a file of that
  format. Adjacent to each tick-box is an entry box for specifying the name of
  the result file.

  Finally, the `Description` box can be used to enter a description relating
  to the estimation, which will be added to the result file(s).

Estimator
---------

  Used for saving (`"pickling" <https://docs.python.org/3/library/pickle.html>`_)
  the :py:class:`nmrespy.core.Estimator` class instance, associated with the
  estimation result.

  + `Save Estimator` - Specifies whether or not to save the estimator to a
    binary file.
  + `Filename` - The filename to save the estimator to.

Directory
---------

  The entry box is used to specify the path to the directory to save **all**
  specified files to. The full path can either be typed out manually, or
  selected, by loading the file navigation window, by pressing the button
  with a folder icon.

Clicking *Save* will result in all the specified files to be saved to the desired
paths. The application will also be closed.
