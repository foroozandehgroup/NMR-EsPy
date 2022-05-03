Estimation Set-up
=================

The following is a screenshot of the NMR-EsPy GUI calculation set-up window.
Key features of the window are annotated:

.. only:: html

  .. image:: ../../media/gui/windows/setup_window.png
    :align: center
    :scale: 60%

.. only:: latex

  .. image:: ../../media/gui/windows/setup_window.png
    :align: center
    :scale: 45%

Plot Navigation
^^^^^^^^^^^^^^^

The Plot navigation toolbar enables you to change the view of the data.
It is an edited version of
`matplotlib's toolbar <https://matplotlib.org/3.2.2/users/navigation_toolbar.html>`_,
with the following available buttons:

.. list-table::
   :header-rows: 1
   :widths: 1 10

   * - Icon
     - Role

   * - .. image:: ../../media/gui/navigation_icons/home.png
          :width: 60%
          :align: center
     - Return to the original plot view.

   * - .. image:: ../../media/gui/navigation_icons/back.png
          :width: 60%
          :align: center
     - Return to the previous plot view.

   * - .. image:: ../../media/gui/navigation_icons/forward.png
          :width: 60%
          :align: center
     - Undo a return to a previous view

   * - .. image:: ../../media/gui/navigation_icons/pan.png
          :width: 60%
          :align: center
     - Pan. Note that panning outside the spectral window is not possible.

   * - .. image:: ../../media/gui/navigation_icons/zoom.png
          :width: 60%
          :align: center
     - Zoom.


Phase Correction
^^^^^^^^^^^^^^^^

The GUI has the following appearance when the `Phase Correction` tab is
selected:

.. only:: html

  .. image:: ../../media/gui/windows/setup_window_phase_tab.png
    :align: center
    :scale: 60%

.. only:: latex

  .. image:: ../../media/gui/windows/setup_window_phase_tab.png
    :align: center
    :scale: 45%

Phase correction can be carried out by editing the
pivot (red line in the above figure), zero-order phase and first-order phase.
This is unlikely to be necessary if you are considering processed data, however
you will probably need to do this if you are considering the raw time-domain
data.

The values may be changed either by adjusting the scale widgets, or by manually
inputting desired values into the adjacent entry boxes.

.. only:: html

  .. note::

     **Validating entry box inputs**

     For the majority of entry boxes in the GUI, you will notice that the box
     turns red after you manually change its contents. This indicates
     that the input must adhere to certain criteria (i.e. it must be a number, a
     valid path on your computer etc.), and it has not been validated. After you
     have changed the value in an entry box, press ``<Return>``. The entry box
     will then turn back to its original colour. If the value you
     provided is valid for the given parameter, the value will be kept. If the
     value provided is invalid, the entry box will revert back to the previous
     valid value.

     The video below illustrates this. Initially, I try to change the value
     of the pivot to 7ppm. As soon as the entry box is changed, it goes red,
     indicating that it needs validating. When ``<Return>`` is pressed, as 7 is
     a valid value for the pivot (it is a number, and is within the spectrum's
     sweep width), the pivot is changed accordingly. Note that it is changed to
     the closest valid value to the nearest 4dp, which happens to be 6.9999ppm
     in this case.

     After this, I try to change the pivot to the value ``invalid`` which of
     course makes no sense in the context of a pivot. As it is invalid, when
     ``<Return>`` is pressed, the pivot entry box reverts back to the last valid
     value it had.

     .. raw:: html

        <video width="640" height="640" style="display:block; margin: 0 auto;" controls autoplay>
          <source src="../../media/gui/entry_widget_example.mp4" type="video/mp4">
          Your browser doesn't support the video tag
        </video>

      Note that if you try to run the estimation routine while at least one entry
      box has not be validated, you will be prevented from doing so:

      .. image:: ../../media/gui/windows/unverified_parameter_window.png
         :align: center
         :scale: 80%

.. only:: latex

  .. note::

    **Validating entry box inputs**

    For the majority of entry boxes in the GUI, you will notice that the box
    turns red after you manually change its contents. This indicates
    that the input adhere to certain criteria (i.e. it must be a number, a
    valid path on your computer etc.), and it has not been validated. After you
    have changed the value in an entry box, press ``<Return>``. The entry box
    will then turn back to its original colour. If the value you
    provided is valid for the given parameter, the value will be kept. If the
    value provided is invalid, the entry box will revert back to the previous
    valid value.

    Note that if you try to run the estimation routine while at least one entry
    box has not be validated, you will be prevented from doing so:

    .. image:: ../../media/gui/windows/unverified_parameter_window.png
      :align: center
      :scale: 70%


Region Selection
^^^^^^^^^^^^^^^^

For typical NMR signals, the estimation routine used in NMR-EsPy is
too expensive to analyse the entire signal. For this reason, it is typically
necessary to generate a signal which has been frequency-filtered, drastically
reducing the computation time, and increasing the accuracy of the estimation
for the region chosen. As a rule of thumb, try to choose a region with fewer
than 30 peaks. Any more than this, and the routine may take too long for you
to bear.

To filter the signal, two regions of the spectrum need to be indicated:

* The region to estimate, highlighted in :regiongreen:`green`.
* A region which appears to contain no signals
  (i.e. is just experimental noise), highlighted in :regionblue:`blue`.

These regions can be adjusted by editing the scale widgets and entry boxes
in the `Region Selection` tab.

Advanced Estimation Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clicking the `Advanced Settings` button will load a window enabling a few
more advanced aspects of the estimation routine to be tweaked:

.. only:: html

  .. image:: ../../media/gui/windows/advanced_settings_window.png
    :align: center
    :scale: 15%

.. only:: latex

  .. image:: ../../media/gui/windows/advanced_settings_window.png
    :align: center
    :scale: 12%

Below is a summary of the meaning of all of these parameters.

.. note::

   For the majority of cases, you should find that the default parameters
   provided will be suitable.


Matrix Pencil Method Options
----------------------------

  The Matrix Pencil Method (MPM) is a singular-value decomposition-based approach
  for estimating signal parameters. It is used in NMR-EsPy to generate an
  initial guess for numerical optimisation. It is possible to either manually
  choose how many oscillators to generate using the MPM, or to
  estimate the number of oscillators using the Minimum Description Length (MDL).

  + `Use MDL` - Whether or not to use the Minimum Description Length.
    By default, the MDL will be used.
  + `Number of Oscillators` - The number of oscillators used in the MPM.
    This can only be specified if `Use MDL` is unticked.

Nonlinear Programming Options
-----------------------------

  The result of the Matrix Pencil Method is fed into a nonlinear programming
  (NLP) routine to determine the final signal parameter estimate.

  + `NLP method` - The optimisation routine. This can be one of
    `Gauss-Newton` (default), `Exact Newton`, or `L-BFGS`.
    The main difference between these methods is the means by which the
    `Hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_ (a matrix
    of second order derivatives) is computed.
  + `Maximum iterations` - The largest number of iterations to perform before
    terminating an returning the optimiser. The default value is dependent on
    the NLP algorithm used (200 if Gauss-Newton is selected, 100 for
    Trust-Region, and 500 for L-BFGS).
  + `Optimise phase variance` - Specifies whether to consider the variance of
    oscillator phases during the estimation routine. If your data is derived
    from a well-phased spectrum, it is advised you have this selected.

Once you are happy with the estimation setup, simply click the *Run* button.
You will find that details of the routine are output to the terminal as it
runs.
