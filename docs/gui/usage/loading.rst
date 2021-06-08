Loading the GUI
===============

.. note::

   On this page, ``<pyexe>`` denotes the symbolic link/path to the Python
   executable you are using.

The GUI can be loaded both from a terminal/command prompt, or from within
TopSpin provided the GUI loader has been installed
(see :doc:`Integrating the GUI into TopSpin <../topspin_install>`).

From a terminal
---------------

To set-up an estimation routine from the terminal/command prompt,
enter the following command:

.. code:: none

   $ <pyexe> -m nmrespy --estimate <path_to_bruker_data>

.. note::

   The shorthand flag ``-e`` can be used in place of ``--estimate``.

``<path_to_bruker_data>`` should be one of the following:

* The path to the parent directory of the raw time-domain data (``fid``).
* The path to the parent directory of the processed data (``1r``).

From TopSpin
------------

To load the GUI from TopSpin, simply select the data you wish to look at,
and then enter the command ``nmrespy`` into the prompt in the bottom left
corner.

You will be asked to select the data you wish to consider (either the
raw time-domain data, or the processed data):

.. only:: html

  .. image:: ../../media/gui/windows/datatype.png
     :align: center
     :scale: 70%

.. only:: latex

  .. image:: ../../media/gui/windows/datatype.png
     :align: center
     :width: 400
