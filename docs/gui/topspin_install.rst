.. _TS_install:

Integrating the GUI into TopSpin
================================

.. note::

   On this page, ``<pyexe>`` denotes the symbolic link/path to the Python
   executable you are using.

It is possible to directly load the NMR-EsPy GUI from within TopSpin. The GUI
can be installed to TopSpin either at the point of installing NMR-EsPy using
``pip install``, or at any subsequent point.

Installation during pip install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As part of the installation of NMR-EsPy from PyPI, you will be asked whether
you would like to install the GUI loader to any TopSpin installations (if you
have any). The installation script searches for directories matching the
following glob pattern in your system:

* UNIX: ``/opt/topspin*``
* Windows: ``C:\Bruker\TopSpin*``

If there are valid directories, you will see a message similar to this:

.. code:: none

   The following TopSpin path(s) were found on your system:
       [1] /opt/topspin3.6.3
       [2] /opt/topspin4.0.8
   For each installation that you would like to install the nmrespy app to,
   provide the corresponding numbers, separated by whitespaces. If you want to
   cancel the install to TopSpin, enter 0. If you want to install to all the
   listed TopSpin installations, press <Return>:


In this example, if you wanted to install the GUI loader to both TopSpin 3.6.3
and TopSpin 4.0.8, you would enter ``1 2`` or simply press ``<Return>``. If you
only wanted to install to TopSpin 4.0.8, you would enter ``2``. To cancel the
install, enter ``0``.

For each specified path to install to, the script will try to copy the GUI
loader to the path ``.../topspin*/exp/stan/nmr/py/user/nmrespy.py``. The
result of the attempted install will be printed to the terminal. Here is
an example where I try to install to both TopSpin 4.0.8 and TopSpin 3.6.3:

.. code:: none

  Installed:
      /opt/topspin3.6.3/exp/stan/nmr/py/user/nmrespy.py

  Installed:
      /opt/topspin4.0.8/exp/stan/nmr/py/user/nmrespy.py

Retrospective installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the TopSpin GUI loader after installing NMR-EsPy, load a
terminal/command prompt, and enter the following command:

.. code:: none

   $ <pyexe> -m nmrespy --install-to-topspin

You then have to follow the procedure outlined in the previous section.

Manual Installation
^^^^^^^^^^^^^^^^^^^

If automatic installation failed, perhaps because TopSpin isn't installed in
the default location, you can still easily get the TopSpin GUI loader
up-and-running. Open a terminal/command prompt and enter the following to
determine where the GUI loading script is located:

.. code:: none

   $ <pyexe> -c "import nmrespy; print(nmrespy.TOPSPINPATH)"
   /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/_topspin.py

Now you simply need to copy this file to your TopSpin installation. It is
recommended you rename the copied file as ``nmrespy.py``:

* UNIX:

  .. code:: none

     $ cp /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/_topspin.py \
     > /path/to/.../topspinx.y.z/exp/stan/nmr/py/user/nmrespy.py

You may need ``sudo`` depending on where your TopSpin directory is.

* Windows:

  .. code:: none

      > copy C:\Users\simon\AppData\Roaming\Python\Python38\site-packages\nmrespy\app\_topspin.py ^
      More? C:\path\to\...\TopSpinx.y.z\exp\stan\nmr\py\user\nmrespy.py

.. note::

   In the UNIX example, ``\`` followed by pressing ``<Return>`` allows
   a single long command to span multiple lines. Similarly, ``^``, followed
   by ``<Return>`` achieves the same thing in Windows.

Now you need to open the newly created file:

1. Load TopSpin
2. Enter ``edpy`` in the bottom-left command prompt
3. Select the ``user`` subdirectory from ``Source``
4. Double click ``nmrespy.py``

Near the top of the file, you will see this:

.. code:: python

   # ------------------------------------------------------------------------
   # exe should be set as the path to the Python executable that you use for
   # nmrespy.
   # One way to determine this that is general for all OSes is to start an
   # interactive Python session from a terminal/command prompt and then enter
   # the following:
   #   >>> import sys
   #   >>> exe = sys.executable.replace('\\', '\\\\')
   #   >>> print(f"\"{exe}\"")
   # Set exe as exactly what the output of this is
   exe = None
   # ------------------------------------------------------------------------

You need to replace ``exe`` with the path to your Python executable. One
way to do this which should be independent of Operating System is to load
a Python interpreter and do the following:

.. code:: pycon

   >>> import sys
   >>> exe = sys.executable.replace('\\', '\\\\') # replace is needed for Windows
   >>> print(f"\"{exe}\"")
   "C:\\Users\\simon\\AppData\\Local\\Programs\\Python\\Python38\\python.exe"

You should set ``exe`` as the **exact** output you get from this:

.. code:: python

   # ------------------------------------------------------------------------
   # --snip--
   exe = "C:\\Users\\simon\\AppData\\Local\\Programs\\Python\\Python38\\python.exe"
   # ------------------------------------------------------------------------

Now everything should be good to go!
