Integrating the GUI into TopSpin
================================

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
   provide the corresponding numbers, separated by whitespaces. If you do not
   want to install to TopSpin, enter 0:

In this example, if you wanted to install the GUI loader to both TopSpin 3.6.3
and TopSpin 4.0.8, you would enter ``1 2``. If you only wanted to install to
TopSpin 4.0.8, you would enter ``2``. And to completely cancel the install,
enter ``0``.

For each specified path to install to, the script will try to copy the GUI
loader to the path ``.../topspin*/exp/stan/nmr/py/user/nmrespy.py``. The
result of the attempted install will be printed to the terminal. Here is
an example where I try to install to both TopSpin 4.0.8 and TopSpin 3.6.3:

.. code:: none

  Failed to copy file
      /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/topspin.py
  to
      /opt/topspin3.6.3/exp/stan/nmr/py/user/nmrespy.py
  with error message:
      [Errno 2] No such file or directory: '/opt/topspin3.6.3/exp/stan/nmr/py/user/nmrespy.py'

  Installed: /opt/topspin4.0.8/exp/stan/nmr/py/user/nmrespy.py

The GUI loader was successfully installed to TopSpin 4.0.8, but failed with
TopSpin 3.6.3. The reason for this is quite simple: TopSpin 3.6.3 isn't really
on my system! I just made an empty directory called ``/opt/topspin3.6.3`` for
the purposes of this tutorial. A failure to install the loader could imply
an issue with permissions, or that your TopSpin installation is faulty.

Retrospective installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the TopSpin GUI loader after installing NMR-EsPy, load a
terminal/command prompt, and enter the following command:

* UNIX:

  .. code:: none

     /home/simon$ python3 -m nmrespy --install-to-topspin

* Windows:

  .. code:: none

     C:\Users\simon> py -3 -m nmrespy --install-to-topspin

You then follow the procedure outlined in the previous section.

Manual Installation
^^^^^^^^^^^^^^^^^^^

If automatic installation failed, perhaps because TopSpin isn't installed in
the default location, you can still easily get the TopSPin GUI loader
up-and-running. Open a terminal/command prompt and enter the following to
determine where the GUI loading script is located and copy the output:

* UNIX:

  .. code:: none

     $ python3 -c "import nmrespy; print(nmrespy.TOPSPINPATH)"
     /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/topspin.py

* Windows:

  .. code:: none

      > py -3 -c "import nmrespy; print(nmrespy.TOPSPINPATH)"
      C:\Users\simon\AppData\Roaming\Python\Python38\site-packages\nmrespy\app\topspin.py

Now you simply need to copy this file to your TopSpin installation. It is
recommended you rename the copied file as ``nmrespy.py``:

* UNIX:

  .. code:: none

     $ cp /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/topspin.py \
     > /path/to/.../topspinx.y.z/exp/stan/nmr/py/user/nmrespy.py

You may need ``sudo`` depending on where your TopSpin directory is.

* Windows:

  .. code:: none

      > copy C:\Users\simon\AppData\Roaming\Python\Python38\site-packages\nmrespy\app\topspin.py
