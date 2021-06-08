Integrating the GUI into TopSpin
================================

It is possible to directly load the NMR-EsPy GUI from within TopSpin. The GUI
can be installed to TopSpin either at the point of installing NMR-EsPy using
``pip install``, or at any subsequent point.

.. note::

   On this page, ``<pyexe>`` denotes the symbolic link/path to the Python
   executable you are using.

Automatic installation
^^^^^^^^^^^^^^^^^^^^^^

After installing NMR-EsPy using ``pip install``, you can set up the TopSpin
GUI loader by entering the following into a terminal:

.. code:: none

   $ <pyexe> -m nmrespy --install-to-topspin

The script searches for directories matching the following glob pattern in your
system:

* UNIX: ``/opt/topspin*``
* Windows: ``C:\Bruker\TopSpin*``

If there are valid directories, you will see a message similar to this:

.. code:: none

    The following TopSpin path(s) were found on your system:
        [1] /opt/topspin3.6.3
        [2] /opt/topspin4.0.8
    For each installation that you would like to install the nmrespy app to,
    provide the corresponding numbers, separated by whitespaces.
    If you want to cancel the install to TopSpin, enter 0.
    If you want to install to all the listed TopSpin installations, press <Return>:


In this example, if you wanted to install the GUI loader to both TopSpin 3.6.3
and TopSpin 4.0.8, you would enter ``1 2`` or simply press ``<Return>``. If you
only wanted to install to TopSpin 4.0.8, you would enter ``2``. To cancel the
install, enter ``0``.

For each specified path to install to, the script will try to copy the GUI
loader to the path ``/.../topspin<x.y.z>/exp/stan/nmr/py/user/nmrespy.py``,
where ``<x.y.z>`` is the TopSpin version number.

The result of the attempted install will be printed to the terminal. Here is
an example where I try to install to both TopSpin 4.0.8 and TopSpin 3.6.3:

.. code:: none

  SUCCESS:
      /opt/topspin3.6.3/exp/stan/nmr/py/user/nmrespy.py

  SUCCESS:
      /opt/topspin4.0.8/exp/stan/nmr/py/user/nmrespy.py

Manual Installation
^^^^^^^^^^^^^^^^^^^

If automatic installation failed, perhaps because TopSpin isn't installed in
the default location, you can still easily get the TopSpin GUI loader
up-and-running with the following steps.

Copying the loader script
-------------------------

Open a terminal/command prompt and enter the following to
determine where the GUI loading script is located:

.. code:: none

   $ <pyexe> -c "import nmrespy; print(nmrespy.TOPSPINPATH)"
   /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/_topspin.py

Now you simply need to copy this file to your TopSpin installation. You should
rename the copied file as ``nmrespy.py``:

* UNIX:

  You may need ``sudo`` depending on where your TopSpin directory is.

  .. code:: none

     $ cp /home/simon/.local/lib/python3.9/site-packages/nmrespy/app/_topspin.py \
     > /path/to/.../topspinx.y.z/exp/stan/nmr/py/user/nmrespy.py

* Windows:

  .. code:: none

      > copy C:\Users\simon\AppData\Roaming\Python\Python38\site-packages\nmrespy\app\_topspin.py ^
      More? C:\path\to\...\TopSpinx.y.z\exp\stan\nmr\py\user\nmrespy.py

.. note::

   In the UNIX example, ``\`` followed by pressing ``<Return>`` allows
   a single long command to span multiple lines. Similarly, ``^``, followed
   by ``<Return>`` achieves the same thing in Windows cmd.

Editing the loader script
-------------------------

Now you need to open the newly created file:

1. Load TopSpin
2. Enter ``edpy`` in the bottom-left command prompt
3. Select the ``user`` subdirectory from ``Source``
4. Double click ``nmrespy.py``

* **Specifying the Python executable path**

  You need to set ``py_exe`` (which is ``None`` initially) with the path to
  your Python executable. One way to determine this which should be independent
  of Operating System is to load a Python interpreter or write a script with
  the following lines (below is an example on Windows):

  .. code:: pycon

     >>> import sys
     >>> exe = sys.executable.replace('\\', '\\\\') # replace is needed for Windows
     >>> print(f"\"{exe}\"")
     "C:\\Users\\simon\\AppData\\Local\\Programs\\Python\\Python38\\python.exe"

  You should set ``py_exe`` as the **EXACT** output you get from this:

  .. code:: python

     py_exe = "C:\\Users\\simon\\AppData\\Local\\Programs\\Python\\Python38\\python.exe"

* **(Optional) Specifying the pdflatex path**

  If you have ``pdflatex`` on your system (see the *LaTeX* section in
  :doc:`Installation <../install>`), and you want to be able to produce
  PDF result files, you will also have to specify the path to the
  ``pdflatex`` executable, given by the variable ``pdflatex_exe``, which
  is set to ``None`` by default. To find this path, load a Python interpreter/
  write a Python script with the following lines:

  + *UNIX*

    .. code:: python

      >>> from subprocess import check_output as co
      >>> exe = check_output("which pdflatex", shell=True)
      >>> exe = str(exe, 'utf-8').rstrip()
      >>> print(f"\"{exe}\"")
      "/usr/bin/pdflatex"

  + *Windows*

    .. code:: python

      >>> from subprocess import check_output
      >>> exe = check_output("where pdflatex", shell=True)
      >>> exe = str(exe, 'utf-8').rstrip().replace("\\", "\\\\")
      >>> print(f"\"{exe}\"")
      "C:\\texlive\\2020\\bin\\win32\\pdflatex.exe"

  You should set ``pdflatex_exe`` as the **EXACT** output you get from this:

  .. code:: python

     pdflatex_exe = "C:\\texlive\\2020\\bin\\win32\\pdflatex.exe"

With the Python path and (optionally) the ``pdflatex`` path set, the script
should now work.
