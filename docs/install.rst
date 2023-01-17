Installation
============

.. note::

   On this page, ``<pyexe>`` denotes the symbolic link/path to the Python
   executable you are using. **You need to be using Python 3.8 or above.**

NMR-EsPy is available via the
`Python Package Index <https://pypi.org/project/nmrespy/>`_. The latest stable
version can be installed using:

.. code:: none

   $ <pyexe> -m pip install nmrespy

If you want to install the development and documentation-building requirements,
use the following command:

.. code:: none

   $ <pyexe> -m pip install nmrespy[dev,docs]

NMR-EsPy has the follwoing dependencies which are installed automatically when
you run ``pip install``.

+-------------------------------------------------------------------+------------+----------------------------------------+
| Package                                                           | Version    | Details                                |
+===================================================================+============+========================================+
| `Numpy <https://numpy.org/>`_                                     | 1.19+      | Ubiquitous                             |
+-------------------------------------------------------------------+------------+----------------------------------------+
| `Scipy <https://www.scipy.org/>`_                                 | 1.5+       | Ubiquitous                             |
+-------------------------------------------------------------------+------------+----------------------------------------+
| `Matplotlib <https://matplotlib.org/stable/index.html>`_          | 3.3+       | Required for result plotting,          |
|                                                                   |            | manual phase-correction, and the GUI.  |
+-------------------------------------------------------------------+------------+----------------------------------------+
| `bruker_utils <https://5hulse.github.io/bruker_utils/>`_          | 0.0.1+     | Required for loading Bruker data.      |
+-------------------------------------------------------------------+------------+----------------------------------------+
| `pybaselines <https://github.com/derb12/pybaselines>`_            | 1.0.0+     | Required for baseline correction.      |
+-------------------------------------------------------------------+------------+----------------------------------------+
| `Colorama <https://pypi.org/project/colorama/>`_                  | 0.4+       | Required on Windows only. Enables      |
|                                                                   |            | ANSI escape character sequences to     |
|                                                                   |            | work, allowing coloured terminal       |
|                                                                   |            | output.                                |
+-------------------------------------------------------------------+------------+----------------------------------------+

.. _TOPSPIN_INSTALL:

Installing the GUI to TopSpin
-----------------------------

NMR-EsPy has an accompanying Graphical User Interface (GUI) which is accessible
via both the terminal and Bruker's TopSpin software. The following two sections
describe ways to install the scripts for accessing the GUI via TopSpin.

Automatic TopSpin installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Use this method if you can, it is less error prone!

Enter the following into a terminal:

.. code:: none

   $ <pyexe> -m nmrespy --install-to-topspin

Directories matching the following glob pattern will be searched for on your
system:

* UNIX: ``/opt/topspin*``
* Windows: ``C:\Bruker\TopSpin*``

If there are valid directories, you will see a message similar to this:

.. code:: none

    The following TopSpin path(s) were found on your system:
        [1] /opt/topspin4.0.8
    For each TopSpin version you would like to install the nmrespy app to,
    provide the corresponding numbers, separated by whitespaces.
    If you want to cancel the install to TopSpin, enter 0.
    If you want to install to all the listed TopSpin installations, press <Return>:

In this example, pressing ``1`` or ``<Return>`` would install the scripts to
TopSpin 4.0.8. Pressing ``0`` would cancel the operation.

For each specified path, two files will be generated:

* ``/<path to>/topspin<x.y.z>/exp/stan/nmr/py/user/espy1d.py``
* ``/<path to>/topspin<x.y.z>/exp/stan/nmr/py/user/espy2dj.py``

where ``<x.y.z>`` is the TopSpin version number.

.. code:: none

    SUCCESS:
        /opt/topspin4.0.8/exp/stan/nmr/py/user/espy1d.py

    SUCCESS:
        /opt/topspin4.0.8/exp/stan/nmr/py/user/espy2dj.py

If you get the following message instead, you either need to install TopSpin,
or manually install the script (see the next section):

.. code:: none

   No TopSpin installations were found on your system! If you don't have
   TopSpin, I guess that makes sense. If you do have TopSpin, perhaps it is
   installed in a non-default location? You'll have to perform a manual
   installation in this case. See the documentation for details.

Manual TopSpin installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If automatic installation failed, perhaps because TopSpin isn't installed in
the default location, you can get the TopSpin GUI loader up-and-running with
the following steps.

Copying the loader scripts
""""""""""""""""""""""""""

Load a Python REPL and enter the following to determine where the GUI loading
scripts are located:

.. code:: pycon

   $ <pyexe>
   >>> from nmrespy import TOPSPINPATHS as tp
   >>> for file in tp:
   ...      print(file)
   ...
   /home/simon/.venv/lib/python3.9/site-packages/nmrespy/app/topspin_scripts/espy2dj.py
   /home/simon/.venv/lib/python3.9/site-packages/nmrespy/app/topspin_scripts/espy1d.py

Copy these files to your TopSpin installation.

* UNIX:

  You may need ``sudo`` depending on where your TopSpin directory is.

  .. code:: none

     $ cp /home/simon/.venv/lib/python3.9/site-packages/nmrespy/app/topspin_scripts/espy2dj.py \
     > /path/to/.../topspinx.y.z/exp/stan/nmr/py/user/
     $ cp /home/simon/.venv/lib/python3.9/site-packages/nmrespy/app/topspin_scripts/espy1d.py \
     > /path/to/.../topspinx.y.z/exp/stan/nmr/py/user/

* Windows:

  .. code:: none

      > copy C:\Users\simon\.venv\Lib\site-packages\nmrespy\app\topspin_scripts\espy2dj.py ^
      More? C:\path\to\...\TopSpinx.y.z\exp\stan\nmr\py\user\
      > copy C:\Users\simon\.venv\Lib\site-packages\nmrespy\app\topspin_scripts\espy1d.py ^
      More? C:\path\to\...\TopSpinx.y.z\exp\stan\nmr\py\user\

Editing the loader scripts
""""""""""""""""""""""""""

Now you need to open the newly created files and make some edits to configure
path information.

#. Load TopSpin
#. Enter ``edpy`` in the bottom-left command prompt
#. Select the ``user`` subdirectory from ``Source``

Then do the following things for both ``espy1d.py`` and ``espy2dj.py``:

#. Double click the file
#. You need to set ``py_exe`` (which is ``None`` initially) with the path to
   your Python executable. One way to determine this which should be independent
   of Operating System is to load a Python REPL and enter the following lines
   (below is an example on Windows):

   .. code:: pycon

      >>> import sys
      >>> exe = sys.executable.replace('\\', '\\\\') # replace is needed for Windows
      >>> print(f"\"{exe}\"")
      "C:\\Users\\simon\\.venv\\Scripts\\python.exe"

   You should set ``py_exe`` as the **EXACT** output you get from this:

   .. code:: python

      py_exe = "C:\\Users\\simon\\.venv\\Scripts\\python.exe"

#. (Optional) If you have ``pdflatex`` on your system (see the *LaTeX* section
   below), and you want to be able to produce PDF result files, you will also
   have to specify the path to the ``pdflatex`` executable, given by the
   variable ``pdflatex_exe``, which is set to ``None`` by default. To find this
   path, enter the following into a REPL:

   + *UNIX*

     .. code:: python

         >>> from subprocess import check_output
         >>> exe = check_output(["which", "pdflatex"])
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

.. _LATEX_INSTALL:

LaTeX (Optional)
----------------

NMR-EsPy provides functionality to save result files to PDF format using LaTeX.
The easiest way to get LaTeX is probably to install `TexLive
<https://tug.org/texlive/>`_.

As a simple check that your system has LaTeX available, the command ``pdflatex``
should exist. Open a terminal.


Enter the following command:

.. code::

   $ pdflatex -v

If you see something similar to the following:

.. code:: none

  pdfTeX 3.14159265-2.6-1.40.20 (TeX Live 2019/Debian)
  kpathsea version 6.3.1
  Copyright 2019 Han The Thanh (pdfTeX) et al.

  --snip--

things should work fine. If you get an error indicating that ``pdflatex``
isn't recognised, you probably haven't got LaTeX installed.

The following is a full list of packages that your LaTeX installation
will need to successfully compile the ``.tex`` files generated by this module:

* `amsmath <https://ctan.org/pkg/amsmath?lang=en>`_
* `array <https://ctan.org/pkg/array?lang=en>`_
* `booktabs <https://ctan.org/pkg/booktabs?lang=en>`_
* `cmbright <https://ctan.org/pkg/cmbright>`_
* `geometry <https://ctan.org/pkg/geometry>`_
* `hyperref <https://ctan.org/pkg/hyperref?lang=en>`_
* `longtable <https://ctan.org/pkg/longtable>`_
* `nicefrac <https://ctan.org/pkg/nicefrac?lang=en>`_
* `siunitx <https://ctan.org/pkg/siunitx?lang=en>`_
* `tcolorbox <https://ctan.org/pkg/tcolorbox?lang=en>`_
* `varwidth <https://www.ctan.org/pkg/varwidth>`_
* `xcolor <https://ctan.org/pkg/xcolor?lang=en>`_

If you wish to check the packages are available, use ``kpsewhich``:

.. code::

    $ kpsewhich booktabs.sty
    /usr/share/texlive/texmf-dist/tex/latex/booktabs/booktabs.sty

If a pathname appears, the package is installed to that path. These packages
are pretty ubiquitous, so it is likely that they have been installed already.

.. _SPINACH_INSTALL:

MATLAB and Spinach (Optional)
-----------------------------

`Spinach <http://spindynamics.org/group/?page_id=12>`_ is a highly
sophisticated library for spin dynamics simulations using `MATLAB
<https://www.mathworks.com/products/matlab.html>`_. NMR-EsPy provides some
routines that enable the generation of datasets via Spinach. For this you will
need:

* MATLAB installed
* Spinach downloaded, and present in the MATLAB path list (see `the
  installation instructions
  <https://spindynamics.org/wiki/index.php?title=Installation>`_)
* The MATLAB Engine for Python installed (see `the installation instructions
  <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_)
