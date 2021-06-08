Installation
============

NMR-EsPy is available via the
`Python Package Index <https://pypi.org/project/nmrespy/>`_. The latest stable
version can be installed using:

.. code::

   $ python3 -m pip install nmrespy

You need to be using Python 3.8 or above.

Python Packages
---------------

+----------------------------------------------------------+------------+---------------------------------------+
| Package                                                  | Version    | Details                               |
+==========================================================+============+=======================================+
| `Numpy <https://numpy.org/>`_                            | 1.20+      | Ubiquitous                            |
+----------------------------------------------------------+------------+---------------------------------------+
| `Scipy <https://www.scipy.org/>`_                        | 1.6+       | Ubiquitous                            |
+----------------------------------------------------------+------------+---------------------------------------+
| `Matplotlib <https://matplotlib.org/stable/index.html>`_ | 3.3+       | Required for result plotting, and for |
|                                                          |            | manual phase-correction               |
+----------------------------------------------------------+------------+---------------------------------------+
| `Colorama <https://pypi.org/project/colorama/>`_         | 0.4+       | Required on Windows only. Enables     |
|                                                          |            | ANSI escape character sequences to    |
|                                                          |            | work, allowing coloured terminal      |
|                                                          |            | output.                               |
+----------------------------------------------------------+------------+---------------------------------------+

LaTeX
-----

NMR-EsPy provides functionality to save result files to PDF format using
LaTeX. Of course, if you wish to use this feature, you will need to have
a LaTeX installed on your machine. The easiest way to get LaTeX is probably
to install `TexLive <https://tug.org/texlive/>`_.

Essentially, in order for generation of PDFs to work, the command ``pdflatex``
should exist. To check this, open a terminal/command prompt.

.. note::

  **Windows users**

  Press ``Win`` + ``R``, and then type ``cmd`` into the
  window that pops up. Finally, press ``<Return>``.

Enter the following command:

.. code::

   $ pdflatex -v

If you see something similar to the following:

.. code::

  pdfTeX 3.14159265-2.6-1.40.20 (TeX Live 2019/Debian)
  kpathsea version 6.3.1
  Copyright 2019 Han The Thanh (pdfTeX) et al.

  --snip--

things should work fine. If you get an error indicating that ``pdflatex``
isn't recognised, you probably haven't got LaTeX installed.

There are a few LaTeX packages required to generate result PDFs. These
are outlined in the *Notes* section of the documentation of
:py:meth:`nmrespy.write.write_result`.
