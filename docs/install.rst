Installation
============

NMR-EsPy is available via the
`Python Package Index <https://pypi.org/project/nmrespy/>`_. The latest stable
version can be installed using:

.. code::

   $ python3 -m pip install nmrespy

Requirements
------------

+----------------------------------------------------------+------------+---------------------------------------+
| Package                                                  | Version    | Details                               |
+==========================================================+============+=======================================+
| Python                                                   | 3.7.0+     |                                       |
+----------------------------------------------------------+------------+---------------------------------------+
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
