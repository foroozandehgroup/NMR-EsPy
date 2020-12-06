.. NMR-EsPy documentation master file, created by
   sphinx-quickstart on Sat Sep  5 10:37:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NMR-EsPy's documentation!
====================================

.. image:: ../nmrespy/pics/nmrespy_full.png
   :height: 2129px
   :width: 3599px
   :scale: 10 %
   :align: center

:Release: |version|
:Data: |today|

NMR-EsPy (**N**\ uclear **M**\ agnetic **R**\ esonance **Es**\ timation in
**Py**\ thon) is a Python package for the estimation of parameters that
describe NMR signals. A GUI is available for direct use within
Bruker's TopSpin software.

As with the majority of packages that do a lot of number crunching in Python,
NMR-EsPy relies heavily on `NumPy <https://numpy.org/>`_ as well as its
trusty sidekick `SciPy <https://www.scipy.org/>`_ to efficiently carry out
its estimation routines.

This package is the culmination of work carried out during my Ph.D project
within the group of Dr. Mohammadali Foroozandeh, in the University of Oxford's
Chemistry Department. To find out more about the group, take a look at our
`website. <http://foroozandeh.chem.ox.ac.uk/home>`_

The following publication(s) are directly linked to NMR-EsPy and provide
an extensive background on the underlying theory and examples of results
obtained by using it:

<No publications yet...>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   theory_overview
   getting_started
   walkthrough
   gui
   references/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
