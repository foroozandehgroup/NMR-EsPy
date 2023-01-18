Reference
=========

.. only:: html

   The following links provide detailed descriptions of classes and functions
   available in the user-facing API of NMR-EsPy.

.. only:: latex

   This chapter provides detailed descriptions of classes and functions
   available in the user-facing API of NMR-EsPy.


The average user is likely to only be concerned with the estimator object they
wish to use (either ``Estimator1D`` or ``Estimator2DJ``). Of potential interest
are also:

* ``ExpInfo``. A parent class of estimator objects. This stores important
  experiment information, such as sweep width, transmitter offset etc., with
  associated methods, such are ones to generate time-point and chemical shift
  arrays.
* ``sig``, a module of functions for processing signals, including FT/IFT,
  apodisation, phasing.

.. toctree::
   :maxdepth: 1

   estimator1d
   estimator2dj
   expinfo
   sig
