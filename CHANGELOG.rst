Changelog for NMR-EsPy
======================

Version 1.0 → Version 1.1
-------------------------

Features
^^^^^^^^

* Added ``latex_nuclei`` property to the ``ExpInfo`` class.

Bug and Error Fixes
^^^^^^^^^^^^^^^^^^^

* Ensured that ``Estimator1D.estimate`` can accept ``region=None`` (enabling
  estimation of entire frequency space).

Version 1.1 → Version 1.2
-------------------------

Features
^^^^^^^^

* Created ``Estimator2DJ``, an estimator class for treating J-resolved (2DJ)
  NMR data.
* Support for generating amplitude- and phase-modulated signals using the
  ``make_fid`` method. Also, routines for processing such signals to derive
  spectral data.
