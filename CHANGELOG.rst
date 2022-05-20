Changelog for NMR-EsPy
======================

Version 1.0 → Version 1.1
-------------------------

Features
^^^^^^^^

* Added ``latex_nuclei`` property to the ``ExpInfo`` class.

Bug and Error Fixes
^^^^^^^^^^^^^^^^^^^

* Ensured that ``Estimator1D.estimate`` can accept ``region=None`` (enabling estimation of entire frequency space).

Version 1.1 → Version 1.1.1
---------------------------

Bug and Error Fixes
^^^^^^^^^^^^^^^^^^^

* ``Estimator1D.write_result`` and ``Estimator1D.plot_result`` now work when estimated region is ``None``.
