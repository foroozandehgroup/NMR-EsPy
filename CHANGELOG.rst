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

Version 1.1 → Version 1.1.1
---------------------------

Bug and Error Fixes
^^^^^^^^^^^^^^^^^^^

* ``Estimator1D.write_result`` and ``Estimator1D.plot_result`` now work when
  estimated region is ``None``.
* Fixed bug involving ``pdflatex`` being ``None`` in the topspin GUI script:
  [https://github.com/foroozandehgroup/NMR-EsPy/issues/1]

Version 1.1.1 → Version 1.1.2
-----------------------------

Features
^^^^^^^^

* `get_result()` methods have been renamed `get_params()` for clarity.
* `make_fid()` now supports the `indirect_modulation` argument for phase- and
  amplitude-modulated signal construction.
* Refactoring of estimator classes: various methods moved from `Estimator1D`
  into the parent `Estimator` class as these behave in a general fashion.
