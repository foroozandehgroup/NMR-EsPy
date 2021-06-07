NMR-EsPy Example: Cyclosporin data
==================================

NMR-EsPy's estimation routine was run on two different regions of a 1D cyclosporin signal.
The result is presented in ``<CITE PAPER WHEN PUBLISHED>`` (Section 3.1).
This data comes as part of TopSpin 4 installations, under
``topspinx.y.z/examdata/exam1d_1H/1``.
The data is saved under ``examples/data/1`` in this repo.
The following is carried out in ``cyclosporin.py``:

  + The stimationm routine is carried out (``estimate()``)
  + The results (pickled estimator instances, result files in ``.pdf``,
    ``.txt`` and ``.csv`` formats, and files with parameters and errors) are
    saved under ``results/1`` and ``results/2``.
  + A result figure is generated (``plot()``). This is similar to the figure
    in the paper, but without special formatting which requires a LaTeX
    installation with ``xelatex``, and custom fonts.

