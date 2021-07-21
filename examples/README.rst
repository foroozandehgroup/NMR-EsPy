NMR-EsPy Examples
=================

This directory contains examples of NMR-EsPy's usage.
Current examples present are:

1. ``cyclosporin/``: Estimation run on two separate regions of a cyclosporin 1D signal (data in ``data/1``)
2. ``artemisinin/``: Estimation run on a very low SNR region of a artemisinin 1D signal (data in ``data/2``)
3. ``cpmg/``: Estimation run on set of 1D datasets acquired in a CPMG expeeriment (data in ``data/3``)

For each example, a script which goes through the estimation routine, and produces various result files and figures
is provided. Note that for each of these, there is a variable called ``ESTIMATE``. This should be set to ``True``
if you want to perform the estimation. If you only want to run the code to generate the final result figure,
and already have run the estimation, you may set this as ``False``.

Some of the examples given here feature in publications relating to NMR-EsPy:

+ ``<CITE 1D PAPER ONCE PUBLISHED>`` Examples 1. and 2.
