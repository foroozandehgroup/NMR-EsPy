.. NMR-EsPy documentation master file, created by
   sphinx-quickstart on Sat Sep  5 10:37:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
    :format: html

NMR-EsPy
========

.. only:: latex

   .. |github| image:: media/github.png
             :scale: 8%

   .. |email| image:: media/email.png
            :scale: 1%

.. only:: html

    **Simon G. Hulse \& Mohammadali Foroozandeh,**:raw-html:`<br />`
    **Department of Chemistry,**:raw-html:`<br />`
    **University of Oxford**

    **Contents**

    .. toctree::
       :maxdepth: 1

       install
       walkthroughs/index
       gui/index
       references/index

**What is NMR-EsPy?**

NMR-EsPy (**N**\ uclear **M**\ agnetic **R**\ esonance **Es**\ timation in
**Py**\ thon) is a Python package for the estimation of parameters that
describe NMR signals.

As with many packages that do number crunching in Python, NMR-EsPy makes heavy
use of the `NumPy ecosystem <https://numpy.org/>`_.

NMR-EsPy package is a product of work carried out during Simon's Ph.D
within the former Foroozandeh group in the University of Oxford's Chemistry
Department.


The primary goal behind the NMR-EsPy package is to quantify NMR datasets. It is
assumed that raw NMR datasets (called FIDs) comprise a number of sinusoidal
oscillators; NMR-EsPy uses numerical techniques in order to estimate the
amplitudes, phases, frequencies, and damping rates of these oscillators. Such
information can be made use of in various applications.

This version of NMR-EsPy provides the means to estimate both 1D NMR datasets
and 2D J-Resolved datasets. Available features include:

* Importing data acquired on Bruker specrometers.
* Simulating data using the Spinach Matlab package.
* Basic pre-processing of NMR data (phasing, baseline correction).
* Estimation of NMR data using numerical optimisation.
* Generating parameter tables.
* Generating result figures.

With 2D-J Resolved datasets, NMR-EsPy can be employed to construct
homodecoupled (pure shift) spectra via estimation (a paper outlining this is
yet to be released).

In future versions, support for other NMR data types and applications will be
included.

**Using the Documentation**

New to NMR-EsPy? Look at the following:

*If you wish to use the API:*

    1. Read the :ref:`installation instructions <INSTALLATION>`. If you are not
       interested in using the GUI, it is possible to skip the section
       titled *Installing the GUI to TopSpin*
    2. Read the :ref:`tutorial on 1D data estimation <ESTIMATOR1D>`.
    3. Read :ref:`further tutorials <TUTORIALS>` that are relevant to your
       interests.
    4. Whenever a moment arises that you wish to understand more about
       particular features in the package, consult the :ref:`API reference
       <REFERENCES>`.

*If you wish to use the GUI:*

    1. Read the :ref:`installation instructions <INSTALLATION>`.
       If you are going to be loading the GUI from within TopSpin,
       it is important to read the section titled *Installing the GUI to
       TopSpin*.
    2. Read the :ref:`instructions on using the GUI <GUI>`.

In you are comfortable with writing Python code, it is recommended that you
make use of NMR-EsPy via the API, rather than the GUI, as it is more feature
rich.

**Issues**

If you come across any unexpected behavior/bugs, please get in touch with
Simon, via email (see the email icon in the sidebar), or `file an issue
<https://github.com/foroozandehgroup/NMR-EsPy/issues>`_. This project is no
longer actively worked on, but if there are bugs that prevent use of the
package I will try to fix them.

**Publications**

* Simon G. Hulse, Mohammadali Foroozandeh. Newton meets Ockham: Parameter
  estimation and model selection of NMR data with NMR-EsPy. *J. Magn. Reson.*
  338 (2022) 107173. :raw-html:`<br />`
  `<https://doi.org/10.1016/j.jmr.2022.107173>`_

.. only:: latex

    .. toctree::
       :maxdepth: 2

       install
       walkthroughs/index
       gui/index
       references/index
