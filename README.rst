.. image:: nmrespy/pics/nmrespy_full.png
   :height: 2129px
   :width: 3599px
   :scale: 1 %
   :align: center

Documentation here: https://nmr-espy.readthedocs.io/en/latest/

I'm yet to chuck this on `PyPI <https://pypi.org/>`_, but if you want
to have a look at it you'll need to do the following:

* clone this repo

  .. code-block:: bash

     $ git clone https://github.com/foroozandehgroup/NMR-EsPy.git

* (Optional) if you'd like to give the TopSpin GUI a go, copy
  ``NMR-EsPy/nmrespy/topspin/nmrespy_win.py`` (Windows) or
  ``NMR-EsPy/nmrespy/topspin/nmrespy_lin.py`` (Linux) to the following
  path in your TopSpin directory: ``<TOPSPIN DIR>/exp/stan/nmr/py/user/``,
  and rename as ``nmrespy.py``

  .. code-block:: bash

     $ cp /path/to/.../NMR-EsPy/nmrespy/topspin/nmrespy_lin.py \
     /opt/topspin4.0.8/exp/stan/nmr/py/user/nmrespy.py

* Determine the path to your Python installation's site packages directory:

  .. code-block:: bash

     $ python3 -m site --user-site
     /home/simon/.local/lib/python3.8/site-packages

* copy the ``NMR-EsPy/nmrespy`` directory to the site packages directory

  .. code-block:: bash

     $ cp -r /path/to/.../NMR-EsPy/nmrespy \
     /home/simon/.local/lib/python3.8/site-packages

Check out the
`Example Walkthrough <https://nmr-espy.readthedocs.io/en/latest/walkthrough.html>`_
page in the docs for a tutorial on NMR-EsPy's typical usage.
