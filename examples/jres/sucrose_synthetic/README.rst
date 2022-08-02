sucrose_synthetic
=================

A illustration of applying `Estimator2DJ` to a simulated ¹H 2DJ dataset of
sucrose. Simulations are performed in MATLAB using `Spinach
<https://spindynamics.org/wiki/index.php?title=Main_Page>`_. You'll need
to have this installed unless you are happy to simply use the sample
dataset provided with 64 and 8192 points in TD1 and TD2, respectively.

Simulation parameters (see ``spinach/sucrose_jres.m`` for full details):

* Field 300MHz
* Transmitter offset: 1000Hz
* Spectral window: 40Hz (F1), 2200Hz (F2)

Files
-----

* ``spinach/jres_seq.m``: Pulse sequence definition.
* ``spinach/sucrose_jres.m``: 2DJ simulation for sucrose.
* ``spinach/sucrose.log``: Log file for sucrose Guassian computation.

