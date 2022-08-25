2DJ Walkthrough
===============

In this example, I will go through some of the most useful features of the
:py:class:`~nmrespy.Estimator2DJ` class for the consideration of data deived
from J-resolved (2DJ) experiments. We will be using a simulated ¹H 2DJ spectrum
of sucrose for this example. The data will be simulated using
`Spinach <https://spindynamics.org/wiki/index.php?title=Main_Page>`_, via the
:py:meth:`~nmrespy.Estimator2DJ.new_spinach` classmethod. In order to use this
method, you need to have a MATLAB installation, configured the `MATLAB Engine
for Python
<https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_,
and installed Spinach. If you do not have MATLAB, I have pre-simulated the data
too so you can still follow along.

Creating the estimator
----------------------


