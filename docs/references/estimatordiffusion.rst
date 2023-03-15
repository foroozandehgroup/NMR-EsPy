Diffusion Estimators
====================

A number of estimators for the consideration of diffusion NMR data exist. These all
the same functionality except for the form of the constant :math:`\Delta^{\prime}`
that is part of the Stejskal-Tanner equation:

.. math::

    I\left(I_0, D, g\right) =
        I_0 \exp \left(-\gamma^2 \sigma^2 \delta^2 \Delta^{\prime} Dg^2\right)


* ``EstimatorDiffusionMonopolar``:

  .. math::

    \Delta^{\prime} = \Delta + 2 \delta \left(\kappa - \lambda\right)

* ``EstimatorDiffusionBipolar``:

  .. math::

    \Delta^{\prime} =
        \Delta + \frac{\delta \left(2 \kappa - 2 \lambda - 1\right)}{4} - \frac{1}{2}

* ``EstimatorDiffusionOneshot``:

  .. math::

    \Delta^{\prime} =
        \Delta +
        \frac{\delta \left(\kappa - \lambda\right)\left(\alpha^2 + 1\right)}{2} +
        \frac{\left(\delta + 2 \tau\right) \left(\alpha^2 - 1\right)}{4}

See `<https://doi.org/10.1002/cmr.a.21223>`_ for details.

.. autoclass:: nmrespy.EstimatorDiffusionMonopolar
   :members:
   :inherited-members:
   :exclude-members: convert, generate_random_signal, oscillator_integrals, spectrum_first_direct, unpack

.. autoclass:: nmrespy.EstimatorDiffusionBipolar
   :members: big_delta_prime

.. autoclass:: nmrespy.EstimatorDiffusionOneshot
   :members: new_from_parameters, big_delta_prime
