import numpy as np
from numpy.random import uniform

from nmrespy import ExpInfo
from nmrespy.nlp import NonlinearProgramming

np.set_printoptions(
    formatter={"float": lambda x: f"{x:.5f}"}
)

expinfo = ExpInfo(
    dim=1,
    sw=1000.0,
    sfo=400.0,
    nuclei="1H",
    default_pts=4096,
)

parameters = np.array(
    [
        [1, 0, 300, 1],
        [2, 0, 290, 1.1],
        [1, 0, 280, 1],
        [1, 0, -150, 1],
        [3, 0, -155, 1.2],
        [3, 0, -160, 1.2],
        [1, 0, -165, 1],
    ]
)

# Make FID with SNR of 30dB
fid = expinfo.make_fid(parameters, snr=30.0)

# Add noise to parameter array for initial guess
initial_guess = parameters + uniform(low=-0.5, high=0.5, size=parameters.shape)

# Optimise parameters using the Gauss-Newton method (default method)
nlp = NonlinearProgramming(
    expinfo,
    fid,
    initial_guess,
)
result = nlp.get_result("hz")
errors = nlp.get_errors("hz")

print(f"Result:\n{result}\nErrors:\n{errors}")

# Ensure that every parameter is at least within 0.01 of its True value
assert np.allclose(parameters, result, atol=1e-2)

# --- Output ---
#
# Result:
# [[0.99669 -0.00051 300.00017 0.99671]
#  [2.00048 -0.00083 290.00006 1.09946]
#  [0.99956 -0.00069 279.99916 0.99938]
#  [0.99962 -0.00070 -150.00034 0.99706]
#  [3.00004 -0.00046 -154.99967 1.20114]
#  [3.00037 -0.00023 -159.99970 1.20171]
#  [1.00073 -0.00039 -164.99948 0.99820]]
# Errors:
# [[0.00277 0.00278 0.00063 0.00394]
#  [0.00291 0.00146 0.00036 0.00227]
#  [0.00278 0.00278 0.00063 0.00395]
#  [0.00280 0.00280 0.00063 0.00395]
#  [0.00310 0.00103 0.00028 0.00174]
#  [0.00310 0.00103 0.00028 0.00174]
#  [0.00280 0.00280 0.00063 0.00396]]
