import numpy as np

from nmrespy import ExpInfo
from nmrespy.mpm import MatrixPencil

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
        [1, 0, -165, 1],
        [3, 0, -160, 1.2],
        [3, 0, -155, 1.2],
        [1, 0, -150, 1],
        [1, 0, 280, 1],
        [2, 0, 290, 1.1],
        [1, 0, 300, 1],
    ]
)

# Make FID with SNR of 30dB
fid = expinfo.make_fid(parameters, snr=30.0)

# Estimation parameters using the MPM with model order selection with the MDL
mpm = MatrixPencil(
    expinfo,
    fid,
)

result = mpm.get_result("hz")

print(f"\nResult\n------\n{result}")

# Ensure that every parameter is at least within 0.01 of its True value
assert np.allclose(parameters, result, atol=1e-2)

# --- Output ---
#
# ===========
# MPM STARTED
# ===========
# --> Pencil Parameter: 1365
# --> Hankel data matrix constructed:
# 	Size:   2731 x 1366
# 	Memory: 56.9236MiB
# --> Performing Singular Value Decomposition...
# --> Computing number of oscillators...
# 	Number of oscillators will be estimated using MDL
# 	Number of oscillations: 7
# --> Computing signal poles...
# --> Computing complex amplitudes...
# --> Checking for oscillators with negative damping...
# 	None found
# ============
# MPM COMPLETE
# ============
# Time elapsed: 0 mins, 5 secs, 488 msecs

# Result
# ------
# [[1.00301 -0.00150 -164.99998 1.00202]
#  [3.00147 0.00071 -159.99990 1.19879]
#  [3.00138 0.00005 -155.00003 1.19939]
#  [0.99621 0.00027 -149.99912 0.99392]
#  [1.00413 -0.00094 280.00033 1.00768]
#  [1.99889 -0.00079 290.00032 1.09956]
#  [0.99982 0.00426 299.99939 1.00186]]
