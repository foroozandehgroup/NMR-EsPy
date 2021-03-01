from nmrespy.core import Estimator

# Path to data. You'll need to change the 4.0.8 bit if you are using a
# different TopSpin version.

# --- Linux users ---
path = "/opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1"

# --- Windows users ---
# path = "C:/Bruker/TopSpin4.0.8/examdata/exam1d_1H/1/pdata/1"

estimator = Estimator.new_bruker(path)
estimator.frequency_filter([[5.54, 5.42]], [[-0.15, -0.3]])
estimator.matrix_pencil()
estimator.nonlinear_programming(phase_variance=True)

msg = "Example estimation result for NMR-EsPy docs."
for fmt in ["txt", "pdf", "csv"]:
    estimator.write_result(path="example", description=msg, fmt=fmt)

# Plot result. Set oscillator colours using the viridis colourmap
plot = estimator.plot_result(oscillator_colors='viridis')

# Shift oscillator labels
label_shifts = [
    (-0.001, 2E5),
    (0.0, 2E5),
    (0.001, 2E5),
    (0.001, 2E5),
    (-0.0015, 2E5),
    (-0.001, 2E5),
    (0.0025, 2E5),
    (0.0025, 2E5),
    (-0.001, 2E5),
]

for i, shifts in enumerate(label_shifts, start=1):
     plot.labels[i].set_position(
        tuple(p + s for p, s in zip(plot.labels[i].get_position(), shifts))
     )

plot.fig.savefig("plot_example_edited.png")
estimator.to_pickle(path="pickle_example")
estimator.save_logfile(path="logfile_example")
