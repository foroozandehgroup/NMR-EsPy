from nmrespy.core import Estimator

# Path to data. You'll need to change the 4.0.8 bit if you are using a
# different TopSpin version.

# --- UNIX users ---
path = "/opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1"

# --- Windows users ---
# path = "C:/Bruker/TopSpin4.0.8/examdata/exam1d_1H/1/pdata/1"

estimator = Estimator.new_bruker(path)

# --- Frequency filter & estimate ---
estimator.frequency_filter([[5.54, 5.42]], [[-0.15, -0.3]])
estimator.matrix_pencil()
estimator.nonlinear_programming(phase_variance=True)

# --- Write result files ---
msg = "Example estimation result for NMR-EsPy docs."
for fmt in ["txt", "pdf", "csv"]:
    estimator.write_result(path="example", description=msg, fmt=fmt)

# --- Plot result ---
# Set oscillator colours using the viridis colourmap
plot = estimator.plot_result(oscillator_colors='viridis')
# Shift oscillator labels
# Move the labels associated with oscillators 1, 2, 5, and 6
# to the right and up.
plot.displace_labels([1,2,5,6], (0.02, 0.01))
# Move the labels associated with oscillators 3, 4, 7, and 8
# to the left and up.
plot.displace_labels([3,4,7,8], (-0.04, 0.01))
# Move oscillators 9's label to the right
plot.displace_labels([9], (0.02, 0.0))

# Save figure as a PNG
plot.fig.savefig("plot_example_edited.png")

# Save the estimator to a binary file.
estimator.to_pickle(path="pickle_example")

# Save a logfile of method calls
estimator.save_logfile(path="logfile_example")
