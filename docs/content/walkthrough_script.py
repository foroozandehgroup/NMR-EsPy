from nmrespy import Estimator1D

# Path to directory containing 1r data file
path = "/opt/topspin4.0.8/examdata/exam1d_1H/1/pdata/1"
# Create estimator object
estimator = Estimator1D.new_bruker(path)
# Spectral regions to be estimated
regions = ((5.285, 5.18), (5.54, 5.42))
# Spectral reion with no discernible peaks
noise_region = (6.48, 6.38)
# Estimate each region in turn
for region in regions:
    estimator.estimate(region, noise_region, region_unit="ppm", phase_variance=True)
# Save results to text file and PDF
for fmt in ("txt", "pdf"):
    estimator.write_result(path="result", fmt=fmt)
# Create and save result figures
plots = estimator.plot_result()
for region, plot in zip(regions, plots):
    path = f"{region[0]}-{region[1]}".replace(".", "_")
    plot.save(path, fmt="png", dpi=600)
# Save logfile
estimator.save_log(path="logfile")
# Pickle the estimator for future use
estimator.to_pickle(path="estimator")
