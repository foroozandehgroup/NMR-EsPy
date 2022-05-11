# NMR-EsPy TODO

## Features To Add

- [ ] 2DJ Module *In progress*
- [ ] Filtering by automated spanning of the frequency space.
- [ ] Visualisation of the optimisation as it runs.
- [ ] Investigate different filtering approaches (FIR filters, interpolation/decimation etc).

## Code Improvements

- [x] `ArgumentChecker`: Specify callables explicitly. **DONE:** Created new, more versatile `sanity_check` module.
- [x] `ExpInfo`: Take number of points out. **DONE:** Restructured `ExpInfo`
  with more features, and included `default_pts` to enable versatility in
  specifying number of points.
- [x] **24-3-22** Clean-up GUI code and check for bugs (1D)
- [ ] `plot.py`: Residual is far from spectrum in examples in `test_plot.py`. Look into this.
