import os
import pathlib
import pickle
import sys
sys.path.insert(0, str(pathlib.Path('../..').resolve()))
from nmrespy import freqfilter, load, mpm, plot, sig, write
from nmrespy.nlp import nlp
import numpy as np


# --- Tweak these as you wish
region = ((4.9, 4.65),)
optimiser_iterations = 200
# ---------------------------

data, expinfo = load.load_bruker('../data/4/1111', ask_convdta=False)
data = sig.phase(data, [2.417], [0.])
virtual_echo = sig.make_virtual_echo([data])
spectrum = sig.ft(virtual_echo) + 1E6
noise_region = ((11.5, 11.),)
filterinfo = freqfilter.filter_spectrum(
    spectrum, region, noise_region, expinfo, region_unit='ppm',
    cut_ratio=1.000000001,
)
filter_expinfo = filterinfo.expinfo
fid = filterinfo.cut_fid
cut_spectrum = filterinfo.cut_spectrum
filter_shifts = sig.get_shifts(filter_expinfo, unit='ppm')[0]

mpm_result = mpm.MatrixPencil(fid, filter_expinfo, M=20)
x0 = mpm_result.get_result()
nlp_result = nlp.NonlinearProgramming(
    fid, x0, filter_expinfo, negative_amps='remove',
    max_iterations=optimiser_iterations,
)
x = nlp_result.get_result()
mpmplot = plot.plot_result(data, x0, expinfo, region=region, plot_model=True)
nlpplot = plot.plot_result(data, x, expinfo, region=region, plot_model=True)

dirname = '-'.join([str(x).replace('.', '_') for x in region[0]])
path = pathlib.Path(dirname).resolve()
os.makedirs(path, mode=0o755, exist_ok=True)

to_save = {
    str(path / 'mpm_result.pkl'): mpm_result,
    str(path / 'nlp_result.pkl'): nlp_result,
}
for p, obj in to_save.items():
    with open(p, 'wb') as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

mpmplot.fig.savefig(str(path / 'mpm_plot.pdf'))
nlpplot.fig.savefig(str(path / 'nlp_plot.pdf'))

# description = ("Gramicidin \\textsuperscript{{1}}H data, region: "
               # f"{region[0][0]} - {region[0][1]}Hz.")
# write.write_result(
    # filter_expinfo, x0, path=str(path / 'mpm_params'),
    # escription=description + " MPM result.", pdf_append_figure=mpmplot,
    # fmt='pdf', force_overwrite=True,
# )
# write.write_result(
    # filter_expinfo, x, errors=nlp_result.get_errors(),
    # path=str(path / 'nlp_params'), description=description + " NLP result.",
    # pdf_append_figure=nlpplot, fmt='pdf', force_overwrite=True
# )
