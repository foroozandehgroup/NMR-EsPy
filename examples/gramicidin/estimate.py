import os
import pathlib
import pickle
import sys
sys.path.insert(0, str(pathlib.Path('../..').resolve()))
from nmrespy import freqfilter, load, mpm, plot, sig, write  # noqa: E402
from nmrespy.nlp import nlp  # noqa: E402


p0 = (2.417,)
p1 = (0.,)
region = ((4.95, 4.60),)
noise_region = ((11.5, 11.),)
optimiser_iterations = 1000

if __name__ == '__main__':
    data, expinfo = load.load_bruker('../data/4/1111', ask_convdta=False)
    data = sig.phase(data, p0, p1)
    virtual_echo = sig.make_virtual_echo([data])
    spectrum = sig.ft(virtual_echo)
    filterinfo = freqfilter.filter_spectrum(
        spectrum, expinfo, region, noise_region, region_unit='ppm',
        sg_power=50.
    )
    fid, filter_expinfo = filterinfo.get_filtered_fid(cut_ratio=1.05)
    filter_shifts = sig.get_shifts(filter_expinfo, unit='ppm')[0]
    # mpm_result = mpm.MatrixPencil(fid, filter_expinfo, M=25)
    # x0 = mpm_result.get_result()
    # nlp_result = nlp.NonlinearProgramming(
        # fid, x0, filter_expinfo, negative_amps='remove',
        # max_iterations=optimiser_iterations,
    # )
    # x = nlp_result.get_result()

    dirname = '-'.join([str(x).replace('.', '_') for x in region[0]])
    path = pathlib.Path(dirname).resolve()
    os.makedirs(path, mode=0o755, exist_ok=True)

    # with open(str(path / 'mpm_result.pkl'), 'wb') as fh:
        # pickle.dump(mpm_result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(str(path / 'nlp_result.pkl'), 'wb') as fh:
        # pickle.dump(nlp_result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(str(path / 'mpm_result.pkl'), 'rb') as fh:
        mpm_result = pickle.load(fh)
    with open(str(path / 'nlp_result.pkl'), 'rb') as fh:
        nlp_result = pickle.load(fh)
    x0 = mpm_result.get_result()
    x = nlp_result.get_result()
    mpmplot = plot.plot_result(
        data, x0, expinfo, region=region, plot_model=True
    )
    nlpplot = plot.plot_result(
        data, x, expinfo, region=region, plot_model=True
    )
    mpmplot.fig.savefig(str(path / 'mpm_plot.pdf'))
    nlpplot.fig.savefig(str(path / 'nlp_plot.pdf'))

    description = ("Gramicidin \\textsuperscript{1}H data, region: "
                   f"{region[0][0]} - {region[0][1]}Hz.")
    write.write_result(
        filter_expinfo, x0, path=str(path / 'mpm_params'),
        description=description + " MPM result.", pdf_append_figure=mpmplot,
        fmt='pdf', force_overwrite=True,
    )
    write.write_result(
        filter_expinfo, x, errors=nlp_result.get_errors(),
        path=str(path / 'nlp_params'),
        description=description + " NLP result.",
        pdf_append_figure=nlpplot, fmt='pdf', force_overwrite=True
    )
