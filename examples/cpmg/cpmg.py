# SGH, 7-7-21
# The current version of nmrespy is not suitable for data other than single
# 1D signals. (nmrespy-0.1.1)
# This script is a hack to showcase how a pseudo-2D experiemnt (CPMG) could
# be analysed with NMR-EsPy.
# The functionality present here will become cleaned up and incorporated into
# a future version of NMR-EsPy

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from nmrespy.core import Estimator
from nmrespy.load import load_bruker
from nmrespy.sig import ft, ift


PATH = Path().cwd().resolve().parents[0] / 'data/3/pdata/1'

@dataclass
class CPMGInfo:
    data: list
    vc: list
    sw: list
    offset: list
    sfo: list
    nuc: list
    path: Path
    fmt: str


def _parse_jcampdx(names, path):
    """Retrieve parameters from files written in JCAMP-DX format."""
    with open(path, 'r') as fh:
        txt = fh.read()

    strinput = isinstance(names, str)
    if strinput:
        names = [names]

    params = []
    for name in names:
        pattern = r'##\$' + name + r'= (\(\d+\.\.\d+\)\n)?(.+?)#'
        try:
            params.append(
                re.search(pattern, txt, re.DOTALL)
                  .groups()[-1]
                  .replace('\n', ' ')
                  .rstrip()
            )
        except Exception as e:
            raise e

    return params[0] if strinput else params


def _get_binary_format(file):
    """Determine the formatting of the binary file (2rr)."""
    names = ['DTYPP', 'BYTORDP']
    encoding, endian = [int(p) for p in _parse_jcampdx(names, file)]
    return (('<' if endian == 0 else '>') + ('i4' if encoding == 0 else 'f8'))


def _unravel_data(data, si, xdim):
    newdata = np.zeros(data.shape, dtype=data.dtype)
    blocks = [i // j for i, j in zip(si, xdim)]
    x = 0
    for i in range(blocks[0]):
        for j in range(xdim[0]):
            for k in range(blocks[1]):
                idx = j + k * xdim[0] + i * blocks[0]
                newdata[x * xdim[1]:(x + 1) * xdim[1]] = \
                    data[idx * xdim[1]:(idx + 1) * xdim[1]]
                x += 1

    return newdata


def load():
    acqus = PATH.parents[1] / 'acqus'
    procs, proc2s = [PATH / f'proc{i}s' for i in ['', '2']]
    vclist = PATH.parents[1] / 'vclist'
    _2rr = PATH / '2rr'

    # Sweep width, offset, transmitter frequency
    sw, offset, sfo = \
        [[float(i)] for i in _parse_jcampdx(['SW_h', 'O1', 'SFO1'], acqus)]
    # Nucleus
    nuc = [re.search(r'<(.+?)>', _parse_jcampdx('NUC1', acqus)).group(1)]
    # Data size
    si = [int(_parse_jcampdx('SI', f)) for f in [proc2s, procs]]
    # Data partitioning
    xdim = [int(_parse_jcampdx('XDIM', f)) for f in [proc2s, procs]]
    # Formatting of 2rr file
    binfmt = _get_binary_format(procs)

    data = np.fromfile(_2rr, dtype=binfmt)
    data = _unravel_data(data, si, xdim).reshape(si)
    import matplotlib.pyplot as plt
    # plt.plot(data[0])
    # plt.show()
    # exit()
    for i in range(data.shape[0]):
        if all(data[i] == 0):
            break
    data = [ift(d)[:d.size // 2] for d in data[:i]]
    plt.plot(data[0])
    plt.show()

    with open(vclist, 'r') as fh:
        vc = [int(i) for i in re.split(r'\n', fh.read())]

    return CPMGInfo(data, vc, sw, offset, sfo, nuc, PATH, binfmt)


def estimate(info):
    region = [[5.1, 4.8]]
    noise_region = [[0.175, 0.075]]

    paths = [Path().cwd().resolve() / f'result/{i}' for i in info.vc]
    for path in paths:
        path.mkdir(mode=0o755, parents=True, exist_ok=True)

    for i, (vc, path, fid) in enumerate(zip(info.vc, paths, info.data)):
        estimator = Estimator(
            source='N/A', data=fid, path=info.path, sw=info.sw,
            off=info.offset, sfo=info.sfo, nuc=info.nuc, fmt=info.fmt,
        )
        estimator.view_data()
        estimator.frequency_filter(region=region, noise_region=noise_region)
        if i != 0:
            estimator.result = old_result

        else:
            estimator.matrix_pencil()

        estimator.nonlinear_programming(phase_variance=True)
        # old_result = estimator.get_result()

        # description = \
        #     f'Cyclosporin CPMG experiment.\nIteration {i + 1} (vc = {vc})'
        #
        # for fmt in ['txt', 'pdf', 'csv']:
        #     if fmt == 'pdf':
        #         description.replace('\n', '\\\\')
        #
        #     estimator.write_result(
        #         path=str(path / 'result'), fmt=fmt, force_overwrite=True,
        #         description=description,
        #     )

        estimator.plot_result().fig.savefig(path / 'figure.pdf')
        estimator.to_pickle(path=str(path / 'estimator'), force_overwrite=True)
        break


if __name__ == '__main__':
    info = load()
    estimate(info)
