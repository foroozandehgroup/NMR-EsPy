import os
from pathlib import Path
import pytest
import re
import subprocess

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from nmrespy._misc import FrequencyConverter
from nmrespy import _cols as cols, plot as nplot, sig


VIEW_PEAKS = False
VIEW_RESULT_PLOTS = True

def make_stylesheet(path):
    # Ensure entries are alphatically in order, except for axes.prop_cycle
    # which is at the end
    txt = ('axes.edgecolor: .15\n'
           'axes.facecolor: white\n'
           'axes.grid: False\n'
           'axes.linewidth: 1.25\n'
           'grid.color: .8\n'
           'patch.facecolor: 006BA4\n'
           'xtick.major.size: 0.0\n'
           'xtick.minor.size: 0.0\n'
           'ytick.major.size: 0.0\n'
           'ytick.minor.size: 0.0\n'
           '# Tableau colorblind 10 palette\n'
           'axes.prop_cycle: cycler(\'color\', [\'006BA4\', '
                 '\'FF800E\', \'ABABAB\', \'595959\', \'5F9ED1\', '
                 '\'C85200\', \'898989\', \'A2C8EC\', \'FFBC79\', '
                 '\'CFCFCF\'])')

    with open(path, 'w') as fh:
        fh.write(txt)

    return txt

def test_extract_rc():
    styledir = Path(mpl.__file__).resolve().parent / "mpl-data/stylelib"
    files = [f for f in styledir.iterdir()]

    # Ensure _extract_rc works when you provide a stylesheet name
    # as well as a path to a stylesheet, and ensure the results are
    # identical
    for f in files:
        name = re.match(r'^(.+?)\.mplstyle$', f.name).group(1)
        with open(f, 'r') as fh:
            # Seems as if some stylesheets are empty...
            if not fh.read():
                continue
        assert nplot._extract_rc(name) == nplot._extract_rc(f)

    with pytest.raises(ValueError):
        nplot._extract_rc('not_a_stylesheet')

    # Create a file that should raise an error
    with open('fail.mplstyle', 'w') as fh:
        fh.write('This is not a valid stylesheet')
    with pytest.raises(ValueError):
        nplot._extract_rc('fail.mplstyle')
    os.remove('fail.mplstyle')

    # Create a stylesheet that should return a valid rc
    with open('pass.mplstyle', 'w') as fh:
        fh.write('# This should pass\n'
                 'axes.grid: False\n'
                 'axes.facecolor: white')
    # Note revesal of attribute order. MPL sorts the attributes
    # alphabetically
    assert nplot._extract_rc('pass.mplstyle') == \
           'axes.facecolor: white\naxes.grid: False'
    os.remove('pass.mplstyle')


def test_create_rc():
    path = 'text.mplstyle'
    original_rc = make_stylesheet(path)
    # Should be the same as `original_rc` without the comment lines,
    # and with axes.prop_cycle line edited.
    edited_rc = nplot._create_rc(path, None, 1)

    # Convert to lists for each newline, and remove any comment lines
    original_rc_list = list(filter(
        lambda ln: ln[0] != '#', original_rc.split('\n')
    ))
    edited_rc_list = edited_rc.split('\n')

    unmatched = []
    for i, (orig, edit) in enumerate(zip(original_rc_list, edited_rc_list)):
        if orig != edit:
            unmatched.append(i)
    # Ensure that only one line is unmatched, and that this is the last
    # line (axes.prop_cycle)
    assert len(unmatched) == 1 and unmatched[0] == len(original_rc_list) - 1
    os.remove(path)


def test_configure_oscillator_colors():
    argss = [
        (None, 5),
        ('viridis', 5),
        ('inferno', 3),
        ('tomato', 3),
        (['forestgreen', '#ff0000', '0.5', (0.7, 0.8, 0.3)], 4)
    ]

    results = [
        ['#1063e0', '#eb9310','#2bb539', '#d4200c'],
        ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        ['#000004', '#bc3754', '#fcffa4'],
        ['#ff6347'],
        ['#228b22', '#ff0000', '#808080', '#b2cc4c'],
    ]

    for args, result in zip(argss, results):
        assert nplot._configure_oscillator_colors(*args) == result

    with pytest.raises(ValueError) as exc_info:
        nplot._configure_oscillator_colors(
            ['blah', 'tomato', (1.2, 0.6, 0.4)], 4
        )

    assert str(exc_info.value) == \
        (f'{cols.R}The following entries in `oscillator_colors` could '
          'not be recognised as valid colours in matplotlib:\n'
         f'--> \'blah\'\n--> (1.2, 0.6, 0.4){cols.END}')


def test_get_region_slice():
    # 1D example
    region_hz = [[3.5, 1.5]]
    hz_result = slice(1, 4, None),
    region_ppm = [[0.05, -0.15]]
    ppm_result = slice(4, 7, None),
    n = [10]
    sw = [9.]
    offset = [0.]
    sfo = [10.]

    assert nplot._get_region_slice('hz', region_hz, n, sw, offset, sfo) == \
           hz_result
    assert nplot._get_region_slice('hz', region_hz, n, sw, offset, None) == \
           hz_result
    assert nplot._get_region_slice('ppm', region_ppm, n, sw, offset, sfo) == \
           ppm_result
    assert nplot._get_region_slice('hz', None, n, sw, offset, sfo) == \
           (slice(0, 10, None),)

    # 2D example
    region_hz = [[3.5, 1.5], [3.5, 1.5]]
    hz_result = (slice(2, 7, None), slice(3, 6, None))
    region_ppm = [[0.2, -0.15], [0.15, -0.15]]
    ppm_result = (slice(5, 14, None), slice(5, 9, None))
    n = [20, 10]
    sw = [9., 9.]
    offset = [0., 2.]
    sfo = [10., 10.]

    assert nplot._get_region_slice('hz', region_hz, n, sw, offset, sfo) == \
           hz_result
    assert nplot._get_region_slice('hz', region_hz, n, sw, offset, None) == \
           hz_result
    assert nplot._get_region_slice('ppm', region_ppm, n, sw, offset, sfo) == \
           ppm_result
    assert nplot._get_region_slice('hz', None, n, sw, offset, sfo) == \
           (slice(0, 20, None), slice(0, 10, None))


class Stuff:
    def __init__(self):
        self.params = np.array([
            [1, 0, 1000, 50],
            [3, 0, 1050, 50],
            [6, 0, 1100, 50],
            [3, 0, 1150, 50],
            [1, 0, 1200, 50]
        ])
        self.n = [4096]
        self.sw = [5000.]
        self.offset = [0.]
        self.sfo = [500.]
        self.region_hz = [[1400., 800.]]
        self.converter = FrequencyConverter(
            self.n, self.sw, self.offset, self.sfo,
        )

    def unpack(self):
        return (
            self.params,
            self.n,
            self.sw,
            self.offset,
            self.sfo,
            self.region_hz,
            self.converter,
        )


def test_generate_peaks():
    result, n, sw, offset, sfo, region_hz, converter = Stuff().unpack()
    region_idx = converter.convert(region_hz, 'hz->idx')
    slice_ = (slice(region_idx[0][0], region_idx[0][1] + 1, None),)
    peaks = nplot._generate_peaks(result, n, sw, offset, slice_)

    if VIEW_PEAKS:
        shifts = sig.get_shifts(n, sw, offset=offset)[0][slice_]
        fig, ax = plt.subplots()
        lines = [ax.plot(shifts, peak) for peak in peaks]
        ax.set_xlim(reversed(ax.get_xlim()))
        ax.set_xlabel('$\omega$ (Hz)')
        ax.set_title('Should have a qunitet with peaks at ' +
                     ', '.join([str(i) for i in range(1000, 1250, 50)]))
        plt.show()


def test_plot_result():
    result, n, sw, offset, sfo, region_hz, converter = Stuff().unpack()
    data = sig.make_fid(result, n, sw, offset, snr=30.)[0]
    region_ppm = converter.convert(region_hz, 'hz->ppm')

    kwargss = [
        {},
        {'region': region_ppm},
        {'region': region_ppm, 'plot_residual': False, 'plot_model': True},
        {'region': region_ppm, 'residual_shift': -100., 'plot_model': True,
         'model_shift': 100.},
        {'region': region_hz, 'shifts_unit': 'hz'},
        {'region': region_ppm, 'nucleus': ['1H']},
        {'region': region_ppm, 'data_color': '#ff0000',
         'residual_color': '#00ff00', 'model_color': '#0000ff',
         'plot_model': True},
        {'region': region_ppm,
         'oscillator_colors': ['#ff0000', '#00ff00', '#0000ff']},
        {'region': region_ppm, 'show_labels': False},
        {'region': region_ppm, 'stylesheet': 'ggplot'}
    ]

    titles = [
        'Default format',
        'Specified region',
        'Residual hidden, Model shown',
        'Residual shift -100, Model shift +100',
        'Region in Hz',
        'Nucleus 1H',
        'Data colour R, Residual colour G, Model colour B',
        'Oscillator colours cycle through R, G, B',
        'No labels',
        'ggplot stylesheet'
    ]

    for kwargs, title in zip(kwargss, titles):
        plot = nplot.plot_result(
            data, result, sw, offset, sfo=sfo, **kwargs
        )

        if VIEW_RESULT_PLOTS:
            plot.ax.set_title(title)
            plot.fig.savefig('test.pdf', dpi=300)
            subprocess.run(['evince', 'test.pdf'])
            os.remove('test.pdf')
