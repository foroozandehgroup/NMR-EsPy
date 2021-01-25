#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift, ifftshift, ifft

import nmrespy.load as load

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

info = load.import_bruker(os.path.join(datadir, '1/pdata/1'), ask_convdta=False)
info.view_data()
info.frequency_filter(
    [5.3,5.18],
    [6.55,6.35],
    region_units='ppm',
    cut=True,
    cut_ratio=4,
)
info.matrix_pencil()
info.nonlinear_programming(phase_variance=True)
fig, *_ = info.plot_result()
fig.savefig('fig.pdf')

# info.write_result('hello', force_overwrite=True)
# info.write_result('hello', format='pdf', force_overwrite=True)
# fig, _, _, _ = info.plot_result()
# fig.savefig('hello.pdf')

# plt.plot(info.get_virtual_echo())
# plt.plot(np.imag(info.get_virtual_echo()))
# plt.show()

# def convert(value, conv):
#
#     sw = 1000
#     off = 500
#     n = 1000
#     sfo = 500
#
#     if conv == 'idx->hz':
#         return float(off + (sw / 2) - ((value * sw) / n))
#
#     elif conv == 'idx->ppm':
#         return float((off + sw / 2 - value * sw / n) / sfo)
#
#     elif conv == 'ppm->idx':
#         return int(round((off + (sw / 2) - sfo * value) * (n / sw)))
#
#     elif conv == 'ppm->hz':
#         return value * sfo
#
#     elif conv == 'hz->idx':
#         return int((n / sw) * (off + (sw / 2) - value))
#
#     elif conv == 'hz->ppm':
#         return value / sfo
#
#
# for i in range(11):
#     print(f'index: {i*100} -> ppm: ', convert(100*i, 'idx->ppm'))
#     print(f'index: {i*100} -> hz: ', convert(100*i, 'idx->hz'))
#
# for j in range(11):
#     print(f'ppm: {j/5} -> idx: ', convert(j/5, 'ppm->idx'))
#     print(f'ppm: {j/5} -> hz: ', convert(j/5, 'ppm->hz'))
#
# for k in range(11):
#     print(f'hz: {k*100} -> idx: ', convert(k*100, 'hz->idx'))
#     print(f'hz: {k*100} -> ppm: ', convert(k*100, 'hz->ppm'))
