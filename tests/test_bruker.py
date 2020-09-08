import unittest
import os

import numpy as np

import nmrespy as espy
import nmrespy.load as load

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class TestBruker(unittest.TestCase):

    maxDiff = None

    def test_import_fid(self):

        path = os.path.join(datadir, '1000')
        info = load.import_bruker_fid(path, ask_convdta=False)
        self.assertEqual(info.get_dtype(), 'raw')
        self.assertEqual(info.get_dim(), 1)
        self.assertIsInstance(info.get_data(), np.ndarray)
        self.assertEqual(info.get_data().shape, (32692,))
        self.assertEqual(info.get_datapath(), path)
        self.assertEqual(round(info.get_sw()[0], 4), 5494.5055)
        self.assertEqual(round(info.get_sw(unit='ppm')[0], 4), 10.9861)
        self.assertEqual(round(info.get_offset()[0], 4), 2249.2060)
        self.assertEqual(round(info.get_offset(unit='ppm')[0], 4), 4.4972)
        self.assertEqual(round(info.get_bf()[0], 4), 500.1300)
        self.assertEqual(round(info.get_sfo()[0], 4), 500.1322)
        self.assertEqual(info.get_nuc()[0], '1H')

    def test_import_pdata(self):

        path = os.path.join(datadir, '1/pdata/1')
        info = load.import_bruker_pdata(path)
        self.assertEqual(info.get_dtype(), 'pdata')
        self.assertEqual(info.get_dim(), 1)
        self.assertIsInstance(info.get_data(pdata_key='1r'), np.ndarray)
        self.assertIsInstance(info.get_data(pdata_key='1i'), np.ndarray)
        self.assertEqual(info.get_data(pdata_key='1r').shape, (32768,))
        self.assertEqual(info.get_data(pdata_key='1i').shape, (32768,))
        self.assertEqual(info.get_datapath(), path)
        self.assertEqual(round(info.get_sw()[0], 4), 5494.5055)
        self.assertEqual(round(info.get_sw(unit='ppm')[0], 4), 10.9861)
        self.assertEqual(round(info.get_offset()[0], 4), 2249.2060)
        self.assertEqual(round(info.get_offset(unit='ppm')[0], 4), 4.4972)
        self.assertEqual(round(info.get_bf()[0], 4), 500.1300)
        self.assertEqual(round(info.get_sfo()[0], 4), 500.1322)
        self.assertEqual(info.get_nuc()[0], '1H')

    def test_pickle(self):

        path = os.path.join(datadir, '1000')
        info = load.import_bruker_fid(path, ask_convdta=False)
        info.pickle_save('pickle_test.pkl')
        info_new = load.pickle_load('pickle_test.pkl')

        # consider string beyond memory location (will differ)
        self.assertEqual(str(info_new).split('>')[1],
                         str(info).split('>')[1])
        os.remove('pickle_test.pkl')

    def test_virt_echo(self):

        path = os.path.join(datadir, '1/pdata/1')
        info = load.import_bruker_pdata(path)
        info.virtual_echo(highs=(5.285,), lows=(5.180,), highs_n=(9.5,),
                          lows_n=(9.2,))

    def test_mpm(self):

        path = os.path.join(datadir, '1/pdata/1')
        info = load.import_bruker_pdata(path)
        info.virtual_echo(highs=(5.285,), lows=(5.180,), highs_n=(9.5,),
                          lows_n=(9.2,))
        info.matrix_pencil(trim=(4096,))

    def test_plot(self):







if __name__ == '__main__':
        unittest.main()
