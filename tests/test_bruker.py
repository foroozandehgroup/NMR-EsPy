#!/usr/bin/python3

import unittest
from pathlib import Path
import pickle

import numpy as np
from nmrespy.core import NMREsPyBruker
import nmrespy._errors as errors
import nmrespy.load as load

datadir = Path().absolute() / 'data'

class TestBruker(unittest.TestCase):

    maxDiff = None

    # tests various aspects of nmrespy.load
    # creates an instance of NMRESPyBruker which is considered in
    # later tests
    def setUp(self):

        # PATHS
        # user's home dir - assume this does not contain any of the
        # requisite files
        invalid_path = Path().home()

        # directory containing sample datasets
        data_dir = Path().cwd() / 'data'
        # path to 3D data set - will raise a MoreTHanTwoDimError
        threedim_path = data_dir / '3'
        # dir containing fid data which has been digitally filtered
        fid_dir = data_dir / '1000'
        # path to processed data
        pdata_dir = data_dir / '1/pdata/1'

        # ---test load._get_param---
        acqus_file = fid_dir / 'acqus'
        self.assertEqual(
            load._get_param('SFO1', acqus_file, type_=float),
            500.132249206,
        )
        self.assertEqual(
            load._get_param('NUC1', acqus_file),
            '<1H>',
        )
        with self.assertRaises(errors.ParameterNotFoundError):
            invalid_param = load._get_param('AAAGH', acqus_file)


        # ---test import_bruker---
        # this simply generates an instance of NMREsPyBruker
        # see other methods below for rigourous test of the class's
        # functionality
        with self.assertRaises(errors.InvalidDirectoryError):
            invalid_info = load.import_bruker(invalid_path)

        with self.assertRaises(errors.MoreThanTwoDimError):
            threedim_info = load.import_bruker(threedim_path)


        self.fid_info = load.import_bruker(fid_dir, ask_convdta=False)
        self.pdata_info = load.import_bruker(pdata_dir)

        for info in [self.fid_info, self.pdata_info]:
            self.assertIsInstance(info, NMREsPyBruker)

        # ---test pickle_load---
        loaded_info = load.pickle_load('NMREsPyBruker.pkl')


    def test_get(self):
        insts = [self.fid_info, self.pdata_info]
        paths = [Path().cwd() / 'data/1000', Path().cwd() / 'data/1/pdata/1']
        dtypes = ['fid', 'pdata']
        ns = [32692, 16384]

        pickle_paths = []
        for dtype in dtypes:#
            sublst = []
            sublst.append(f'shifts_1d_{dtype}_hz.pkl')
            sublst.append(f'shifts_1d_{dtype}_ppm.pkl')
            sublst.append(f'tp_1d_{dtype}.pkl')
            pickle_paths.append(iter(sublst))

        for inst, path, dtype, n, ppaths in zip(insts, paths, dtypes, ns, pickle_paths):
            self.assertEqual(inst.get_datapath(), path)
            self.assertEqual(inst.get_datapath(type_='str'), str(path))
            self.assertEqual(inst.get_dtype(), dtype)
            self.assertEqual(inst.get_dim(), 1)
            self.assertIsInstance(inst.get_data(), np.ndarray)
            self.assertEqual(inst.get_data().shape, (n,))

            n_ = inst.get_n()
            sw_h = inst.get_sw()
            sw_p = inst.get_sw(unit='ppm')
            off_h = inst.get_offset()
            off_p = inst.get_offset(unit='ppm')
            bf = inst.get_bf()
            sfo = inst.get_sfo()
            nuc = inst.get_nucleus()

            objects = [
                n_, sw_h, sw_p, off_h, off_p, bf, sfo, nuc,
            ]
            values = [
                n, 5494.505, 10.986, 2249.206, 4.497, 500.130, 500.132, '1H',
            ]

            for i, (obj, value) in enumerate(zip(objects, values)):
                self.assertIsInstance(obj, list)
                self.assertEqual(len(obj), 1)

                if i in [0, 7]:
                    # n and nucleus (int and str types)
                    self.assertEqual(obj[0], value)
                else:
                    # other parameters are floats
                    self.assertEqual(round(obj[0], 3), value)

            with open(next(ppaths), 'rb') as fh:
                shifts_hz = pickle.load(fh)

            with open(next(ppaths), 'rb') as fh:
                shifts_ppm = pickle.load(fh)

            with open(next(ppaths), 'rb') as fh:
                tp = pickle.load(fh)


            self.assertTrue(
                np.array_equal(inst.get_shifts(), shifts_hz)
            )
            self.assertTrue(
                np.array_equal(inst.get_shifts(unit='ppm'), shifts_ppm)
            )
            self.assertTrue(
                np.array_equal(inst.get_tp(), tp)
            )

            # get methods that are None at the point of initialisation using
            # bruker_load
            none_methods = [
                inst.get_region,
                inst.get_noise_region,
                inst.get_p0,
                inst.get_p1,
                inst.get_filtered_spectrum,
                inst.get_virtual_echo,
                inst.get_half_echo,
                inst.get_filtered_n,
                inst.get_filtered_sw,
                inst.get_filtered_offset,
                inst.get_theta0,
                inst.get_theta,
            ]

            for method in none_methods:
                with self.assertRaises(errors.AttributeIsNoneError):
                    value = method()
                self.assertIsNone(method(kill=False))





    # def test_import_pdata(self):
    #
    #     path = os.path.join(datadir, '1/pdata/1')
    #     info = load.import_bruker_pdata(path)
    #     self.assertEqual(info.get_dtype(), 'pdata')
    #     self.assertEqual(info.get_dim(), 1)
    #     self.assertIsInstance(info.get_data(pdata_key='1r'), np.ndarray)
    #     self.assertIsInstance(info.get_data(pdata_key='1i'), np.ndarray)
    #     self.assertEqual(info.get_data(pdata_key='1r').shape, (32768,))
    #     self.assertEqual(info.get_data(pdata_key='1i').shape, (32768,))
    #     self.assertEqual(info.get_datapath(), path)
    #     self.assertEqual(round(info.get_sw()[0], 4), 5494.5055)
    #     self.assertEqual(round(info.get_sw(unit='ppm')[0], 4), 10.9861)
    #     self.assertEqual(round(info.get_offset()[0], 4), 2249.2060)
    #     self.assertEqual(round(info.get_offset(unit='ppm')[0], 4), 4.4972)
    #     self.assertEqual(round(info.get_bf()[0], 4), 500.1300)
    #     self.assertEqual(round(info.get_sfo()[0], 4), 500.1322)
    #     self.assertEqual(info.get_nuc()[0], '1H')
    #
    # def test_pickle(self):
    #
    #     path = os.path.join(datadir, '1000')
    #     info = load.import_bruker_fid(path, ask_convdta=False)
    #     info.pickle_save('pickle_test.pkl')
    #     info_new = load.pickle_load('pickle_test.pkl')
    #
    #     # consider string beyond memory location (will differ)
    #     self.assertEqual(str(info_new).split('>')[1],
    #                      str(info).split('>')[1])
    #     os.remove('pickle_test.pkl')
    #
    # def test_mpm(self):
    #
    #     path = os.path.join(datadir, '1/pdata/1')
    #     info = load.import_bruker_pdata(path)
    #     info.virtual_echo(highs=(5.285,), lows=(5.180,), highs_n=(9.5,),
    #                       lows_n=(9.2,))
    #     info.matrix_pencil(trim=(4096,))


if __name__ == '__main__':
        unittest.main()
