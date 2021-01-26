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

    # tests various aspects of nmrespy.load
    # creates an instance of NMRESPyBruker which is considered in
    # later tests
    def setUp(self):
        """Tests nmrespy.load and simulataneously creates two instances
        for testing in later test methods:
        self.fid_info
        self.pdata_info
        """

        self.pickle_dir = Path().cwd() / 'pickled_objects'

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
        loaded_info = load.pickle_load(self.pickle_dir / 'NMREsPyBruker.pkl')

    def test_dunder_methods(self):
        """Simply checks no errors are raised when __str__ and __repr__
        are called"""

        # test __str___
        string = self.fid_info.__str__()
        # test __repr__
        representaiton = self.fid_info.__repr__()


    def test_get_initial(self):
        """Test the various methods that get infomation from the class.
        At this stage, attributes related to frequency filtration and
        estimation will be None"""

        # loop over fid and pdata instances
        insts = [self.fid_info, self.pdata_info]
        paths = [Path().cwd() / 'data/1000', Path().cwd() / 'data/1/pdata/1']
        dtypes = ['fid', 'pdata']
        ns = [32692, 16384]

        # paths to pickled data (chemical shift arrays and time-point arrays
        # for comparison
        pickle_paths = []
        for dtype in dtypes:
            sublst = []
            sublst.append(self.pickle_dir / f'shifts_1d_{dtype}_hz.pkl')
            sublst.append(self.pickle_dir / f'shifts_1d_{dtype}_ppm.pkl')
            sublst.append(self.pickle_dir / f'tp_1d_{dtype}.pkl')
            pickle_paths.append(iter(sublst))

        # test the get methods!
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
                # check objects are length-1 lists
                self.assertIsInstance(obj, list)
                self.assertEqual(len(obj), 1)

                # check list elements are the correct value
                if i in [0, 7]:
                    # n and nucleus (int and str types)
                    self.assertEqual(obj[0], value)
                else:
                    # other parameters are floats
                    self.assertEqual(round(obj[0], 3), value)

            # check get_shifts and get_tp
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

            # get methods that correspond to attributes which are  None at
            # the point of initialisation using bruker_load
            none_methods = [
                inst.get_region,
                inst.get_noise_region,
                inst.get_p0,
                inst.get_p1,
                inst.get_filtered_spectrum,
                inst.get_filtered_signal,
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


    def test_filter(self):
        # TODO
        region = [[5.06, 4.76]]
        noise_region = [[9.7, 9.3]]
        self.pdata_info.frequency_filter(region, noise_region, cut_ratio=5)
        self.pdata_info.matrix_pencil()



if __name__ == '__main__':
        unittest.main()
