from pathlib import Path
import pytest
import subprocess
from context import nmrespy
import numpy as np
import numpy.linalg as nlinalg
from nmrespy import ExpInfo, sig, write as nwrite

USER_INPUT = True
FILE = Path(__file__).resolve().parent / 'file.pdf'


class Stuff:
    def __init__(self, dim: int = 1, inc_sfo: bool = True):
        self.params = np.zeros((5, 2 + 2 * dim))
        self.params[:, 0] = np.array([1, 3, 6, 3, 1])
        self.params[:, 1] = np.zeros(5)
        for i in range(2, 2 + dim):
            self.params[:, i] = np.linspace(1000, 1200, 5)
        for i in range(2 + dim, 2 + 2 * dim):
            self.params[:, i] = 50 * np.ones(5)
        self.errors = self.params / 100
        pts = 1024
        sw = 5000.
        offset = 0.
        sfo = 500. if inc_sfo else None
        self.expinfo = ExpInfo(pts=pts, sw=sw, offset=offset, sfo=sfo, dim=dim)

        self.integrals = np.array([sig.oscillator_integral(osc, self.expinfo)
                                   for osc in self.params])

    def unpack(self, *attrs):
        return (self.__dict__[attr] for attr in attrs)


class TestMakeParameterTable():
    def test_onedim_sfo_given(self):
        params, expinfo, integrals = \
            Stuff().unpack('params', 'expinfo', 'integrals')
        table = nwrite._make_parameter_table(params, expinfo)
        assert table.shape == (params.shape[0], 8)
        assert np.array_equal(table[:, 0], np.arange(1, params.shape[0] + 1))
        assert np.array_equal(table[:, 1], params[:, 0])
        assert np.array_equal(table[:, 2], params[:, 1])
        assert np.array_equal(table[:, 3], params[:, 2])
        assert np.array_equal(table[:, 4], params[:, 2] / expinfo.sfo[0])
        assert np.array_equal(table[:, 5], params[:, 3])
        assert np.array_equal(table[:, 6], integrals)
        assert np.array_equal(table[:, 7], integrals / nlinalg.norm(integrals))

    def test_onedim_sfo_none(self):
        params, expinfo, integrals = \
            Stuff(inc_sfo=False).unpack('params', 'expinfo', 'integrals')
        table = nwrite._make_parameter_table(params, expinfo)
        assert table.shape == (params.shape[0], 7)
        assert np.array_equal(table[:, 0], np.arange(1, params.shape[0] + 1))
        assert np.array_equal(table[:, 1], params[:, 0])
        assert np.array_equal(table[:, 2], params[:, 1])
        assert np.array_equal(table[:, 3], params[:, 2])
        assert np.array_equal(table[:, 4], params[:, 3])
        assert np.array_equal(table[:, 5], integrals)
        assert np.array_equal(table[:, 6], integrals / nlinalg.norm(integrals))

    def test_twodim_sfo_given(self):
        params, expinfo, integrals = \
            Stuff(dim=2).unpack('params', 'expinfo', 'integrals')
        table = nwrite._make_parameter_table(params, expinfo)
        assert table.shape == (params.shape[0], 11)
        assert np.array_equal(table[:, 0], np.arange(1, params.shape[0] + 1))
        assert np.array_equal(table[:, 1], params[:, 0])
        assert np.array_equal(table[:, 2], params[:, 1])
        assert np.array_equal(table[:, 3], params[:, 2])
        assert np.array_equal(table[:, 4], params[:, 3])
        assert np.array_equal(table[:, 5], params[:, 2] / expinfo.sfo[0])
        assert np.array_equal(table[:, 6], params[:, 3] / expinfo.sfo[1])
        assert np.array_equal(table[:, 7], params[:, 4])
        assert np.array_equal(table[:, 8], params[:, 5])
        assert np.array_equal(table[:, 9], integrals)
        assert np.array_equal(
            table[:, 10], integrals / nlinalg.norm(integrals))

    def test_twodim_sfo_none(self):
        params, expinfo, integrals = \
            Stuff(inc_sfo=False, dim=2).unpack(
                'params', 'expinfo', 'integrals'
            )
        table = nwrite._make_parameter_table(params, expinfo)
        assert table.shape == (params.shape[0], 9)
        assert np.array_equal(table[:, 0], np.arange(1, params.shape[0] + 1))
        assert np.array_equal(table[:, 1], params[:, 0])
        assert np.array_equal(table[:, 2], params[:, 1])
        assert np.array_equal(table[:, 3], params[:, 2])
        assert np.array_equal(table[:, 4], params[:, 3])
        assert np.array_equal(table[:, 5], params[:, 4])
        assert np.array_equal(table[:, 6], params[:, 5])
        assert np.array_equal(table[:, 7], integrals)
        assert np.array_equal(table[:, 8], integrals / nlinalg.norm(integrals))


class TestMakeErrorTable():
    def test_onedim_sfo_given(self):
        errors, expinfo, integrals = \
            Stuff().unpack('errors', 'expinfo', 'integrals')
        table = nwrite._make_error_table(errors, expinfo)
        assert table.shape == (errors.shape[0], 8)
        assert np.all(np.isnan(table[:, 0]))
        assert np.array_equal(table[:, 1], errors[:, 0])
        assert np.array_equal(table[:, 2], errors[:, 1])
        assert np.array_equal(table[:, 3], errors[:, 2])
        assert np.array_equal(table[:, 4], errors[:, 2] / expinfo.sfo[0])
        assert np.array_equal(table[:, 5], errors[:, 3])
        assert np.all(np.isnan(table[:, 6:]))

    def test_onedim_sfo_none(self):
        errors, expinfo, integrals = \
            Stuff(inc_sfo=False).unpack('errors', 'expinfo', 'integrals')
        table = nwrite._make_error_table(errors, expinfo)
        assert table.shape == (errors.shape[0], 7)
        assert np.all(np.isnan(table[:, 0]))
        assert np.array_equal(table[:, 1], errors[:, 0])
        assert np.array_equal(table[:, 2], errors[:, 1])
        assert np.array_equal(table[:, 3], errors[:, 2])
        assert np.array_equal(table[:, 4], errors[:, 3])
        assert np.all(np.isnan(table[:, 5:]))

    def test_twodim_sfo_given(self):
        errors, expinfo, integrals = \
            Stuff(dim=2).unpack('errors', 'expinfo', 'integrals')
        table = nwrite._make_error_table(errors, expinfo)
        assert table.shape == (errors.shape[0], 11)
        assert np.all(np.isnan(table[:, 0]))
        assert np.array_equal(table[:, 1], errors[:, 0])
        assert np.array_equal(table[:, 2], errors[:, 1])
        assert np.array_equal(table[:, 3], errors[:, 2])
        assert np.array_equal(table[:, 4], errors[:, 3])
        assert np.array_equal(table[:, 5], errors[:, 2] / expinfo.sfo[0])
        assert np.array_equal(table[:, 6], errors[:, 3] / expinfo.sfo[1])
        assert np.array_equal(table[:, 7], errors[:, 4])
        assert np.array_equal(table[:, 8], errors[:, 5])
        assert np.all(np.isnan(table[:, 9:]))

    def test_twodim_sfo_none(self):
        errors, expinfo, integrals = \
            Stuff(inc_sfo=False, dim=2).unpack(
                'errors', 'expinfo', 'integrals'
            )
        table = nwrite._make_error_table(errors, expinfo)
        assert table.shape == (errors.shape[0], 9)
        assert np.all(np.isnan(table[:, 0]))
        assert np.array_equal(table[:, 1], errors[:, 0])
        assert np.array_equal(table[:, 2], errors[:, 1])
        assert np.array_equal(table[:, 3], errors[:, 2])
        assert np.array_equal(table[:, 4], errors[:, 3])
        assert np.array_equal(table[:, 5], errors[:, 4])
        assert np.array_equal(table[:, 6], errors[:, 5])
        assert np.all(np.isnan(table[:, 7:]))


def test_format_error_table():
    errors, expinfo = Stuff().unpack('errors', 'expinfo')
    table = nwrite._make_error_table(errors, expinfo)
    fmtstr = lambda x: nwrite._strval(x, 4, (-2, 3), 'txt')
    fmttable = nwrite._format_error_table(table, fmtstr)


def test_construct_table():
    params, errors, expinfo = Stuff().unpack('params', 'errors', 'expinfo')
    def fmtval(value):
        return nwrite._format_value(value, 5, (-2, 3), 'txt')
    table = nwrite._construct_paramtable(
        params, errors, expinfo, 'txt', fmtval
    )
    print(nwrite._txt_tabular(table, titles=False))


def test_format_string():
    fmt = 'txt'
    sig_figs = 4
    sci_lims = (-2, 3)
    tests = {
        123456789012: '1.235e+11',
        1435678.349: '1.436e+6',
        0.0000143: '1.43e-5',
        -0.000004241: '-4.241e-6',
        1000.: '1e+3',
        -999.9: '-999.9',
        999.99: '1e+3',
        0.: '0',
        0.01: '1e-2',
        0.09999: '9.999e-2',
        0.1: '0.1'
    }

    for value, result in tests.items():
        assert nwrite._format_value(value, sig_figs, sci_lims, fmt) == result


def test_me():
    assert nwrite._configure_save_path('file', 'pdf', True) == (True, FILE)

    # Make file `file.pdf` and assert that same result is returned when
    # `force_overwrite` is `True`
    subprocess.run(['touch', 'file.pdf'])
    assert nwrite._configure_save_path('file', 'pdf', True) == \
        (True, FILE)

    if USER_INPUT:
        # Set `force_overwrite` to False and ensure the user is prompted
        print(f"\n{cols.R}PLEASE PRESS y{cols.END}")
        assert nwrite._configure_save_path('file', 'pdf', False) == \
            (True, FILE)

        print(f"{cols.R}PLEASE PRESS n{cols.END}")
        assert nwrite._configure_save_path('file', 'pdf', False) == \
            (False,
             f'{cols.R}Overwrite of file {str(FILE)} denied. Result '
             f'file will not be written.{cols.END}')

    subprocess.run(['rm', 'file.pdf'])

    # Non-existent directory
    invalid_path = FILE.parent / 'extra_dir' / 'file'
    assert nwrite._configure_save_path(invalid_path, 'pdf', False) == \
        (False,
         f'{cols.R}The directory specified by `path` does not exist:\n'
         f'{invalid_path.parent}{cols.END}')


def test_raise_error():
    msg = 'Value is not valid.'
    with pytest.raises(ValueError) as exc_info:
        nwrite.raise_error(ValueError, msg, True)
    assert str(exc_info.value) == msg

    assert nwrite.raise_error(ValueError, msg, False) is None
