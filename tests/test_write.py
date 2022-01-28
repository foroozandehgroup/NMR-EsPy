from pathlib import Path
import pytest
import subprocess

import numpy as np
import numpy.linalg as nlinalg

from nmrespy import RED, END, ExpInfo, sig, write as nwrite

USER_INPUT = True


class Stuff:
    def __init__(
        self, dim: int = 1, inc_nuc: bool = True, inc_sfo: bool = True
    ) -> None:
        self.params = np.zeros((5, 2 + 2 * dim))
        self.params[:, 0] = np.array([1, 3, 6, 3, 1])
        self.params[:, 1] = np.zeros(5)
        for i in range(2, 2 + dim):
            self.params[:, i] = np.linspace(1000, 1200, 5)
        for i in range(2 + dim, 2 + 2 * dim):
            self.params[:, i] = 50 * np.ones(5)
        self.errors = self.params / 100
        pts = 1024
        sw = 5000.0
        offset = 0.0
        sfo = 500.0 if inc_sfo else None
        nuclei = "1H" if inc_nuc else None
        self.expinfo = ExpInfo(
            pts=pts, sw=sw, offset=offset, sfo=sfo, nuclei=nuclei, dim=dim
        )
        self.integrals = np.array(
            [sig.oscillator_integral(osc, self.expinfo) for osc in self.params]
        )

    def unpack(self, *attrs):
        return (self.__dict__[attr] for attr in attrs)


class TestMakeParameterTable:
    def test_onedim_sfo_given(self):
        params, expinfo, integrals = Stuff().unpack("params", "expinfo", "integrals")
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
        params, expinfo, integrals = Stuff(inc_sfo=False).unpack(
            "params", "expinfo", "integrals"
        )
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
        params, expinfo, integrals = Stuff(dim=2).unpack(
            "params", "expinfo", "integrals"
        )
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
        assert np.array_equal(table[:, 10], integrals / nlinalg.norm(integrals))

    def test_twodim_sfo_none(self):
        params, expinfo, integrals = Stuff(inc_sfo=False, dim=2).unpack(
            "params", "expinfo", "integrals"
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


class TestMakeErrorTable:
    def test_onedim_sfo_given(self):
        errors, expinfo, integrals = Stuff().unpack("errors", "expinfo", "integrals")
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
        errors, expinfo, integrals = Stuff(inc_sfo=False).unpack(
            "errors", "expinfo", "integrals"
        )
        table = nwrite._make_error_table(errors, expinfo)
        assert table.shape == (errors.shape[0], 7)
        assert np.all(np.isnan(table[:, 0]))
        assert np.array_equal(table[:, 1], errors[:, 0])
        assert np.array_equal(table[:, 2], errors[:, 1])
        assert np.array_equal(table[:, 3], errors[:, 2])
        assert np.array_equal(table[:, 4], errors[:, 3])
        assert np.all(np.isnan(table[:, 5:]))

    def test_twodim_sfo_given(self):
        errors, expinfo, integrals = Stuff(dim=2).unpack(
            "errors", "expinfo", "integrals"
        )
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
        errors, expinfo, integrals = Stuff(inc_sfo=False, dim=2).unpack(
            "errors", "expinfo", "integrals"
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


class TestConstructInfotable:
    def test_onedim(self):
        (expinfo,) = Stuff(inc_sfo=False, inc_nuc=False).unpack("expinfo")
        infotable = nwrite._construct_infotable(expinfo)
        assert len(infotable) == 3
        assert infotable[0] == ["Parameter", "F1"]
        assert infotable[1] == ["Sweep width (Hz)", "5000.0"]
        assert infotable[2] == ["Transmitter offset (Hz)", "0.0"]

    def test_onedim_sfo(self):
        (expinfo,) = Stuff(inc_nuc=False).unpack("expinfo")
        infotable = nwrite._construct_infotable(expinfo)
        assert len(infotable) == 6
        assert infotable[0] == ["Parameter", "F1"]
        assert infotable[1] == ["Transmitter frequency (MHz)", "500.0"]
        assert infotable[2] == ["Sweep width (Hz)", "5000.0"]
        assert infotable[3] == ["Sweep width (ppm)", "10.0"]
        assert infotable[4] == ["Transmitter offset (Hz)", "0.0"]
        assert infotable[5] == ["Transmitter offset (ppm)", "0.0"]

    def test_onedim_sfo_nuc(self):
        (expinfo,) = Stuff().unpack("expinfo")
        infotable = nwrite._construct_infotable(expinfo)
        assert len(infotable) == 7
        assert infotable[0] == ["Parameter", "F1"]
        assert infotable[1] == ["Nucleus", "1H"]
        assert infotable[2] == ["Transmitter frequency (MHz)", "500.0"]
        assert infotable[3] == ["Sweep width (Hz)", "5000.0"]
        assert infotable[4] == ["Sweep width (ppm)", "10.0"]
        assert infotable[5] == ["Transmitter offset (Hz)", "0.0"]
        assert infotable[6] == ["Transmitter offset (ppm)", "0.0"]


def test_format_string():
    fmt = "txt"
    sig_figs = 4
    sci_lims = (-2, 3)
    tests = {
        123456789012: "1.235e+11",
        1435678.349: "1.436e+6",
        0.0000143: "1.43e-5",
        -0.000004241: "-4.241e-6",
        1000.0: "1e+3",
        -999.9: "-999.9",
        999.99: "1e+3",
        0.0: "0",
        0.01: "1e-2",
        0.09999: "9.999e-2",
        0.1: "0.1",
    }

    for value, result in tests.items():
        assert nwrite._format_value(value, sig_figs, sci_lims, fmt) == result


class TestConfigureSavePath:
    filepath = Path(__file__).resolve().parent / "file.pdf"

    def test_file_doesnt_exist(self):
        assert nwrite._configure_save_path("file", "pdf", True) == self.filepath

    def test_file_exists_force_true(self):
        subprocess.run(["touch", "file.pdf"])
        assert nwrite._configure_save_path("file", "pdf", True) == self.filepath
        subprocess.run(["rm", "file.pdf"])

    def test_file_exists_force_false_input_yes(self, monkeypatch):
        subprocess.run(["touch", "file.pdf"])
        monkeypatch.setattr("builtins.input", lambda: "y")
        assert nwrite._configure_save_path("file", "pdf", False) == self.filepath
        subprocess.run(["rm", "file.pdf"])

    def test_file_exists_force_false_input_no(self, monkeypatch):
        subprocess.run(["touch", "file.pdf"])
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert nwrite._configure_save_path("file", "pdf", False) is None
        subprocess.run(["rm", "file.pdf"])

    def test_invalid_dir(self):
        invalid_path = self.filepath.parent / "extra_dir" / "file"
        with pytest.raises(ValueError) as exc_info:
            nwrite._configure_save_path(invalid_path, "pdf", False)
        assert str(exc_info.value) == (
            f"{RED}The directory specified by `path` does not exist:\n"
            f"{invalid_path.parent}{END}"
        )
