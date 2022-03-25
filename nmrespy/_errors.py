# _errors.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 10:49:22 GMT

"""NMR-EsPy-specific errors."""

from nmrespy._colors import RED, CYA, END, USE_COLORAMA
from nmrespy._paths_and_links import GITHUBLINK

if USE_COLORAMA:
    import colorama

    colorama.init()


class MoreThanTwoDimError(Exception):
    """Raise when user tries importing data that is >2D."""

    def __init__(self):
        self.msg = f"{RED}nmrespy does'nt support >2D data.{END}"
        super().__init__(self.msg)


class TwoDimUnsupportedError(Exception):
    """Raise when user tries running a method that doesn't support 2D data."""

    def __init__(self):
        self.msg = (
            f"{RED}Unfortunately 2D virtual echo creation isn't"
            " supported yet. Check if there are any more recent"
            " versions of nmrespy with this feature:\n"
            f"{CYA}{GITHUBLINK}{END}"
        )
        super().__init__(self.msg)


class InvalidUnitError(Exception):
    """Raise when the specified unit is invalid."""

    def __init__(self, *args):
        # args are strings with the valid unit names.
        valid_units = ", ".join([repr(unit) for unit in args])
        self.msg = f"{RED}unit should be one of the following:" f" {valid_units}{END}"
        super().__init__(self.msg)


class InvalidDirectoryError(Exception):
    """Raise when the a directory does not have the requisite files."""

    def __init__(self, dir):
        self.msg = (
            f"{RED}{str(dir)} does not contain the necessary files"
            f" for importing data.{END}"
        )
        super().__init__(self.msg)


class ParameterNotFoundError(Exception):
    """Raise when a desired parameter is not present in an acqus/procs file."""

    def __init__(self, param_name, path):
        self.msg = (
            f"{RED}Could not find parameter {param_name} in file" f"{str(path)}{END}"
        )
        super().__init__(self.msg)


class NoParameterEstimateError(Exception):
    """Raise when instance does not possess a valid parameter array."""

    def __init__(self):
        self.msg = (
            f"{RED}No attribute corresponding to a parameter array"
            f" could be found. Perhaps you need to run an"
            f" estimation routine first?{END}"
        )
        super().__init__(self.msg)


class PhaseVarianceAmbiguityError(Exception):
    """Raise when phase_variance is True, but 'p' is not specified in mode."""

    def __init__(self, mode):
        self.msg = (
            f"{RED}You have specified you want to minimise phase"
            " varaince (phase_variance=True) but you have not"
            " asked for the phases to be be optimised"
            f" (mode = '{mode}'). The phase variance cannot change"
            f" if you don't include 'p' in mode.{END}"
        )
        super().__init__(self.msg)


class AttributeIsNoneError(Exception):
    """Raise when a `get_<attr>` method is called and the attribute is None."""

    def __init__(self, attribute, method):
        self.msg = f"{RED}The attribute `{attribute}` is None."
        if method is not None:
            self.msg += f"\nPerhaps you are yet to call `{method}` on the estimator?"
        self.msg += END
        super().__init__(self.msg)


class LaTeXFailedError(Exception):
    """Raise when an issue in compiling with LaTeX arises."""

    def __init__(self, texpath):
        self.msg = (
            f"{RED}The file {texpath} failed to compile using"
            " pdflatex. Make you sure have a LaTeX installation"
            f" by opening a terminal and entering:\n{CYA}"
            f" pdflatex\n{RED}If you have pdflatex, run:\n"
            f" {CYA}pdflatex {texpath}\n{RED}and try to fix"
            f" whatever it is not happy with{END}"
        )

        super().__init__(self.msg)
