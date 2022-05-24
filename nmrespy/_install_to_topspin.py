# _install_to_topspin.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 23 May 2022 11:34:19 BST

import glob
import pathlib
import platform
import subprocess
import sys


def get_opsys():
    """Determine the operating system.

    This will determine the directory pattern that will be used to determine
    whether any TopSpin directories exist.
    """
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        return "unix"
    elif system == "Windows":
        return "windows"
    else:
        print(
            "Your operating system is not supported for automatic "
            "installation of nmrespy into TopSpin. See the documentation "
            "for guidance on manual installation."
        )
        return None


def get_topspin_paths(opsys):
    """Determine whether TopSpin installations exist in the default path."""
    if opsys == "unix":
        pattern = "/opt/topspin*"
    elif opsys == "windows":
        pattern = "C:/Bruker/TopSpin*"

    topspin_paths = glob.glob(pattern)

    if not topspin_paths:
        print(
            "\nNo TopSpin installations were found on your system! "
            "If you don't have TopSpin, I guess that makes sense. If you "
            "do have TopSpin, perhaps it is installed in a non-default "
            "location? You'll have to perform a manual installation in "
            "this case. See the documentation for details."
        )

        topspin_paths = None

    return topspin_paths


def get_install_paths(topspin_paths):
    """Get the user to specify which paths to install the GUI loader to."""
    path_list = "\n\t".join(
        [f"{[i]} {path}" for i, path in enumerate(topspin_paths, start=1)]
    )

    print(
        "\nThe following TopSpin path(s) were found on your system:"
        f"\n\t{path_list}\n"
        "For each installation that you would like to install the nmrespy "
        "app to, provide the corresponding numbers, separated by "
        "whitespaces.\nIf you want to cancel the install to TopSpin, enter "
        "0.\nIf you want to install to all the listed TopSpin "
        "installations, press <Return>:"
    )

    user_input = input()

    # Get user input
    # If valid, deal accoridngly.
    # If invalid, re-ask the user for input.
    while True:
        indices = parse_user_input(user_input, len(topspin_paths))
        if indices is False:
            print("Invalid input. Please try again:")
            user_input = input()
        else:
            return [topspin_paths[idx] for idx in indices]


def parse_user_input(user_input, number):
    """Determine the paths the user wants to install the GUI loader to."""
    if user_input == "":
        # User pressed <Return>. Return list of all valid indices.
        return list(range(number))

    if user_input == "0":
        # User pressed 0. Return empty list (no installation will take place)
        print("No installation of the nmrespy app will " "occur...")
        return []

    # Split input at whitespace (filter out any empty elements)
    values = list(filter(lambda x: x != "", user_input.split(" ")))

    for value in values:
        # Check each element is numeric and of valid value
        if not (value.isnumeric() and int(value) <= number):
            return False

    # Return indices coresponding to TopSPin paths of interest
    return [int(value) - 1 for value in values]


def get_pdflatex_executable(opsys):
    """Find the path to the pdflatex executable.

    If no executable can be found, `None` is returned
    """
    # Check pdflatex exists (return code will be 0 if it does).
    if opsys == "unix":
        command = "which"
    elif opsys == "windows":
        command = "where"

    which_pdflatex = subprocess.run(
        f"{command} pdflatex",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    if which_pdflatex.returncode == 0:
        return (
            which_pdflatex
            .stdout.decode("utf-8")
            .rstrip("\n\r")
            .replace("\\", "\\\\")
        )

    else:
        print(
            "I was unable to find a pdflatex executable on your"
            "system. You will not be able to generate PDF's of your "
            "results"
        )
        return None


def install(install_paths, txt):
    """Try to write ``txt`` to each file path."""
    for path in install_paths:
        try:
            dst = pathlib.Path(path) / "exp/stan/nmr/py/user/nmrespy.py"
            with open(dst, "w") as fh:
                fh.write(txt)
            print(f"\nSUCCESS:\n\t{str(dst)}")

        except Exception as e:
            print(f"\nFAIL:\n\t{str(dst)}\n" f"ERROR MESSAGE:\n\t{e}")


def main():
    """Configure installation of GUI loader to TopSpin."""
    opsys = get_opsys()
    if opsys is None:
        return

    topspin_paths = get_topspin_paths(opsys)
    if topspin_paths is None:
        return

    install_paths = get_install_paths(topspin_paths)
    if not install_paths:
        return

    # Python executable
    py_exe = sys.executable.replace("\\", "\\\\")
    # pdflatex executable (if present)
    pdflatex_exe = get_pdflatex_executable(opsys)

    # --- Write executables to app._topspin.py ---------------------------
    path = pathlib.Path(__file__).parent.resolve() / "app" / "_topspin.py"
    with open(path, "r") as fh:
        txt = fh.read()

    txt = txt.replace("py_exe = None", f"py_exe = \"{py_exe}\"")
    if pdflatex_exe is not None:
        txt = txt.replace(
            "pdflatex_exe = \"None\"",
            f"pdflatex_exe = \"{pdflatex_exe}\"",
        )

    install(install_paths, txt)


if __name__ == "__main__":
    main()
