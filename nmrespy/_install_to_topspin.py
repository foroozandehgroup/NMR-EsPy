import glob
import pathlib
import platform
import shutil
import sys

from nmrespy import NMRESPYPATH
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama

def main():
    # --- Determine OS ---------------------------------------------------
    # From OS info, determine the required directory pattern to match
    # and the name of the default Python 3 executable command.
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        pattern = "/opt/topspin*"
        exe = "python3"

    elif system == "Windows":
        pattern = "C:/Bruker/TopSpin*"
        exe = "py -3"

    else:
        raise OSError(
            f"{cols.R}Your operating system is not supported for automatic "
             "installation of nmrespy into TopSpin. See the documentation "
            f"for guidance on manual installation.{cols.END}"
        )

    # --- Get user to specify the Python 3 executable --------------------
    msg = (
        f"{cols.O}I need to know the Python3 executable that you would "
         "like to use when running nmrespy. By default, I will assume this "
        f"is:\n\t{exe}\nIf this is fine, simply press <Return>. If you are "
         "using a non-default Python executable, please state the full path "
        f"to it here: {cols.END}"
    )

    exe_input = input(msg)

    while True:
        if exe_input == "":
            break
        elif pathlib.Path(exe_input).is_file():
            exe = exe_input
            break
        else:
            exe_input = input(
                f"{cols.R}Invalid path specified. Please try again: {cols.END}"
            )

    # --- Write executable to app.topspin --------------------------------
    with open(NMRESPYPATH / "app/topspin.py", "r") as fh:
        txt = fh.read()
        txt = txt.replace("<EXECUTABLE>", f"\"{exe}\"")

    # --- Determine whether there are any TopSpin paths ------------------
    topspin_paths = glob.glob(pattern)

    if not topspin_paths:
        raise RuntimeError(
            f"{cols.R}\nNo TopSpin installations were found on your system! "
             "If you don't have TopSpin, I guess that makes sense. If you "
             "do have TopSpin, perhaps it is installed in a non-default "
             "location? You'll have to perform a manual installation in this "
            f"case. See the documentation for details.{cols.END}"
        )

    # --- Get user to specify desired install paths ----------------------
    path_list = '\n\t'.join(
        [f"{[i]} {path}" for i, path in enumerate(topspin_paths, start=1)]
    )
    msg = (
        f"{cols.O}\nThe following TopSpin path(s) were found on your system:"
        f"\n\t{path_list}\n"
         "For each installation that you would like to install the nmrespy "
         "app to, provide the corresponding numbers, separated by "
         "whitespaces. If you want to cancel the install to TopSpin, enter "
         "0. If you want to install to all the listed TopSpin installations, "
        f"press <Return>: {cols.END}"
    )

    user_input = input(msg)


    while True:
        indices = parse_user_input(user_input, len(topspin_paths))
        if indices:
            install_paths = [topspin_paths[idx] for idx in indices]
            break
        else:
            user_input = input(
                f"{cols.R}Invalid input. Please try again: {cols.END}"
            )

    # -- Try to write to each file path ----------------------------------
    for path in install_paths:
        try:
            dst = pathlib.Path(path) / "exp/stan/nmr/py/user/nmrespy.py"
            with open(dst, "w") as fh:
                fh.write(txt)
            print(f"\nInstalled:\n\t{cols.G}{str(dst)}{cols.END}")

        except Exception as e:
            print(
                f"{cols.R}\nFailed to install to:\n\t{str(dst)}\n"
                f"with error message:\n\t{e}{cols.END}"
            )


def parse_user_input(user_input, number):
    if user_input == "":
        return(list(range(number)))

    if user_input == '0':
        print(
            f"{cols.R}Cancelling installation of the nmrespy app to "
            f"TopSpin...{cols.END}"
        )
        exit()

    values = list(filter(lambda x: x != '', user_input.split(' ')))

    for value in values:
        if not (value.isnumeric() and int(value) <= number):
            return False

    return [int(value)-1 for value in values]


if __name__ == "__main__":
    main()
