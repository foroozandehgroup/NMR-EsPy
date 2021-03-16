import glob
import pathlib
import platform
import shutil
import sys

import nmrespy._cols as cols

def get_opsys():
    opsys = platform.system()
    if opsys in ['Darwin', 'Linux']:
        return 'unix'
    elif opsys == 'Windows':
        return 'windows'
    else:
        raise OSError(
            f"{cols.R}Your operating system is not supported for manual "
             "installation of nmrespy into TopSpin. See the documentation "
            f"for guidance on manual installation.{cols.END}"
        )


def get_topspin_paths(opsys):
    if opsys == 'unix':
        pattern = "/opt/topspin*"
    elif opsys == 'windows':
        pattern = "C:/Bruker/TopSpin*"

    topspin_paths = glob.glob(pattern) # glob.glob.glob.glob

    if not topspin_paths:
        raise RuntimeError(
            f"{cols.R}No TopSpin installations were found on your system! "
             "If you don't have TopSpin, I guess that makes sense. If you "
             "do have TopSpin, perhaps it is installed in a non-default "
             "location? You'll have to perform a manual installation in this "
            f"case. See the documentation for details.{cols.END}"
        )

    else:
        return topspin_paths


def get_desired_topspin_paths(paths):

    path_list = '\n\t'.join(
        [f"{[i]} {path}" for i, path in enumerate(paths, start=1)]
    )
    message = (
        f"{cols.O}The following TopSpin path(s) were found on your system:"
        f"\n\t{path_list}\n"
         "For each installation that you would like to install the nmrespy "
         "app to, provide the corresponding numbers, separated by "
         "whitespaces. If you want to cancel the install to TopSpin, enter "
        f"0: {cols.END}"
    )

    def parse_path_selection(desired_paths, number):
        if desired_paths == '0':
            print(
                f"{cols.R}Cancelling installation of the nmrespy app to "
                f"TopSpin...{cols.END}"
            )
            exit()

        values = list(filter(lambda x: x != '', desired_paths.split(' ')))
        for value in values:
            if not (value.isnumeric() and int(value) <= number):
                return False
        return [int(value)-1 for value in values]

    desired_paths = input(message)
    while True:
        indices = parse_path_selection(desired_paths, len(paths))
        if indices:
            return [paths[idx] for idx in indices]
        else:
            desired_paths = input(
                f"{cols.R}Invalid input. Please try again: {cols.END}"
            )


def install(path):
    src = pathlib.Path(__file__).parent / "app/topspin.py"
    dst = pathlib.Path(path) / "exp/stan/nmr/py/user/nmrespy.py"

    try:
        shutil.copyfile(src, dst)
        print(f"Installed: {cols.G}{str(dst)}{cols.END}\n")

    except Exception as e:
        print(
            f"{cols.R}Failed to copy file\n\t{str(src)}\nto\n\t{str(dst)}\n"
            f"with error message:\n\t{e}{cols.END}\n"
        )


def main():
    opsys = get_opsys()
    topspin_paths = get_topspin_paths(opsys)
    install_paths = get_desired_topspin_paths(topspin_paths)
    for path in install_paths:
        install(path)


if __name__ == "__main__":
    main()
