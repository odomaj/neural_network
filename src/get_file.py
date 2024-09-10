from pathlib import Path
import sys
import platform


def get_this_filepath() -> str:
    """returns the path of the called filepath from the users
    current working directory"""
    return sys.argv[0]


def path_absolute(path: str) -> bool:
    """returns True if the passed path is absolute
    returns False if the passed path is relative
    currently only supports Windows and Linux"""
    operating_system = platform.system()
    if operating_system == "Windows":
        if len(path) < 3:
            return False
        if path[:3] == "C:\\" or path[:3] == "C:/":
            return True
        return False
    if operating_system == "Linux":
        if len(path) < 1:
            return False
        if path[0] == "/":
            return True
        return False
    if operating_system == "Darwin":
        print("[WARNING] absolute file paths not supported on Mac OS")
        return False
    print("[WARNING] unsupported uperating system for absolute file path")
    return False


def get_file(path: str) -> Path:
    """concatenates the passed filepath with the directory
    of the called file"""
    if path_absolute(path):
        return Path(path)
    running_path = Path(get_this_filepath()).absolute()
    return running_path.parent.joinpath(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] not enough args passed")
    else:
        print(get_file(sys.argv[1]))
