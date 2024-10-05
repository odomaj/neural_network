import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parent.parent.absolute()))

from src.get_file import get_file


FORMAT_STR = " "


def dump_contents(contents: list) -> None:
    for line in contents:
        print(list(filter(None, line.split(FORMAT_STR))))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--path", "-p", default="", help="path to file to be dumped"
    )
    args = vars(arg_parser.parse_args())

    if args["path"] == "":
        arg_parser.print_help()
        sys.exit()

    with get_file(args["path"]).open("r") as file:
        dump_contents(file.readlines())
