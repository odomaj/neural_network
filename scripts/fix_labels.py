import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parent.parent.absolute()))

from src.get_file import get_file


def fix_labels(contents: str) -> str:
    contents_list = list(filter(None, contents.split("\n")))
    for i in range(len(contents_list)):
        contents_list[i] = float(contents_list[i])
    return str(contents_list)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--input", "-i", default="", help="path to label file to be fixed"
    )
    arg_parser.add_argument(
        "--output", "-o", default="", help="path to fixed label file"
    )

    args = vars(arg_parser.parse_args())
    if args["input"] == "" or args["output"] == "":
        arg_parser.print_help()
        sys.exit()
    with get_file(args["input"]).open("r") as file:
        contents = file.read()
    with get_file(args["output"]).open("w") as file:
        file.write(fix_labels(contents))
