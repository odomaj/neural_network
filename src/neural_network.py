from get_file import get_file
import numpy as np
import sys


DEFAULT_DATA_INPUT_PATH = "../data/second_test/a2-test-data.txt"
DEFAULT_LABEL_INPUT_PATH = "../data/second_test/a2-test-label.txt"
INTEGER_CHARS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"}


def normal_distribution_error(distribution: np.array) -> float:
    return np.mean(distribution)


class Data:
    """class to hold a data point
    contains a numpy array and an integer value representing
    positive or negative"""

    def __init__(self, data_string="", label="") -> None:
        """if a data_string and a label is passed will initialize
        the data object"""
        self.initialized = False
        if data_string != "" and label != "":
            self.from_str(data_string, label)

    def get_array(self, strings: list) -> np.array:
        """takes a list of strings and outputs an np.array of floats"""
        floats = []
        for string in strings:
            floats.append(float(string))
        return np.array(floats)

    def from_str(self, string: str, label: int) -> None:
        """takes a line from the data input file and the
        corresponding integer label and updates the initialization state"""
        self.array = self.get_array(list(filter(None, string.split(" "))))
        self.label = label
        self.initialized = True

    def __getitem__(self, key: int) -> float:
        """returns the item in the numpy array if the data object is
        initialized and the item exists"""
        if not self.initialized:
            print("[ERROR] tried to get item from unitialized Data object")
            return None
        if key - 1 > len(self.array):
            print(
                f"[ERROR] tried to read from index {key} from a Data object of"
                f" length {len(self.array)}"
            )
            return None
        return self.array[key]

    def __str__(self) -> str:
        """outputs a string containing all of the elements in the numpy
        array separated by spaces"""
        if not self.initialized:
            return ""
        string = ""
        for element in self.array:
            string += f"{element} "
        return string

    def full_str(self) -> str:
        """outputs a string containing all of the elements in the numpy array
        separated by spaces, followed by the label"""
        if not self.initialized:
            return ""
        return f"{self}{self.label}"


class DataSet:
    """object to store a list of Data objects"""

    def __init__(self, data_strings=[], label_string="") -> None:
        """if data_strings and label_string are not the default values
        initializes the DataSet with the passed values"""
        self.initialized = False
        self.data_list = None
        if data_strings != [] and label_string != "":
            self.init(data_strings, label_string)

    def parse_label(label: str) -> int:
        """takes a string containing a numerical value
        strips all non numeric characters and outputs the integer value"""
        new_label = ""
        for c in label:
            if c == ".":
                break
            if c in INTEGER_CHARS:
                new_label += c
        return int(new_label)

    def init(self, data_strings: list, label_string: str) -> None:
        """constructs a list of Data objects from the passed data_strings
        and label_string and updates the internal intialization state"""
        labels = label_string.split(",")
        for i in range(len(labels)):
            labels[i] = parse_label(labels[i])

        if len(data_strings) != len(labels):
            print("[WARNING] data is incompatible with labels")

        data_count = min(len(data_strings), len(labels))
        self.data_list = []
        for i in range(data_count):
            self.data_list.append(
                Data(data_string=data_strings[i], label=labels[i])
            )
        self.initialized = True

    def __getitem__(self, key: int) -> Data:
        """returns the Data object at the passed index"""
        if not self.initialized:
            return None
        return self.data_list[key]

    def __str__(self) -> str:
        """returns a string containing all the Data objects separated by
        newline characters"""
        if not self.initialized:
            return ""
        output = ""
        for data in self.data_list:
            output += f"{data}\n"
        return output

    def full_str(self) -> str:
        """returns a string containing all the Data objects full strings
        separated by newline characters"""
        if not self.initialized:
            return ""
        output = ""
        for data in self.data_list:
            output += f"{data.full_str()}\n"
        return output


class Neuron:
    def __init__(self) -> None:
        self.weights = None

    def square_error(
        self,
        value: np.array,
        distribution_error,
    ) -> float:
        if len(self.weights) != len(value):
            -2.0
        square_errors = np.zeros(len(self.weights))
        for i in range(len(value)):
            square_errors[i] = 0.5
        return distribution_error(square_errors)

    def __getitem__(self, key: int) -> float:
        return self.weights[key]

    def __str__(self) -> str:
        string = ""
        for weight in self.weights:
            string += f"{weight} "
        return string


class Layer:
    def __init__(self) -> None:
        self.neurons = None

    def __getitem__(self, key: int) -> Neuron:
        return self.neurons[key]

    def __str__(self) -> str:
        string = ""
        for neuron in self.neurons:
            string += f"{neuron} "
        return string


class Network:
    def __init__(self) -> None:
        self.layers = None

    def parse_string(self, string: str) -> bool:
        return True

    def backpropagation(self) -> None:
        pass

    def __getitem__(self, key: int) -> Layer:
        return self.layers[key]

    def __str__(self) -> str:
        string = ""
        for layer in self.layers:
            string += f"{layer} "
        return string


def parse_label(lable: str) -> int:
    new_lable = ""
    for c in lable:
        if c == ".":
            break
        if c in INTEGER_CHARS:
            new_lable += c
    return int(new_lable.strip())


def get_data(input_data_file_path: str, input_label_file_path: str) -> DataSet:
    data_file = get_file(input_data_file_path)
    label_file = get_file(input_label_file_path)
    with data_file.open("r") as file:
        data = file.readlines()
    with label_file.open("r") as file:
        labels = file.read()

    if len(labels.split(",")) != len(data):
        print(
            f"[WARNING] data from {data_file.absolute()} is incompatible with"
            f" labels from {label_file.absolute()}"
        )

    return DataSet(data, labels)


if __name__ == "__main__":
    input_data_file_path = DEFAULT_DATA_INPUT_PATH
    input_label_file_path = DEFAULT_LABEL_INPUT_PATH
    if len(sys.argv) >= 3:
        input_data_file_path = sys.argv[1]
        input_label_file_path = sys.argv[2]

    data_list = get_data(input_data_file_path, input_label_file_path)
