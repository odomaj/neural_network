from get_file import get_file
import numpy as np
import sys
from math import tanh, sinh, cosh
from random import seed, random


DEFAULT_DATA_INPUT_PATH = "../data/second_test/a2-test-data.txt"
DEFAULT_LABEL_INPUT_PATH = "../data/second_test/a2-test-label.txt"
INTEGER_CHARS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"}

HIDDEN_LAYER_STRUCTURE = [128, 16]
NUMBER_OF_OUTPUTS = 1


def normal_distribution_error(distribution: np.array) -> float:
    return np.mean(distribution)


class Data:
    """class to hold a data point
    contains a numpy array and an integer value representing
    positive or negative"""

    def __init__(
        self,
        data_string=None,
        data=None,
        label=None,
    ) -> None:
        """if a data_string and a label is passed will initialize
        the data object"""
        self.initialized = False
        self.label = None
        self.array = None
        if label == None:
            return
        if data_string is not None and data is not None:
            print(
                "[WARNING] cannot initalize a Data object from a string and an"
                " array"
            )
        elif data_string != None:
            self.from_str(data_string, label)
        elif data is not None:
            self.from_array(data, label)

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

    def from_array(self, data: np.array, label: int) -> None:
        self.array = np.array(data)
        self.label = label
        self.initialized = True

    def __getitem__(self, key: int) -> float:
        """returns the item in the numpy array if the data object is
        initialized and the item exists"""
        if not self.initialized:
            print("[ERROR] tried to get item from unitialized Data object")
            return None
        if key + 1 > len(self.array):
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

    def parse_label(self, label: str) -> int:
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
            labels[i] = self.parse_label(labels[i])

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
        if key + 1 > len(self.data_list):
            print(
                f"[ERROR] tried to read from index {key} from a DataSet object"
                f" of length {len(self.data_list)}"
            )
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
    def __init__(self, number_weights=0) -> None:
        self.weights = None
        self.bias = None
        self.initialized = False
        if number_weights != 0:
            self.init(number_weights)

    def init(self, number_weights: int) -> None:
        """"""
        self.weights = np.array([random() for i in range(number_weights)])
        self.bias = random()
        self.initialized = True

    def front_propagate(self, data: Data) -> float:
        return self.sigmoid(self.activation(data))

    def back_propagate(self, next_neuron) -> float:
        pass

    def cost(self, data: Data) -> float:
        """calculates the square error"""
        error = self.sigmoid(self.activation(data)) - data.label
        return error * error

    def activation(self, data: Data) -> float:
        if len(data.array) != len(self.weights):
            print(
                "[WARNING] differing lengths arrays when calculating"
                " activation functions"
            )
            return self.bias
        return self.bias + np.dot(data.array, self.weights)

    def delta_bias(self, data: Data) -> float:
        activation_result = self.activation(data)
        return (
            2
            * (activation_result - data.label)
            * self.sigmoid_prime(activation_result)
        )

    def delta_weight(self, data: Data) -> np.array:
        activation_result = self.activation(data)
        delta_weight = (
            2
            * (activation_result - data.label)
            * self.sigmoid_prime(activation_result)
        )
        return np.multiply(self.weights, delta_weight)

    def delta_cost(self, data: Data, previous_activation: float) -> float:
        activation_result = self.activation(data)
        return (
            2
            * (activation_result - data.label)
            * self.sigmoid_prime(activation_result)
            * previous_activation
        )

    def sigmoid(self, real: float) -> float:
        return tanh(real / 2)

    def sigmoid_prime(self, real: float) -> float:
        """derivative of sigmoid function"""
        # sigmoid = tanh(real/2)
        # tanh(x)' = ((cosh(x))^2 - (sinh(x))^2) / (cosh(x))^2
        real = real / 2
        s = sinh(real)
        c = cosh(real)
        s = s * s
        c = c * c
        # divide by 2 for chain rule
        return (c - s) / (2 * c)

    def __getitem__(self, key: int) -> float:
        return self.weights[key]

    def __str__(self) -> str:
        string = ""
        for weight in self.weights:
            string += f"{weight} "
        return string


class Layer:
    def __init__(
        self,
        number_neurons=0,
        number_weights=0,
    ) -> None:
        self.neurons = None
        self.initialized = False
        if number_neurons != 0 and number_weights != 0:
            self.init(number_neurons, number_weights)

    def init(
        self,
        number_neurons: int,
        number_weights: int,
    ) -> None:
        self.neurons = [Neuron(number_weights) for i in range(number_neurons)]
        self.initialized = True

    def front_propagation(self, data: Data) -> Data:
        new_data = Data()
        new_data.label = data.label
        new_data.array = np.zeros(len(self.neurons))
        for i in range(len(new_data.array)):
            new_data.array[i] = self.neurons[i].front_propagate(data)
        new_data.initialized = True
        return new_data

    def __getitem__(self, key: int) -> Neuron:
        return self.neurons[key]

    def __str__(self) -> str:
        string = ""
        for neuron in self.neurons:
            string += f"{neuron} "
        return string


class Network:
    def __init__(
        self,
        number_hidden_layers=0,
        number_output_neurons=0,
        number_weights=0,
    ) -> None:
        self.hidden_layers = None
        self.output_layer = None
        self.initialized = False
        if (
            number_hidden_layers != 0
            and number_output_neurons != 0
            and number_weights != 0
        ):
            self.init(
                number_hidden_layers,
                number_output_neurons,
                number_weights,
            )

    def init(
        self,
        hidden_layers: list,
        number_output_neurons: int,
        number_weights: int,
    ) -> None:
        self.hidden_layers = []
        # number of neurons in the previous layer determines the number
        # of weights in the current layer
        for hidden_layer in hidden_layers:
            self.hidden_layers.append(
                Layer(
                    number_neurons=hidden_layer,
                    number_weights=number_weights,
                )
            )
            number_weights = hidden_layer
        self.output_layer = Layer(number_output_neurons, number_weights)
        self.initialized = True

    def cost(self, dataset: DataSet) -> float:
        cost = 0
        for data in dataset.data_list:
            output = self.front_propagation(data)
            value = output.array[0] - output.label
            cost += value * value
        return cost / len(data_list.data_list)

    def train(self, training_data: DataSet) -> None:
        pass

    def back_propagation(self, dataset: DataSet) -> None:
        pass

    def front_propagation(self, data: Data) -> Data:
        previous_layer = data
        for layer in self.hidden_layers:
            previous_layer = layer.front_propagation(previous_layer)
        return self.output_layer.front_propagation(previous_layer)

    def __str__(self) -> str:
        string = ""
        for layer in self.hidden_layers:
            string += f"{layer}\n"
        string += f"{self.output_layer}\n"
        return string


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

    seed(1)

    data_list = get_data(input_data_file_path, input_label_file_path)

    if len(data_list.data_list) == 0:
        print("[ERROR] no data")
        sys.exit()

    network = Network(
        number_hidden_layers=HIDDEN_LAYER_STRUCTURE,
        number_output_neurons=NUMBER_OF_OUTPUTS,
        number_weights=len(data_list[0].array),
    )
    network.train(data_list)
    print(network.cost(data_list))
