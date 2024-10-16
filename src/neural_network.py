from get_file import get_file
import numpy as np
import sys
from argparse import ArgumentParser
from math import tanh, sinh, cosh
from random import seed, random
from threading import Event

DEFAULT_TRAINING_DATA_INPUT_PATH = "../data/first_test/a2-train-data.txt"
DEFAULT_TRAINING_DATA_LABEL_PATH = "../data/first_test/a2-train-label.txt"

DEFAULT_TEST_DATA_INPUT_PATH = "../data/second_test/a2-test-data.txt"
DEFAULT_TEST_DATA_LABEL_PATH = "../data/second_test/a2-test-label.txt"

HIDDEN_LAYER_STRUCTURE = [256]
NUMBER_OF_OUTPUTS = 1

INTEGER_CHARS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"}


do_quit = Event()


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
            if string != "\n":
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


class WeightAdjustments:
    def __init__(
        self,
        output_weights=None,
        output_bias=None,
        hidden_weights=None,
        hidden_bias=None,
    ):
        self.output_weights = output_weights
        self.output_bias = output_bias
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias


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

    def front_propagation(self, data: Data) -> float:
        return self.sigmoid(self.activation(data))

    def adjust_weights(self, gradiant: np.array, learning_rate: float) -> None:
        self.weights = np.subtract(
            self.weights, np.multiply(gradiant, learning_rate)
        )

    def adjust_bias(self, gradiant: float, learning_rate: float) -> None:
        self.bias -= gradiant * learning_rate

    def activation(self, data: Data) -> float:
        if len(data.array) != len(self.weights):
            print(
                "[WARNING] Neuron.activation ran with"
                f" {len(self.weights)} weights and {len(data.array)} data"
                " values"
            )
            return self.bias
        return self.bias + np.dot(data.array, self.weights)

    def sigmoid(self, real: float) -> float:
        return tanh(real / 2)

    def sigmoid_prime(self, real: float) -> float:
        """derivative of sigmoid function"""
        # sigmoid = tanh(real/2)
        # tanh(x)' = ((cosh(x))^2 - (sinh(x))^2) / (cosh(x))^2
        real = real / 2
        try:
            s = sinh(real)
            c = cosh(real)
            s = s * s
            c = c * c
            # divide by 2 for chain rule
            output = (c - s) / (2 * c)
            if np.isnan(output):
                return 0
            return output
        except:
            return 0

    def __getitem__(self, key: int) -> float:
        return self.weights[key]

    def __str__(self) -> str:
        string = ""
        for weight in self.weights:
            string += f"{weight} "
        return string + f"{self.bias}"


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

    def activation(self, data: Data) -> Data:
        new_data = Data()
        new_data.label = data.label
        new_data.array = np.zeros(len(self.neurons))
        for i in range(len(new_data.array)):
            new_data.array[i] = self.neurons[i].activation(data)
        new_data.initialized = True
        return new_data

    def front_propagation(self, data: Data) -> Data:
        new_data = Data()
        new_data.label = data.label
        new_data.array = np.zeros(len(self.neurons))
        for i in range(len(new_data.array)):
            new_data.array[i] = self.neurons[i].front_propagation(data)
        new_data.initialized = True
        return new_data

    def get_all_activations(self, data: Data) -> np.array:
        activations = np.zeros(len(self.neurons))
        for i in range(len(self.neurons)):
            activations[i] = self.neurons[i].activation(data)
        return activations

    def get_all_sigmoids(self, data: Data) -> np.array:
        sigmoids = np.zeros(len(self.neurons))
        for i in range(len(self.neurons)):
            sigmoids[i] = self.neurons[i].sigmoid(
                self.neurons[i].activation(data)
            )
        return sigmoids

    def get_all_sigmoid_primes(self, data: Data) -> np.array:
        sigmoid_primes = np.zeros(len(self.neurons))
        for i in range(len(self.neurons)):
            sigmoid_primes[i] = self.neurons[i].sigmoid_prime(
                self.neurons[i].activation(data)
            )
        return sigmoid_primes

    def sum_error_terms(
        self, error_terms: np.array, weight_index: int
    ) -> float:
        """outputs the sum of the error_terms multiplied by the weight_index
        weight in each neuron"""
        if len(error_terms) != len(self.neurons):
            print(
                "[ERROR] Layer.sum_error_terms called with"
                f" {len(error_terms)} error terms given for"
                f" {len(self.neurons)} neurons"
            )
            return
        sum = 0
        for i in range(len(error_terms)):
            sum += error_terms[i] * self.neurons[i].weights[weight_index]
        return sum

    def adjust_neurons(
        self,
        weight_adjustments: list,
        bias_adjustments: list,
        learning_rate: float,
    ) -> None:
        if not self.initialized:
            return
        if len(weight_adjustments) != len(self.neurons):
            print(
                "[ERROR] Layer.adjust_neurons called with"
                f" {len(weight_adjustments)} weight_adjustments and"
                f" {len(self.neurons)} neurons"
            )
            return
        if len(bias_adjustments) != len(self.neurons):
            print(
                "[ERROR] Layer.adjust_neurons called with"
                f" {len(bias_adjustments)} bias_adjustments and"
                f" {len(self.neurons)} neurons"
            )
            return
        for i in range(len(self.neurons)):
            self.neurons[i].adjust_weights(
                weight_adjustments[i], learning_rate
            )
            self.neurons[i].adjust_bias(bias_adjustments[i], learning_rate)

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
        return cost / len(dataset.data_list)

    def average_training_results(self, gradiants: list) -> WeightAdjustments:
        adjustments = WeightAdjustments(
            output_weights=[],
            output_bias=[],
            hidden_weights=[],
            hidden_bias=[],
        )
        if len(gradiants) < 1:
            return None
        for n in range(len(gradiants[0].output_weights)):
            weights = np.zeros(len(gradiants[0].output_weights[n]))
            for w in range(len(weights)):
                sum = 0
                for i in range(len(gradiants)):
                    sum += gradiants[i].output_weights[n][w]
                weights[w] = sum / len(gradiants)
            adjustments.output_weights.append(weights)

        for n in range(len(gradiants[0].output_bias)):
            sum = 0
            for i in range(len(gradiants)):
                sum += gradiants[i].output_bias[n]
            adjustments.output_bias.append(sum / len(gradiants))

        for l in range(len(gradiants[0].hidden_weights)):
            adjustments.hidden_weights.append([])
            for n in range(len(gradiants[0].hidden_weights[l])):
                weights = np.zeros(len(gradiants[0].hidden_weights[l][n]))
                for w in range(len(weights)):
                    sum = 0
                    for i in range(len(gradiants)):
                        sum += gradiants[i].hidden_weights[l][n][w]
                    weights[w] = sum / len(gradiants)
                adjustments.hidden_weights[l].append(weights)

        for l in range(len(gradiants[0].hidden_bias)):
            adjustments.hidden_bias.append([])
            for n in range(len(gradiants[0].hidden_bias[l])):
                sum = 0
                for i in range(len(gradiants)):
                    sum += gradiants[i].hidden_bias[l][n]
                adjustments.hidden_bias[l].append(sum / len(gradiants))

        return adjustments

    def train(
        self,
        training_data: DataSet,
        epoch: int,
        learning_rate: float,
        batch_size=0,
    ) -> None:
        previous_cost = self.cost(training_data)
        min_cost = previous_cost
        print(f"[INFO] training with an initial cost of {previous_cost}")

        if batch_size == 0:
            batch_size = len(training_data.data_list)
        number_of_batches = len(training_data.data_list) // batch_size
        if number_of_batches * batch_size < len(training_data.data_list):
            number_of_batches += 1
        i = 0
        while previous_cost > min_cost or i < epoch:
            i += 1
            for j in range(number_of_batches):
                self.train_set(
                    training_data,
                    learning_rate,
                    j * batch_size,
                    (j + 1) * batch_size,
                )
                print(
                    "[VERBOSE] trained batch new cost is"
                    f" {self.cost(training_data)}"
                )
            new_cost = self.cost(training_data)
            print(
                "[INFO] trained on data set with a delta cost of "
                f" {previous_cost - new_cost}, new cost is {new_cost} after"
                f" {i} training runs with a learning rate of {learning_rate}"
            )
            previous_cost = new_cost
            min_cost = min(min_cost, previous_cost)

    def train_set(
        self,
        training_data: DataSet,
        learning_rate: float,
        start_index: int,
        end_index: int,
    ) -> None:
        if len(training_data.data_list) == 0:
            return
        gradiants = []
        if len(training_data.data_list) < end_index + 1:
            end_index = len(training_data.data_list) + 1
        for data in training_data.data_list[start_index:end_index]:
            gradiants.append(self.back_propagation(data))
        self.adjust_weights(
            self.average_training_results(gradiants), learning_rate
        )

    def adjust_weights(
        self, weight_adjustments: WeightAdjustments, learning_rate: float
    ) -> None:
        if len(weight_adjustments.hidden_weights) != len(self.hidden_layers):
            print(
                "[ERROR] Network.adjust_weights called with"
                f" {len(weight_adjustments.hidden_weights)} hidden weights"
                f" vectors and {len(self.hidden_layers)} layers"
            )
            return
        if len(weight_adjustments.hidden_bias) != len(self.hidden_layers):
            print(
                "[ERROR] Network.adjust_weights called with"
                f" {len(weight_adjustments.hidden_bias)} hidden biases"
                f" vectors and {len(self.hidden_layers)} layers"
            )
            return
        self.output_layer.adjust_neurons(
            weight_adjustments.output_weights,
            weight_adjustments.output_bias,
            learning_rate,
        )
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].adjust_neurons(
                weight_adjustments.hidden_weights[i],
                weight_adjustments.hidden_bias[i],
                learning_rate,
            )

    def get_layer_error_terms(
        self,
        layer: Layer,
        subsequent_layer: Layer,
        data: Data,
        error_terms: np.array,
    ) -> np.array:
        """outputs a numpy array containing the error terms for each neuron in
        the passed layer. Subsequent error term multiplied by the subsequent
        layers weights (handled by Layer.sum_error_terms, then again multiplied
        by this layers sigmoid primes"""
        sigmoid_primes = layer.get_all_sigmoid_primes(data)
        new_error_terms = np.zeros(len(layer.neurons))
        for weight_index in range(len(error_terms)):
            new_error_terms[weight_index] = subsequent_layer.sum_error_terms(
                error_terms, weight_index
            )
            new_error_terms[weight_index] *= sigmoid_primes[weight_index]
        return new_error_terms

    def get_layer_weight_adjustments(
        self,
        previous_layer_output: np.array,
        error_terms: np.array,
    ):
        """outputs a list of numpy array containing the weight adjustments for
        each neuron in a layer. calculated by multiplying previous layer's output
        with the current layer's error term"""
        weight_adjustments = [
            np.zeros(len(previous_layer_output))
            for i in range(len(error_terms))
        ]
        for i in range(len(error_terms)):
            for j in range(len(previous_layer_output)):
                weight_adjustments[i][j] = (
                    error_terms[i] * previous_layer_output[j]
                )
        return weight_adjustments

    def back_propagation(self, data: Data) -> WeightAdjustments:
        if not self.initialized:
            return None
        weight_adjustments = WeightAdjustments()
        # generate a list of all the data outputs for each layer
        previous_layers = [data]
        for layer in self.hidden_layers:
            previous_layers.append(
                layer.front_propagation(previous_layers[-1])
            )
        current_network_output = self.output_layer.front_propagation(
            previous_layers[-1]
        )
        # calculate scalar multiplied for delta_weight for the output layer
        error_terms = np.array(
            [
                2
                * (
                    current_network_output.array[0]
                    - current_network_output.label
                )
                * self.output_layer.neurons[0].sigmoid_prime(
                    self.output_layer.neurons[0].activation(
                        previous_layers[-1]
                    )
                )
            ]
        )
        # multiply that constant across all the
        # sigmoids of the last hidden layer
        weight_adjustments.output_weights = [
            np.multiply(
                self.hidden_layers[-1].get_all_sigmoids(previous_layers[-2]),
                error_terms[0],
            )
        ]
        weight_adjustments.output_bias = error_terms
        weight_adjustments.hidden_weights = []
        weight_adjustments.hidden_bias = []
        # for every layer calculate the error terms
        # then multiply the error terms by the previous layers outputs

        # calculate the error terms for the last hidden layer
        # error term = current_layers_sigmoid_prime(previous_layers_output) * subsequent_layers_weight * subsequent_layers_error
        error_terms = self.get_layer_error_terms(
            self.hidden_layers[-1],
            self.output_layer,
            previous_layers[-2],
            error_terms,
        )
        for i in range(len(self.hidden_layers) - 1, 0, -1):
            weight_adjustments.hidden_weights.insert(
                0,
                self.get_layer_weight_adjustments(
                    previous_layers[i].array, error_terms
                ),
            )
            weight_adjustments.hidden_bias.insert(0, error_terms)
            error_terms = self.get_layer_error_terms(
                self.hidden_layers[i - 1],
                self.hidden_layers[i],
                previous_layers[i - 1],
                error_terms,
            )
        weight_adjustments.hidden_weights.insert(
            0,
            self.get_layer_weight_adjustments(
                previous_layers[0].array, error_terms
            ),
        )
        weight_adjustments.hidden_bias.insert(0, error_terms)

        return weight_adjustments

    def front_propagation(self, data: Data) -> Data:
        if not self.initialized:
            print(
                "[WARNING] Network.front_propagation ran with Network object"
                " not initialized"
            )
            return
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


def dump_network(file_path: str, network: Network) -> None:
    hidden_units = len(network.hidden_layers[0].neurons)
    with get_file(file_path).open("w") as file:
        file.write(f"{hidden_units}\n")
        file.write(f"{network.output_layer.neurons[0]}\n")
        for neuron in network.hidden_layers[0].neurons:
            file.write(f"{neuron}\n")


def dump_output(file_path: str, network: Network, data_set: DataSet) -> None:
    with get_file(file_path).open("w") as file:
        for data in data_set.data_list:
            output = network.front_propagation(data).array[0]
            if output < 0:
                file.write("-1 ")
            else:
                file.write("+1 ")


def load_neuron(line: str) -> Neuron:
    neuron = Neuron()
    neuron_weights = Data().get_array(line.split(" "))
    neuron.weights = np.array(neuron_weights[:-1])
    neuron.bias = neuron_weights[-1]
    neuron.initialized = True
    return neuron


def load_network(file_path: str) -> Network:
    with get_file(file_path).open("r") as file:
        network = Network()
        network.hidden_layers = [Layer()]
        network.hidden_layers[0].neurons = [None] * int(file.readline())
        network.output_layer = Layer()
        network.output_layer.neurons = [load_neuron(file.readline())]
        for i in range(len(network.hidden_layers[0].neurons)):
            network.hidden_layers[0].neurons[i] = load_neuron(file.readline())
        network.output_layer.initialized = True
        network.hidden_layers[0].initialized = True
        network.initialized = True
        return network


def initialize() -> tuple:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--train_data",
        default=DEFAULT_TRAINING_DATA_INPUT_PATH,
        help="path to training data file",
    )
    arg_parser.add_argument(
        "--train_label",
        default=DEFAULT_TRAINING_DATA_LABEL_PATH,
        help="path to training data label file",
    )
    arg_parser.add_argument(
        "--test_data",
        default=DEFAULT_TEST_DATA_INPUT_PATH,
        help="path to test data file",
    )
    arg_parser.add_argument(
        "--test_label",
        default=DEFAULT_TEST_DATA_LABEL_PATH,
        help="path to test data label file",
    )
    arg_parser.add_argument(
        "--network_file", "-n", default="", help="network to load from save"
    )
    args = arg_parser.parse_args()

    seed(1)

    training_data = get_data(args.train_data, args.train_label)
    test_data = get_data(args.test_data, args.test_label)

    if len(training_data.data_list) == 0 or len(test_data.data_list) == 0:
        print("[ERROR] no data")
        return 1

    if args.network_file == "":
        network = Network(
            number_hidden_layers=HIDDEN_LAYER_STRUCTURE,
            number_output_neurons=NUMBER_OF_OUTPUTS,
            number_weights=len(training_data[0].array),
        )
    else:
        network = load_network(args.network_file)
    return (training_data, test_data, network)


def update_saved(network: Network, data: DataSet, save_index: int) -> int:
    filepath = f"../saves/network_save_{save_index}.txt"
    if save_index == 0:
        dump_network(filepath, network)
        return 1
    previous_filepath = f"../saves/network_save_{save_index-1}.txt"
    previous_network = load_network(previous_filepath)
    if network.cost(data) < previous_network.cost(data):
        dump_network(filepath, network)
        return save_index + 1
    return save_index


def main(network: Network, training_data: DataSet, test_data: DataSet) -> int:
    save_index = 0
    while not do_quit.is_set():
        network.train(training_data, 60, 1.5, batch_size=50)
        network.train(training_data, 60, 2, batch_size=100)
        network.train(training_data, 30, 1, batch_size=300)
        network.train(training_data, 10, 0.5)
        save_index = update_saved(network, training_data, save_index)
    update_saved(network, training_data, save_index)
    return 1


if __name__ == "__main__":
    training_data, test_data, network = initialize()
    try:
        exit(main(network, training_data, test_data))
    except KeyboardInterrupt:
        do_quit.set()
        exit(main(network, training_data, test_data))
