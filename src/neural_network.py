from get_file import get_file
import numpy as np
import sys


DEFAULT_INPUT_PATH = "../data/input.txt"


def normal_distribution_error(distribution: np.array) -> float:
    return np.mean(distribution)


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


if __name__ == "__main__":
    input_file_path = DEFAULT_INPUT_PATH
    if len(sys.argv) > 2:
        input_file_path = sys.argv[1]

    neural_network = Network()
    with get_file(input_file_path).open("r") as file:
        if not neural_network.parse_string(file.read()):
            print("[ERROR] network string failed to parse")
            sys.exit()

    neural_network.backpropagation()
