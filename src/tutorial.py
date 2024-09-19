from random import seed
from random import random
from math import exp

#Initialize a newtwork
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    #creating n hidden neurons with number of inputs +1 weights 
    #one weight for each input column in dataset and one bias
    hidden_layer = [{'weights': [random() for i in range(n_inputs +1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    #creating output layer with n_output neurons and each has N-hidden+1 weights
    #each nwueron in the output layer has a weightfor each neuron in the hidden layer 
    output_layer = [{'weights': [random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    #calculate the neuron activation for an input
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i] #assumes that the bias is the last weight
    return activation

def transfer(activation):
    #implements the sigmoid function
    return 1.0/(1.0 + exp(-activation))

def forward_propogate(network,row):
    #forward propogate input to a network output
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation) #a neruons output value is stored in outputs
            new_inputs.append(neuron['output']) #output layer is collected in new_inputs and will be used as the inputs for the following layer
        inputs = new_inputs
    return inputs

def transfer_derivative(output):
    #calculating the derivative of the sigmoid function to get the slope of an output value from a neuron
    return output * (1.0-output)

def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    #update network weights with error using an input row of data and a learning rate
    #assume forward and backward propogaton have already been run
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    #train a network for a fixed number of epochs
    for epoch in range(n_epoch):
        sum_error =0
        for row in train:
            outputs = forward_propogate(network,row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] =1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate(network, expected)
            update_weights(network,row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' %(epoch, l_rate,sum_error ))

def predict(network,row):
    #Make a prediction with a network
    outputs = forward_propogate(network,row)
    return outputs.index(max(outputs))

"""Testing first network initialization"""
#seed(1)
#network = initialize_network(2,1,2) #creating a network with two nerons in the hidden layer
#for layer in network:
    #print(layer)

"""test forward propogation"""
#network = [[{'weights': [0.13436424411240122,0.8474337369372327,0.763774618976614]}],[{'weights': [0.2550690257394217,0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#row = [1,0,None] #propogating on pattern [1,0]
#output = forward_propogate(network,row)
#print(output) #output layer has two neurons so will get list of two numbers

"""test backward propogation"""
#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'output': 0.6213859615555266,'weights': [0.2550690257394217, 0.49543508709194095]},{'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763 ]}]]
#expected = [0,1]
#backward_propagate(network,expected)
#for layer in network:
    #print(layer)

"""Testing training backprop alg"""
#seed(1)
#dataset = [[2.7810836,2.550537003,0],
	#[1.465489372,2.362125076,0],
	#[3.396561688,4.400293529,0],
	#[1.38807019,1.850220317,0],
	#[3.06407232,3.005305973,0],
	#[7.627531214,2.759262235,1],
	#[5.332441248,2.088626775,1],
	#[6.922596716,1.77106367,1],
	#[8.675418651,-0.242068655,1],
	#[7.673756466,3.508563011,1]]
#n_inputs = len(dataset[0]) -1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = initialize_network(n_inputs, 2, n_outputs)
#train_network(network,dataset,.5,20,n_outputs)
#for layer in network:
  #  print(layer)

"""Testing predictions"""
#network numbers came from running the training
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))