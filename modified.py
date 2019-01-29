# Citation:
  # YASAR, Ali & Kaya, Esra & Saritas, Ismail. (2016).
  # Classification of Wheat Types by Artificial Neural Network.
  # International Journal of Intelligent Systems and Applications in Engineering.
  # 4. 12. 10.18201/ijisae.64198.


import csv
import random
import numpy as np
from math import exp
# number of neurons in each layer
layer_config = [8,5,3]
learning_rate = 0.3
n_folds = 5

n_epoch = 100
n_hidden = 5

def main():
  random.seed(1)
  # read data from csv
  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")
  printarr("trainset",train_set)
  # shuffle data
  # random.shuffle(train_set)
  # random.shuffle(test_set)
  wholeset = train_set+test_set
  random.shuffle(train_set)

  minmax = dataset_minmax(wholeset)
  normalize_dataset(wholeset, minmax)
  train_set = wholeset[0:len(train_set)]
  test_set = wholeset[len(train_set):]
  # view data samples

  printarr("train set", train_set,len(train_set))
  printarr("test set", test_set)



  network = initialize_network(layer_config[0],layer_config[1],layer_config[2])
  n_outputs = 3
  for epoch in range(n_epoch):
    for row in train_set:
      outputs = forward_propagate(network, row)
      expected = [0 for i in range(n_outputs)]
      expected[int(row[-1])] = 1
      print(expected)
      print(outputs)
      backward_propagate_error(network, expected)
      update_weights(network, row,learning_rate)

  print(accuracy(network,train_set))

def dataset_minmax(dataset):
  minmax = list()
  stats = [[min(column), max(column)] for column in zip(*dataset)]
  return stats

def str_column_to_int(dataset, column):
  class_values = [row[column] for row in dataset]
  unique = set(class_values)
  lookup = dict()
  for i, value in enumerate(unique):
    lookup[value] = i
  for row in dataset:
    row[column] = lookup[row[column]]
  return lookup

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
  for row in dataset:
    for i in range(len(row) - 1):
      row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def accuracy(network,data):
  correct = 0
  for row in data:
    output = forward_propagate(network,row)
    ans = output.index(max(output))
    if ans == int(row[-1]):
      correct += 1
  return correct/len(data)



def update_weights(network, row, l_rate):
  for i in range(len(network)):
    inputs = row[:-1]
    if i != 0:
      inputs = [neuron['output'] for neuron in network[i - 1]]
    for neuron in network[i]:
      for j in range(len(inputs)):
        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
      neuron['weights'][-1] += l_rate * neuron['delta']

# calculate the gradient of error w.r.t every neuron


def backward_propagate_error(network, expected):
  for i in reversed(range(len(network))):
    layer = network[i]
    errors = list()
    if i != len(network) - 1:
      for j in range(len(layer)):
        error = 0.0
        for neuron in network[i + 1]:
          error += (neuron['weights'][j] * neuron['delta'])
        errors.append(error)
    else:
      for j in range(len(layer)):
        neuron = layer[j]
        errors.append(expected[j] - neuron['output'])
    for j in range(len(layer)):
      neuron = layer[j]
      neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def transfer_derivative(output):
  return output * (1.0 - output)

# Forward propagate input to a network output
def forward_propagate(network, row):
  inputs = row
  for layer in network:
    new_inputs = []
    for neuron in layer:
      activation = activate(neuron['weights'], inputs)
      neuron['output'] = transfer(activation)
      new_inputs.append(neuron['output'])
    inputs = new_inputs
  return inputs

def transfer(activation):
  return 1.0 / (1.0 + exp(-activation))

def accuracy_metric(actual, predicted):
  correct = 0
  for i in range(len(actual)):
    if actual[i] == predicted[i]:
      correct += 1
  return correct / float(len(actual)) * 100.0



def activate(weights, inputs):
  activation = weights[-1]
  for i in range(len(weights) - 1):
    activation += weights[i] * inputs[i]
  return activation


def initialize_network(n_inputs, n_hidden, n_outputs):
  network = list()
  hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
  network.append(hidden_layer)
  output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
  network.append(output_layer)
  return network

def printarr(name,arr, end=5):
  print(name)
  for i in range(end):
    line = ""
    for j in range(len(arr[0])):
      line += "\t"+str(arr[i][j])
    print(line)

def readcsv(filename):
  arr = []
  with open(filename,'r') as csvfile:
    for row in csvfile:
      row = row[0:-1].split(',')
      row[-1] = int(row[-1])-1
      arr.append([float(i) for i in row])
      arr[-1][-1] = int(arr[-1][-1])
  return arr

main()