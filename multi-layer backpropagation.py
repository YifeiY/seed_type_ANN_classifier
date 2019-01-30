# Citation:
  # YASAR, Ali & Kaya, Esra & Saritas, Ismail. (2016).
  # Classification of Wheat Types by Artificial Neural Network.
  # International Journal of Intelligent Systems and Applications in Engineering.
  # 4. 12. 10.18201/ijisae.64198.

# This model (layers = [8,5,3]) is different from the one in citation (layers = [8,10,1])
# But it is able to achieve above 99.99% accuracy after ~320 iterations

import csv
import random
import numpy as np
import math
import matplotlib.pyplot as plt
# number of neurons in each layer
layer_config = [8,5,3]
learning_rate = 0.3


def main():
  random.seed(0)
  # read data from csv
  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")

  wholeset = train_set+test_set
  minmax = dataset_minmax(wholeset)
  normalize_dataset(wholeset, minmax)

  train_set = wholeset[0:len(train_set)]
  test_set = wholeset[len(train_set):]

  random.shuffle(train_set)
  random.shuffle(test_set)

  printarr("train set", train_set)
  printarr("test set", test_set)



  network = make_network(layer_config)
	
  accuracy_plot = []
  for i in range(320):
    #print(i)
    for trail in range (len(train_set)):
      inputs = train_set[trail]
      outputs = feed_forward(network,inputs)
      expected = [0 for i in range(len(outputs))]
      expected[int(inputs[-1])] = 1
      feed_error_back(network,expected) # calculate 'delta's
      update_weights(network,inputs[:-1])
    accuracy_value = accuracy(network,test_set)
    accuracy_plot.append([accuracy_value,i])
    print(accuracy_value) # using test set to test the outcome of training on the training data set
  


    
    
  plt.plot([accuracy_plot[i][1] for i in range(len(accuracy_plot))],[accuracy_plot[i][0] for i in range(len(accuracy_plot))])
  plt.show()



#####need !!!!!!!!!!!!!!!
def dataset_minmax(dataset):
  minmax = list()
  stats = [[min(column), max(column)] for column in zip(*dataset)]
  return stats


#####need !!!!!!!!!!!!!!!

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
  for row in dataset:
    for i in range(len(row) - 1):
      row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def accuracy(network,data):
  correct = 0
  for row in data:
    output = feed_forward(network,row)
    ans = output.index(max(output))
    if ans == int(row[-1]):
      correct += 1
  return correct/len(data)




def update_weights(network, inputs):
  # updating the weights in the first hidden layer
  for neuron in network[0]:
    for j in range(len(inputs)):
      neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
    neuron['weights'][-1] += learning_rate * neuron['delta']
  for i in range(1,len(network)):
    for neuron in network[i]:
      inputs = [neuron['output'] for neuron in network[i - 1]]
      for j in range(len(inputs)):
        neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
      neuron['weights'][-1] += learning_rate * neuron['delta'] # bias


# calculate the gradient of error w.r.t every neuron
def feed_error_back(network, expected):
  for i in reversed(range(len(network))):
    layer = network[i]
    errors = []
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
      neuron['delta'] = errors[j] * calculate_derivative(neuron['output'])

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


def calculate_derivative(x):
  return x * (1 - x)

def feed_forward(network,inputs):
  for layer in network:
    outputs = [] # outputs of the current layer
    for neuron in layer:
      weights = neuron['weights']
      for i in range (len(weights)-1):
        neuron['output'] += weights[i] * inputs[i]
      neuron['output'] = sigmoid(neuron['output'])
      outputs.append(neuron['output'])
    inputs = outputs # the inputs of the next layer is the output of the current layer
  return inputs # the output of the last layer is the input of next imaginary operation


def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

def make_network(layers):
  network = []
  # input layer is the data itself
  # hidden layer(s), output layer
  for i in range(1,len(layers)):
    layer = [{'weights':[random.random() for n in range (layers[i-1]+1)],
              'output':0,'delta':0} for n in range(layers[i]+1)]
    network.append(layer)
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
