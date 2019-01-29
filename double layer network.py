import numpy
import math
import random
import matplotlib.pyplot as plt

iterations = 20000
learning_rate = 0.001

def main():
  global learning_rate
  random.seed(0)
  # read data from csv
  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")

  #wholeset = train_set + test_set
  #normalize_dataset(wholeset)

  #train_set = wholeset[0:len(train_set)]
  #test_set = wholeset[len(train_set):]


  
  # initialize the weights of the perceptrons in the output layer
  p1 = [random.random()] * len(train_set[0])
  p2 = [random.random()] * len(train_set[0])
  p3 = [random.random()] * len(train_set[0])
  neurons = [p1,p2,p3]
	
  accuracy = 0
  accuracy_plot = []
  for iteration in range(iterations):
    random.shuffle(train_set)
    random.shuffle(test_set)
    #learning_rate = (1 - accuracy)/300
    for row in train_set:
      outputs = feed_data(neurons,row)
      expected = [0, 0, 0]
      expected[int(row[-1]) - 1] = 1
      adjust_weights(neurons,row,outputs,expected)
    accuracy = test(test_set,neurons)
    accuracy_plot.append([accuracy,iteration])
    print(iteration,"-- accuracy =",int(accuracy*100))
    
    
  plt.plot([accuracy_plot[i][1] for i in range(len(accuracy_plot))],[accuracy_plot[i][0] for i in range(len(accuracy_plot))])
  plt.show()


def adjust_weights(neurons,inputs,outputs,expected):
  inputs = inputs[:-1] +[1]
  for j in range(len(neurons)):
    neuron = neurons[j]
    for i in range (len(neuron)):
      if outputs[j] == 1 and expected[j] == 0:
        neuron[i] -= learning_rate * inputs[i]
      elif outputs[j] == 0 and expected[j] == 1:
        neuron[i] += learning_rate * inputs[i]

def test(test_set,neurons):
  total_length = len(test_set)
  correct = 0
  for row in test_set:
    expected = [0, 0, 0]
    expected[int(row[-1]) - 1] = 1
    if (expected == feed_data(neurons,row)):
      correct += 1
  return correct/total_length

def feed_data(neurons,row):
  output = []
  for neuron in neurons:
    summation = 0
    for i in range(len(neuron)):
      summation += neuron[i] * row[i]
    summation += neuron[-1] # add the bias
    output.append(activation(summation))
  return output


def activation(x):
  if x >0:
    return 1
  else:
    return 0


def normalize_dataset(dataset):
  min_max = [[min(column), max(column)] for column in zip(*dataset)]
  for row in dataset:
    for i in range(len(row) - 1):
      row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

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
