import numpy
import math
import random

iterations = 40
learning_rate = 0.01

def main():
  random.seed(0)
  # read data from csv
  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")

  wholeset = train_set + test_set
  minmax = dataset_minmax(wholeset)
  normalize_dataset(wholeset, minmax)

  train_set = wholeset[0:len(train_set)]
  test_set = wholeset[len(train_set):]

  random.shuffle(train_set)
  random.shuffle(test_set)

  printarr("train set", train_set)
  printarr("test set", test_set)

  # initialize the weights of the perceptrons in the output layer
  p1 = [random.random()] * len(train_set[0])
  p2 = [random.random()] * len(train_set[0])
  p3 = [random.random()] * len(train_set[0])
  neurons = [p1,p2,p3]

  for iteration in range(iterations):
    for row in train_set:
      outputs = feed_data(neurons,row)
      expected = [0, 0, 0]
      expected[int(row[-1]) - 1] = 1
      adjust_weights(neurons,row,outputs,expected)
    print("accuracy =", test(test_set,neurons))

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