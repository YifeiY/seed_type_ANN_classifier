import random
import matplotlib.pyplot as plt

iterations = 20000
learning_rate = 0.005

def main():
  global learning_rate
  random.seed(0)
  # read data from csv
  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")

  random.shuffle(train_set)

  # initialize the weights of the perceptrons in the output layer
  p1 = [random.random()] * len(train_set[0])
  p2 = [random.random()] * len(train_set[0])
  p3 = [random.random()] * len(train_set[0])
  neurons = [p1,p2,p3]

  accuracy = 0
  accuracy_plot = []

  for iteration in range(iterations):
    for row in train_set:
      outputs = feed_data(neurons,row)# get the output of the network
      expected = [0, 0, 0]
      expected[int(row[-1])] = 1
      adjust_weights(neurons,row,outputs,expected) # adjust the weights according to real outputs and expected outputs
    accuracy = test(test_set,neurons)
    accuracy_plot.append([accuracy,iteration]) # for visualization of improvement once the training is done
    print(iteration,"-- accuracy =",int(accuracy*100))

  plt.plot([accuracy_plot[i][1] for i in range(len(accuracy_plot))],[accuracy_plot[i][0] for i in range(len(accuracy_plot))])
  plt.show()


def adjust_weights(neurons,inputs,outputs,expected):
  # +[1] correspond to the multiplier of the bias of a neuron, the bias itself is stored as a weight
  inputs = inputs[:-1] + [1]
  for j in range(len(neurons)):
    neuron = neurons[j]
    for i in range (len(neuron)):
      if outputs[j] == 1 and expected[j] == 0:
        neuron[i] -= learning_rate * inputs[i]
      elif outputs[j] == 0 and expected[j] == 1:
        neuron[i] += learning_rate * inputs[i]

# test calcualte the accuracy of the network, using test data
def test(test_set,neurons):
  total_length = len(test_set)
  correct = 0
  for row in test_set:
    expected = [0, 0, 0]
    expected[int(row[-1])] = 1
    if (expected == feed_data(neurons,row)):
      correct += 1
  return correct/total_length

def feed_data(neurons,row):
  output = []
  for neuron in neurons:
    summation = 0
    # calculate the potential sum
    for i in range(len(neuron)):
      summation += neuron[i] * row[i]
    summation += neuron[-1] # add the bias
    # check if the potential sum exceeds the
    output.append(activation(summation))
  return output

# simple threshold activation function
def activation(x):
  if x >=0:
    return 1
  else:
    return 0

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
