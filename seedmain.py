import csv
import random
import numpy as np
import math

# This implementation uses a 8 - 5 - 3 ANN,
# layer0 is the input layer,
# layer1 is the hidden layer,
# layer3 is the output layer.
learning_rate = 0.01
def main():



  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")

  printarr("train set",train_set)
  printarr("test set",test_set)

  network_config = [7,5,3] #number of neurons in each layer
  network = [[]]

  layer0 = []
  layer1 = []
  layer2 = []

  for i in range(network_config[1]):
    weights = []
    for i in range(network_config[0]):
      weights.append(random.random())
    layer1.append(Perceptron(weights,0))

  for i in range(network_config[2]):
    weights = []
    for i in range(network_config[1]):
      weights.append(random.random())
    layer2.append(Perceptron(weights,0))

  network.append(layer1)
  network.append(layer2)


  trails = 10000
  correct_rate = 0.3
  for t in range(trails):
    random.shuffle(train_set)
    correct = 0
    for n in range (len(train_set)):
      layer0 = train_set[n][:-1]

      layer0_out=layer0
      layer1_out = calculate_layer_output(layer0_out,layer1)
      layer2_out = calculate_layer_output(layer1_out,layer2)
      #print(layer1_out)
      ans = [0,0,0]
      ans[int(train_set[n][-1])-1] = 1

      ans_out = 0
      #error = calculate_error(layer2_out,ans)
      error = 1-correct_rate
      if layer2_out[0]>layer2_out[1]and layer2_out[0]>layer2_out[2]:
        ans_out = 1
      elif layer2_out[1]>layer2_out[0]and layer2_out[1]>layer2_out[2]:
        ans_out = 2
      else:
        ans_out=3

      if ans[ans_out-1] == 1:
        correct+=1

      #print(layer2_out)
      for j in range(len(layer2)):
        p = layer2[j]
        for i in range(len(layer1)):
          p.delta = (ans[j] - layer2_out[j]) * layer2_out[j] * (1-layer2_out[j])
          p.weights[i] += learning_rate * layer1_out[i] * p.delta

      for i in range(len(layer1)):
        p = layer1[i]
        for h in range(len(layer0)):
          delta_h_i = 0
          for j in range(len(layer2)):
            delta_h_i += layer2[j].delta * layer2[j].weights[i]

          delta = delta_h_i * layer0_out[i] * (1-layer0_out[i])

          p.delta = delta
          p.weights[i] += learning_rate * delta * layer0_out[h]
    correct_rate = correct/len(train_set)
    print("correct rate:",correct_rate)



class Perceptron:
  def __init__(self,weights,bias):

    self.weights = weights
    self.bias = bias
    self.delta = 0

  def output(self,inputs):
    return sigmoid(np.dot(inputs,self.weights)-self.bias)










def calculate_error(a,b):
  if (len(a) != len(b)):
    print("ERROR:,","dim a != dim b,("+str(len(a))+" ,"+str(len(b))+'),')
    exit(9)
  else:
    return np.sum(np.square(np.subtract(a, b)))

def calculate_layer_output(input,layer):
  outputs = []
  for p in layer:
    outputs.append(p.output(input))
  return outputs

def readcsv(filename):
  arr = []
  with open(filename,'r') as csvfile:
    for row in csvfile:
      row = row[0:-1].split(',')
      arr.append([float(i) for i in row])
  return arr

def printarr(name,arr, end=5):
  print(name)
  for i in range(end):
    line = ""
    for j in range(len(arr[0])):
      line += "\t"+str(arr[i][j])
    print(line)

def sigmoid(x):
  if x>700:
    return 0
  if x<-700:
    return 1
  return 1 / (1 + math.exp(-x))

main()