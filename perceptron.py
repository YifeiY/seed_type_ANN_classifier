import random

learning_rate = 0.005
output_file_name = "perceptron confusion matrix.txt"

def main():
  global learning_rate
  random.seed(0)

  # read data from csv
  train_set = readcsv("trainSeeds.csv")
  test_set = readcsv("testSeeds.csv")

  random.shuffle(train_set)

  # initialize the weights of the perceptrons in the output layer
  p1 = [random.random() for _ in range(len(train_set[0]) + 1)]
  p2 = [random.random() for _ in range(len(train_set[0]) + 1)]
  p3 = [random.random() for _ in range(len(train_set[0]) + 1)]
  neurons = [p1,p2,p3]
  initial_weights = [[neuron[j] for j in range(len(neuron))] for neuron in neurons]

  # these three variable is used to terminate the program once the accuracy rate has settled for 100 iterations
  iteration = 0
  same_accuracy = 0
  last_accuracy = 0
  settle_counts = 100

  while same_accuracy < settle_counts:
    for row in train_set:
      outputs = feed_data(neurons,row)# get the output of the network
      expected = [0, 0, 0]
      expected[int(row[-1])] = 1
      adjust_weights(neurons,row,outputs,expected) # adjust the weights according to real outputs and expected outputs
    accuracy = test(test_set,neurons)
    # whether the program is generating consistent accuracy rate or not
    if accuracy == last_accuracy:
      same_accuracy += 1
    else:
      last_accuracy = accuracy
      same_accuracy = 0
    iteration += 1
    print("iteration:",iteration)

  # output log file
  output_files(train_set,test_set,initial_weights,neurons,iteration,settle_counts)
  print("please see",output_file_name,"file for details")


def output_files(train_set,test_set,initial_weights,neurons,iteration,settle_counts):

  output_file = open(output_file_name, "w")
  output_data_set = train_set + test_set
  output_data = [["area", "pmeter", "compact", "length", "width", "asym", "klength", "type", "prediction"]]
  type_dict = {1: 'Kama', 2: 'Rosa', 3: 'Canada'}

  # writing predictions
  for i in range(len(output_data_set)):
    result = feed_data(neurons, output_data_set[i])
    output_data.append(
      output_data_set[i][:-1] + [type_dict[output_data_set[i][-1] + 1]] + [type_dict[result.index(max(result)) + 1]])
  for row in output_data:
    line = ""
    for item in row:
      line += str(item) + '\t'
    output_file.write(line + '\n')
  output_file.write("\niterations used: " + str(iteration))
  output_file.write("\nterminating criteria: " + "after " + str(settle_counts) + " numbers of the same accuracy occurred\n\n")

  # writing the weights
  for i in range(len(neurons)):
    output_file.write('\nneuron ' + str(i) + ':\n')
    for j in range(len(neurons[i])):
      output_file.write(str(neurons[i][j]) +'\t--->\t' +str(initial_weights[i][j])+'\n')

  # writing confusion matrix
  output_file.write('\n\n')
  confusion_matrix = []
  for type in range (1,4):
    this_type = type_dict[type]
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in output_data:
      true_type = row[-2]
      predicted_type = row[-1]
      if true_type == this_type:
        if true_type == predicted_type:
          true_positive += 1
        else:
          false_negative += 1
      else:
        if this_type != predicted_type:
          true_negative += 1
        else:
          false_positive += 1
    confusion_matrix.append([this_type,true_positive,false_positive,true_negative,false_negative,
                             (true_positive)/(true_positive+false_positive),(true_positive)/(true_positive+false_negative)])
  for type in confusion_matrix:
    output_file.write(type[0]+':\n') # type name
    output_file.write("Precision: " + str(type[-2]) + '\n')
    output_file.write("Recall: " + str(type[-1]) + '\n')
    output_file.write("T+ = " + str(type[1]) +'\t\t')
    output_file.write("F+ = " + str(type[2]) +'\t\n')
    output_file.write("T- = " + str(type[3]) +'\t')
    output_file.write("F- = " + str(type[4]) +'\t\n\n')

  output_file.close()


def feed_data(neurons,row):
  output = []
  for neuron in neurons:
    summation = 0
    # calculate the potential sum
    for i in range(len(neuron)-1):
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

def adjust_weights(neurons,inputs,outputs,expected):
  # +[1] correspond to the multiplier of the bias of a neuron, the bias itself is stored as a weight
  inputs = inputs[:-1] + [1]
  for j in range(len(neurons)):
    neuron = neurons[j]
    for i in range (len(neuron)-1):
      if outputs[j] == 1 and expected[j] == 0:
        neuron[i] -= learning_rate * inputs[i]
      elif outputs[j] == 0 and expected[j] == 1:
        neuron[i] += learning_rate * inputs[i]
    if outputs[j] == 1 and expected[j] == 0:
        neuron[-1] -= learning_rate
    elif outputs[j] == 0 and expected[j] == 1:
        neuron[-1] += learning_rate

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
