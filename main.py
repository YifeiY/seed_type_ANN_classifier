import csv
import random
def main():


  train_set = readcsv("trainSeeds")
  test_set = readcsv("testSeeds")

  number_of_params = len(train_set[0])-1
  layer_1 = [[Perceptron()] * number_of_params] # first layer of perceptrons that take the inputs
class Perceptron:

  def __init__(self,w = random.random(),b=random.random()):
    self.w = w
    self.b = b

def arrprint(arr,end = 5):
  print("\tarea\tperim\tcmpct\tlength\twidth\tasymty\tkernel\tkind\tprediction")
  for row in arr[0:end]:
    content = ""
    for item in row:
      content += "\t" + item
    print(content)


def readcsv(filename):
  arr = []
  with open('trainSeeds.csv','r') as csvfile:
    for row in csvfile:
      arr.append(row[0:-1].split(','))
  return arr

main()