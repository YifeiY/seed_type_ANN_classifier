from sklearn.linear_model import Perceptron
output_file_name = "scikit-learn.Perceptron output file.txt"

def main():
  train_data, train_answer = readcsv("trainSeeds.csv")
  test_data, test_answer = readcsv("testSeeds.csv")

  clf = Perceptron(tol=1e-3, random_state=0,n_iter_no_change=100)
  clf.fit(train_data,train_answer)

  predicted_answer = clf.predict(train_data + test_data)

  whole_data, whole_answer = train_data + test_data, train_answer + test_answer

  output_data = [] # the prediction of every data points, concentrated
  for i in range (len(whole_data)):
    output_data.append(whole_data[i] + [whole_answer[i]] + [predicted_answer[i]])

  output_file = open(output_file_name, "w")
  type_dict = {1: 'Kama', 2: 'Rosa', 3: 'Canada'}
  confusion_matrix = []
  for type in range(1, 4):
    this_type = type_dict[type]
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in output_data:
      true_type = type_dict[row[-2]]
      predicted_type = type_dict[row[-1]]
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
    confusion_matrix.append([this_type, true_positive, false_positive, true_negative, false_negative,
                             (true_positive) / (true_positive + false_positive),
                             (true_positive) / (true_positive + false_negative)])

  for type in confusion_matrix:
    output_file.write(type[0] + ':\n')  # type name
    output_file.write("Precision: " + str(type[-2]) + '\n')
    output_file.write("Recall: " + str(type[-1]) + '\n')
    output_file.write("T+ = " + str(type[1]) + '\t\t')
    output_file.write("F+ = " + str(type[2]) + '\t\n')
    output_file.write("T- = " + str(type[3]) + '\t')
    output_file.write("F- = " + str(type[4]) + '\t\n\n')

def readcsv(filename):
  data_arr = []
  ans_arr = []
  with open(filename,'r') as csvfile:
    for row in csvfile:
      row = row[0:-1].split(',')
      row[-1] = int(row[-1])
      data_arr.append([float(i) for i in row[:-1]])
      ans_arr.append(int(row[-1]))
  return data_arr,ans_arr

main()