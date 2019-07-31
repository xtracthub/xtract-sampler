import csv
import json

prediction_results_file = open('prediction_results.json', 'r')
true_reults_file = csv.reader(open('new_naivetruth.csv', 'r'))

prediction_results_dict = json.load(prediction_results_file)
true_reults_dict = {}
for row in true_reults_file:
    true_reults_dict[row[0]] = row[2]

total_count = 0
wrong_count = 0
right_count = 0
print(true_reults_dict)
print(prediction_results_dict)
for result in true_reults_dict:
    print(result)
    try:
        if prediction_results_dict.get(result) == true_reults_dict.get(result):
            right_count += 1
        else:
            #print("{}: {}".format(result, prediction_results_dict.get(result)))
            #print("{}: {}".format(result, true_reults_dict.get(result)))
            print(prediction_results_dict.get(result))
            print(true_reults_dict.get(result))
            wrong_count += 1
        total_count += 1
    except Exception as e:
        print(e)
        pass

print("Number right: {}".format(right_count))
print("Number wrong: {}".format(wrong_count))
print("Total accuracy: {}".format(right_count / total_count))


