import numpy as np
import csv
import pickle

# Separate classes
ones = []
zeros = []
with open('data/creditcard.csv', 'r') as f:
    data = csv.reader(f, delimiter=',')
    for line in data:
        if line[-1] == '0':
            zeros.append(line)
        else:
            ones.append(line)

test_set = np.array(ones[:-93] + zeros[:-50000])
training_set = np.array(ones[-93:] + zeros[-50000:])

np.random.shuffle(test_set)
np.random.shuffle(training_set)

test_set_x = test_set[:, :-1]
test_set_y = test_set[:, -1]
training_set_x = training_set[:, :-1]
training_set_y = training_set[:, -1]

data_set = [training_set_x, training_set_y,
            test_set_x, test_set_y]

with open('data_set.pickle', 'wb') as handle:
    pickle.dump(data_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""
with open('data_set.pickle', 'rb') as handle:
    b = pickle.load(handle)
"""
