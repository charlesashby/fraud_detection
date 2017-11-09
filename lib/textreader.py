import numpy as np
import pickle


class TextReader(object):
    """ Data Set Handler """

    def __init__(self):
        with open('data/data_set.pickle', 'rb') as handler:
            self.training_set_x, self.training_set_y, \
                self.test_set_x, self.test_set_y = pickle.load(handler)

    def iterate_mini_batch(self, size=128):
        n_batch = int(len(self.training_set_y) / size)

        for i in range(n_batch):
            inputs, targets = self.training_set_x[i * size: (i + 1) * size], \
                                self.training_set_y[i * size: (i + 1) * size]
            yield inputs, targets
    