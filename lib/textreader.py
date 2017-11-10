import numpy as np
import pickle


'''
class TextReader(object):
    """ Data Set Handler """

    def __init__(self):
        with open('data/data_set.pickle', 'rb') as handler:
            self.training_set_x, self.training_set_y, \
                self.test_set_x, self.test_set_y = pickle.load(handler)

    def iterate_mini_batch(self, size=128, data_set='TRAIN'):
        if data_set == 'TRAIN':
            n_batch = int(len(self.training_set_y) / size)
            data_x, data_y = self.training_set_x, self.training_set_y
        else:
            n_batch = int(len(self.test_set_y) / size)
            data_x, data_y = self.test_set_x, self.test_set_y

        for i in range(n_batch):
            inputs, targets = data_x[i * size: (i + 1) * size], \
                                data_y[i * size: (i + 1) * size]
            yield inputs, targets

'''


class TextReader(object):
    def __init__(self):
        # ones: fraud
        # zeros: no fraud
        with open('data/data_set_10.pickle', 'rb') as handler:
            self.training_set_ones, self.training_set_zeros, \
                self.test_set_ones, self.test_set_zeros = pickle.load(handler)

    def iterate_mini_batch(self, size=128, data_set='TRAIN'):
        """ Sample Mini-Batches """
        # Strong imbalance between the classes forces us
        # to over sample from the positive class

        if data_set == 'TRAIN':
            n_zeros = self.training_set_zeros.shape[0]
            n_ones = self.training_set_ones.shape[0]
            n_batch = int(n_zeros / size)
        else:
            n_zeros = self.test_set_zeros.shape[0]
            n_ones = self.test_set_ones.shape[0]
            n_batch = int(n_zeros / size)

        ones = np.random.randint(low=0, high=n_ones, size=[n_batch, int(size/2)])
        zeros = np.random.randint(low=0, high=n_zeros, size=[n_batch, int(size/2)])

        for i in range(n_batch):
            #data = self.training_set_ones[ones[i], :] + self.training_set_zeros[zeros[i], :]
            data = np.concatenate((self.training_set_ones[ones[i], :],
                                   self.training_set_zeros[zeros[i], :]), axis=0)
            inputs, targets = data[:, :-2], data[:, -2:]
            yield inputs, targets


if __name__ == '__main__':
    t = TextReader()
    m = t.iterate_mini_batch()
