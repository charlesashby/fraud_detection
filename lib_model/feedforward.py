import tensorflow as tf
import numpy as np


class FeedForwardNN(object):
    """ Feed Forward Neural Network Implementation """

    def __init__(self):
        self.hparams = self.get_hparams()
        self.X = tf.placeholder('float32', shape=[None, 30], name='X')
        self.Y = tf.placeholder('float32', shape=[None, 2], name='Y')

    def build(self):
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        


    def get_hparams(self):
        """ Get Hyper-Parameters """
        return {
            'BATCH_SIZE':       128,
            'EPOCHS':           500,
            'learning_rate':    0.0001,
        }