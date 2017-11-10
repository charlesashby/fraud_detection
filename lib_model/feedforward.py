import tensorflow as tf
import numpy as np
from lib.ops import MLP
from lib.textreader import TextReader

SAVE_PATH = 'checkpoints/mlp'
LOGGING_PATH = 'data/log.txt'


class FeedForwardNN(object):
    """ Feed Forward Neural Network Implementation """

    def __init__(self):
        self.hparams = self.get_hparams()
        self.X = tf.placeholder('float32', shape=[None, 30], name='X')
        self.Y = tf.placeholder('float32', shape=[None, 2], name='Y')

    def build(self):
        """ Build the Network """

        mlp = MLP(self.X, out_dim=2, size=128, scope='mlp')

        self.prediction = tf.nn.softmax(mlp)

    def train(self):
        """ Train the Network """
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        EPOCHS = self.hparams['EPOCHS']
        learning_rate = self.hparams['learning_rate']
        patience = self.hparams['patience']

        pred = self.prediction

        cost = - tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

        predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

        acc = tf.reduce_mean(tf.cast(predictions, 'float32'))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # parameters for saving and early stopping
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            best_acc = 0.0
            DONE = False
            epoch = 0

            while epoch <= EPOCHS and not DONE:
                loss = 0.0
                batch = 1
                epoch += 1

                reader = TextReader()
                for mini_batch in reader.iterate_mini_batch(BATCH_SIZE, data_set='TRAIN'):
                    batch_x, batch_y = mini_batch

                    _, c, a = sess.run([optimizer, cost, acc], feed_dict={self.X: batch_x, self.Y: batch_y})

                    loss += c
                    n_batch = int(reader.training_set_zeros.shape[0] / BATCH_SIZE)
                    if batch % 100 == 0:
                        # Compute Accuracy on the Training set and print some info
                        print('Epoch: %5d/%5d -- batch: %5d/%5d -- Loss: %.4f -- Train Accuracy: %.4f' %
                              (epoch, EPOCHS, batch, n_batch, loss/batch, a))

                        # Write loss and accuracy to some file
                        log = open(LOGGING_PATH, 'a')
                        log.write('%s, %6d, %.5f, %.5f \n' % ('train', epoch * batch, loss/batch, a))
                        log.close()

                    # --------------
                    # EARLY STOPPING
                    # --------------

                    # Compute Accuracy on the Validation set, check if validation has improved, save model, etc
                    if batch % 500 == 0:
                        accuracy = []

                        # Compute accuracy on validation set
                        for mb in reader.iterate_mini_batch(BATCH_SIZE, data_set='TEST'):
                            valid_x, valid_y = mb
                            a = sess.run([acc], feed_dict={self.X: valid_x, self.Y: valid_y})
                            accuracy.append(a)
                        mean_acc = np.mean(accuracy)

                        # if accuracy has improved, save model and boost patience
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            save_path = saver.save(sess, SAVE_PATH)
                            patience = self.hparams['patience']
                            print('Model saved in file: %s' % save_path)

                        # else reduce patience and break loop if necessary
                        else:
                            patience -= 500
                            if patience <= 0:
                                DONE = True
                                break
                        n_batch = int(reader.test_set_zeros.shape[0] / BATCH_SIZE)

                        print('Epoch: %5d/%5d -- batch: %5d/%5d -- Valid Accuracy: %.4f' %
                             (epoch, EPOCHS, batch, n_batch, mean_acc))

                        # Write validation accuracy to log file
                        log = open(LOGGING_PATH, 'a')
                        log.write('%s, %6d, %.5f \n' % ('valid', epoch * batch, mean_acc))
                        log.close()

                    batch += 1

    def get_hparams(self):
        """ Get Hyper-Parameters """
        return {
            'BATCH_SIZE':       128,
            'EPOCHS':           500,
            'learning_rate':    0.0001,
            'patience':         5000000
        }

if __name__ == '__main__':
    network = FeedForwardNN()
    network.build()
    network.train()
