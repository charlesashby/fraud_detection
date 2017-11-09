from lib_model.feedforward import FeedForwardNN


if __name__ == '__main__':
    network = FeedForwardNN()
    network.build()
    network.train()

