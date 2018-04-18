import numpy as np
from abc import abstractmethod


class Sequence(object):
    """
    from keras.utils.data_utils import Sequence
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass


def sequence_test(generator: Sequence, epochs=1):
    """
    Flow copied from keras.engine.training.py#Model#fit_generator
    
    :param generator: 
    :param epochs: 
    :return: 
    """
    assert isinstance(generator, Sequence)
    assert isinstance(epochs, int)
    print("this is a sequence")

    output_generator = iter(generator)
    steps_per_epoch = len(generator)
    epoch = 0

    while epoch < epochs:
        steps_done = 0
        batch_index = 0
        while steps_done < steps_per_epoch:
            generator_output = next(output_generator)

            if not hasattr(generator_output, '__len__'):
                errmsg = 'Output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: ' + str(
                    generator_output)
                raise ValueError(errmsg)

            if len(generator_output) == 2:
                x, y = generator_output
                print("obs: " + str(type(x)) + " " + str(x.shape))
                print("lbl: " + str(type(y)) + " " + str(y.shape))
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                errmsg = 'Output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: ' + str(
                    generator_output)
                raise ValueError(errmsg)

            batch_index += 1
            steps_done += 1

        generator.on_epoch_end()
        epoch += 1


class RandomSequence(Sequence):
    """
    From keras tests keras.engine.test_training.py#RandomSequence
    """
    def __init__(self, batch_size, sequence_length=12):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self):
        print("__len__()")
        return self.sequence_length

    def __getitem__(self, idx):
        print("__getitem__(" + str(idx) + ")")
        x_shape = (self.batch_size, 500, 12)
        y_shape = (self.batch_size, 4)
        x = np.random.random(x_shape)
        y = np.random.random(y_shape)
        return x, y

    def on_epoch_end(self):
        print("on_epoch_end()")
        pass


class TestSequence(Sequence):
    """
    Just prints
    typically wrapped by iter()
    """

    def __init__(self, n, batch_size):
        self.n = n
        self.k = batch_size

    def __len__(self):
        print("__len__()")
        return self.n

    def __getitem__(self, idx):
        """
        Yields one batch of observations and labels
        :param idx: 
        :return: tuple of ndarray (obs, labels)
        """
        print("__getitem__(" + str(idx) + ")")
        batch_x = [0] * self.k
        batch_y = [1] * self.k
        x = np.array(batch_x)
        y = np.array(batch_y)
        return x, y

    def on_epoch_end(self):
        print("on_epoch_end()")



#sequence_test(TestSequence(10, 2), 3)
sequence_test(RandomSequence(3, 5), 3)
