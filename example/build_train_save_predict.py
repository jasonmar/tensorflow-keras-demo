from tensorflow.contrib.keras.api.keras.activations import relu
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.contrib.keras.api.keras.losses import mean_squared_logarithmic_error
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.utils import Sequence
from numpy import ndarray, random
import os
import tempfile
import numpy as np


class RandomSequence(Sequence):
    """
    From keras tests keras.engine.test_training.py#RandomSequence
    """
    def __init__(self, batch_size=8, sequence_length=32, x_shape=(500, 20, 1), y_shape=(4,), debug=False):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.debug = debug

    def __len__(self):
        if self.debug:
            print("__len__()")
        return self.sequence_length

    def __getitem__(self, idx):
        if self.debug:
            print("__getitem__(" + str(idx) + ")")
        x_shape = (self.batch_size,) + self.x_shape
        y_shape = (self.batch_size,) + self.y_shape
        x = np.random.random(x_shape)
        y = np.random.random(y_shape)
        return x, y

    def on_epoch_end(self):
        if self.debug:
            print("on_epoch_end()")
        pass


class Model1(object):
    def __init__(self, name='model1', save_path=None):
        print(name)
        if save_path is None:
            self.save_path = tempfile.mkdtemp()
        else:
            self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("created " + self.save_path)

        self.name = name
        self.model = Sequential(name=self.name)
        self.epochs = 1
        self.batch_size = 8
        self.input_shape = (500, 20, 1)
        self.output_shape = 4
        self.generator = RandomSequence(batch_size=self.batch_size, x_shape=self.input_shape, y_shape=(self.output_shape,))

    def build_model(self):
        print("Building")
        # From https://navoshta.com/end-to-end-deep-learning/
        # https://keras.io/layers/convolutional/#conv2d
        # https://keras.io/layers/core/#Dense
        m = self.model
        m.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=self.input_shape, activation=relu))
        m.add(Conv2D(filters=32, kernel_size=(3, 3), activation=relu))
        m.add(Conv2D(filters=64, kernel_size=(3, 3), activation=relu))
        m.add(Flatten())
        m.add(Dense(500, activation=relu))
        m.add(Dropout(0.27))
        m.add(Dense(100, activation=relu))
        m.add(Dense(20, activation=relu))
        m.add(Dense(self.output_shape))
        print("Build complete")

    def compile(self):
        print("Compiling")
        self.model.compile(optimizer='adam',
                           loss=mean_squared_logarithmic_error,
                           metrics=['accuracy'])
        print("Compilation complete")

    def train(self):
        print("Training")
        self.model.fit_generator(generator=self.generator,
                                 steps_per_epoch=self.batch_size,
                                 epochs=self.epochs)
        print("Training complete")

    def save(self):
        print("saving to " + self.save_path)
        json_path = os.path.join(self.save_path, self.name + '.json')
        with open(json_path, "w") as f:
            f.write(self.model.to_json())
            print("wrote " + json_path)
        h5_path = os.path.join(self.save_path, self.name + '.h5')
        self.model.save_weights(filepath=h5_path)
        print("wrote " + h5_path)

    def predict(self, x: ndarray):
        """
        Generates predictions
        :param x: ndarray with shape: (batch_size,) + input_shape
        :return: ndarray with shape: (batch_size,) + output_shape
        """
        assert isinstance(x, ndarray)
        return self.model.predict(x)

    def pipeline(self):
        self.build_model()
        self.compile()
        self.train()
        self.save()
        random_input = random.random((1,) + self.input_shape)
        prediction = self.predict(random_input)
        print("random_prediction: " + str(prediction[0]))


m = Model1()
m.pipeline()
