from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.utils import Sequence
from numpy import ndarray
import os
import tempfile


def compile_adam_msle(model: Sequential):
    from keras.losses import mean_squared_logarithmic_error
    model.compile(optimizer='adam',
                  loss=mean_squared_logarithmic_error,
                  metrics=['accuracy'])


class Model1(object):
    def __init__(self, save_path=None):
        self.defined = False
        self.compiled = False
        self.trained = False
        self.input_shape = (500, 20, 1)
        if save_path is None:
            self.save_path = tempfile.mkdtemp()
        else:
            self.save_path = save_path
        self.name = 'model1'
        self.model = Sequential(name=self.name)

    def build_model(self):
        from keras.layers import Conv2D, Dense, Flatten
        from keras.activations import relu
        # From https://navoshta.com/end-to-end-deep-learning/
        layers = [
            Conv2D(filters=16, kernel_size=(3, 3), input_shape=self.input_shape, activation=relu),
            Conv2D(filters=32, kernel_size=(3, 3), activation=relu),
            Conv2D(filters=64, kernel_size=(3, 3), activation=relu),
            Flatten(),
            Dense(500, activation=relu),
            Dense(100, activation=relu),
            Dense(20, activation=relu),
            Dense(1)
        ]
        self.model = Sequential(layers=layers, name=self.name)
        self.defined = True

    def compile(self):
        assert self.defined
        compile_adam_msle(self.model)
        self.compiled = True

    def train(self, generator: Sequence, epochs: int, batch_size: int=None):
        assert isinstance(generator, Sequence)
        assert self.compiled
        # Train the model, iterating on the data in batches
        # self.model.fit(data, labels, epochs=epochs, batch_size=batch_size)
        self.model.fit_generator(generator=generator,
                                 steps_per_epoch=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=None,
                                 validation_data=None,
                                 validation_steps=None,
                                 class_weight=None,
                                 max_queue_size=10,
                                 workers=1,
                                 use_multiprocessing=False,
                                 shuffle=True,
                                 initial_epoch=0)
        self.trained = True

    def save(self):
        assert self.defined
        print(self.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(os.path.join(self.save_path, self.name + '.json'), "w") as text_file:
            text_file.write(self.model.to_json())
        self.model.save_weights(filepath=os.path.join(self.save_path, self.name + '.h5'))

    def predict(self, x: ndarray):
        assert isinstance(x, ndarray)
        assert self.trained
        self.model.predict(x)

    def pipeline(self):
        self.build_model()
        self.compile()
        self.save()


model = Model1()
model.pipeline()
