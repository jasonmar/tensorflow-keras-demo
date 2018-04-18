from tensorflow.python.client.session import Session
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Dense, Flatten, Dropout, TimeDistributed, LSTM, Input, ELU, SimpleRNN, add
from tensorflow.contrib.keras.api.keras.activations import relu, softmax
from tensorflow.contrib.keras.api.keras.losses import mean_squared_logarithmic_error
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.python.summary.summary import merge_all, Summary, FileWriter
from tensorflow.python.keras._impl.keras.backend import get_session
from numpy import ndarray, random
from nn.util import RandomSequence
import os
import tempfile
import time


class TestCallback(Callback):
    """
    From tensorflow.contrib.keras.api.keras.callbacks#TensorBoard
    """
    def __init__(self, log_dir='./logs'):
        super(TestCallback, self).__init__()
        self.log_dir = log_dir
        self.merged = None
        self.write_graph = True
        self.writer: FileWriter = None
        self.sess: Session = None
        self.model: Model = None

    def set_model(self, model: Model):
        self.model = model
        self.sess = get_session()
        self.merged = merge_all()
        if self.write_graph:
            self.writer = FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Write keras metrics and loss
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)

        # Write custom metrics
        summary = Summary()
        test_value = summary.value.add()
        test_value.simple_value = 42.0
        test_value.tag = 'TestValue'
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()


class TestModel(object):
    def __init__(self, name='testmodel', save_path=None):
        print(name)
        self.name = name
        self.model = Sequential(name=self.name)
        self.epochs = 1
        self.time_step = 8
        self.batch_size = 16
        self.input_shape = (500, 20, 1)
        self.batch_input_shape = (self.time_step,) + self.input_shape
        self.output_shape = 4
        self.generator = RandomSequence(batch_size=self.batch_size, x_shape=self.batch_input_shape, y_shape=(self.output_shape,))

        if save_path is None:
            self.save_path = tempfile.mkdtemp()
        else:
            self.save_path = save_path

        self.log_dir = os.path.join(self.save_path, "logs_" + str(int(time.time()))[-6:])
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print("Created log_dir: " + self.log_dir)

        self.testcallback = TestCallback(log_dir=self.log_dir)

    def build_model(self):
        print("Building")

        keep_prob = 0.288

        # From https://navoshta.com/end-to-end-deep-learning/
        # https://keras.io/layers/convolutional/#conv2d
        # https://keras.io/layers/core/#Dense
        m = Sequential(name='vision')
        m.add(Conv2D(filters=64, kernel_size=(16, 3), strides=(8, 1), input_shape=self.input_shape))
        m.add(Dropout(keep_prob))

        m.add(Conv2D(filters=64, kernel_size=(8, 2), strides=(2, 1)))
        m.add(Dropout(keep_prob))

        m.add(Conv2D(filters=64, kernel_size=(8, 2), strides=(1, 1)))
        m.add(Dropout(keep_prob))

        m.add(Conv2D(filters=64, kernel_size=(8, 2), strides=(1, 1)))
        m.add(Dropout(keep_prob))

        for k in [1024, 512, 256]:
            m.add(Dense(k, activation=relu))
            m.add(Dropout(keep_prob))

        m.add(Dense(128, activation=None))
        m.add(Dropout(keep_prob))
        m.add(Flatten())

        # https://keras.io/getting-started/functional-api-guide/
        # https://keras.io/getting-started/sequential-model-guide/
        batch = Input(shape=self.batch_input_shape, batch_size=self.batch_size)
        series = TimeDistributed(m)(batch)
        memory = LSTM(128, dropout=keep_prob, stateful=True, batch_input_shape=self.batch_input_shape)(series)
        fc1 = Dense(32, activation=softmax)(memory)
        output = Dense(self.output_shape)(fc1)

        self.model = Model(inputs=batch, outputs=output, name=self.name)

        print("Build complete")

    def compile(self):
        print("Compiling")
        self.model.compile(optimizer='adam',
                           metrics=['mae', 'acc'],
                           loss=mean_squared_logarithmic_error)
        print("Compilation complete")

    def train(self):
        print("Training")
        self.model.fit_generator(generator=self.generator,
                                 steps_per_epoch=self.batch_size,
                                 epochs=self.epochs,
                                 callbacks=[self.testcallback])
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
        random_input = random.random((self.batch_size,) + self.batch_input_shape)
        prediction = self.predict(random_input)
        print("random_prediction: " + str(prediction[0]))


m = TestModel()
m.pipeline()
