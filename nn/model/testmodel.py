from tensorflow.python.client.session import Session
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Dense, Flatten, Dropout, TimeDistributed, LSTM, Input, ELU, SimpleRNN, add
from tensorflow.contrib.keras.api.keras.activations import relu, softmax
from tensorflow.contrib.keras.api.keras.losses import mean_squared_logarithmic_error
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.python.framework.ops import name_scope
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
    def __init__(self, name='model3', save_path=None):
        print(name)
        self.name = name
        self.model = Sequential(name=self.name)
        self.epochs = 1
        self.time_step = 8
        self.batch_size = 16
        self.input_node_names = ['memory/images:0']
        self.output_op_names = ['output/scores/BiasAdd']
        self.output_node_names = ['output/scores/BiasAdd:0']
        self.input_shape = (500, 20, 1)
        self.batch_input_shape = (self.time_step,) + self.input_shape
        self.shape = (self.batch_size, self.time_step) + self.input_shape
        self.output_shape = 4
        self.generator = RandomSequence(batch_size=self.batch_size, x_shape=self.batch_input_shape, y_shape=(self.output_shape,))

        if save_path is None:
            self.save_path = tempfile.mkdtemp()
        else:
            self.save_path = save_path

        self.log_dir = os.path.join(self.save_path, "logs_" + str(int(time.time()))[-6:])

        # SavedModelBuilder.save() will fail if directory exists
        self.tf_serving_dir = os.path.join(self.save_path, "tfs_model_" + str(int(time.time()))[-6:])

        for p in [self.log_dir]:
            if not os.path.exists(p):
                os.makedirs(p)
                print("Created " + p)

        self.tf_frozen_model = os.path.join(self.tf_serving_dir, self.name + "_frozen.pb")
        self.tf_inference_model = os.path.join(self.tf_serving_dir, self.name + "_inference.pb")
        self.tf_checkpoint = os.path.join(self.save_path, self.name + ".ckpt")
        self.testcallback = TestCallback(log_dir=self.log_dir)

    def build_model(self):
        print("Building")

        keep_prob = 0.288

        # From https://navoshta.com/end-to-end-deep-learning/
        # https://keras.io/layers/convolutional/#conv2d
        # https://keras.io/layers/core/#Dense
        with name_scope('vision'):
            m = Sequential()
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
        with name_scope('memory'):
            batch = Input(shape=self.batch_input_shape, batch_size=self.batch_size, name='images')
            series = TimeDistributed(m)(batch)
            memory = LSTM(128, dropout=keep_prob, stateful=True, batch_input_shape=self.batch_input_shape)(series)

        with name_scope('output'):
            fc1 = Dense(32, activation=softmax)(memory)
            output = Dense(self.output_shape, name='scores')(fc1)

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
        self.save_keras()
        self.export_saved_model()
        self.freeze()
        self.optimize()

    def save_keras(self):
        print("Saving keras model")
        json_path = os.path.join(self.save_path, self.name + '.json')
        with open(json_path, "w") as f:
            f.write(self.model.to_json())
            print("Wrote " + json_path)
        h5_path = os.path.join(self.save_path, self.name + '.h5')
        self.model.save_weights(filepath=h5_path)
        print("Wrote " + h5_path)
        print("Finished saving keras model")

    def export_saved_model(self):
        print("Writing GraphDef for Tensorflow-Serving")
        # Save tensorflow-serving model
        # https://github.com/tensorflow/serving/issues/310
        from tensorflow.python.saved_model.builder import SavedModelBuilder
        from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
        from tensorflow.python.saved_model import tag_constants

        builder = SavedModelBuilder(export_dir=self.tf_serving_dir)
        signature = predict_signature_def(inputs={'images': self.model.input},
                                          outputs={'scores': self.model.output})
        if os.path.exists(self.tf_serving_dir):
            print("Created " + self.tf_serving_dir)
        builder.add_meta_graph_and_variables(sess=get_session(),
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()
        print("Wrote " + os.path.join(self.tf_serving_dir, "saved_model.pb"))

    def checkpoint(self):
        print("Writing checkpoint")
        from tensorflow.python.training.saver import Saver
        from tensorflow.python.ops.variables import global_variables, global_variables_initializer
        get_session().run(global_variables_initializer())
        Saver(global_variables()).save(sess=get_session(),
                                       save_path=self.tf_checkpoint,
                                       write_meta_graph=False)
        print("Wrote " + self.tf_checkpoint)

    def freeze(self):
        print("Freezing graph")
        # Inlined from tensorflow.python.tools.freeze_graph#freeze_graph
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.tools.saved_model_utils import get_meta_graph_def
        from tensorflow.python.framework.importer import import_graph_def
        from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants
        from tensorflow.python.saved_model.loader_impl import load, _parse_saved_model
        from tensorflow.python.platform.gfile import GFile
        from google.protobuf.message import Message
        from tensorflow.core.protobuf.saved_model_pb2 import SavedModel

        graph_def = get_meta_graph_def(saved_model_dir=self.tf_serving_dir,
                                       tag_set=tag_constants.SERVING).graph_def

        ops = import_graph_def(graph_def=graph_def,
                               name="")

        print("Parsing saved model from " + self.tf_serving_dir)
        saved_model: SavedModel = _parse_saved_model(self.tf_serving_dir)
        for meta_graph_def in saved_model.meta_graphs:
            print("MetaGraphDef tags: " + str(meta_graph_def.meta_info_def.tags))

        meta_graph_def = load(sess=get_session(),
                              tags=[tag_constants.SERVING],
                              export_dir=self.tf_serving_dir)

        graph_def_frozen: Message = convert_variables_to_constants(sess=get_session(),
                                                                   input_graph_def=graph_def,
                                                                   output_node_names=self.output_op_names,
                                                                   variable_names_whitelist=None,
                                                                   variable_names_blacklist=None)

        with GFile(self.tf_frozen_model, "wb") as f:
            f.write(graph_def_frozen.SerializeToString())

        # full invocation
        # from tensorflow.python.tools.freeze_graph import freeze_graph
        # from tensorflow.core.protobuf import saver_pb2
        # freeze_graph(input_graph="",
        #              input_saver="",
        #              input_binary=False,
        #              input_checkpoint="",
        #              output_node_names="frozen",
        #              restore_op_name="",
        #              filename_tensor_name="",
        #              output_graph=self.tf_frozen_dir,
        #              clear_devices=True,
        #              initializer_nodes="",
        #              variable_names_whitelist="",
        #              variable_names_blacklist="",
        #              input_meta_graph=None,
        #              input_saved_model_dir=self.tf_serving_dir,
        #              saved_model_tags=tag_constants.SERVING,
        #              checkpoint_version=saver_pb2.SaverDef.V2)
        print("Finished writing frozen graph")

    def optimize(self):
        """
        If you've finished training your model and want to deploy it on a server or a mobile device, you'll want it to
        run as fast as possible, and with as few non-essential dependencies as you can.

        This recipe removes all of the nodes that aren't called during inference,
        shrinks expressions that are always constant into single nodes,
        and optimizes away some multiply operations used during batch normalization
        by pre-multiplying the weights for convolutions.
        https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#optimizing-for-deployment
        https://github.com/tensorflow/tensorflow/issues/9914
        """
        from tensorflow.python.platform.gfile import Open, FastGFile
        from tensorflow.core.framework.graph_pb2 import GraphDef
        from tensorflow.tools.graph_transforms import TransformGraph

        print("Parsing GraphDef from " + self.tf_frozen_model)
        graph_def_frozen = GraphDef()
        with Open(self.tf_frozen_model, "rb") as f:
            data = f.read()
            graph_def_frozen.ParseFromString(data)

        shape_csv = ','.join(map(str, list(self.shape)))
        transforms = [
            'add_default_attributes',
            'strip_unused_nodes(type=float, shape="' + shape_csv + '")',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms',
            'strip_unused_nodes',
            'sort_by_execution_order',
            'obfuscate_names'
        ]

        print("Optimizing for inference with TransformGraph")
        output_graph_def = TransformGraph(input_graph_def=graph_def_frozen,
                                          inputs=self.input_node_names,
                                          outputs=self.output_node_names,
                                          transforms=transforms)

        print("Writing " + self.tf_inference_model)
        with FastGFile(self.tf_inference_model, "w") as f:
            f.write(output_graph_def.SerializeToString())
        print("Finished optimization")

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
        #random_input = random.random((self.batch_size,) + self.batch_input_shape)
        #prediction = self.predict(random_input)
        #print("random_prediction: " + str(prediction[0]))


if __name__ == "__main__":
    m = TestModel()
    m.pipeline()
