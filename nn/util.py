from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import Graph
from tensorflow.contrib.keras.api.keras.utils import Sequence
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


def graph_def_from_pb(model_pb_file_path) -> GraphDef:
    from tensorflow.python.platform.gfile import Open
    graph_def: GraphDef = GraphDef()
    with Open(model_pb_file_path, "rb") as f:
        data = f.read()
        graph_def.ParseFromString(data)
    return graph_def


def graph_from_pb(model_pb_file_path) -> Graph:
    from tensorflow.python.framework.importer import import_graph_def
    graph_def = graph_def_from_pb(model_pb_file_path)
    graph: Graph = Graph()
    with graph.as_default():
        import_graph_def(graph_def, name='')
    return graph


def session_from_pb(model_pb_file_path) -> Session:
    from tensorflow.core.protobuf.config_pb2 import ConfigProto, GPUOptions
    graph = graph_from_pb(model_pb_file_path)
    config = ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True,
        gpu_options=GPUOptions(per_process_gpu_memory_fraction=1)
    )
    return Session(graph=graph, config=config)


def predict(sess: Session, input_op_name, output_op_name, x):
    input_op = sess.graph.get_operation_by_name(input_op_name)
    input_tensor = input_op.outputs[0]
    output_op = sess.graph.get_operation_by_name(output_op_name)
    output_tensor = output_op.outputs[0]
    return sess.run(output_tensor, {input_tensor: x})


def optimize(model_pb, out_pb, input_node_names, output_node_names):
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
    from tensorflow.python.platform.gfile import FastGFile
    from tensorflow.tools.graph_transforms import TransformGraph

    print("Parsing GraphDef from " + model_pb)
    graph_def_frozen = graph_def_from_pb(model_pb)

    transforms = [
        'add_default_attributes',
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
                                      inputs=input_node_names,
                                      outputs=output_node_names,
                                      transforms=transforms)

    print("Writing " + out_pb)
    with FastGFile(out_pb, "w") as f:
        f.write(output_graph_def.SerializeToString())
    print("Finished optimization")
