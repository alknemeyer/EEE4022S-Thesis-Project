import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from pathlib import Path

m_dir = Path('model')
m_name = 'mobilenet_v1_1.0_224'

# ---

#with tf.Session(graph=tf.Graph()) as sess:
#  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], model_dir)

new_saver = tf.train.import_meta_graph(m_name + '.ckpt.meta')
new_saver.restore(sess, 'my_project')





# this works! but I get other errors
def printTensors(pb_file):
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    # print operations
    for op in graph.get_operations():
        print(op.name)

printTensors('saved_model.pb')








from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
ckpt_path = "model.ckpt"
print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=True, all_tensor_names=True)
