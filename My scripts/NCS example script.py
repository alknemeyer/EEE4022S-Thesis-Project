# https://movidius.github.io/ncsdk/ncapi/ncapi2/py_api/readme.html

from mvnc import mvncapi
import cv2
import numpy

# Get a list of available device identifiers
device_list = mvncapi.enumerate_devices()

# Initialize a Device
device = mvncapi.Device(device_list[0])

# Initialize the device and open communication
device.open()

# Load graph file data
GRAPH_FILEPATH = './graph'
with open(GRAPH_FILEPATH, mode='rb') as f:
    graph_buffer = f.read()

# Initialize a Graph object
graph = mvncapi.Graph('graph1')

# Allocate the graph to the device and create input and output Fifos with
# default arguments
input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_file_buffer)

# Allocate the graph to the device and create input and output Fifos with
# keyword arguments (default values shown)
input_fifo, output_fifo = graph.allocate_with_fifos(
    device, graph_file_buffer,
    input_fifo_type=mvncapi.FifoType.HOST_WO,
    input_fifo_data_type=mvncapi.FifoDataType.FP32,
    input_fifo_num_elem=2,
    output_fifo_type=mvncapi.FifoType.HOST_RO,
    output_fifo_data_type=mvncapi.FifoDataType.FP32,
    output_fifo_num_elem=2)

# Read an image from file
tensor = cv2.imread('img.jpg')
# Do pre-processing specific to this network model (resizing, subtracting
# network means, etc.)

# Convert an input tensor to 32FP data type
# Tensor data should be stored in a numpy ndarray.
tensor = tensor.astype(numpy.float32)

# Write the tensor to the input_fifo and queue an inference
graph.queue_inference_with_fifo_elem(
    input_fifo, output_fifo, tensor, 'user object')

# Get the results from the output queue
output, user_obj = output_fifo.read_elem()

# Clean up
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()
