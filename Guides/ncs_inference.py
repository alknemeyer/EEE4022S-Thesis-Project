# --- IMPORT LIBRARIES
import mvnc.mvncapi as mvnc
import numpy

# -- SET UP CONSTANTS
GRAPH_PATH = ''
IMAGES_PATH = ''
IMAGE_MEAN = ''

# --- OPEN DEVICE
# Look for enumerated Intel Movidius NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

# Get a handle to the first enumerated device and open it
device = mvnc.Device(devices[0])
device.OpenDevice()

# --- UPLOAD PRETRAINED MODEL
# Read the graph file into a buffer
with open(GRAPH_PATH, mode='rb') as f:
    blob = f.read()

# Load the graph buffer into the NCS
graph = device.AllocateGraph(blob)

# --- PREPROCESSING
# Read & resize image [Image size is defined during training]
img = print_img = skimage.io.imread(IMAGES_PATH)
img = skimage.transform.resize(img, IMAGE_DIM, preserve_range=True)

# Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
img = img[:, :, ::-1]

# Mean subtraction & scaling [A common technique used to center the data]
img = img.astype(numpy.float32)
img = (img — IMAGE_MEAN) * IMAGE_STDDEV

# --- LOAD IMAGE TO NCS
# Load the image as a half-precision floating point array
graph.LoadTensor(img.astype(numpy.float16), 'user object')

# --- GET THE RESULTS FROM NCS
output, userobj = graph.GetResult()

# --- PRINT THE RESULTS
print('\n — — — — predictions — — — — ')
labels = numpy.loadtxt(LABELS_FILE_PATH, str, delimiter='\t')
order = output.argsort()[::-1][:6]

for i in range(5):
    print('prediction ' + str(i) + ' is' + labels[order[i]])

# --- CLEAR AND SHUTDOWN NCS
graph.DeallocateGraph()
device.CloseDevice()
