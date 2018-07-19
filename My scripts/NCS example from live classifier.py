# source:
# https://github.com/movidius/ncappzoo/blob/ncsdk2/apps/live-image-classifier/live-image-classifier.py
# https://github.com/movidius/ncappzoo/tree/ncsdk2/apps

import mvnc.mvncapi as mvnc
import os
import cv2
import numpy

# Variable to store commandline arguments
ARGS = None

# OpenCV object for video capture
cam = None


def open_ncs_device():
    """ Step 1: Open the enumerated device and get a handle to it """

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print("No devices found")
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device(devices[0])
    device.open()

    return device


def load_graph(device):
    """ Step 2: Load a graph file onto the NCS device """

    # Read the graph file into a buffer
    with open(ARGS.graph, mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = mvnc.Graph(ARGS.graph)
    # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

    return graph, fifo_in, fifo_out


def pre_process_image():
    """ Step 3: Pre-process the images """

    # Grab a frame from the camera
    ret, frame = cam.read()
    height, width, channels = frame.shape

    # Extract/crop a section of the frame and resize it
    x1 = int(width / 3)
    y1 = int(height / 4)
    x2 = int(width * 2 / 3)
    y2 = int(height * 3 / 4)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    img = frame[y1:y2, x1:x2]

    # Resize image [Image size if defined by choosen network, during training]
    img = cv2.resize(img, tuple(ARGS.dim))

    # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
    if (ARGS.colormode == "BGR"):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = (img - ARGS.mean) * ARGS.scale

    return img, frame


def infer_image(graph, img, frame, fifo_in, fifo_out):
    """ Step 4: Read & print inference results from the NCS """

    # Load the labels file
    labels = [line.rstrip('\n') for line in open(ARGS.labels)
              if line != 'classes\n']

    # Load the image as a half-precision floating point array
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out,
                                         img.astype(numpy.float32), None)

    # Get the results from NCS
    output, userobj = fifo_out.read_elem()

    # Find the index of highest confidence
    top_prediction = output.argmax()

    # Get execution time
    inference_time = graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)

    print("I am %3.1f%%" % (100.0 * output[top_prediction]) + " confident"
          + " you are " + labels[top_prediction]
          + " ( %.2f ms )" % (numpy.sum(inference_time)))

    # If a display is available, show image on which inference was performed
    if 'DISPLAY' in os.environ:
        frame = cv2.flip(frame, 1)
        cv2.imshow('NCS live inference', frame)


def clean_up(device, graph, fifo_in, fifo_out):
    """ Step 5: Close/clean up fifos, graph, and device """
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()
    cam.release()
    cv2.destroyAllWindows()


def main():
    device = open_ncs_device()
    graph, fifo_in, fifo_out = load_graph(device)

    while(True):
        img, frame = pre_process_image()
        infer_image(graph, img, frame, fifo_in, fifo_out)

        # Display the frame for 5ms, and close the window so that the next
        # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    clean_up(device, graph, fifo_in, fifo_out)
