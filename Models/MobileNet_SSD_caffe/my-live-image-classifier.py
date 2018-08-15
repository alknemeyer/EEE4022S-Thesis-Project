#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Perform inference on a LIVE camera feed using DNNs on 
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import cv2
import sys
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform

import mvnc.mvncapi as mvnc

import picamera
from picamera.array import PiRGBArray

import time

network_type = 'SSD'

# Variable to store commandline arguments
ARGS = None

# PiCam objects
camera = None
rawCapture = None

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():
    print('Opening NCS device...')

    # Look for enumerated NCS device(s); quit program if none found.
    # devices = mvnc.EnumerateDevices()
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    # device.OpenDevice()
    device.open()

    print('Device opened succesfully')

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):
    print('Loading graph onto NCS device...')

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    # graph = device.AllocateGraph( blob )
    graph = mvnc.Graph( ARGS.graph )

    # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

    return graph, fifo_in, fifo_out

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image(frame):

    # Grab a frame from the camera
    height, width, channels = frame.shape

    # Extract/crop a section of the frame and resize it
    #x1 = int( width / 3 )
    #y1 = int( height / 4 )
    #x2 = int( width * 2 / 3 )
    #y2 = int( height * 3 / 4 )

    #cv2.rectangle( frame, ( x1, y1 ) , ( x2, y2 ), ( 0, 255, 0 ), 2 )
    #img = frame[ y1 : y2, x1 : x2 ]
    img = frame

    # Resize image [Image size if defined by choosen network, during training]
    img = cv2.resize( img, tuple( ARGS.dim ) )

    # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
    #if (ARGS.colormode == 'BGR'):
    #    img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    #img = img.astype( numpy.float16 )
    #img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale
    img = img - 127.5
    img = img * 0.007843

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image(graph, img, img_unprocessed, fifo_in, fifo_out, labels):

    # Load the image as a half-precision floating point array
    # graph.LoadTensor( img, 'user object' )
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img.astype(numpy.float32), None)

    # Get the results from NCS
    # output, userobj = graph.GetResult()
    output, userobj = fifo_out.read_elem()

    # print('output = ', output)
    # print('len(output) = ', len(output))
    # time.sleep(2)

    # Decode the output
    decode_ncs_ssd_output(output, labels, img_unprocessed)

    # Get execution time
    # inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )
    inference_time = graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)

#    print('I am %3.1f%%' % (100.0 * output[top_prediction] ) + ' confidant'
#            + ' you are ' + labels[top_prediction]
#            + ' ( %.2f ms )' % (numpy.sum(inference_time)))

    # If a display is available, show the image on which inference was performed
    # if 'DISPLAY' in os.environ:
    #     cv2.imshow('NCS live inference', frame)


def decode_ncs_ssd_output(output, labels, image_to_classify):
    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0
    # number of boxes returned
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))

    for box_index in range(num_valid_boxes):
        base_index = 7+ box_index * 7
        if (not numpy.isfinite(output[base_index]) or
                not numpy.isfinite(output[base_index + 1]) or
                not numpy.isfinite(output[base_index + 2]) or
                not numpy.isfinite(output[base_index + 3]) or
                not numpy.isfinite(output[base_index + 4]) or
                not numpy.isfinite(output[base_index + 5]) or
                not numpy.isfinite(output[base_index + 6])):
            # boxes with non infinite (inf, nan, etc) numbers must be ignored
            print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
            continue

        # clip the boxes to the image size incase network returns boxes outside of the image
        x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
        y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
        x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
        y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))

        x1_ = str(x1)
        y1_ = str(y1)
        x2_ = str(x2)
        y2_ = str(y2)

        print('box at index: ' + str(box_index) + ' : ClassID: ' + labels[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')

        # overlay boxes and labels on the original image to classify
        overlay_on_image(image_to_classify, output[base_index:base_index + 7], labels)

# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info, labels):

    # the minimal score for a box to be shown
    min_score_percent = 60

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = labels[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    if 'DISPLAY' in os.environ:
        cv2.imshow('NCS live inference', display_image)


# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device(device, graph, fifo_in, fifo_out):
    # graph.DeallocateGraph()
    # device.CloseDevice()
    # cam.release()
    # cv2.destroyAllWindows()
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()
    #cam.release()
    cv2.destroyAllWindows()

# ---- Main function (entry point for this script ) --------------------------

def main():

    device = open_ncs_device()
    graph, fifo_in, fifo_out = load_graph(device)
    
    # Load the labels file 
    labels = [line.rstrip('\n') for line in open( ARGS.labels )
                                if line not in ('classes\n', '\n')]
    print('labels = ', labels)

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        # grap the numpy array representing the image, then initialize the timestamp and occupied/unoccupied text
        img_unprocessed = frame.array
        # print('image[0,:,:] = ', image[0,:,:])
        # print('image shape =\n', image.shape)

        # print('Preprocessing image...')
        img = pre_process_image(img_unprocessed)
        # print('img.size = ', img.size)
        # print('img.shape = ', img.shape)

        # print('Inferring image...')
        infer_image(graph, img, img_unprocessed, fifo_in, fifo_out, labels)

        # Display the frame for 5ms, and close the window so that the next frame 
        # can be displayed. Close the window if 'q' or 'Q' is pressed.
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

    close_ncs_device(device, graph, fifo_in, fifo_out)

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description='Image classifier using \
                         Intel® Movidius™ Neural Compute Stick.' )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/SSD_MobileNet/graph',
                         help='Absolute path to the neural network graph file.' )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../caffe/SSD_MobileNet/categories.txt',
                         help='Absolute path to labels file.' )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[78.42633776, 87.76891437, 114.89584775],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help='Absolute path to labels file.' )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[227, 227],
                         help='Image dimensions. ex. -D 224 224' )

    parser.add_argument( '-c', '--colormode', type=str,
                         default='RGB',
                         help='RGB vs BGR color sequence. \
                               Defined during model training.' )


    ARGS = parser.parse_args()

    #initialize the camera and grap a referance to the raw camera capture
    camera = picamera.PiCamera()
    camera.resolution = (720, 576) #default resolution
    camera.vflip = True

    rawCapture = PiRGBArray(camera)


    main()

# ==== End of file ===========================================================
