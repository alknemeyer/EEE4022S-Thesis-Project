import argparse
import cv2
from pathlib import Path


def extractImages(path_in, path_out, t):
    count = 0
    vidcap = cv2.VideoCapture(path_in)
    success, image = vidcap.read()

    while success is True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, 1000*count*t)
        success, image = vidcap.read()
        print('Read a new frame: ', success)

        if success:
            cv2.imwrite(path_out/'frame%d.png' % count, image)

        count += 1


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('-i', '--input_video', help='path to input video')
    a.add_argument('-o', '--output_dir', help='path to output images')
    a.add_argument('-t', '--time_between_frames',
                   help='time in seconds between frames', default=1)

    args = a.parse_args()
    extractImages(Path(args.input_video),
                  Path(args.output_dir),
                  a.time_between_frames)
