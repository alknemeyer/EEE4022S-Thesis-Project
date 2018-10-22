# EEE4022S Thesis Project: Autonomous Object Identification and Tracking using Neural Networks

Git repository to backup work for my fourth year thesis project at the University of Cape Town. The repo is mostly just to back up my work across multiple work stations, so a lot of the commit descriptions are horrible. Woops.

## Abstract
Many systems require a camera to autonomously identify the objects in its field of view, and then rotate the camera to keep it aimed at a certain class of object.

This report describes the design and implementation of such a system on a low-powered
device. The system uses a Convolutional Neural Network to identify the position of
objects from photos, making use of a neural accelerator device to achieve near real-time
inferences. These measurements are improved through the use of a Kalman Filter, which
estimates the angular state of the tracked object. A multiprocess pipeline is used to
manage the control on a non-real time OS. Finally, a gimbal is designed and built to
rotate the unique payload.

The tracking system is shown to improve over a stationary camera setup.

## Coding style:

I find development using a notebook to be quite a bit easier than developing using a regular python file. Unfortunately, you can't import a `.ipynb` as a module. So, here's the workflow:
1. Use each `.ipynb` file to understand the code and make changes.
2. When you want to commit a change, click `Kernal > Restart and Clear Output` to remove your outputs + make the file a bit smaller (shows up as fewer lines in the git commit).
3. Run the command `jupyter nbconvert --to=python FILENAME.ipynb` to generate a `.py` file which can be imported as a module. Just make sure that any debugging code doesn't run if this is imported as a module into another file!



Alexander Knemeyer
