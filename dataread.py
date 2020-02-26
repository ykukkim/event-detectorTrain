import sys
import re
import os
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.signal import argrelextrema
from numpy import matlib as mb

sys.path.append("/Users/YKK/Documents/GitHub/LMBTrain/btkmac")
import btk

# Data conversion from .c3d to csv
# Directory settings
Input_Dir = "/Users/YKK/Desktop/untitled folder/untitled folder/"
Output_Dir = "/Users/YKK/Desktop/untitled folder/untitled folder/csv"


def derivative(traj, nFrames):
    traj_der = traj[1:nFrames, :] - traj[0:(nFrames - 1), :]
    return np.append(traj_der, [[0, 0, 0]], axis=0)


def data_filter(acq,traj, nFrames):
    # b, a = signal.butter(4,1,'lowpass',acq.GetPointFrequency())
    b, a = signal.butter(4, 8/(acq.GetPointFrequency()/2))
    Mean = np.mean(traj, axis=0)
    Minput = traj - mb.repmat(Mean, nFrames, 1)
    Minput = signal.filtfilt(b, a, Minput, axis=0)
    Moutput = Minput + np.matlib.repmat(Mean, nFrames, 1)
    return Moutput


def compute(leg, Filename_In):
    m = re.match(Input_Dir + "(?P<name>.+).c3d", Filename_In)
    name = m.group('name').replace(" ", "-")
    output_file = "%s/%s%s.csv" % (Output_Dir, leg, name)
    print("Trying %s" % (Filename_In))

    # Read files in .c3d and read data
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(Filename_In)
    reader.Update()
    acq = reader.GetOutput()

    nFrames = acq.GetPointFrameNumber()
    first_frame = acq.GetFirstFrame()

    # Heel, Ankle, Hallux, Toe
    # 2 * 4 * 3 = 24 Marker Trajectories
    # 2 * 4 * 3 = 24 Velocity
    # 2 * 4 * 3 = 24 Acceleration
    markers = ["ANK", "TOE", "HLX", "HEE"]
    opposite = {'L': 'R', 'R': 'L'}

    # Check if there are any marker data in the file
    brk = True
    for point in btk.Iterate(acq.GetPoints()):
        if point.GetLabel() == "L" + markers[0]:
            brk = False
            break

    if brk:
        print("No Datain %s!" % (Filename_In,))
        return

    # Marker Data extraction and filtering.
    traj = [None] * (len(markers) * 2)
    for i, v in enumerate(markers):
        try:
            traj[i] = acq.GetPoint(leg + v).GetValues()
            traj[len(markers) + i] = acq.GetPoint(opposite[leg] + v).GetValues()
        except:
            return

        traj[i][:, 0] = traj[i][:, 0]  # * incrementX
        traj[len(markers) + i][:, 0] = traj[len(markers) + i][:, 0]  # * incrementX
        traj[i][:, 2] = traj[i][:, 2]  # * incrementX
        traj[len(markers) + i][:, 2] = traj[len(markers) + i][:, 2]  # * incrementXq

    # filter then getting position, velocity and accleration
    filtered_traj = [data_filter(acq,ax, nFrames) for ax in traj]
    vel = [derivative(bx, nFrames) for bx in filtered_traj]
    acc = [derivative(cx, nFrames) for cx in vel]

    curves = np.concatenate(filtered_traj + vel + acc, axis = 1)

    # Add events as output using annotation on data
    outputs = np.array([[0] * nFrames, [0] * nFrames]).T
    for event in btk.Iterate(acq.GetEvents()):
        if event.GetFrame() >= nFrames:
            print("Event happened too far")
            return
        if len(event.GetContext()) == 0:
            print("No events found")
            return
        if event.GetContext()[0] == leg:
            if event.GetLabel() == "Foot Strike":
                outputs[event.GetFrame() - first_frame, 0] = 1
            elif event.GetLabel() == "Foot Off":
                outputs[event.GetFrame() - first_frame, 1] = 1

    if (np.sum(outputs) == 0):
        print("No events in %s!" % (Filename_In,))
        return

    arr = np.concatenate((curves, outputs), axis=1)

    print("Writing %s" % Filename_In)
    np.savetxt(output_file, arr, delimiter=',')


# files = os.listdir(Input_Dir)
for Filename_In in os.listdir(Input_Dir):
    if not Filename_In.startswith('.') and os.path.isfile(os.path.join(Input_Dir, Filename_In)):
        for leg in ['L', 'R']:
            compute(leg, Input_Dir + Filename_In)
            print(Filename_In)

