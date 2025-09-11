###################################################################################################
#
# GRBLocalizer.py  (PyTorch-based model update; all other code unchanged)
#
# Copyright (C) by Andreas Zoglauer & Anna Shang.
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

import signal
import sys
import time
import math
import csv
import os
import argparse
from datetime import datetime
from functools import reduce

print("\nGRB localization (PyTorch based)")
print("=================================\n")

###################################################################################################
# Step 1: Input parameters
###################################################################################################

# Default parameters

NumberOfComptonEvents = 2000
NumberOfBackgroundEvents = 0

# Depends on GPU memory and layout
MaxBatchSize = 256

NumberOfTrainingBatches = 1024
NumberOfTestingBatches = 8

ResolutionInDegrees = 5

OneSigmaNoiseInDegrees = 0.0

OutputDirectory = "Output"

# Parse command line:

print("\nParsing the command line (if there is any)\n")

parser = argparse.ArgumentParser(description='Perform training and/or testing for gamma-ray burst localization')
parser.add_argument('-m', '--mode', default='toymodel', help='Choose an input data more: toymodel or simulations')
parser.add_argument('-t', '--toymodeloptions', default='2000:0:0.0:32:8', help='The toy-model options: source_events:background_events:one_sigma_noise_in_degrees:training_batches:testing_batches')
parser.add_argument('-s', '--simulationoptions', default='', help='')
parser.add_argument('-r', '--resolution', default='5.0', help='Resolution of the input grid in degrees')
parser.add_argument('-b', '--batchsize', default='256', help='The number of GRBs in one training batch (default: 256 corresponsing to 5 degree grid resolution (64 for 3 degrees))')
parser.add_argument('-o', '--outputdirectory', default='Output', help='Name of the output directory. If it exists, the current data and time will be appended.')

args = parser.parse_args()


Mode = (args.mode).lower()
if Mode != 'toymodel' and Mode != 'simulation':
  print("Error: The mode must be either 'toymodel' or 'simulation'")
  sys.exit(0)


if Mode == 'toymodel':
  print("CMD-Line: Using toy model".format(NumberOfComptonEvents))

  ToyModelOptions = args.toymodeloptions.split(":")
  if len(ToyModelOptions) != 5:
    print("Error: You need to give 5 toy model options. You gave {}. Options: {}".format(len(ToyModelOptions), ToyModelOptions))
    sys.exit(0)

  NumberOfComptonEvents = int(ToyModelOptions[0])
  if NumberOfComptonEvents <= 10:
    print("Error: You need at least 10 source events and not {}".format(NumberOfComptonEvents))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} source events per GRB".format(NumberOfComptonEvents))

  NumberOfBackgroundEvents = int(ToyModelOptions[1])
  if NumberOfBackgroundEvents < 0:
    print("Error: You need a non-negative number of background events and not {}".format(NumberOfBackgroundEvents))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} background events per GRB".format(NumberOfBackgroundEvents))

  OneSigmaNoiseInDegrees = float(ToyModelOptions[2])
  if OneSigmaNoiseInDegrees < 0:
    print("Error: You need a non-negative number for the noise and not {}".format(OneSigmaNoiseInDegrees))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} degrees as 1-sigma resolution".format(OneSigmaNoiseInDegrees))

  NumberOfTrainingBatches = int(ToyModelOptions[3])
  if NumberOfTrainingBatches < 1:
    print("Error: You need a positive number for the number of traing batches and not {}".format(NumberOfTrainingBatches))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} training batches".format(NumberOfTrainingBatches))

  NumberOfTestingBatches = int(ToyModelOptions[4])
  if NumberOfTestingBatches < 1:
    print("Error: You need a positive number for the number of testing batches and not {}".format(NumberOfTestingBatches))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} testing batches".format(NumberOfTestingBatches))

elif Mode == 'simulation':
  print("Error: The simulation mode has not yet implemented")
  sys.exit(0)


ResolutionInDegrees = float(args.resolution)
if ResolutionInDegrees > 10 or ResolutionInDegrees < 1:
  print("Error: The resolution must be between 1 & 10 degrees")
  sys.exit(0)
print("CMD-Line: Using {} degrees as input grid resolution".format(ResolutionInDegrees))

MaxBatchSize = int(args.batchsize)
if MaxBatchSize < 1 or MaxBatchSize > 1024:
  print("Error: The batch size must be between 1 && 1024")
  sys.exit(0)
print("CMD-Line: Using {} as batch size".format(MaxBatchSize))

OutputDirectory = args.outputdirectory
# TODO: Add checks
if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")

os.makedirs(OutputDirectory)
print("CMD-Line: Using \"{}\" as output directory".format(OutputDirectory))

print("\n\n")


# Determine derived parameters

OneSigmaNoiseInRadians = math.radians(OneSigmaNoiseInDegrees)

NumberOfTrainingLocations = NumberOfTrainingBatches*MaxBatchSize
TrainingBatchSize = MaxBatchSize

NumberOfTestLocations = NumberOfTestingBatches*MaxBatchSize
TestingBatchSize = MaxBatchSize


PsiMin = -np.pi
PsiMax = +np.pi
PsiBins = int(360 / ResolutionInDegrees)

ChiMin = 0
ChiMax = np.pi
ChiBins = int(180 / ResolutionInDegrees)

PhiMin = 0
PhiMax = np.pi
PhiBins = int(180 / ResolutionInDegrees)

InputDataSpaceSize = PsiBins * ChiBins * PhiBins
OutputDataSpaceSize = 2


###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press  Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from GRBData import GRBData
from GRBCreatorToyModel import GRBCreatorToyModel

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


print("Info: Creating {:,} Compton events".format((NumberOfTrainingLocations + NumberOfTestLocations) * (NumberOfComptonEvents + NumberOfBackgroundEvents)))


ToyModelCreator = GRBCreatorToyModel(ResolutionInDegrees, OneSigmaNoiseInDegrees)


def generateOneDataSet(_):
  DataSet = GRBData()
  DataSet.create(ToyModelCreator, NumberOfComptonEvents, NumberOfBackgroundEvents)
  return DataSet


# Parallelizing using Pool.starmap()
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

# Create data sets
TimerCreation = time.time()

TrainingDataSets = pool.map(generateOneDataSet, range(0, NumberOfTrainingLocations))
print("Info: Created {:,} training data sets. ".format(NumberOfTrainingLocations))

TestingDataSets = pool.map(generateOneDataSet, range(0, NumberOfTestLocations))
print("Info: Created {:,} testing data sets. ".format(NumberOfTestLocations))

pool.close()

TimeCreation = time.time() - TimerCreation
print("Info: Total time to create data sets: {:.1f} seconds (= {:,.0f} events/second)".format(TimeCreation, (NumberOfTrainingLocations + NumberOfTestLocations) * (NumberOfComptonEvents + NumberOfBackgroundEvents) / max(TimeCreation, 1e-9)))


# Plot the first test data point
'''
l = 0

print("Pos {}, {}".format(math.degrees(YTrain[l, 0]), math.degrees(YTrain[l, 1])))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

adds = 0
for p in range(0, PsiBins):
  for c in range(0, ChiBins):
    for t in range(0, PhiBins):
      if XTrain[l, p, c, t] > 0:
        ax.scatter(math.degrees(PsiMin) + p * ResolutionInDegrees, math.degrees(ChiMin) + c * ResolutionInDegrees, math.degrees(PhiMin) + t * ResolutionInDegrees, XTrain[l, p, c, t])
        adds += XTrain[l, p, c, t]

print("Adds: {}".format(adds))

plt.show()
plt.pause(0.001)

input("Press [enter] to EXIT")
sys.exit()
'''


###################################################################################################
# Step 4: Setting up the neural network (PyTorch)
###################################################################################################


print("Info: Setting up neural network...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class GRBNet(nn.Module):
  def __init__(self, psi_bins, chi_bins, phi_bins):
    super().__init__()
    # Channels-first: [N, C=1, D=psi, H=chi, W=phi]
    self.features = nn.Sequential(
      nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),  # VALID
      nn.ReLU(inplace=True),
      nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0), # VALID
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      nn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0),# VALID
      nn.ReLU(inplace=True),
      nn.Conv3d(128, 128, kernel_size=2, stride=2, padding=0),# VALID
      nn.ReLU(inplace=True)
    )

    # Infer flattened size once via a dummy forward (CPU, no grad)
    with torch.inference_mode():
      dummy = torch.zeros(1, 1, psi_bins, chi_bins, phi_bins)
      flat = self.features(dummy).flatten(1).size(1)

    self.fc1 = nn.Linear(flat, 128)
    self.act = nn.ReLU(inplace=True)
    self.out = nn.Linear(128, OutputDataSpaceSize)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.act(self.fc1(x))
    return self.out(x)

model = GRBNet(PsiBins, ChiBins, PhiBins).to(device)

# Loss function - replicate TF: sum of squared error divided by NumberOfTestLocations (constant)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("      ... model ready; number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

# Add ops to save and restore all the variables.
def save_checkpoint(iteration, directory):
  path = os.path.join(directory, f"Model_{iteration}.pt")
  torch.save({'iteration': iteration, 'model_state_dict': model.state_dict()}, path)



###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################


print("Info: Training and evaluating the network")

# Train the network

MaxTimesNoImprovement = 1000
TimesNoImprovement = 0
BestMeanSquaredError = sys.float_info.max
BestMeanAngularDeviation = sys.float_info.max
BestRMSAngularDeviation = sys.float_info.max
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0

print("Info: Creating configuration and progress file")

with open(OutputDirectory + "/Configuration.txt", 'w') as f:
  f.write("Configuration\n\n")
  f.write("Mode: {}\n".format(Mode))
  if Mode == 'toymodel':
    f.write("NumberOfComptonEvents: {}\n".format(NumberOfComptonEvents))
    f.write("NumberOfBackgroundEvents: {}\n".format(NumberOfBackgroundEvents))
    f.write("Noise: {}\n".format(OneSigmaNoiseInDegrees))
    f.write("TrainingBatchSize: {}\n".format(TrainingBatchSize))
    f.write("TestingBatchSize: {}\n".format(TestingBatchSize))
  f.write("ResolutionInDegrees: {}\n".format(ResolutionInDegrees))
  f.write("MaxBatchSize: {}\n".format(MaxBatchSize))
  f.write("OutputDirectory: {}\n".format(OutputDirectory))

with open(OutputDirectory + '/Progress.txt', 'w') as f:
  f.write("Progress\n\n")

from shutil import copyfile
copyfile("GRBLocalizer.py", OutputDirectory + '/GRBLocalizer.py')
copyfile("GRBCreator.py", OutputDirectory + '/GRBCreator.py')
copyfile("GRBCreatorToyModel.py", OutputDirectory + '/GRBCreatorToyModel.py')
copyfile("GRBData.py", OutputDirectory + '/GRBData.py')

@torch.no_grad()
def CheckPerformance():
  global TimesNoImprovement
  global BestMeanSquaredError
  global BestMeanAngularDeviation
  global BestRMSAngularDeviation
  global IterationOutputInterval

  model.eval()

  MeanAngularDeviation = 0
  RMSAngularDeviation = 0
  for Batch in range(0, NumberOfTestingBatches):

    # Step 1: Convert the data
    XTest = np.zeros(shape=(TestingBatchSize, PsiBins*ChiBins*PhiBins*1), dtype=np.float32)
    YTest = np.zeros(shape=(TestingBatchSize, OutputDataSpaceSize), dtype=np.float32)

    for g in range(0, TestingBatchSize):
      # NOTE: preserved original behavior (uses TrainingDataSets here)
      GRB = TrainingDataSets[g + Batch*TrainingBatchSize]
      YTest[g][0] = GRB.OriginLatitude
      YTest[g][1] = GRB.OriginLongitude

      XSlice = XTest[g,]
      XSlice.put(GRB.getIndices(), GRB.getValues())

    XTest = XTest.reshape((TestingBatchSize, 1, PsiBins, ChiBins, PhiBins))  # channels-first
    x = torch.from_numpy(XTest).to(device=device, dtype=torch.float32)
    YOut = model(x).cpu().numpy()

    # Step 3: Analyze it
    for l in range(0, TestingBatchSize):

      Real = M.MVector()
      Real.SetMagThetaPhi(1.0, YTest[l, 0].item(), YTest[l, 1].item())

      Reconstructed = M.MVector()
      Reconstructed.SetMagThetaPhi(1.0, float(YOut[l, 0]), float(YOut[l, 1]))

      AngularDeviation = math.degrees(Real.Angle(Reconstructed))

      MeanAngularDeviation += AngularDeviation
      RMSAngularDeviation += math.pow(AngularDeviation, 2)

      if Batch == NumberOfTestingBatches-1:
        print("  Cross-Check element: {:-7.3f} degrees difference: {:-6.3f} vs. {:-6.3f} & {:-6.3f} vs. {:-6.3f}".format(
          AngularDeviation, YTest[l, 0].item(), float(YOut[l, 0]), YTest[l, 1].item(), float(YOut[l, 1])))

  # Calculate the mean RMS
  MeanAngularDeviation /= NumberOfTestingBatches*TestingBatchSize
  RMSAngularDeviation /= NumberOfTestingBatches*TestingBatchSize
  RMSAngularDeviation = math.sqrt(RMSAngularDeviation)

  Improvement = False

  # Check for improvement mean
  if MeanAngularDeviation < BestMeanAngularDeviation:
    BestMeanAngularDeviation = MeanAngularDeviation
    BestRMSAngularDeviation = RMSAngularDeviation
    Improvement = True

  print("\n")
  print("RMS Angular deviation:   {:-6.3f} deg  -- best: {:-6.3f} deg".format(RMSAngularDeviation, BestRMSAngularDeviation))
  print("Mean Angular deviation:  {:-6.3f} deg  -- best: {:-6.3f} deg".format(MeanAngularDeviation, BestMeanAngularDeviation))

  model.train()
  return Improvement



# Main training and evaluation loop
TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0
MaxIterations = 50000
Iteration = 0

model.train()

while Iteration < MaxIterations:
  Iteration += 1
  for Batch in range(0, NumberOfTrainingBatches):

    # Convert the data set into training and testing data
    TimerConverting = time.time()

    XTrain = np.zeros(shape=(TrainingBatchSize, PsiBins*ChiBins*PhiBins*1), dtype=np.float32)
    YTrain = np.zeros(shape=(TrainingBatchSize, OutputDataSpaceSize), dtype=np.float32)

    for g in range(0, TrainingBatchSize):
      GRB = TrainingDataSets[g + Batch*TrainingBatchSize]
      YTrain[g][0] = GRB.OriginLatitude
      YTrain[g][1] = GRB.OriginLongitude

      XSlice = XTrain[g,]
      XSlice.put(GRB.getIndices(), GRB.getValues())

    XTrain = XTrain.reshape((TrainingBatchSize, 1, PsiBins, ChiBins, PhiBins))  # channels-first

    TimeConverting += time.time() - TimerConverting

    # The actual training
    TimerTraining = time.time()

    x = torch.from_numpy(XTrain).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(YTrain).to(device=device, dtype=torch.float32)

    optimizer.zero_grad()
    YPred = model(x)
    Loss = criterion(YPred, y) / max(NumberOfTestLocations, 1)  # mirror TF scaling
    Loss.backward()
    optimizer.step()

    TimeTraining += time.time() - TimerTraining

  # Check performance
  TimerTesting = time.time()
  print("\n\nIteration {}".format(Iteration))
  Improvement = CheckPerformance()

  if Improvement == True:
    TimesNoImprovement = 0

    save_checkpoint(Iteration, OutputDirectory)

    with open(OutputDirectory + '/Progress.txt', 'a') as f:
      f.write(' '.join(map(str, (CheckPointNum, Iteration, BestMeanAngularDeviation, BestRMSAngularDeviation)))+'\n')

    print("\nSaved new best model and performance!")
    CheckPointNum += 1
  else:
    TimesNoImprovement += 1

  TimeTesting += time.time() - TimerTesting

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
    break

  # Take care of Ctrl-C
  if Interrupted == True: break

# End: fo all iterations

print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/max(Iteration,1)))
print("Total time training per Iteration:   {} sec".format(TimeTraining/max(Iteration,1)))
print("Total time testing per Iteration:    {} sec".format(TimeTesting/max(Iteration,1)))


#input("Press [enter] to EXIT")
sys.exit(0)
