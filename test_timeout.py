from __future__ import print_function, division
import sys
sys.path.append('core')
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from raft import RAFT
import evaluate
import datasets
from torch.cuda.amp import GradScaler
import tqdm as tq

if __name__ == '__main__':
    print("Done")