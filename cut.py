from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from beginner_source.dcgan_faces_tutorial import (
    dataset,
    dataloader,
    device,
    real_batch,
    fixed_noise,
    netG,
    input_to_export,
)

PATH = "DCGAN-trained-16x16-full"
netG.load_state_dict(torch.load(PATH + '.pickle'))
layers = list(list(netG.children())[0].children())
previous_layer_output = input_to_export
for i, layer in enumerate(layers):
    torch.onnx.export(layer, previous_layer_output, f"{PATH}-{i}.onnx")
    previous_layer_output = layer(previous_layer_output)
