import argparse
import sys
import os
import time
import numpy as np
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='input/test/A/sy2.jpg', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=384, help='size of the data (squared assumed)')

parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='mymodels/cartoon/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='mymodels/cartoon/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)


###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

netG_A2B.cuda()
netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor 
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                transforms.Resize((opt.size,opt.size)) 
                ]
img = cv2.imread(opt.dataroot).astype(np.float32)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
shape = img.shape
img = img / 255.0
img = transforms.Compose(transforms_)(img)
img = np.expand_dims(img, axis=0)
img = Tensor(img)

st = time.time()
fake_B = 0.5*(netG_A2B(img).data + 1.0)
print ('cost..',time.time()-st)
fake_A = 0.5*(netG_B2A(img).data + 1.0)


save_image(fake_A, 'output/a.png')
save_image(fake_B, 'output/bb.png')

###################################
# python test.py --cuda --generator_A2B mymodels/cartoon/netG_A2B.pth --generator_B2A mymodels/cartoon/netG_B2A.pth --dataroot input