
from re import X
import torch
import sys
import os
import onnxruntime as ort
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

sys.path.append('../')
from models import Generator

def get_input():
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   transforms.Resize((256, 256))]

    img = cv2.imread('../input/test/A/0.jpg').astype(np.float32)
    img = transforms.Compose(transforms_)(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def convert():
    model = Generator(3, 3)
    torch_model = torch.load(sys.argv[1], map_location='cpu')
    model.load_state_dict(torch_model)
    model.eval()

    input_names = ['input']
    output_names = ['output']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 512, 512, 3) #???not working?

    dummy_input = get_input()
    dummy_input = torch.tensor(dummy_input)


    torch.onnx.export(model, dummy_input, '../mymodels/model.onnx', opset_version=10, input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': [0], 'output': [0]})

if __name__ == '__main__':
    convert()

#python convert_to_onnx.py ../mymodels/cartoon/netG_A2B.pth