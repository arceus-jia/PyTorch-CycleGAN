
from re import X
import torch
import sys
import os
import onnxruntime as ort
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time

def get_input():
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   transforms.Resize((320, 320))]

    img = cv2.imread(sys.argv[2]).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = img.shape
    img = img / 255.0
    img = transforms.Compose(transforms_)(img)
    img = np.expand_dims(img, axis=0)
    return img,shape


def test():
    onnx = os.path.join(sys.argv[1])
    session = ort.InferenceSession(
        onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    img,shape = get_input()

    st = time.time()
    fake_img = session.run(None, {x: img})
    print ('time cost:',time.time()-st)

    fake_img = np.array(fake_img)
    fake_img = np.squeeze(fake_img)
    fake_img = fake_img.transpose(1, 2, 0)
    fake_img = 0.5 * (fake_img + 1) * 255

    fake_img = cv2.resize(fake_img, (shape[1],shape[0]))
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite('../output/result1.jpg', fake_img)


def test2():
    onnx = os.path.join(sys.argv[1])
    session = ort.InferenceSession(
        onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   transforms.Resize((384, 384))]

    img = transforms.Compose(transforms_)(
        Image.open('../input/test/A/xin.jpg'))
    img = np.expand_dims(img, axis=0)

    fake_img = session.run(None, {x: img})
    fake_img = np.array(fake_img)
    fake_img = np.squeeze(fake_img)
    fake_img = fake_img.transpose(1, 2, 0)
    fake_img = 0.5 * (fake_img + 1) * 255
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite('../output/result.jpg', fake_img)


if __name__ == '__main__':
    test()
#  python test_onnx.py ../mymodels/baby.onnx ../input/chen_face.jpg