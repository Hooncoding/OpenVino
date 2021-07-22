
#  ### OpenVino에서 xml, bin 파일 추론

import os

from PIL import Image
import numpy as np


from openvino import inference_engine as ie
from openvino.inference_engine import IECore
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

def pre_process_image(imagePath, img_height=224):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    processedImg = image.resize((h,w), resample=Image.BILINEAR)

    processedImg = (np.array(processedImg) - 0) / 255.0
    
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg = processedImg.reshape((n, c, h, w))

    return image, processedImg, imagePath

model_xml = 'C:/dev-tf/models/tf/frozen_mobilenet.xml'
model_bin = 'C:/dev-tf/models/tf/frozen_mobilenet.bin'

ie =IECore()

net = ie.read_network(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
exec_net = ie.load_network(network=net, device_name='CPU')

image, processedImg, imagePath = pre_process_image('C:/dev-tf/facedata/test/SangHoon/22095217.png')

res = exec_net.infer(inputs={input_blob: processedImg})

output_node_name = list(res.keys())[0]
res = res[output_node_name]

idx = np.argsort(res[0])[-1]

with open('C:/dev-tf/models/class6.pickle', 'rb') as f:
    labels = pickle.load(f)

plt.imshow(load_img('C:/dev-tf/facedata/test/SangHoon/22095217.png'))
plt.show()
print('결과는: {}'.format(labels[idx].upper()))