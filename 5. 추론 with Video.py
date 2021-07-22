import os
import cv2
from PIL import Image
import numpy as np
import pickle
from openvino.inference_engine import IECore

def pre_process_image(image, img_height=224):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    procecced_img = cv2.resize(image,(h,w),interpolation=cv2.INTER_AREA)

    procecced_img = (np.array(procecced_img) - 0) / 255.0
    procecced_img = procecced_img.transpose((2, 0, 1))
    procecced_img = procecced_img.reshape((n, c, h, w))

    return image, procecced_img

model_xml = 'C:/dev-tf/models/tf/frozen_mobilenet.xml'
model_bin = 'C:/dev-tf/models/tf/frozen_mobilenet.bin'

ie =IECore()

net = ie.read_network(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

exec_net = ie.load_network(network=net, device_name='CPU')
del net

with open('C:/dev-tf/models/class6.pickle', 'rb') as f:
    labels = pickle.load(f)

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
try:
    capture = cv2.VideoCapture(0)
except:
    print('카메라 안됨')

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    if not ret:
        print('비디오 읽기 오류')
        break

    image, processedImg = pre_process_image(frame)
    res = exec_net.infer(inputs={input_blob: processedImg})
    output_node_name = list(res.keys())[0]
    res = res[output_node_name]

    idx = idx = np.argsort(res[0])[-1]

    prob = res[0][idx] * 100

    info = 'Predicted : None'
    color = (0, 0, 0)

    if prob > 50:
        info = 'Predicted: {} ({:.2f}%)'.format(labels[idx].upper(), prob)
        color = (0, 0, 255)

    cv2.putText(frame, info, (10, 30), 0, 1, color, 2)
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(1) > 0:
        break

capture.release()
cv2.destroyAllWindows()