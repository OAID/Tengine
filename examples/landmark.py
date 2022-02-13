from tengine import tg
import numpy as np
import cv2
import argparse
import os
import time

DEFAULT_REPEAT_COUNT = 1
DEFAULT_THREAD_COUNT = 1

img_h = 144
img_w = 144
mean_value = [128.0, 128.0, 128.0]
scale = [0.0039, 0.0039, 0.0039]

parser = argparse.ArgumentParser(description='landmark')
parser.add_argument('-m', '--model', default='./models/landmark.tmfile', type=str)
parser.add_argument('-i', '--image', default='./images/mobileface02.jpg', type=str)
parser.add_argument('-r', '--repeat_count', default=f'{DEFAULT_REPEAT_COUNT}', type=str)
parser.add_argument('-t', '--thread_count', default=f'{DEFAULT_THREAD_COUNT}', type=str)

def get_current_time():
    return time.time() * 1000


def draw_circle(data_out, x, y, radius=2, r=0, g=255, b=0,w=0,h=0):
    startX = x - radius
    startY = y - radius
    endX = x + radius
    endY = y + radius
    if startX<0:
        startX=0
    if startY<0:
        startY=0
    if endX>w:
        endX=w
    if endY>h:
        endY=h

    for j in range(startY,endY):
        for i in range(startX,endX):
            num1 = (i - x) * (i - x) + (j - y) * (j - y)
            num2 = radius * radius
            if num1<=num2:
                data_out[j, i, 0] = r
                data_out[j, i, 1] = g
                data_out[j, i, 2] = b

def main(args):
    image_file = args.image
    tm_file = args.model
    assert os.path.exists(tm_file), 'Model: {tm_file} not found'
    assert os.path.exists(image_file), 'Image: {image_file} not found'
    assert len(mean_value) == 3, 'The number of mean_value should be 3, e.g. 104.007,116.669,122.679'
    img_mean = np.array(mean_value).reshape((1, 1, 3))
    data = cv2.imread(image_file)
    h=data.shape[0]
    w=data.shape[1]
    data = cv2.resize(data, (img_w, img_h))
    data = ((data - img_mean) * scale[0]).astype(np.float32)
    data = data.transpose((2, 0, 1))
    assert data.dtype == np.float32
    data = data.copy()

    graph = tg.Graph(None, 'tengine', tm_file)
    input_tensor = graph.getInputTensor(0, 0)
    dims = [1, 3, img_h, img_w]
    input_tensor.shape = dims
    graph.preRun()
    input_tensor.buf = data
    graph.run(1) # 1 is blocking
    output_tensor = graph.getOutputTensor(0, 0)
    output = output_tensor.getNumpyData()
    output=np.squeeze(output)
    img_shape=output.shape
    
    data_out = cv2.imread(image_file)
    for i in range(int(img_shape[0]/2)):
        x=int(output[2*i]*float(w/144))
        y=int(output[2*i+1]*float(h/144))
        draw_circle(data_out, x, y, radius=2, r=0, g=255, b=0,w=w,h=h)

    out_path="./landmark_out.jpg"
    cv2.imwrite(out_path,data_out)
    print("save_img path:",out_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

