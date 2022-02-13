from tengine import tg
import numpy as np
import cv2
import argparse
import os
import time


DEFAULT_MAX_BOX_COUNT = 100
DEFAULT_REPEAT_COUNT = 1
DEFAULT_THREAD_COUNT = 1
DEFAULT_IMG_H = 300
DEFAULT_IMG_W = 300
DEFAULT_SCALE = 0.008
DEFAULT_MEAN1 = 127.5
DEFAULT_MEAN2 = 127.5
DEFAULT_MEAN3 = 127.5
SHOW_THRESHOLD = 0.5

parser = argparse.ArgumentParser(description='mobilenet_ssd')
parser.add_argument('-m', '--model', default='./models/mobilenet_ssd.tmfile', type=str)
parser.add_argument('-i', '--image', default='./images/dog.jpg', type=str)
parser.add_argument('-r', '--repeat_count', default=f'{DEFAULT_REPEAT_COUNT}', type=str)
parser.add_argument('-t', '--thread_count', default=f'{DEFAULT_THREAD_COUNT}', type=str)


def get_current_time():
    return time.time() * 1000

def draw_box(img, x1, y1, x2, y2, w=2, r=125, g=0, b=125):
    im_h,im_w,im_c = img.shape
    #print("draw_box", im_h, im_w, x1, x2, y1, y2)
    x1 = np.clip(x1, 0, im_w)
    x2 = np.clip(x2, 0, im_w)
    y1 = np.clip(y1, 0, im_h)
    y2 = np.clip(y2, 0, im_h)
    img[y1:y2, x1:x1+w] = [r, g, b]
    img[y1:y2, x2:x2+w] = [r, g, b]
    img[y1:y1+w, x1:x2] = [r, g, b]
    img[y2:y2+w, x1:x2] = [r, g, b]


def main(args):
    class_name=["background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                "sofa",       "train",     "tvmonitor"]
    image_file = args.image
    tm_file = args.model
    assert os.path.exists(tm_file), 'Model: {tm_file} not found'
    assert os.path.exists(image_file), 'Image: {image_file} not found'

    img_h,img_w =map(int,(DEFAULT_IMG_H, DEFAULT_IMG_W))
    scale = list(map(float,(DEFAULT_SCALE,DEFAULT_SCALE,DEFAULT_SCALE)))
    mean_value = list(map(float,(DEFAULT_MEAN1,DEFAULT_MEAN2,DEFAULT_MEAN3)))
    show_threshold = float(SHOW_THRESHOLD)
    assert len(mean_value) == 3, 'The number of mean_value should be 3, e.g. 104.007,116.669,122.679'
    img_mean = np.array(mean_value).reshape((1, 1, 3))
    data = cv2.imread(image_file)
    h=data.shape[0]
    w=data.shape[1]
    data = cv2.resize(data, (img_w, img_h))
    data = ((data - img_mean) * DEFAULT_SCALE).astype(np.float32)
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
    num=output.shape[1]
    output=np.squeeze(output)

    print('detect result num: %d'%num)
    BOX=[]
    for i in range(int(num)):
        box={}
        score = output[i][1]
        if score>show_threshold:
            class_idx = output[i][0]
            x0 = output[i][2] * w
            y0 = output[i][3] * h
            x1 = output[i][4] * w
            y1 = output[i][5] * h
            box["x0"]=int(x0)
            box["y0"]=int(y0)
            box["x1"]=int(x1)
            box["y1"]=int(y1)
            print('class_names[%d]:%s\tscore: %.2f%%'%(int(class_idx), class_name[int(class_idx)],(score*100)),'  BOX:( %d , %d ),( %d , %d )'%(x0,y0,x1,y1))
        BOX.append(box)
    
    data_out = cv2.imread(image_file)
    for i in range(int(num)):
        draw_box(img=data_out, x1= BOX[i]["x0"], y1=BOX[i]["y0"], x2=BOX[i]["x1"], y2=BOX[i]["y1"], w=2, r=125, g=0, b=125)
        
    out_path="./mobilenet_ssd_out.jpg"
    cv2.imwrite(out_path,data_out)
    print("save_img path:",out_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


