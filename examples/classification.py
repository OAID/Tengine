from tengine import tg
import numpy as np
import cv2
import os
import time
import numpy as np

DEFAULT_MODEL_NAME = "squeezenet"
DEFAULT_LABEL_FILE = "./models/synset_words.txt"
DEFAULT_IMG_H = 227
DEFAULT_IMG_W = 227
DEFAULT_SCALE = 1.0
DEFAULT_MEAN1 = 104.007
DEFAULT_MEAN2 = 116.669
DEFAULT_MEAN3 = 122.679
DEFAULT_REPEAT_CNT = 1

def get_current_time():
    return time.time() * 1000

def read_synset_words():
    outs = []
    with open('synset_words.txt', 'r') as fin:
        for line in fin.readlines():
            outs.append(line.strip())
    return outs

def main():
    image_file = './cat.jpg'
    tm_file = './mobilenet.tmfile'
    assert os.path.exists(tm_file)
    img_h = 224
    img_w = 224
    scale = 0.017
    img_mean = np.array([DEFAULT_MEAN1, DEFAULT_MEAN2, DEFAULT_MEAN3]).reshape((1, 1, 3))
    data = cv2.imread(image_file)
    data = cv2.resize(data, (img_w, img_h))
    data = ((data - img_mean) * scale).astype(np.float32)
    data = np.ascontiguousarray(data.transpose((2, 0, 1)))
    assert data.dtype == np.float32

    graph = tg.Graph(None, 'tengine', tm_file)
    input_tensor = graph.getInputTensor(0, 0)
    dims = [1, 3, img_h, img_w]
    input_tensor.shape = dims
    graph.preRun()
    input_tensor.buf = data
    graph.run(1) # 1 is blocking
    output_tensor = graph.getOutputTensor(0, 0)
    output = np.array(output_tensor.buf)

    classes = read_synset_words()
    k = 5
    idx = output.argsort()[-1:-k-1:-1]
    for i in idx:
        print(classes[i], output[i])

if __name__ == '__main__':
    main()
