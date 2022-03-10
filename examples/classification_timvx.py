from tengine import tg
import numpy as np
import cv2
import argparse
import os
import time
import numpy as np
import time

DEFAULT_LABEL_FILE = "./synset_words.txt"
DEFAULT_IMG_H = 224
DEFAULT_IMG_W = 224
DEFAULT_SCALE = 0.017
DEFAULT_MEAN1 = 104.007
DEFAULT_MEAN2 = 116.669
DEFAULT_MEAN3 = 122.679

parser = argparse.ArgumentParser(description='classification')
parser.add_argument('-m', '--model', default='./mobilenet_uint8.tmfile', type=str)
parser.add_argument('-i', '--image', default='./cat.jpg', type=str)
parser.add_argument('-g', '--image-size', default=f'{DEFAULT_IMG_H},{DEFAULT_IMG_W}', type=str, help='image size: height, width')
parser.add_argument('-w', '--mean-value', default=f'{DEFAULT_MEAN1},{DEFAULT_MEAN2},{DEFAULT_MEAN3}', type=str, help='mean value: mean1, mean2, mean3')
parser.add_argument('-s', '--scale', default=f'{DEFAULT_SCALE}', type=str)
parser.add_argument('-l', '--label', default=f'{DEFAULT_LABEL_FILE}', type=str, help='the default path of labels.txt (e.g. synset_words.txt)')

def get_current_time():
    return time.time() * 1000

def read_labels_file(fname):
    outs = []
    with open(fname, 'r') as fin:
        for line in fin.readlines():
            outs.append(line.strip())
    return outs

def main(args):
    image_file = args.image
    tm_file = args.model
    assert os.path.exists(args.label), f'Label File: {args.label} not found'
    assert os.path.exists(image_file), f'Image: {image_file} not found'
    assert os.path.exists(tm_file), f'Model: {tm_file} not found'
    labels = read_labels_file(args.label)

    img_h, img_w = map(int, args.image_size.split(','))
    context = tg.Context("timvx", 1)
    context.addDev("TIMVX")
    graph = tg.Graph(context, 'tengine', tm_file)
    input_tensor = graph.getInputTensor(0, 0)
    dims = [1, 3, img_h, img_w]
    input_tensor.shape = dims
    graph.preRun()

    input_scale, input_zero_point = input_tensor.getQuantParam(1)
    scale = float(args.scale)
    mean_value = list(map(float, args.mean_value.split(',')))
    assert len(mean_value) == 3, 'The number of mean_value should be 3, e.g. 104.007,116.669,122.679'
    img_mean = np.array(mean_value).reshape((1, 1, 3))
    data = cv2.imread(image_file)
    data = cv2.resize(data, (img_w, img_h))
    data = (((data - img_mean) * scale)/ input_scale + input_zero_point).astype(np.uint8)
    data = np.ascontiguousarray(data.transpose((2, 0, 1)))
    assert data.dtype == np.uint8
    data = data.copy()

    input_tensor.buf = data
    begin = time.time()
    graph.run(1) # 1 is blocking
    print("time:",(time.time() - begin)*1000, "ms")
    output_tensor = graph.getOutputTensor(0, 0)
    output_scale, output_zero_point = output_tensor.getQuantParam(1)
    output = np.array(output_tensor.buf)
    output = (output - output_zero_point) * output_scale

    k = 5
    idx = output.argsort()[-1:-k-1:-1]
    for i in idx:
        print(labels[i], output[i])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
