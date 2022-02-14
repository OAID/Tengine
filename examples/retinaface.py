from tengine import tg
import numpy as np
import cv2
import argparse
import os
import time
import math

DEFAULT_LABEL_FILE = "./synset_words.txt"
DEFAULT_IMG_H = 224
DEFAULT_IMG_W = 224
DEFAULT_SCALE = 0.017
DEFAULT_MEAN1 = 104.007
DEFAULT_MEAN2 = 116.669
DEFAULT_MEAN3 = 122.679

parser = argparse.ArgumentParser(description='retinaface')
parser.add_argument('-m', '--model', default='./models/retinaface.tmfile', type=str)
parser.add_argument('-i', '--image', default='./images/mtcnn_face4.jpg', type=str)


CONF_THRESH = 0.8
NMS_THRESH = 0.4
input_name = 'data'

bbox_name = ["face_rpn_bbox_pred_stride32", "face_rpn_bbox_pred_stride16", "face_rpn_bbox_pred_stride8"]
score_name = ["face_rpn_cls_prob_reshape_stride32", "face_rpn_cls_prob_reshape_stride16", "face_rpn_cls_prob_reshape_stride8"]
landmark_name = ["face_rpn_landmark_pred_stride32", "face_rpn_landmark_pred_stride16", "face_rpn_landmark_pred_stride8"]

stride = [32, 16, 8]
scales = [[32.0, 16.0], [8.0, 4.0], [2.0, 1.0]]

#face
#face[0] = score
#face[1] = [x, y, w, h]
#face[2] = [[x, y], [x, y], [x, y], [x, y], [x, y]]
#

def draw_box(img, x1, y1, x2, y2, w, r, g, b):
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
    return img

def draw_circle(img, x, y, radius, r, g, b):
    im_h,im_w,im_c = img.shape
    startX = x - radius
    startY = y - radius
    endX = x + radius
    endY = y + radius
    if startX < 0:
        startX = 0
    if startY < 0:
        startY = 0
    if endX > im_w:
        endX = im_w
    if endY > im_h:
        endY = im_h

    for j in range(startY, endY):
        for i in range(startX, endX):
            num1 = (i - x) * (i - x) + (j - y) * (j - y)
            num2 = radius * radius
            if num1 <= num2:
                img[j, i] = [r, g, b]
    return img


def draw_target(all_pred_boxes, img):
    class_name = 'faces'
    print ("detected face num: %d" %(len(all_pred_boxes)))

    for b in range(len(all_pred_boxes)):
        box = all_pred_boxes[b]
        print ("BOX %.2f:( %g , %g ),( %g , %g )" %(box[0], box[1][0], box[1][1], box[1][2], box[1][3]))
        img = draw_box(img, int(box[1][0]), int(box[1][1]), int(box[1][0] + box[1][2]), int(box[1][1] + box[1][3]), 2, 0, 255, 0)

        for l in range(5):
            img = draw_circle(img, int(box[2][l][0]), int(box[2][l][1]), 1, 0, 128, 128)
    out_path = "./retinaface_out.jpg"
    cv2.imwrite(out_path, img)
    print("save_img path:", out_path)


def iou(face_a, face_b):
    area_a = face_a[1][2] * face_a[1][3]
    area_b = face_b[1][2] * face_b[1][3]

    xx1 = max(face_a[1][0], face_b[1][0])
    yy1 = max(face_a[1][1], face_b[1][1])
    xx2 = min(face_a[1][0] + face_a[1][2], face_b[1][0] + face_b[1][2])
    yy2 = min(face_a[1][1] + face_a[1][3], face_b[1][1] + face_b[1][3])

    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)

    inter = w * h
    over = float(inter) / (area_a + area_b - inter)
    return over

def nms_sorted_boxes(face_objects, nms_threshold):
    picked = []
    areas = []
    for face in face_objects:
        areas.append(face[1][2] * face[1][3])

    for i in range(len(face_objects)):
        a = face_objects[i]

        keep = 1
        for j in range(len(picked)):
            b = face_objects[picked[j]]

            inter_area = iou(a, b)
            if inter_area > nms_threshold:
                keep = 0
        if keep:
            picked.append(i)
    return picked

def qsort_descent_inplace(face_objects, left=None, right=None):
    if(left == None) and (right == None):
        if len(face_objects) == 0:
            return
        qsort_descent_inplace(face_objects, 0, len(face_objects) - 1)
    else:
        i = left
        j = right
        p = face_objects[(left + right)//2][0]
        while(i <= j):
            while(face_objects[i][0] > p):
                i = i + 1
            while(face_objects[j][0] < p):
                j = j - 1
            if i <= j:
                tmp = face_objects[i]
                face_objects[i] = face_objects[j]
                face_objects[j] = tmp
                i = i + 1
                j = j - 1
                
        if left < j:
            qsort_descent_inplace(face_objects, left, j)
        if i < right:
            qsort_descent_inplace(face_objects, i, right)


#anchor = [x1, y1, x2, y2]
def generate_anchors(base_size, ratios, scales):
    num_ratio = len(ratios)
    num_scale = len(scales)

    cx = base_size * 0.5
    cy = base_size * 0.5

    anchors = []

    for i in range(num_ratio):
        ar = ratios[i]
        r_w = int(round(float(base_size) / math.sqrt(ar)))
        r_h = int(round(float(r_w * ar)))

        for j in range(num_scale):
            scale = scales[j]
            rs_w = r_w * scale
            rs_h = r_h * scale

            box = []
            box.append(cx - rs_w * 0.5)
            box.append(cy - rs_h * 0.5)
            box.append(cx + rs_w * 0.5)
            box.append(cy + rs_h * 0.5)

            anchors.append(box)
    return anchors

def generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faces):
    n, c, h, w = bbox_blob.shape
    offset = w * h
    num_anchors = len(anchors)  
    
    score_blob_v = np.resize(score_blob, score_blob.size)
    bbox_blob_v = np.resize(bbox_blob, bbox_blob.size)
    landmark_blob_v = np.resize(landmark_blob, landmark_blob.size)

    for q in range(num_anchors):
        anchor = anchors[q]
        score_offset = (q + num_anchors) * offset
        bbox_offset = (q * 4) * offset
        landmark_offset = (q * 10) * offset

        anchor_y = anchor[1]
        anchor_w = anchor[2] - anchor[0]
        anchor_h = anchor[3] - anchor[1]

        for i in range(h):
            anchor_x = anchor[0]
            for j in range(w):
                index = i * w + j
                prob = score_blob_v[score_offset + index]

                if prob >= prob_threshold:
                    #apply center size
                    dx = bbox_blob_v[bbox_offset + index + offset * 0]
                    dy = bbox_blob_v[bbox_offset + index + offset * 1]
                    dw = bbox_blob_v[bbox_offset + index + offset * 2]
                    dh = bbox_blob_v[bbox_offset + index + offset * 3]

                    cx = anchor_x + anchor_w * 0.5
                    cy = anchor_y + anchor_h * 0.5

                    pb_cx = cx + anchor_w * dx
                    pb_cy = cy + anchor_h * dy

                    pb_w = anchor_w * math.exp(dw)
                    pb_h = anchor_h * math.exp(dh)

                    x0 = pb_cx - pb_w * 0.5
                    y0 = pb_cy - pb_h * 0.5
                    x1 = pb_cx + pb_w * 0.5
                    y1 = pb_cy + pb_h * 0.5

                    obj = []
                    obj.append(prob)
                    rect = []
                    rect.append(x0)
                    rect.append(y0)
                    rect.append(x1 - x0 + 1)
                    rect.append(y1 - y0 + 1)
                    obj.append(rect)

                    landmarks = []
                    point = []
                    point.append(cx + (anchor_w + 1) * landmark_blob_v[landmark_offset + index + offset * 0])
                    point.append(cy + (anchor_h + 1) * landmark_blob_v[landmark_offset + index + offset * 1])
                    landmarks.append(point)

                    point = []
                    point.append(cx + (anchor_w + 1) * landmark_blob_v[landmark_offset + index + offset * 2])
                    point.append(cy + (anchor_h + 1) * landmark_blob_v[landmark_offset + index + offset * 3])
                    landmarks.append(point)

                    point = []
                    point.append(cx + (anchor_w + 1) * landmark_blob_v[landmark_offset + index + offset * 4])
                    point.append(cy + (anchor_h + 1) * landmark_blob_v[landmark_offset + index + offset * 5])
                    landmarks.append(point)

                    point = []
                    point.append(cx + (anchor_w + 1) * landmark_blob_v[landmark_offset + index + offset * 6])
                    point.append(cy + (anchor_h + 1) * landmark_blob_v[landmark_offset + index + offset * 7])
                    landmarks.append(point)

                    point = []
                    point.append(cx + (anchor_w + 1) * landmark_blob_v[landmark_offset + index + offset * 8])
                    point.append(cy + (anchor_h + 1) * landmark_blob_v[landmark_offset + index + offset * 9])
                    landmarks.append(point)

                    obj.append(landmarks)

                    faces.append(obj)

                anchor_x = anchor_x + feat_stride

            anchor_y = anchor_y + feat_stride

def get_input_data(image_file, max_size=None, target_size=None):
    if (max_size == None) and (target_size == None):
        img = cv2.imread(image_file)
        im_h, im_w, im_c = img.shape
        ori_size = [0,0]

        return img, ori_size, ori_size, 1

    else:
        img = cv2.imread(image_file)

        im_h, im_w, im_c = img.shape

        ori_size = [im_w,im_h]
        dst_size = [0,0]

        #img = image_permute(img); don't change any thing?
        im_size_min = min(im_h, im_w)
        im_size_max = max(im_h, im_w)

        scale = float(target_size) / float(im_size_min)

        if scale * float(im_size_max) > float(max_size):
            scale = float(max_size) / im_size_max

        dst_size[0] = int(round(im_w * scale))
        dst_size[1] = int(round(im_h * scale))

        resImg = resize_image(img, dst_size[0], dst_size[1])

        return resImg, ori_size, dst_size, scale


def main(args):
    image_file = args.image
    tm_file = args.model
    assert os.path.exists(image_file), f'Image: {image_file} not found'
    assert os.path.exists(tm_file), f'Model: {tm_file} not found'


    graph = tg.Graph(None, 'tengine', tm_file)

    target_size = 1024
    max_size = 1980
    image_data, _, _, _ = get_input_data(image_file)
    img_h, img_w, img_c = image_data.shape
    
    #model need RGB-type input
    image_data = cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)

    #image_data = image_data.transpose((2, 0, 1)).astype(np.float32)
    image_data = np.ascontiguousarray(image_data.transpose((2, 0, 1)).astype(np.float32))
    image_data = image_data.copy()
    #print("img_h, img_w, img_c: %d, %d, %d" %(img_h, img_w, img_c))

    input_tensor = graph.getTensorByName(input_name)
    dims = [1, 3, img_h, img_w]
    input_tensor.shape = dims

    input_tensor.buf = image_data

    graph.preRun()
    graph.run(1) # 1 is blocking

    
    face_proposals = []

    # process the detection result
    for stride_index in range(3):
        score_blob_tensor = graph.getTensorByName(score_name[stride_index])
        bbox_blob_tensor = graph.getTensorByName(bbox_name[stride_index])
        landmark_blob_tensor = graph.getTensorByName(landmark_name[stride_index])
        
        score_blob = np.resize(np.array(score_blob_tensor.buf), score_blob_tensor.shape)
        bbox_blob = np.resize(np.array(bbox_blob_tensor.buf), bbox_blob_tensor.shape)
        landmark_blob = np.resize(np.array(landmark_blob_tensor.buf), landmark_blob_tensor.shape)

        base_size = 16
        feat_stride = stride[stride_index]
        current_ratios = [1.0]

        current_scales = scales[stride_index]

        threshold = CONF_THRESH

        anchors = generate_anchors(base_size, current_ratios, current_scales)

        face_objects = []

        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, threshold, face_objects)

        for face in face_objects:
            face_proposals.append(face)

    #sort all proposals by score from highest to lowest
    qsort_descent_inplace(face_proposals)

    #apply nms with nms_threshold
    picked = nms_sorted_boxes(face_proposals, NMS_THRESH)

    face_count = len(picked)

    face_objects = []
    for i in range(face_count):
        face = face_proposals[picked[i]]
    
        x0 = face[1][0]
        y0 = face[1][1]
        x1 = x0 + face[1][2]
        y1 = y0 + face[1][3]

        x0 = max(min(x0, img_w - 1), 0)
        y0 = max(min(y0, img_h - 1), 0)
        x1 = max(min(x1, img_w - 1), 0)
        y1 = max(min(y1, img_h - 1), 0)

        face[1][0] = x0
        face[1][1] = y0
        face[1][2] = x1 - x0
        face[1][3] = y1 - y0
        face_objects.append(face)

    img = cv2.imread(image_file)
    draw_target(face_objects, img)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

