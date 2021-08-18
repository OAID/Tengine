/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include "yolo.hpp"
#include "timer.hpp"

#define MODE_WIDTH  416
#define MODE_HEIGHT 416

#define DEF_IMAGE "images/ssd_dog.jpg"
#define DET_MODEL "models/yolov3_uint8.tmfile"

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

static const unsigned char colors[81][3] = {
    {226, 255, 0},
    {56, 0, 255},
    {0, 94, 255},
    {0, 37, 255},
    {0, 255, 94},
    {255, 226, 0},
    {0, 18, 255},
    {255, 151, 0},
    {170, 0, 255},
    {0, 255, 56},
    {255, 0, 75},
    {0, 75, 255},
    {0, 255, 169},
    {255, 0, 207},
    {75, 255, 0},
    {207, 0, 255},
    {37, 0, 255},
    {0, 207, 255},
    {94, 0, 255},
    {0, 255, 113},
    {255, 18, 0},
    {255, 0, 56},
    {18, 0, 255},
    {0, 255, 226},
    {170, 255, 0},
    {255, 0, 245},
    {151, 255, 0},
    {132, 255, 0},
    {75, 0, 255},
    {151, 0, 255},
    {0, 151, 255},
    {132, 0, 255},
    {0, 255, 245},
    {255, 132, 0},
    {226, 0, 255},
    {255, 37, 0},
    {207, 255, 0},
    {0, 255, 207},
    {94, 255, 0},
    {0, 226, 255},
    {56, 255, 0},
    {255, 94, 0},
    {255, 113, 0},
    {0, 132, 255},
    {255, 0, 132},
    {255, 170, 0},
    {255, 0, 188},
    {113, 255, 0},
    {245, 0, 255},
    {113, 0, 255},
    {255, 188, 0},
    {0, 113, 255},
    {255, 0, 0},
    {0, 56, 255},
    {255, 0, 113},
    {0, 255, 188},
    {255, 0, 94},
    {255, 0, 18},
    {18, 255, 0},
    {0, 255, 132},
    {0, 188, 255},
    {0, 245, 255},
    {0, 169, 255},
    {37, 255, 0},
    {255, 0, 151},
    {188, 0, 255},
    {0, 255, 37},
    {0, 255, 0},
    {255, 0, 170},
    {255, 0, 37},
    {255, 75, 0},
    {0, 0, 255},
    {255, 207, 0},
    {255, 0, 226},
    {255, 245, 0},
    {188, 255, 0},
    {0, 255, 18},
    {0, 255, 75},
    {0, 255, 151},
    {255, 56, 0},
    {245, 255, 0}};

int main(int argc, char* argv[])
{
    /* initial the capture */
    cv::VideoCapture vp(0);

    if (!vp.isOpened())
    {
        printf("Open camera error.\n");

        return -1;
    }

    fprintf(stdout, "Camera is opened.\n");

    vp.set(cv::CAP_PROP_FPS, 30);

    cv::Mat image_ori, image_flip;
    vp >> image_ori;

    if (image_ori.empty())
    {
        fprintf(stderr, "Reading image was failed.\n");
        return -1;
    }

    fprintf(stdout, "Init tengine...\n");

    init_tengine();

    fprintf(stdout, "Tengine was inited.\n");

    std::array<float, 3> image_scale = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    std::array<float, 3> image_shift = {0.f, 0.f, 0.f};

    YOLO detector(DET_MODEL, MODE_WIDTH, MODE_HEIGHT, image_scale, image_shift);

    std::vector<Object> objects;

    cv::namedWindow("Tengine Khadas YOLOv3 Demo", cv::WINDOW_AUTOSIZE);

    while (true)
    {
        vp >> image_ori;

        objects.clear();

        Timer det_timer;
        detector.detect(image_ori, objects);

        det_timer.Stop();

        fprintf(stdout, "detect cost %.2fms.\n", det_timer.Cost());

        /* result show */
        for (auto& object : objects)
        {
            // box
            cv::Rect2f rect(object.box.x, object.box.y, object.box.width, object.box.height);

            const unsigned char* color = colors[object.label];

            cv::rectangle(image_ori, rect, cv::Scalar(color[0], color[1], color[2]), 2);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[object.label], object.score * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = rect.x;
            int y = rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image_ori.cols)
                x = image_ori.cols - label_size.width;

            cv::rectangle(image_ori, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(image_ori, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0));
        }

        cv::imshow("Tengine Khadas YOLOv3 Demo", image_ori);
        if (27 == cv::waitKey(1))
        {
            break;
        }
    }

    release_tengine();

    return 0;
}
