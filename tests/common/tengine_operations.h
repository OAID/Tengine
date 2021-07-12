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
 * Copyright (c) 2020, OPEN AI LAB
 */

#ifndef __TENGINE_OPERATIONS_H__
#define __TENGINE_OPERATIONS_H__

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    GRAY2BGR,
    RGB2GRAY,
    RGB2BGR
} IMFORMAT;

typedef enum
{
    NONE,
    CAFFE,
    TENSORFLOW,
    MXNET,
    TFLITE
} FUNCSTYLE;

typedef struct image
{
    int w;
    int h;
    int c;
    float* data;
} image;

typedef struct
{
    int id;
    float score;
} cls_score;

/**
 * read image to buffer
 * @param [in] filename: image name
 * @param [in] channels: c
 */
image load_image_stb(const char* filename, int channels);

/**
 * calloc a buffer for empty image
 * @param [in] w: image's w
 * @param [in] h: image's h
 * @param [in] c: image's c
 */
image make_image(int w, int h, int c);

/**
 * parameters setting
 * @param [in] w: image's w
 * @param [in] h: image's h
 * @param [in] c: image's c
 */
image make_empty_image(int w, int h, int c);

/**
 * return ture if file exist
 * @param [in] file_name: file name
 * @return 1: success, 0: fail.
 */
int check_file_exist(const char* file_name);

/**
 * add a image from source to dest.
 * @param [in] source:image
 * @param [out] dest:image
 * @param [in] dx: x bias
 * @param [in] dy: y bias
 */
void add_image(image source, image dest, int dx, int dy);

/**
 * @param [in] image_file: image file
 * @param [out] input_data: data buffer
 * @param [in] img_h: image's h
 * @param [in] img_w: image's w
 * @param [in] mean: means
 * @param [in] scale: scale
 */
void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean,
                    const float* scale);

/**
 * read image and output with nhwc format
 * @param [in] filename: image name
 * @param [in] img_w: resize width
 * @param [in] img_h: resize height
 * @param [in] means: image means value, needs to be size of three: means[0], means[1], means[2]
 * @param [in] scale: image scale value, needs to be size of three: scale[0], scale[1], scale[2]
 * @param [in] convert: image data format, RGB, BGR, GRAY
 */
image imread2tf(image im, int img_w, int img_h, float* means, float* scale);

/**
 * read image and output with nchw format
 * @param [in] filename: image name
 * @param [in] img_w: resize width
 * @param [in] img_h: resize height
 * @param [in] means: image means value, needs to be size of three: means[0], means[1], means[2]
 * @param [in] scale: image scale value, needs to be size of three: scale[0], scale[1], scale[2]
 * @param [in] convert: image data format, RGB, BGR, GRAY
 */
image imread2caffe(image im, int img_w, int img_h, float* means, float* scale);

/**
 * read image and output with nchw format
 * @param [in] filename: image name
 * @param [in] img_w: resize width
 * @param [in] img_h: resize height
 * @param [in] means: image means value, needs to be size of three: means[0], means[1], means[2]
 * @param [in] scale: image scale value, needs to be size of three: scale[0], scale[1], scale[2]
 * @param [in] convert: image data format, RGB, BGR, GRAY
 */
image imread2mxnet(image im, int img_w, int img_h, float* means, float* scale);

/**
 * read image and output with nhwc format
 * @param [in] filename: image name
 * @param [in] img_w: resize width
 * @param [in] img_h: resize height
 * @param [in] means: image means value, needs to be size of three: means[0], means[1], means[2]
 * @param [in] scale: image scale value, needs to be size of three: scale[0], scale[1], scale[2]
 * @param [in] convert: image data format, RGB, BGR, GRAY
 */
image imread2tflite(image im, int img_w, int img_h, float* means, float* scale);
/*
 * resize the image, and then return the image type
 * @param [in] im : input image
 * @param [in] w: resied width
 * @param [in] h: resized height
 */
image resize_image(image im, int w, int h);

/**
 * load image, support JPG, PNG, BMP, TGA formats
 * @param [in] filename: file name
 */
image imread(const char* filename);

/**
 * load image, support JPG, PNG, BMP, TGA formats
 * @param [in] filename: file name
 * @param [in] img_w : image's w
 * @param [in] img_h : image's h
 * @param [in] means : means
 * @param [in] scale : scale
 */
image imread_process(const char* filename, int img_w, int img_h, float* means, float* scale);

/**
 * load image, support JPG, PNG, BMP, TGA formats
 */
image imread2post(const char* filename);

/**
 * convert image pixels to nchw layout
 * src: origin image
 * return: converted image
 */
image image_permute(image src);

/**
 * convert image pixels from RGB to BGR formats
 * src: origin image
 * return: converted image
 */
image rgb2bgr_permute(image src);

/**
 * convert image pixels from GRAY to BGR formats
 * src: origin image
 * return: converted image
 */
image gray2bgr(image src);

/**
 * tranpose matrix
 * @param [in] src: origin image
 * return: transposed image
 */
image tranpose(image src);

/**
 * draw circle on the image
 * @param [in] im: target image needed to draw circle
 * @param [in] x: circle's x position center
 * @param [in] y: circle's y position center
 * @param [in] radius: circle's radius
 * @param [in] r: RGB color for red
 * @param [in] g: RGB color for green
 * @param [in] b: RGB color for blue
 */
void draw_circle(image im, int x, int y, int radius, int r, int g, int b);

/**
 *  do subtract between two image (image a and imge b)
 * @param [in] a: input image
 * @param [in] b: input image
 * @param [out] c: output image
 */
void subtract(image a, image b, image c);

/**
 * multiply image's pixels by value
 * @param [in] a: image needed to do multiply
 * @param [in] value: multiply value
 * @param [out] b : output image
 */
void multi(image a, float value, image b);

/**
 * convert bgr format to gray format
 * @param [in] src: origin image
 * @return : converted image
 */
image rgb2gray(image src);

/**
 * copy image
 * @param [in] p: origin image needed to be copied
 * @return : copied image
 */
image copy_image(image p);

/**
 * Extend the image with constant value
 * @param [in] im: target image
 * @param [in] top: top area of image to extend
 * @param [in] bottom: bottom area of image to extend
 * @param [in] left: left area of image to extend
 * @param [in] right: right area of image to extend
 * @param [in] value: value to fill the extended area
 */
image copyMaker(image im, int top, int bottom, int left, int right, float value);

/**
 * save image
 * @param [in] im: image need to be saved
 * @param [in] name: name of saved image
 * @param [in] fabs: image format
 */
void save_image(image im, const char* name);

/**
 * draw box on the image
 * @param [in] a: target image
 * @param [in] x1: top-left's x position
 * @param [in] y1: top-left's y position
 * @param [in] x2: bottom-right's x position
 * @param [in] y2: bottom-right's y position
 * @param [in] w: rectangle's width value
 * @param [in] r: RGB color of red
 * @param [in] g: RGB color of green
 * @param [in] b: RGB color of blue
 */
void draw_box(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);

/**
 * free image data from memory
 * @param [in] src : image data
 * @return : NULL
 */
void free_image(image m);

/**
 * Paste text on a graphic box
 * @param [in] im : image
 * @param [in] w: width
 * @param [in] h: height
 */
image letterbox(image im, int w, int h);

/**
 * resize data
 * @param [in] input : data of input
 * @param [out] output: data of output
 * @param [in] img_w: image's w
 * @param [in] img_h: image's h
 * @param [in] c:  channel
 * @param [in] h: height
 * @param [in] w: width
 */
void tengine_resize_f32(float* input, float* output, int img_w, int img_h, int c, int h, int w);

/**
 * sort class by score from big to small
 * @param [in] array: the array of class's score
 * @param [in] left: the left score
 * @param [in] right: the right s
 */
static void sort_cls_score(cls_score* array, int left, int right);

/**
 * print the class of top num and scores
 * @param [in] data : data
 * @param [in] total_num: total class num
 * @param [in] topk : top k
 */
void print_topk(float* data, int total_num, int topk);

#ifdef __cplusplus
}
#endif

#endif
