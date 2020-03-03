#ifndef __TENGINE_OPERATIONS_H__
#define __TENGINE_OPERATIONS_H__

#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <string.h>
#include <cmath>
#include <algorithm>

struct image;
typedef struct image image;

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


image load_image_stb(const char* filename, int channels);
image make_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
void draw_label(image a, int r, int c, image label, const float* rgb);
image get_label(const char* string, int size);
image addText_images(image a, image b, int dx);
void combination_image(image source, image dest, int dx, int dy);
void add_image(image source, image dest, int dx, int dy);

image imread(const char* filename, int img_w, int img_h, float* means, float* scale, FUNCSTYLE func);
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

image imread2tflite(image im, int img_w, int img_h, float* means, float* scale);
/*
 * resize the image, and then return the image format
 *  im : input image
 *  h: resized height
 *  w: resied width
 */
image resize_image(image im, int h, int w);

/**
 * load image, support JPG, PNG, BMP, TGA formats
 */
image imread(const char* filename);

/**
 * load image, support JPG, PNG, BMP, TGA formats
 */
image imread2post(const char* filename);


/**
 * convert image pixels from RGB to BGR formats
 * src: origin image
 * return: converted image
 */
image rgb2bgr_premute(image src);

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
 * do subtract between two image (image a and imge b)
 * a: input image
 * b: input image
 * c: output image
 */
void subtract(image a, image b, image c);

/**
 * multiply image's pixels by value
 * a: image needed to do multiply
 * value: multiply value
 * b: output image
 */
void multi(image a, float value, image b);

/**
 * convert bgr format to gray format
 * src: origin image
 * return: converted image
 */
image rgb2gray(image src);

/**
 * copy image
 * @param [in] p: origin image needed to be copied
 * return: copied image
 */
image copy_image(image p);

/**
 * put text words on image, supporting a-z, A-Z, 0-9 and normal sign
 * @param [in] im: target image
 * @param [in] string: text context
 * @param [in] size: size of string on image
 * @param [in] x: x position for start
 * @param [in] y: y position for start
 * @param [in] r: RGB color of red
 * @param [in] g: RGB color of green
 * @param [in] b: RGB color of blue
 */
void put_label(image im, const char* string, float size, int x, int y, int r, int g, int b);

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
 * src: image data
 * return: NULL
 */
void free_image(image m);

image letterbox(image im, int w, int h);
void tengine_resize_f32(float* input, float* output, int img_w, int img_h, int c, int h, int w);
void tengine_resize_uint8(uint8_t* input, float* output, int img_w, int img_h, int c, int h, int w);

template<typename T>
void tengine_resize(T* input, float* output, int img_w, int img_h, int c, int h, int w){
    if(sizeof(T) == sizeof(float))
    	tengine_resize_f32((float*)input, output, img_w, img_h, c, h, w);
    if(sizeof(T) == sizeof(uint8_t))
	    tengine_resize_uint8((uint8_t*)input, output, img_w, img_h, c, h, w);
}

#endif
