#ifndef __REF_YOLOV3_DETECTIONOUTPUT_KERNEL_H__
#define __REF_YOLOV3_DETECTIONOUTPUT_KERNEL_H__

#include <stdint.h>
#include <vector>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C"{
#endif

struct  Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

struct YOLOV3_ddo_parm
{
    int num_classes;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    std::vector<float> bias;
    std::vector<float> mask;
    std::vector<float> anchors_scale;
    std::vector<std::vector<int>> input_dims;
    int mask_group_num;
    float out_scale;
    std::vector<Box> output_box;
};

typedef int (*ref_YOLOV3DetectionOutput_kernel_t)(void* input, YOLOV3_ddo_parm* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_yolov3detectionoutput_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_yolov3detectionoutput_fp16.c"
#endif
#ifdef CONFIG_KERNEL_INT8
#include "ref_yolov3detectionoutput_int8.c"
#endif                                            

#ifdef __cplusplus
}
#endif

#endif