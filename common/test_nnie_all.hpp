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
 * Copyright (c) 2019, Open AI Lab
 * Author: cmeng@openailab.com
 */

#ifndef __TEST_NNIE_ALL_HPP__
#define __TEST_NNIE_ALL_HPP__

#include <vector>
#include <sys/time.h>
#include <arm_neon.h>
#include "hi_comm_svp.h"
#include "hi_nnie.h"
#include "mpi_nnie.h"

/*16Byte align*/
#define TEST_NNIE_ALIGN_16                          16
#define TEST_NNIE_ALIGN16(u32Num)                   ((u32Num + TEST_NNIE_ALIGN_16 - 1) / TEST_NNIE_ALIGN_16 * TEST_NNIE_ALIGN_16)
#define TEST_NNIE_COORDI_NUM                        4    /*coordinate numbers*/
#define TEST_NNIE_QUANT_BASE                        4096 /*the base value*/
#define TEST_NNIE_PROPOSAL_WIDTH                    6    /*the number of proposal values*/
#define TEST_NNIE_SSD_REPORT_NODE_NUM               12
#define TEST_NNIE_MAX_SOFTWARE_MEM_NUM              4
#define TEST_NNIE_SSD_REPORT_NODE_NUM               12
#define TEST_NNIE_SSD_PRIORBOX_NUM                  6
#define TEST_NNIE_SSD_SOFTMAX_NUM                   6
#define TEST_NNIE_SSD_ASPECT_RATIO_NUM              6
#define TEST_NNIE_YOLOV3_REPORT_BLOB_NUM            3  /*yolov3 report blob num*/
#define TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM 85 /*yolov3 inference result num of each bbox*/
#define TEST_NNIE_YOLOV3_EACH_GRID_BIAS_NUM         6  /*yolov3 bias num of each grid*/
#define TEST_NNIE_SCORE_NUM                         2  /*the num of RPN scores*/

#define TEST_NNIE_COORDI_NUM 4    /*coordinate numbers*/
#define TEST_COORDI_NUM      4    /*num of coordinates*/
#define TEST_NNIE_HALF       0.5f /*the half value*/
#define TEST_NNIE_MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define TEST_NNIE_MIN(a, b)  (((a) < (b)) ? (a) : (b))

#define TEST_NNIE_SIGMOID(x)       (HI_FLOAT)(1.0f / (1 + fast_exp(-x)))
#define TEST_NNIE_SIGMOID_NOEXP(x) (HI_FLOAT)(1.0f / (1 + x))

inline float32x4_t vexpq10_f32(float32x4_t x)
{
    x = vmlaq_n_f32(vdupq_n_f32(1.0f), x, 0.0009765625f); // n = 10
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    return x;
}

void fast_exp_4f(const float* a, float* xx)
{
    float32x4_t x = vld1q_f32(a);
    x = vexpq10_f32(x);
    vst1q_f32(xx, x);
    return;
}

float fast_exp(float x)
{
    x = 1.0 + x * 0.0009765625f;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}

/*FasterRcnn software parameter*/
typedef struct hiTEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S
{
    HI_U32 au32Scales[9];
    HI_U32 au32Ratios[9];
    HI_U32 au32ConvHeight[2];
    HI_U32 au32ConvWidth[2];
    HI_U32 au32ConvChannel[2];
    HI_U32 u32ConvStride;
    HI_U32 u32NumRatioAnchors;
    HI_U32 u32NumScaleAnchors;
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 u32MinSize;
    HI_U32 u32SpatialScale;
    HI_U32 u32NmsThresh;
    HI_U32 u32FilterThresh;
    HI_U32 u32NumBeforeNms;
    HI_U32 u32MaxRoiNum;
    HI_U32 u32ClassNum;
    HI_U32 au32ConfThresh[21];
    HI_U32 u32ValidNmsThresh;
    HI_S32* aps32Conv[2];
    SVP_MEM_INFO_S stRpnTmpBuf;
    SVP_DST_BLOB_S stRpnBbox;
    SVP_DST_BLOB_S stClassRoiNum;
    SVP_DST_BLOB_S stDstRoi;
    SVP_DST_BLOB_S stDstScore;
    SVP_MEM_INFO_S stGetResultTmpBuf;
    HI_CHAR* apcRpnDataLayerName[2];
} TEST_NNIE_FASTERRCNN_SOFTWARE_PARAM_S;

typedef struct hiTEST_NNIE_CNN_GETTOPN_UNIT_S
{
    HI_U32 u32ClassId;
    HI_U32 u32Confidence;
} TEST_NNIE_CNN_GETTOPN_UNIT_S;

typedef struct hiTEST_NNIE_CNN_SOFTWARE_PARAM_S
{
    HI_U32 u32TopN;
    SVP_DST_BLOB_S stGetTopN;
    SVP_MEM_INFO_S stAssistBuf;
} TEST_NNIE_CNN_SOFTWARE_PARAM_S;

/*SSD software parameter*/
typedef struct hiTEST_NNIE_SSD_SOFTWARE_PARAM_S
{
    /*----------------- Model Parameters ---------------*/
    HI_U32 au32ConvHeight[12];
    HI_U32 au32ConvWidth[12];
    HI_U32 au32ConvChannel[12];
    /*----------------- PriorBox Parameters ---------------*/
    HI_U32 au32PriorBoxWidth[6];
    HI_U32 au32PriorBoxHeight[6];
    HI_FLOAT af32PriorBoxMinSize[6][1];
    HI_FLOAT af32PriorBoxMaxSize[6][1];
    HI_U32 u32MinSizeNum;
    HI_U32 u32MaxSizeNum;
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 au32InputAspectRatioNum[6];
    HI_FLOAT af32PriorBoxAspectRatio[6][2];
    HI_FLOAT af32PriorBoxStepWidth[6];
    HI_FLOAT af32PriorBoxStepHeight[6];
    HI_FLOAT f32Offset;
    HI_BOOL bFlip;
    HI_BOOL bClip;
    HI_S32 as32PriorBoxVar[4];
    /*----------------- Softmax Parameters ---------------*/
    HI_U32 au32SoftMaxInChn[6];
    HI_U32 u32SoftMaxInHeight;
    HI_U32 u32ConcatNum;
    HI_U32 u32SoftMaxOutWidth;
    HI_U32 u32SoftMaxOutHeight;
    HI_U32 u32SoftMaxOutChn;
    /*----------------- DetectionOut Parameters ---------------*/
    HI_U32 u32ClassNum;
    HI_U32 u32TopK;
    HI_U32 u32KeepTopK;
    HI_U32 u32NmsThresh;
    HI_U32 u32ConfThresh;
    HI_U32 au32DetectInputChn[6];
    HI_U32 au32ConvStride[6];
    SVP_MEM_INFO_S stPriorBoxTmpBuf;
    SVP_MEM_INFO_S stSoftMaxTmpBuf;
    SVP_DST_BLOB_S stClassRoiNum;
    SVP_DST_BLOB_S stDstRoi;
    SVP_DST_BLOB_S stDstScore;
    SVP_MEM_INFO_S stGetResultTmpBuf;
} TEST_NNIE_SSD_SOFTWARE_PARAM_S;

typedef struct hiTEST_NNIE_YOLOV1_SOFTWARE_PARAM_S
{
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 u32BboxNumEachGrid;
    HI_U32 u32ClassNum;
    HI_U32 u32GridNumHeight;
    HI_U32 u32GridNumWidth;
    HI_U32 u32NmsThresh;
    HI_U32 u32ConfThresh;
    SVP_MEM_INFO_S stGetResultTmpBuf;
    SVP_DST_BLOB_S stClassRoiNum;
    SVP_DST_BLOB_S stDstRoi;
    SVP_DST_BLOB_S stDstScore;
} TEST_NNIE_YOLOV1_SOFTWARE_PARAM_S;

typedef struct hiTEST_NNIE_YOLOV1_SCORE
{
    HI_U32 u32Idx;
    HI_S32 s32Score;
} TEST_NNIE_YOLOV1_SCORE_S;

/*Yolov2 software parameter*/
typedef struct hiTEST_NNIE_YOLOV2_SOFTWARE_PARAM_S
{
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 u32BboxNumEachGrid;
    HI_U32 u32ClassNum;
    HI_U32 u32GridNumHeight;
    HI_U32 u32GridNumWidth;
    HI_U32 u32NmsThresh;
    HI_U32 u32ConfThresh;
    HI_U32 u32MaxRoiNum;
    HI_FLOAT af32Bias[10];
    SVP_MEM_INFO_S stGetResultTmpBuf;
    SVP_DST_BLOB_S stClassRoiNum;
    SVP_DST_BLOB_S stDstRoi;
    SVP_DST_BLOB_S stDstScore;
} TEST_NNIE_YOLOV2_SOFTWARE_PARAM_S;

typedef struct hiTEST_NNIE_YOLOV2_BBOX
{
    HI_FLOAT f32Xmin;
    HI_FLOAT f32Xmax;
    HI_FLOAT f32Ymin;
    HI_FLOAT f32Ymax;
    HI_S32 s32ClsScore;
    HI_U32 u32ClassIdx;
    HI_U32 u32Mask;
} TEST_NNIE_YOLOV2_BBOX_S;

typedef TEST_NNIE_YOLOV2_BBOX_S TEST_NNIE_YOLOV3_BBOX_S;

/*Yolov3 software parameter*/
typedef struct hiTEST_NNIE_YOLOV3_SOFTWARE_PARAM_S
{
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 u32BboxNumEachGrid;
    HI_U32 u32ClassNum;
    HI_U32 au32GridNumHeight[3];
    HI_U32 au32GridNumWidth[3];
    HI_U32 u32NmsThresh;
    HI_U32 u32ConfThresh;
    HI_U32 u32MaxRoiNum;
    HI_FLOAT af32Bias[3][6];
    SVP_MEM_INFO_S stGetResultTmpBuf;
    SVP_DST_BLOB_S stClassRoiNum;
    SVP_DST_BLOB_S stDstRoi;
    SVP_DST_BLOB_S stDstScore;
} TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S;

/*stack for sort*/
typedef struct hiTEST_NNIE_STACK
{
    HI_S32 s32Min;
    HI_S32 s32Max;
} TEST_NNIE_STACK_S;

HI_S32 SAMPLE_COMM_SVP_MallocMem(const HI_CHAR* pszMmb, const HI_CHAR* pszZone, HI_U64* pu64PhyAddr, HI_VOID** ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;

    s32Ret = HI_MPI_SYS_MmzAlloc(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);

    return s32Ret;
}

#endif
