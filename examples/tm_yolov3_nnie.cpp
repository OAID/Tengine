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

#include <unistd.h>
#include <sys/stat.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <math.h>
#include "mpi_sys.h"
#include "mpi_vb.h"
#include <sys/time.h>
#include <float.h>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include "tengine_nnie_plugin.h"
#include "test_nnie_all.hpp"

/*
*Malloc memory with cached
*/
HI_S32 TEST_COMM_MallocCached(const HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);

    return s32Ret;
}

/*
*Fulsh cached
*/
HI_S32 TEST_COMM_FlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = HI_MPI_SYS_MmzFlushCache(u64PhyAddr, pvVirAddr, u32Size);
    return s32Ret;
}

bool get_input_data(const char *image_file, void *input_data, int input_length)
{
    FILE *fp = fopen(image_file, "rb");
    if (fp == nullptr)
    {
        std::cout << "Open input data file failed: " << image_file << "\n";
        return false;
    }

    int res = fread(input_data, 1, input_length, fp);
    if (res != input_length)
    {
        std::cout << "Read input data file failed: " << image_file << "\n";
        return false;
    }
    fclose(fp);
    return true;
}

static HI_FLOAT s_af32ExpCoef[10][16] = {
    {1.0f, 1.00024f, 1.00049f, 1.00073f, 1.00098f, 1.00122f, 1.00147f, 1.00171f, 1.00196f, 1.0022f, 1.00244f, 1.00269f, 1.00293f, 1.00318f, 1.00342f, 1.00367f},
    {1.0f, 1.00391f, 1.00784f, 1.01179f, 1.01575f, 1.01972f, 1.02371f, 1.02772f, 1.03174f, 1.03578f, 1.03984f, 1.04391f, 1.04799f, 1.05209f, 1.05621f, 1.06034f},
    {1.0f, 1.06449f, 1.13315f, 1.20623f, 1.28403f, 1.36684f, 1.45499f, 1.54883f, 1.64872f, 1.75505f, 1.86825f, 1.98874f, 2.117f, 2.25353f, 2.39888f, 2.55359f},
    {1.0f, 2.71828f, 7.38906f, 20.0855f, 54.5981f, 148.413f, 403.429f, 1096.63f, 2980.96f, 8103.08f, 22026.5f, 59874.1f, 162755.0f, 442413.0f, 1.2026e+006f, 3.26902e+006f},
    {1.0f, 8.88611e+006f, 7.8963e+013f, 7.01674e+020f, 6.23515e+027f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f},
    {1.0f, 0.999756f, 0.999512f, 0.999268f, 0.999024f, 0.99878f, 0.998536f, 0.998292f, 0.998049f, 0.997805f, 0.997562f, 0.997318f, 0.997075f, 0.996831f, 0.996588f, 0.996345f},
    {1.0f, 0.996101f, 0.992218f, 0.98835f, 0.984496f, 0.980658f, 0.976835f, 0.973027f, 0.969233f, 0.965455f, 0.961691f, 0.957941f, 0.954207f, 0.950487f, 0.946781f, 0.94309f},
    {1.0f, 0.939413f, 0.882497f, 0.829029f, 0.778801f, 0.731616f, 0.687289f, 0.645649f, 0.606531f, 0.569783f, 0.535261f, 0.502832f, 0.472367f, 0.443747f, 0.416862f, 0.391606f},
    {1.0f, 0.367879f, 0.135335f, 0.0497871f, 0.0183156f, 0.00673795f, 0.00247875f, 0.000911882f, 0.000335463f, 0.00012341f, 4.53999e-005f, 1.67017e-005f, 6.14421e-006f, 2.26033e-006f, 8.31529e-007f, 3.05902e-007f},
    {1.0f, 1.12535e-007f, 1.26642e-014f, 1.42516e-021f, 1.60381e-028f, 1.80485e-035f, 2.03048e-042f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

static HI_FLOAT TEST_NNIE_QuickExp(HI_S32 s32Value)
{
    if (s32Value & 0x80000000)
    {
        s32Value = ~s32Value + 0x00000001;
        return s_af32ExpCoef[5][s32Value & 0x0000000F] * s_af32ExpCoef[6][(s32Value >> 4) & 0x0000000F] * s_af32ExpCoef[7][(s32Value >> 8) & 0x0000000F] * s_af32ExpCoef[8][(s32Value >> 12) & 0x0000000F] * s_af32ExpCoef[9][(s32Value >> 16) & 0x0000000F];
    }
    else
    {
        return s_af32ExpCoef[0][s32Value & 0x0000000F] * s_af32ExpCoef[1][(s32Value >> 4) & 0x0000000F] * s_af32ExpCoef[2][(s32Value >> 8) & 0x0000000F] * s_af32ExpCoef[3][(s32Value >> 12) & 0x0000000F] * s_af32ExpCoef[4][(s32Value >> 16) & 0x0000000F];
    }
}

static void TEST_NNIE_Argswap(HI_S32 *ps32Src1, HI_S32 *ps32Src2)
{
    HI_U32 i = 0;
    HI_S32 u32Tmp = 0;
    for (i = 0; i < TEST_NNIE_PROPOSAL_WIDTH; i++)
    {
        u32Tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = u32Tmp;
    }
}

static void TEST_NNIE_Yolov1_Argswap(HI_S32 *ps32Src1, HI_S32 *ps32Src2,
                                     HI_U32 u32ArraySize)
{
    HI_U32 i = 0;
    HI_S32 s32Tmp = 0;
    for (i = 0; i < u32ArraySize; i++)
    {
        s32Tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = s32Tmp;
    }
}

static HI_S32 TEST_NNIE_Yolo_NonRecursiveArgQuickSort(HI_S32 *ps32Array,
                                                      HI_S32 s32Low, HI_S32 s32High, HI_U32 u32ArraySize, HI_U32 u32ScoreIdx,
                                                      TEST_NNIE_STACK_S *pstStack)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;
    HI_S32 s32KeyConfidence = ps32Array[u32ArraySize * s32Low + u32ScoreIdx];
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    while (s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        s32KeyConfidence = ps32Array[u32ArraySize * s32Low + u32ScoreIdx];

        while (i < j)
        {
            while ((i < j) && (s32KeyConfidence > ps32Array[j * u32ArraySize + u32ScoreIdx]))
            {
                j--;
            }
            if (i < j)
            {
                TEST_NNIE_Yolov1_Argswap(&ps32Array[i * u32ArraySize], &ps32Array[j * u32ArraySize], u32ArraySize);
                i++;
            }

            while ((i < j) && (s32KeyConfidence < ps32Array[i * u32ArraySize + u32ScoreIdx]))
            {
                i++;
            }
            if (i < j)
            {
                TEST_NNIE_Yolov1_Argswap(&ps32Array[i * u32ArraySize], &ps32Array[j * u32ArraySize], u32ArraySize);
                j--;
            }
        }

        if (s32Low < i - 1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = s32Low;
            pstStack[s32Top].s32Max = i - 1;
        }

        if (s32High > i + 1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = i + 1;
            pstStack[s32Top].s32Max = s32High;
        }
    }
    return HI_SUCCESS;
}

static HI_S32 TEST_NNIE_SoftMax(HI_FLOAT *pf32Src, HI_U32 u32Num)
{
    HI_FLOAT f32Max = 0;
    HI_FLOAT f32Sum = 0;
    HI_U32 i = 0;

    for (i = 0; i < u32Num; ++i)
    {
        if (f32Max < pf32Src[i])
        {
            f32Max = pf32Src[i];
        }
    }

    for (i = 0; i < u32Num; ++i)
    {
        pf32Src[i] = (HI_FLOAT)TEST_NNIE_QuickExp((HI_S32)((pf32Src[i] - f32Max) * TEST_NNIE_QUANT_BASE));
        f32Sum += pf32Src[i];
    }

    for (i = 0; i < u32Num; ++i)
    {
        pf32Src[i] /= f32Sum;
    }
    return HI_SUCCESS;
}

static HI_FLOAT TEST_NNIE_Yolov2_GetMaxVal(HI_FLOAT *pf32Val, HI_U32 u32Num,
                                           HI_U32 *pu32MaxValueIndex)
{
    HI_U32 i = 0;
    HI_FLOAT f32MaxTmp = 0;

    f32MaxTmp = pf32Val[0];
    *pu32MaxValueIndex = 0;
    for (i = 1; i < u32Num; i++)
    {
        if (pf32Val[i] > f32MaxTmp)
        {
            f32MaxTmp = pf32Val[i];
            *pu32MaxValueIndex = i;
        }
    }

    return f32MaxTmp;
}

static HI_DOUBLE TEST_NNIE_Yolov2_Iou(TEST_NNIE_YOLOV2_BBOX_S *pstBbox1,
                                      TEST_NNIE_YOLOV2_BBOX_S *pstBbox2)
{
    HI_FLOAT f32InterWidth = 0.0;
    HI_FLOAT f32InterHeight = 0.0;
    HI_DOUBLE f64InterArea = 0.0;
    HI_DOUBLE f64Box1Area = 0.0;
    HI_DOUBLE f64Box2Area = 0.0;
    HI_DOUBLE f64UnionArea = 0.0;

    f32InterWidth = TEST_NNIE_MIN(pstBbox1->f32Xmax, pstBbox2->f32Xmax) - TEST_NNIE_MAX(pstBbox1->f32Xmin, pstBbox2->f32Xmin);
    f32InterHeight = TEST_NNIE_MIN(pstBbox1->f32Ymax, pstBbox2->f32Ymax) - TEST_NNIE_MAX(pstBbox1->f32Ymin, pstBbox2->f32Ymin);

    if (f32InterWidth <= 0 || f32InterHeight <= 0)
        return 0;

    f64InterArea = f32InterWidth * f32InterHeight;
    f64Box1Area = (pstBbox1->f32Xmax - pstBbox1->f32Xmin) * (pstBbox1->f32Ymax - pstBbox1->f32Ymin);
    f64Box2Area = (pstBbox2->f32Xmax - pstBbox2->f32Xmin) * (pstBbox2->f32Ymax - pstBbox2->f32Ymin);
    f64UnionArea = f64Box1Area + f64Box2Area - f64InterArea;

    return f64InterArea / f64UnionArea;
}

static HI_S32 TEST_NNIE_Yolov2_NonMaxSuppression(TEST_NNIE_YOLOV2_BBOX_S *pstBbox,
                                                 HI_U32 u32BboxNum, HI_U32 u32NmsThresh, HI_U32 u32MaxRoiNum)
{
    HI_U32 i, j;
    HI_U32 u32Num = 0;
    HI_DOUBLE f64Iou = 0.0;

    for (i = 0; i < u32BboxNum && u32Num < u32MaxRoiNum; i++)
    {
        if (pstBbox[i].u32Mask == 0)
        {
            u32Num++;
            for (j = i + 1; j < u32BboxNum; j++)
            {
                if (pstBbox[j].u32Mask == 0)
                {
                    f64Iou = TEST_NNIE_Yolov2_Iou(&pstBbox[i], &pstBbox[j]);
                    if (f64Iou >= (HI_DOUBLE)u32NmsThresh / TEST_NNIE_QUANT_BASE)
                    {
                        pstBbox[j].u32Mask = 1;
                    }
                }
            }
        }
    }

    return HI_SUCCESS;
}

HI_U32 TEST_NNIE_Yolov3_GetResultTmpBuf(graph_t graph,
                                        TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 u32TotalSize = 0;
    HI_U32 u32AssistStackSize = 0;
    HI_U32 u32TotalBboxNum = 0;
    HI_U32 u32TotalBboxSize = 0;
    HI_U32 u32DstBlobSize = 0;
    HI_U32 u32MaxBlobSize = 0;
    HI_U32 i = 0;
    node_t outputNode = get_graph_output_node(graph, 0);
    HI_U32 outputNum = get_node_output_number(outputNode);

    for (i = 0; i < outputNum; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, i);
        u32DstBlobSize = get_tensor_buffer_size(output_tensor);
        if (u32MaxBlobSize < u32DstBlobSize)
        {
            u32MaxBlobSize = u32DstBlobSize;
        }
        u32TotalBboxNum += pstSoftwareParam->au32GridNumWidth[i] * pstSoftwareParam->au32GridNumHeight[i] *
                           pstSoftwareParam->u32BboxNumEachGrid;
    }
    u32AssistStackSize = u32TotalBboxNum * sizeof(TEST_NNIE_STACK_S);
    u32TotalBboxSize = u32TotalBboxNum * sizeof(TEST_NNIE_YOLOV3_BBOX_S);
    u32TotalSize += (u32MaxBlobSize + u32AssistStackSize + u32TotalBboxSize);

    return u32TotalSize;
}

static HI_S32 TEST_NNIE_Yolov3_SoftwareInit(graph_t graph, TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[4]; //NCHW
    int dimssize = 4;
    get_tensor_shape(input_tensor, dims, dimssize); //NCHW
    // fprintf(stderr, "input tensor dims[%d:%d:%d:%d]\n", dims[0], dims[1], dims[2], dims[3]);

    pstSoftWareParam->u32OriImHeight = dims[2]; //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = dims[3];  //pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 3;
    pstSoftWareParam->u32ClassNum = 80;
    pstSoftWareParam->au32GridNumHeight[0] = 13;
    pstSoftWareParam->au32GridNumHeight[1] = 26;
    pstSoftWareParam->au32GridNumHeight[2] = 52;
    pstSoftWareParam->au32GridNumWidth[0] = 13;
    pstSoftWareParam->au32GridNumWidth[1] = 26;
    pstSoftWareParam->au32GridNumWidth[2] = 52;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.5f * TEST_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0][0] = 116;
    pstSoftWareParam->af32Bias[0][1] = 90;
    pstSoftWareParam->af32Bias[0][2] = 156;
    pstSoftWareParam->af32Bias[0][3] = 198;
    pstSoftWareParam->af32Bias[0][4] = 373;
    pstSoftWareParam->af32Bias[0][5] = 326;
    pstSoftWareParam->af32Bias[1][0] = 30;
    pstSoftWareParam->af32Bias[1][1] = 61;
    pstSoftWareParam->af32Bias[1][2] = 62;
    pstSoftWareParam->af32Bias[1][3] = 45;
    pstSoftWareParam->af32Bias[1][4] = 59;
    pstSoftWareParam->af32Bias[1][5] = 119;
    pstSoftWareParam->af32Bias[2][0] = 10;
    pstSoftWareParam->af32Bias[2][1] = 13;
    pstSoftWareParam->af32Bias[2][2] = 16;
    pstSoftWareParam->af32Bias[2][3] = 30;
    pstSoftWareParam->af32Bias[2][4] = 33;
    pstSoftWareParam->af32Bias[2][5] = 23;

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;

    u32TmpBufTotalSize = TEST_NNIE_Yolov3_GetResultTmpBuf(graph, pstSoftWareParam);
    u32DstRoiSize = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    u32DstScoreSize = TEST_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    u32ClassRoiNumSize = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = TEST_COMM_MallocCached("SAMPLE_YOLOV3_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                    (void **)&pu8VirAddr, u32TotalSize);
    if (HI_SUCCESS != s32Ret)
        fprintf(stderr, "Error,Malloc memory failed!\n");

    memset(pu8VirAddr, 0, u32TotalSize);
    TEST_COMM_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /*set each tmp buffer addr*/
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);

    /*set result blob*/
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                             pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) * TEST_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
                                                        pstSoftWareParam->u32MaxRoiNum * TEST_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum *
                                                               pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
                                                 u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32TmpBufTotalSize +
                                                          u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = TEST_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

static HI_S32 TEST_NNIE_Yolov3_GetResult(HI_S32 **pps32InputData, HI_U32 au32GridNumWidth[],
                                         HI_U32 au32GridNumHeight[], HI_U32 au32Stride[], HI_U32 u32EachGridBbox, HI_U32 u32ClassNum, HI_U32 u32SrcWidth,
                                         HI_U32 u32SrcHeight, HI_U32 u32MaxRoiNum, HI_U32 u32NmsThresh, HI_U32 u32ConfThresh,
                                         HI_FLOAT af32Bias[TEST_NNIE_YOLOV3_REPORT_BLOB_NUM][TEST_NNIE_YOLOV3_EACH_GRID_BIAS_NUM],
                                         HI_S32 *ps32TmpBuf, HI_S32 *ps32DstScore, HI_S32 *ps32DstRoi, HI_S32 *ps32ClassRoiNum)
{
    HI_S32 *ps32InputBlob = NULL;
    HI_FLOAT *pf32Permute = NULL;
    TEST_NNIE_YOLOV3_BBOX_S *pstBbox = NULL;
    HI_S32 *ps32AssistBuf = NULL;
    HI_U32 u32TotalBboxNum = 0;
    HI_U32 u32ChnOffset = 0;
    HI_U32 u32HeightOffset = 0;
    HI_U32 u32BboxNum = 0;
    HI_U32 u32GridXIdx;
    HI_U32 u32GridYIdx;
    HI_U32 u32Offset;
    HI_FLOAT f32StartX;
    HI_FLOAT f32StartY;
    HI_FLOAT f32Width;
    HI_FLOAT f32Height;
    HI_FLOAT f32ObjScore;
    HI_U32 u32MaxValueIndex = 0;
    HI_FLOAT f32MaxScore;
    HI_S32 s32ClassScore;
    HI_U32 u32ClassRoiNum;
    HI_U32 i = 0, j = 0, k = 0, c = 0, h = 0, w = 0;
    HI_U32 u32BlobSize = 0;
    HI_U32 u32MaxBlobSize = 0;

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        u32BlobSize = au32GridNumWidth[i] * au32GridNumHeight[i] * sizeof(HI_U32) *
                      TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM * u32EachGridBbox;
        if (u32MaxBlobSize < u32BlobSize)
        {
            u32MaxBlobSize = u32BlobSize;
        }
    }

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        u32TotalBboxNum += au32GridNumWidth[i] * au32GridNumHeight[i] * u32EachGridBbox;
    }

    //get each tmpbuf addr
    pf32Permute = (HI_FLOAT *)ps32TmpBuf;
    pstBbox = (TEST_NNIE_YOLOV3_BBOX_S *)(pf32Permute + u32MaxBlobSize / sizeof(HI_S32));
    ps32AssistBuf = (HI_S32 *)(pstBbox + u32TotalBboxNum);

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        //permute
        u32Offset = 0;
        ps32InputBlob = pps32InputData[i];
        u32ChnOffset = au32GridNumHeight[i] * au32GridNumWidth[i];
        u32HeightOffset = au32GridNumWidth[i];

        for (h = 0; h < au32GridNumHeight[i]; h++)
        {
            for (w = 0; w < au32GridNumWidth[i]; w++)
            {
                for (c = 0; c < TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM * u32EachGridBbox; c++)
                {
                    pf32Permute[u32Offset++] = (HI_FLOAT)(ps32InputBlob[c * u32ChnOffset + h * u32HeightOffset + w]) / TEST_NNIE_QUANT_BASE;
                }
            }
        }

        //decode bbox and calculate score
        for (j = 0; j < au32GridNumWidth[i] * au32GridNumHeight[i]; j++)
        {
            u32GridXIdx = j % au32GridNumWidth[i];
            u32GridYIdx = j / au32GridNumWidth[i];
            for (k = 0; k < u32EachGridBbox; k++)
            {
                u32MaxValueIndex = 0;
                u32Offset = (j * u32EachGridBbox + k) * TEST_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM;
                //decode bbox
                float a[4] = {-pf32Permute[u32Offset + 0], -pf32Permute[u32Offset + 1], pf32Permute[u32Offset + 2], pf32Permute[u32Offset + 3]};
                float x[4] = {0.f, 0.f, 0.f, 0.f};
                fast_exp_4f(a, x);
                f32StartX = ((HI_FLOAT)u32GridXIdx + TEST_NNIE_SIGMOID_NOEXP(x[0])) / au32GridNumWidth[i];
                f32StartY = ((HI_FLOAT)u32GridYIdx + TEST_NNIE_SIGMOID_NOEXP(x[1])) / au32GridNumHeight[i];
                f32Width = (HI_FLOAT)((x[2]) * af32Bias[i][2 * k]) / u32SrcWidth;
                f32Height = (HI_FLOAT)((x[3]) * af32Bias[i][2 * k + 1]) / u32SrcHeight;

                //calculate score
                f32ObjScore = TEST_NNIE_SIGMOID(pf32Permute[u32Offset + 4]);
                (void)TEST_NNIE_SoftMax(&pf32Permute[u32Offset + 5], u32ClassNum);
                f32MaxScore = TEST_NNIE_Yolov2_GetMaxVal(&pf32Permute[u32Offset + 5], u32ClassNum, &u32MaxValueIndex);
                s32ClassScore = (HI_S32)(f32MaxScore * f32ObjScore * TEST_NNIE_QUANT_BASE);
                //filter low score roi
                if (s32ClassScore > (HI_S32)u32ConfThresh)
                {
                    pstBbox[u32BboxNum].f32Xmin = (HI_FLOAT)(f32StartX - f32Width * 0.5f);
                    pstBbox[u32BboxNum].f32Ymin = (HI_FLOAT)(f32StartY - f32Height * 0.5f);
                    pstBbox[u32BboxNum].f32Xmax = (HI_FLOAT)(f32StartX + f32Width * 0.5f);
                    pstBbox[u32BboxNum].f32Ymax = (HI_FLOAT)(f32StartY + f32Height * 0.5f);
                    pstBbox[u32BboxNum].s32ClsScore = s32ClassScore;
                    pstBbox[u32BboxNum].u32Mask = 0;
                    pstBbox[u32BboxNum].u32ClassIdx = (HI_S32)(u32MaxValueIndex + 1);
                    u32BboxNum++;
                }
            }
        }
    }

    //quick sort
    (void)TEST_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32 *)pstBbox, 0, u32BboxNum - 1,
                                                  sizeof(TEST_NNIE_YOLOV3_BBOX_S) / sizeof(HI_U32), 4, (TEST_NNIE_STACK_S *)ps32AssistBuf);
    (void)TEST_NNIE_Yolov2_NonMaxSuppression(pstBbox, u32BboxNum, u32NmsThresh, sizeof(TEST_NNIE_YOLOV3_BBOX_S) / sizeof(HI_U32));

    //Get result
    for (i = 1; i < u32ClassNum; i++)
    {
        u32ClassRoiNum = 0;
        for (j = 0; j < u32BboxNum; j++)
        {
            if ((0 == pstBbox[j].u32Mask) && (i == pstBbox[j].u32ClassIdx) && (u32ClassRoiNum < u32MaxRoiNum))
            {
                *(ps32DstRoi++) = TEST_NNIE_MAX((HI_S32)(pstBbox[j].f32Xmin * u32SrcWidth), 0);
                *(ps32DstRoi++) = TEST_NNIE_MAX((HI_S32)(pstBbox[j].f32Ymin * u32SrcHeight), 0);
                *(ps32DstRoi++) = TEST_NNIE_MIN((pstBbox[j].f32Xmax * u32SrcWidth), u32SrcWidth);
                *(ps32DstRoi++) = TEST_NNIE_MIN((pstBbox[j].f32Ymax * u32SrcHeight), u32SrcHeight);
                *(ps32DstScore++) = pstBbox[j].s32ClsScore;
                u32ClassRoiNum++;
            }
        }
        *(ps32ClassRoiNum + i) = u32ClassRoiNum;
    }
    return HI_SUCCESS;
}

HI_S32 TEST_NNIE_Yolov3_GetResult(graph_t graph,
                                  TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftwareParam)
{
    HI_U32 i = 0;
    HI_S32 *aps32InputBlob[TEST_NNIE_YOLOV3_REPORT_BLOB_NUM] = {0};
    HI_U32 au32Stride[TEST_NNIE_YOLOV3_REPORT_BLOB_NUM] = {0};

    for (i = 0; i < TEST_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, i);
        void *output_data = get_tensor_buffer(output_tensor);
        aps32InputBlob[i] = (HI_S32 *)output_data; //pstNnieParam->astSegData[0].astDst[i].u64VirAddr;
        au32Stride[i] = 0;
    }
    return TEST_NNIE_Yolov3_GetResult(aps32InputBlob, pstSoftwareParam->au32GridNumWidth,
                                      pstSoftwareParam->au32GridNumHeight, au32Stride, pstSoftwareParam->u32BboxNumEachGrid,
                                      pstSoftwareParam->u32ClassNum, pstSoftwareParam->u32OriImWidth,
                                      pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32NmsThresh,
                                      pstSoftwareParam->u32ConfThresh, pstSoftwareParam->af32Bias,
                                      (HI_S32 *)pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
                                      (HI_S32 *)pstSoftwareParam->stDstScore.u64VirAddr,
                                      (HI_S32 *)pstSoftwareParam->stDstRoi.u64VirAddr,
                                      (HI_S32 *)pstSoftwareParam->stClassRoiNum.u64VirAddr);
}

static HI_S32 TEST_NNIE_Detection_Yolov3_Result(SVP_BLOB_S *pstDstScore,
                                                     SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32 *ps32Score = (HI_S32 *)pstDstScore->u64VirAddr;
    HI_S32 *ps32Roi = (HI_S32 *)pstDstRoi->u64VirAddr;
    HI_S32 *ps32ClassRoiNum = (HI_S32 *)pstClassRoiNum->u64VirAddr;
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;
    HI_S32 s32XMin = 0, s32YMin = 0, s32XMax = 0, s32YMax = 0;

    u32RoiNumBias += ps32ClassRoiNum[0];

    for (i = 1; i < u32ClassNum; i++)
    {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * TEST_NNIE_COORDI_NUM;
        /*if the confidence score greater than result threshold, the result will be printed*/
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / TEST_NNIE_QUANT_BASE >=
                f32PrintResultThresh &&
            ps32ClassRoiNum[i] != 0)
        {
            fprintf(stderr, "==== The %dth class box info====\n", i);
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / TEST_NNIE_QUANT_BASE;
            if (f32Score < f32PrintResultThresh)
            {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 1];
            s32XMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 2];
            s32YMax = ps32Roi[u32BboxBias + j * TEST_NNIE_COORDI_NUM + 3];
            fprintf(stderr, "%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    return HI_SUCCESS;
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count]\n");
}

int main(int argc, char *argv[])
{
    int repeat_count = 1;
    char* model_file = nullptr;
    char* image_file = nullptr;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, nullptr, 10);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* prepare input data */
    struct stat statbuf;
    stat(image_file, &statbuf);
    int input_length = statbuf.st_size;

    void *input_data = malloc(input_length);
    if (!get_input_data(image_file, input_data, input_length))
        return -1;

    /* create NNIE plugin backend */
    context_t nnie_context = create_context("nnie", 1);
    int rtt = add_context_device(nnie_context, "nnie");
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device HISI NNIE DEVICE failed.\n");
        return -1;
    }	    

    graph_t graph = create_graph(nnie_context, "nnie", model_file, "noconfig");
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* get input tensor */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    /* setup input buffer */
    if (set_tensor_buffer(input_tensor, input_data, input_length) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* run the graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "\nmodel file : %s\n", model_file);
    fprintf(stderr, "image file : %s\n", image_file);
    fprintf(stderr, "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    TEST_NNIE_YOLOV3_SOFTWARE_PARAM_S stSoftWareParam;
    TEST_NNIE_Yolov3_SoftwareInit(graph, &stSoftWareParam);
    TEST_NNIE_Yolov3_GetResult(graph, &stSoftWareParam);

    fprintf(stderr, "print result, this sample has 81 classes:\n");
    fprintf(stderr, "class 0:background      class 1:person       class 2:bicycle         class 3:car            class 4:motorbike      class 5:aeroplane\n");
    fprintf(stderr, "class 6:bus             class 7:train        class 8:truck           class 9:boat           class 10:traffic light\n");
    fprintf(stderr, "class 11:fire hydrant   class 12:stop sign   class 13:parking meter  class 14:bench         class 15:bird\n");
    fprintf(stderr, "class 16:cat            class 17:dog         class 18:horse          class 19:sheep         class 20:cow\n");
    fprintf(stderr, "class 21:elephant       class 22:bear        class 23:zebra          class 24:giraffe       class 25:backpack\n");
    fprintf(stderr, "class 26:umbrella       class 27:handbag     class 28:tie            class 29:suitcase      class 30:frisbee\n");
    fprintf(stderr, "class 31:skis           class 32:snowboard   class 33:sports ball    class 34:kite          class 35:baseball bat\n");
    fprintf(stderr, "class 36:baseball glove class 37:skateboard  class 38:surfboard      class 39:tennis racket class 40bottle\n");
    fprintf(stderr, "class 41:wine glass     class 42:cup         class 43:fork           class 44:knife         class 45:spoon\n");
    fprintf(stderr, "class 46:bowl           class 47:banana      class 48:apple          class 49:sandwich      class 50orange\n");
    fprintf(stderr, "class 51:broccoli       class 52:carrot      class 53:hot dog        class 54:pizza         class 55:donut\n");
    fprintf(stderr, "class 56:cake           class 57:chair       class 58:sofa           class 59:pottedplant   class 60bed\n");
    fprintf(stderr, "class 61:diningtable    class 62:toilet      class 63:vmonitor       class 64:laptop        class 65:mouse\n");
    fprintf(stderr, "class 66:remote         class 67:keyboard    class 68:cell phone     class 69:microwave     class 70:oven\n");
    fprintf(stderr, "class 71:toaster        class 72:sink        class 73:refrigerator   class 74:book          class 75:clock\n");
    fprintf(stderr, "class 76:vase           class 77:scissors    class 78:teddy bear     class 79:hair drier    class 80:toothbrush\n");
    fprintf(stderr, "Yolov3 result:\n");
    HI_FLOAT f32PrintResultThresh = 0.8f;
    TEST_NNIE_Detection_Yolov3_Result(&stSoftWareParam.stDstScore, &stSoftWareParam.stDstRoi, &stSoftWareParam.stClassRoiNum, f32PrintResultThresh);    
    HI_MPI_SYS_MmzFree(stSoftWareParam.stGetResultTmpBuf.u64PhyAddr, (void *)stSoftWareParam.stGetResultTmpBuf.u64VirAddr);

    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
    release_tengine();

    return 0;
}
