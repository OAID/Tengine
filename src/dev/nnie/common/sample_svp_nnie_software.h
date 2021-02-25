#ifndef _SAMPLE_SVP_USER_KERNEL_H_
#define _SAMPLE_SVP_USER_KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "hi_comm_svp.h"
#include "hi_nnie.h"
#include "mpi_nnie.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define SAMPLE_SVP_NNIE_NUM_TO_STR(Num) #Num
#define SAMPLE_SVP_NNIE_ROUND(value) (vx_float32)(((vx_float32)(value) > 0)? floor((vx_float32)(value)+0.5):ceil((vx_float32)(value)-0.5))
#define SAMPLE_SVP_NNIE_FLOOR(value) ((vx_int32)((value) + ((value) >= 0 ? 0.5 : -0.5))) - ((vx_float32)((value)-(vx_int32)((value) + ((value) >= 0 ? 0.5 : -0.5))) < 0)
#define SAMPLE_SVP_NNIE_CEIL(value)  ((vx_int32)((value) + ((value) >= 0 ? 0.5 : -0.5))) + ((vx_float32)((vx_int32)((value) + ((value) >= 0 ? 0.5 : -0.5))-(value)) < 0)
#define SAMPLE_SVP_NNIE_ADDR_ALIGN_16          16  /*16 byte alignment*/
#define SAMPLE_SVP_NNIE_DIM_OF(x) (sizeof(x)/sizeof(x[0]))
#define SAMPLE_SVP_NNIE_MAX(a,b)    (((a) > (b)) ? (a) : (b))
#define SAMPLE_SVP_NNIE_MIN(a,b)    (((a) < (b)) ? (a) : (b))
#define SAMPLE_SVP_NNIE_SIGMOID(x)   (HI_FLOAT)(1.0f/(1+exp(-x)))

#define SAMPLE_SVP_NNIE_KERNEL_LOCAL_SIZE (1024)
#define SAMPLE_SVP_NNIE_COORDI_NUM  4      /*coordinate numbers*/
#define SAMPLE_SVP_NNIE_PROPOSAL_WIDTH  6  /*the number of proposal values*/
#define SAMPLE_SVP_NNIE_QUANT_BASE 4096    /*the base value*/
#define SAMPLE_SVP_NNIE_IFDT_END_FLAG 5000 /*the end flag*/
#define SAMPLE_SVP_NNIE_SCORE_NUM  2       /*the num of RPN scores*/
#define SAMPLE_SVP_NNIE_HALF 0.5f          /*the half value*/
#define SAMPLE_SVP_NNIE_PRINT_RESULT 1     /*the switch of result print*/
#define SAMPLE_SVP_NNIE_MAX_RATIO_ANCHORS_NUM 32
#define SAMPLE_SVP_NNIE_MAX_SCALE_ANCHORS_NUM 32
#define SAMPLE_SVP_NNIE_MAX_VLAUE_OF_MIN_ANCHOR_SIZE 100 /*The maximum value of minimum anchor size*/
#define SAMPLE_SVP_NNIE_MAX_VLAUE_OF_SPATIAL_SCALE 2*4096 /*The maximum value of minimum anchor size*/
#define SAMPLE_SVP_NNIE_MAX_VLAUE_OF_ANCHOR_RATIO  100*4096 /*The maximum value of minimum anchor size*/
#define SAMPLE_SVP_NNIE_MAX_VLAUE_OF_ANCHOR_SCALE  500*4096 /*The maximum value of minimum anchor size*/
#define SAMPLE_SVP_NNIE_USE_MUL_THREAD            1 /*Use multi thread*/
#define SAMPLE_SVP_NNIE_MAX_THREAD_NUM            4 /*Multi thread number*/
#define SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM    3 /*yolov3 report blob num*/
#define SAMPLE_SVP_NNIE_YOLOV3_EACH_GRID_BIAS_NUM 6 /*yolov3 bias num of each grid*/
#define SAMPLE_SVP_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM   85 /*yolov3 inference result num of each bbox*/
/*CNN GetTopN unit*/
typedef struct hiSAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S
{
    HI_U32   u32ClassId;
    HI_U32   u32Confidence;
}SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S;

/*stack for sort*/
typedef struct hiSAMPLE_SVP_NNIE_STACK
{
    HI_S32 s32Min;
    HI_S32 s32Max;
}SAMPLE_SVP_NNIE_STACK_S;

/*stack for sort*/
typedef struct hiSAMPLE_SVP_NNIE_YOLOV1_SCORE
{
    HI_U32 u32Idx;
    HI_S32 s32Score;
}SAMPLE_SVP_NNIE_YOLOV1_SCORE_S;

typedef struct hiSAMPLE_SVP_NNIE_YOLOV2_BBOX
{
    HI_FLOAT f32Xmin;
    HI_FLOAT f32Xmax;
    HI_FLOAT f32Ymin;
    HI_FLOAT f32Ymax;
    HI_S32 s32ClsScore;
    HI_U32 u32ClassIdx;
    HI_U32 u32Mask;
}SAMPLE_SVP_NNIE_YOLOV2_BBOX_S;

typedef SAMPLE_SVP_NNIE_YOLOV2_BBOX_S SAMPLE_SVP_NNIE_YOLOV3_BBOX_S;

/*CNN*/
HI_S32 SAMPLE_SVP_NNIE_Cnn_GetTopN(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstSoftwareParam);

/*FasterRcnn*/
HI_U32 SAMPLE_SVP_NNIE_RpnTmpBufSize(HI_U32 u32RatioAnchorsNum,
    HI_U32 u32ScaleAnchorsNum, HI_U32 u32ConvHeight, HI_U32 u32ConvWidth);

HI_U32 SAMPLE_SVP_NNIE_FasterRcnn_GetResultTmpBufSize(
    HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum);

HI_S32 TNG_NNIE_OP_CPU_Proposal(SAMPLE_SVP_NNIE_PARAM_S* pstNnieParam,
                                      SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_Rpn(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam);


/*Pvanet*/
HI_U32 SAMPLE_SVP_NNIE_Pvanet_GetResultTmpBufSize(
    HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum);

HI_S32 SAMPLE_SVP_NNIE_Pvanet_Rpn(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_Pvanet_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam);

/*RFCN*/
HI_U32 SAMPLE_SVP_NNIE_Rfcn_GetResultTmpBuf(HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum);

HI_S32 SAMPLE_SVP_NNIE_Rfcn_Rpn(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_Rfcn_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S* pstSoftwareParam);

/*SSD*/
HI_U32 SAMPLE_SVP_NNIE_Ssd_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_Ssd_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S* pstSoftwareParam);

/*YOLOV1*/
HI_U32 SAMPLE_SVP_NNIE_Yolov1_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_Yolov1_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S* pstSoftwareParam);

/*YOLOV2*/
HI_U32 SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_Yolov2_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftwareParam);

/*YOLOV3*/
HI_U32 SAMPLE_SVP_NNIE_Yolov3_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftwareParam);

HI_S32 SAMPLE_SVP_NNIE_Yolov3_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftwareParam);
#ifdef __cplusplus
}
#endif

#endif /* _SAMPLE_SVP_USER_KERNEL_H_ */