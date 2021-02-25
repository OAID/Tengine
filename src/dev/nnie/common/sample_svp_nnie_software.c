#include"sample_svp_nnie_software.h"
#include <math.h>

#ifdef __cplusplus    // If used by C++ code,
extern "C" {          // we need to export the C interface
#endif

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
    {1.0f, 1.12535e-007f, 1.26642e-014f, 1.42516e-021f, 1.60381e-028f, 1.80485e-035f, 2.03048e-042f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

/*****************************************************************************
* Prototype :   SVP_NNIE_QuickExp
* Description : this function is used to quickly get exp result
* Input :     HI_S32    s32Value           [IN]   input value
*
*
*
*
* Output :
* Return Value : HI_FLOAT: output value.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_FLOAT SVP_NNIE_QuickExp( HI_S32 s32Value )
{
    if( s32Value & 0x80000000 )
    {
        s32Value = ~s32Value + 0x00000001;
        return s_af32ExpCoef[5][s32Value & 0x0000000F] * s_af32ExpCoef[6][(s32Value>>4) & 0x0000000F] * s_af32ExpCoef[7][(s32Value>>8) & 0x0000000F] * s_af32ExpCoef[8][(s32Value>>12) & 0x0000000F] * s_af32ExpCoef[9][(s32Value>>16) & 0x0000000F ];
    }
    else
    {
        return s_af32ExpCoef[0][s32Value & 0x0000000F] * s_af32ExpCoef[1][(s32Value>>4) & 0x0000000F] * s_af32ExpCoef[2][(s32Value>>8) & 0x0000000F] * s_af32ExpCoef[3][(s32Value>>12) & 0x0000000F] * s_af32ExpCoef[4][(s32Value>>16) & 0x0000000F ];
    }
}

/*****************************************************************************
* Prototype :   SVP_NNIE_SoftMax
* Description : this function is used to do softmax
* Input :     HI_FLOAT*         pf32Src           [IN]   the pointer to source data
*             HI_U32             u32Num           [IN]   the num of source data
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_SoftMax( HI_FLOAT* pf32Src, HI_U32 u32Num)
{
    HI_FLOAT f32Max = 0;
    HI_FLOAT f32Sum = 0;
    HI_U32 i = 0;

    for(i = 0; i < u32Num; ++i)
    {
        if(f32Max < pf32Src[i])
        {
            f32Max = pf32Src[i];
        }
    }

    for(i = 0; i < u32Num; ++i)
    {
        pf32Src[i] = (HI_FLOAT)SVP_NNIE_QuickExp((HI_S32)((pf32Src[i] - f32Max)*SAMPLE_SVP_NNIE_QUANT_BASE));
        f32Sum += pf32Src[i];
    }

    for(i = 0; i < u32Num; ++i)
    {
        pf32Src[i] /= f32Sum;
    }
    return HI_SUCCESS;
}


/*****************************************************************************
* Prototype :   SVP_NNIE_SSD_SoftMax
* Description : this function is used to do softmax for SSD
* Input :       HI_S32*           pf32Src          [IN]   the pointer to input array
*               HI_S32            s32ArraySize     [IN]   the array size
*               HI_S32*           ps32Dst          [OUT]  the pointer to output array
*
*
*
*
* Output :
* Return Value : void
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-03-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_SSD_SoftMax(HI_S32* ps32Src, HI_S32 s32ArraySize, HI_S32* ps32Dst)
{
    /***** define parameters ****/
    HI_S32 s32Max = 0;
    HI_S32 s32Sum = 0;
    HI_S32 i = 0;
    for (i = 0; i < s32ArraySize; ++i)
    {
        if (s32Max < ps32Src[i])
        {
            s32Max = ps32Src[i];
        }
    }
    for (i = 0; i < s32ArraySize; ++i)
    {
        ps32Dst[i] = (HI_S32)(SAMPLE_SVP_NNIE_QUANT_BASE* exp((HI_FLOAT)(ps32Src[i] - s32Max) / SAMPLE_SVP_NNIE_QUANT_BASE));
        s32Sum += ps32Dst[i];
    }
    for (i = 0; i < s32ArraySize; ++i)
    {
        ps32Dst[i] = (HI_S32)(((HI_FLOAT)ps32Dst[i] / (HI_FLOAT)s32Sum) * SAMPLE_SVP_NNIE_QUANT_BASE);
    }
    return HI_SUCCESS;
}


/*****************************************************************************
* Prototype :   SVP_NNIE_Argswap
* Description : this function is used to exchange array data
* Input :       HI_FLOAT*           pf32Src1          [IN]   the pointer to the first array
*               HI_S32*             ps32Src2          [OUT]  the pointer to the second array
*
*
*
*
* Output :
* Return Value : void
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-03-10
* Author :
* Modification : Create
*
*****************************************************************************/
static void SVP_NNIE_Argswap(HI_S32* ps32Src1, HI_S32* ps32Src2)
{
    HI_U32 i = 0;
    HI_S32 u32Tmp = 0;
    for( i = 0; i < SAMPLE_SVP_NNIE_PROPOSAL_WIDTH; i++ )
    {
        u32Tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = u32Tmp;
    }
}


/*****************************************************************************
* Prototype :   SVP_NNIE_NonRecursiveArgQuickSort
* Description : this function is used to do quick sort
* Input :       HI_S32*             ps32Array         [IN]   the array need to be sorted
*               HI_S32              s32Low            [IN]   the start position of quick sort
*               HI_S32              s32High           [IN]   the end position of quick sort
*               SAMPLE_SVP_NNIE_STACK_S *  pstStack   [IN]   the buffer used to store start positions and end positions
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-03-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_NonRecursiveArgQuickSort(HI_S32* ps32Array,
    HI_S32 s32Low, HI_S32 s32High, SAMPLE_SVP_NNIE_STACK_S *pstStack,HI_U32 u32MaxNum)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;
    HI_S32 s32KeyConfidence = ps32Array[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * s32Low + 4];
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    while(s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        s32KeyConfidence = ps32Array[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * s32Low + 4];

        while(i < j)
        {
            while((i < j) && (s32KeyConfidence > ps32Array[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 4]))
            {
                j--;
            }
            if(i < j)
            {
                SVP_NNIE_Argswap(&ps32Array[i*SAMPLE_SVP_NNIE_PROPOSAL_WIDTH], &ps32Array[j*SAMPLE_SVP_NNIE_PROPOSAL_WIDTH]);
                i++;
            }

            while((i < j) && (s32KeyConfidence < ps32Array[i*SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 4]))
            {
                i++;
            }
            if(i < j)
            {
                SVP_NNIE_Argswap(&ps32Array[i*SAMPLE_SVP_NNIE_PROPOSAL_WIDTH], &ps32Array[j*SAMPLE_SVP_NNIE_PROPOSAL_WIDTH]);
                j--;
            }
        }

        if(s32Low <= u32MaxNum)
        {
                if(s32Low < i-1)
                {
                    s32Top++;
                    pstStack[s32Top].s32Min = s32Low;
                    pstStack[s32Top].s32Max = i-1;
                }

                if(s32High > i+1)
                {
                    s32Top++;
                    pstStack[s32Top].s32Min = i+1;
                    pstStack[s32Top].s32Max = s32High;
                }
        }
    }
    return HI_SUCCESS;
}


/*****************************************************************************
* Prototype :   SVP_NNIE_Overlap
* Description : this function is used to calculate the overlap ratio of two proposals
* Input :     HI_S32              s32XMin1          [IN]   first input proposal's minimum value of x coordinate
*               HI_S32              s32YMin1          [IN]   first input proposal's minimum value of y coordinate of first input proposal
*               HI_S32              s32XMax1          [IN]   first input proposal's maximum value of x coordinate of first input proposal
*               HI_S32              s32YMax1          [IN]   first input proposal's maximum value of y coordinate of first input proposal
*               HI_S32              s32XMin1          [IN]   second input proposal's minimum value of x coordinate
*               HI_S32              s32YMin1          [IN]   second input proposal's minimum value of y coordinate of first input proposal
*               HI_S32              s32XMax1          [IN]   second input proposal's maximum value of x coordinate of first input proposal
*               HI_S32              s32YMax1          [IN]   second input proposal's maximum value of y coordinate of first input proposal
*             HI_FLOAT            *pf32IoU          [INOUT]the pointer of the IoU value
*
*
* Output :
* Return Value : HI_FLOAT f32Iou.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-03-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Overlap(HI_S32 s32XMin1, HI_S32 s32YMin1, HI_S32 s32XMax1, HI_S32 s32YMax1, HI_S32 s32XMin2,
    HI_S32 s32YMin2, HI_S32 s32XMax2, HI_S32 s32YMax2,  HI_S32* s32AreaSum, HI_S32* s32AreaInter)
{
    /*** Check the input, and change the Return value  ***/
    HI_S32 s32Inter = 0;
    HI_S32 s32Total = 0;
    HI_S32 s32XMin = 0;
    HI_S32 s32YMin = 0;
    HI_S32 s32XMax = 0;
    HI_S32 s32YMax = 0;
    HI_S32 s32Area1 = 0;
    HI_S32 s32Area2 = 0;
    HI_S32 s32InterWidth = 0;
    HI_S32 s32InterHeight = 0;

    s32XMin = SAMPLE_SVP_NNIE_MAX(s32XMin1, s32XMin2);
    s32YMin = SAMPLE_SVP_NNIE_MAX(s32YMin1, s32YMin2);
    s32XMax = SAMPLE_SVP_NNIE_MIN(s32XMax1, s32XMax2);
    s32YMax = SAMPLE_SVP_NNIE_MIN(s32YMax1, s32YMax2);

    s32InterWidth = s32XMax - s32XMin + 1;
    s32InterHeight = s32YMax - s32YMin + 1;

    s32InterWidth = ( s32InterWidth >= 0 ) ? s32InterWidth : 0;
    s32InterHeight = ( s32InterHeight >= 0 ) ? s32InterHeight : 0;

    s32Inter = s32InterWidth * s32InterHeight;
    s32Area1 = (s32XMax1 - s32XMin1 + 1) * (s32YMax1 - s32YMin1 + 1);
    s32Area2 = (s32XMax2 - s32XMin2 + 1) * (s32YMax2 - s32YMin2 + 1);

    s32Total = s32Area1 + s32Area2 - s32Inter;

    *s32AreaSum = s32Total;
    *s32AreaInter = s32Inter;
    return HI_SUCCESS;
}

/*****************************************************************************
* Prototype :   SVP_NNIE_FilterLowScoreBbox
* Description : this function is used to remove low score bboxes, in order to speed-up Sort & RPN procedures.
* Input :      HI_S32*         ps32Proposals     [IN]   proposals
*              HI_U32          u32NumAnchors     [IN]   input anchors' num
*              HI_U32          u32FilterThresh   [IN]   rpn configuration
*              HI_U32*         u32NumAfterFilter [OUT]  output num of anchors after low score filtering
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-03-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_FilterLowScoreBbox(HI_S32* ps32Proposals, HI_U32 u32AnchorsNum,
    HI_U32 u32FilterThresh, HI_U32* u32NumAfterFilter)
{
    HI_U32 u32ProposalCnt = u32AnchorsNum;
    HI_U32 i = 0;

    if( u32FilterThresh > 0 )
    {
        for( i = 0; i < u32AnchorsNum; i++ )
        {
            if( ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 4 ] < (HI_S32)u32FilterThresh )
            {
                ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 5 ] = 1;
            }
        }

        u32ProposalCnt = 0;
        for( i = 0; i < u32AnchorsNum; i++ )
        {
            if( 0 == ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 5 ] )
            {
                ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32ProposalCnt ] = ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i ];
                ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 1 ] = ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 1 ];
                ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 2 ] = ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 2 ];
                ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 3 ] = ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 3 ];
                ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 4 ] = ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 4 ];
                ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32ProposalCnt + 5 ] = ps32Proposals[ SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i + 5 ];
                u32ProposalCnt++;
            }
        }
    }
    *u32NumAfterFilter = u32ProposalCnt;
    return HI_SUCCESS;
}
/*****************************************************************************
* Prototype :   SVP_NNIE_NonMaxSuppression
* Description : this function is used to do non maximum suppression
* Input :       HI_S32*           ps32Proposals     [IN]   proposals
*               HI_U32            u32AnchorsNum     [IN]   anchors num
*               HI_U32            u32NmsThresh      [IN]   non maximum suppression threshold
*               HI_U32            u32MaxRoiNum      [IN]  The max roi num for the roi pooling
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-03-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_NonMaxSuppression( HI_S32* ps32Proposals, HI_U32 u32AnchorsNum,
    HI_U32 u32NmsThresh,HI_U32 u32MaxRoiNum)
{
    /****** define variables *******/
    HI_S32 s32XMin1 = 0;
    HI_S32 s32YMin1 = 0;
    HI_S32 s32XMax1 = 0;
    HI_S32 s32YMax1 = 0;
    HI_S32 s32XMin2 = 0;
    HI_S32 s32YMin2 = 0;
    HI_S32 s32XMax2 = 0;
    HI_S32 s32YMax2 = 0;
    HI_S32 s32AreaTotal = 0;
    HI_S32 s32AreaInter = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 u32Num = 0;
    HI_BOOL bNoOverlap  = HI_TRUE;
    for (i = 0; i < u32AnchorsNum && u32Num < u32MaxRoiNum; i++)
    {
        if( ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i+5] == 0 )
        {
            u32Num++;
            s32XMin1 =  ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i];
            s32YMin1 =  ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i+1];
            s32XMax1 =  ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i+2];
            s32YMax1 =  ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i+3];
            for(j= i+1;j< u32AnchorsNum; j++)
            {
                if( ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j+5] == 0 )
                {
                    s32XMin2 = ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j];
                    s32YMin2 = ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j+1];
                    s32XMax2 = ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j+2];
                    s32YMax2 = ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j+3];
                    bNoOverlap = (s32XMin2>s32XMax1)||(s32XMax2<s32XMin1)||(s32YMin2>s32YMax1)||(s32YMax2<s32YMin1);
                    if(bNoOverlap)
                    {
                        continue;
                    }
                    (void)SVP_NNIE_Overlap(s32XMin1, s32YMin1, s32XMax1, s32YMax1, s32XMin2, s32YMin2, s32XMax2, s32YMax2, &s32AreaTotal, &s32AreaInter);
                    if(s32AreaInter*SAMPLE_SVP_NNIE_QUANT_BASE > ((HI_S32)u32NmsThresh*s32AreaTotal))
                    {
                        if( ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i+4] >= ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j+4] )
                        {
                            ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*j+5] = 1;
                        }
                        else
                        {
                            ps32Proposals[SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*i+5] = 1;
                        }
                    }
                }
            }
        }
    }
    return HI_SUCCESS;
}

/*****************************************************************************
*   Prototype    : SVP_NNIE_Cnn_GetTopN
*   Description  : Cnn get top N
*   Input        : HI_S32   *ps32Fc       [IN]  FC data pointer
*                  HI_U32   u32FcStride   [IN]  FC stride
*                  HI_U32   u32ClassNum   [IN]  Class Num
*                  HI_U32   u32BatchNum   [IN]  Batch Num
*                  HI_U32   u32TopN       [IN]  TopN
*                  HI_S32   *ps32TmpBuf   [IN]  assist buffer pointer
*                  HI_U32   u32TopNStride [IN]  TopN result stride
*                  HI_S32   *ps32GetTopN  [OUT]  TopN result
*
*   Output       :
*   Return Value : HI_S32
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-03-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Cnn_GetTopN(HI_S32 *ps32Fc, HI_U32 u32FcStride,
    HI_U32 u32ClassNum,HI_U32 u32BatchNum, HI_U32 u32TopN, HI_S32 *ps32TmpBuf,
    HI_U32 u32TopNStride,HI_S32*ps32GetTopN)
{
    HI_U32 i = 0, j = 0, n = 0;
    HI_U32 u32Id = 0;
    HI_S32* ps32Score = NULL;
    SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S stTmp = {0};
    SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S *pstTopN = NULL;
    SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S *pstTmpBuf = (SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S*)ps32TmpBuf;
    for(n = 0; n < u32BatchNum; n++)
    {
        ps32Score = (HI_S32 *)((HI_U8*)ps32Fc + n * u32FcStride);
        pstTopN = (SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S*)((HI_U8*)ps32GetTopN + n * u32TopNStride);
        for(i = 0; i < u32ClassNum; i++)
        {
            pstTmpBuf[i].u32ClassId = i;
            pstTmpBuf[i].u32Confidence = (HI_U32)ps32Score[i];
        }

        for(i = 0; i < u32TopN; i++)
        {
            u32Id = i;
            pstTopN[i].u32ClassId   = pstTmpBuf[i].u32ClassId;
            pstTopN[i].u32Confidence = pstTmpBuf[i].u32Confidence;
            for(j = i+1; j < u32ClassNum; j++)
            {
                if(pstTmpBuf[u32Id].u32Confidence < pstTmpBuf[j].u32Confidence)
                {
                    u32Id = j;
                }
            }

            stTmp.u32ClassId    = pstTmpBuf[u32Id].u32ClassId;
            stTmp.u32Confidence = pstTmpBuf[u32Id].u32Confidence;

            if(i!=u32Id)
            {
                pstTmpBuf[u32Id].u32ClassId   = pstTmpBuf[i].u32ClassId;
                pstTmpBuf[u32Id].u32Confidence = pstTmpBuf[i].u32Confidence;
                pstTmpBuf[i].u32ClassId   = stTmp.u32ClassId;
                pstTmpBuf[i].u32Confidence = stTmp.u32Confidence;

                pstTopN[i].u32ClassId   = stTmp.u32ClassId;
                pstTopN[i].u32Confidence = stTmp.u32Confidence;
            }
        }
    }

    return HI_SUCCESS;
}


/*****************************************************************************
* Prototype :   SVP_NNIE_Rpn
* Description : this function is used to do RPN
* Input :     HI_S32** pps32Src              [IN] convolution data
*             HI_U32 u32NumRatioAnchors      [IN] Ratio anchor num
*             HI_U32 u32NumScaleAnchors      [IN] scale anchor num
*             HI_U32* au32Scales             [IN] scale value
*             HI_U32* au32Ratios             [IN] ratio value
*             HI_U32 u32OriImHeight          [IN] input image height
*             HI_U32 u32OriImWidth           [IN] input image width
*             HI_U32* pu32ConvHeight         [IN] convolution height
*             HI_U32* pu32ConvWidth          [IN] convolution width
*             HI_U32* pu32ConvChannel        [IN] convolution channel
*             HI_U32  u32ConvStride          [IN] convolution stride
*             HI_U32 u32MaxRois              [IN] max roi num
*             HI_U32 u32MinSize              [IN] min size
*             HI_U32 u32SpatialScale         [IN] spatial scale
*             HI_U32 u32NmsThresh            [IN] NMS thresh
*             HI_U32 u32FilterThresh         [IN] filter thresh
*             HI_U32 u32NumBeforeNms         [IN] num before doing NMS
*             HI_U32 *pu32MemPool            [IN] assist buffer
*             HI_S32 *ps32ProposalResult     [OUT] proposal result
*             HI_U32* pu32NumRois            [OUT] proposal num
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Rpn(HI_S32** pps32Src,HI_U32 u32NumRatioAnchors,
    HI_U32 u32NumScaleAnchors,HI_U32* au32Scales,HI_U32* au32Ratios,HI_U32 u32OriImHeight,
    HI_U32 u32OriImWidth,HI_U32* pu32ConvHeight,HI_U32* pu32ConvWidth,HI_U32* pu32ConvChannel,
    HI_U32  u32ConvStride,HI_U32 u32MaxRois,HI_U32 u32MinSize,HI_U32 u32SpatialScale,
    HI_U32 u32NmsThresh,HI_U32 u32FilterThresh,HI_U32 u32NumBeforeNms,HI_U32 *pu32MemPool,
    HI_S32 *ps32ProposalResult,HI_U32* pu32NumRois)
{
    /******************** define parameters ****************/
    HI_U32 u32Size = 0;
    HI_S32* ps32Anchors = NULL;
    HI_S32* ps32BboxDelta = NULL;
    HI_S32* ps32Proposals = NULL;
    HI_U32* pu32Ptr = NULL;
    HI_S32* ps32Ptr = NULL;
    HI_U32 u32NumAfterFilter = 0;
    HI_U32 u32NumAnchors = 0;
    HI_FLOAT f32BaseW = 0;
    HI_FLOAT f32BaseH = 0;
    HI_FLOAT f32BaseXCtr = 0;
    HI_FLOAT f32BaseYCtr = 0;
    HI_FLOAT f32SizeRatios = 0;
    HI_FLOAT* pf32RatioAnchors = NULL;
    HI_FLOAT* pf32Ptr = NULL;
    HI_FLOAT *pf32Ptr2 = NULL;
    HI_FLOAT* pf32ScaleAnchors = NULL;
    HI_FLOAT* pf32Scores = NULL;
    HI_FLOAT f32Ratios = 0;
    HI_FLOAT f32Size = 0;
    HI_U32 u32PixelInterval = 0;
    HI_U32 u32SrcBboxIndex = 0;
    HI_U32 u32SrcFgProbIndex = 0;
    HI_U32 u32SrcBgProbIndex = 0;
    HI_U32 u32SrcBboxBias = 0;
    HI_U32 u32SrcProbBias = 0;
    HI_U32 u32DesBox = 0;
    HI_U32 u32BgBlobSize = 0;
    HI_U32 u32AnchorsPerPixel = 0;
    HI_U32 u32MapSize = 0;
    HI_U32 u32LineSize = 0;
    HI_S32* ps32Ptr2 = NULL;
    HI_S32* ps32Ptr3 = NULL;
    HI_S32 s32ProposalWidth = 0;
    HI_S32 s32ProposalHeight = 0;
    HI_S32 s32ProposalCenterX = 0;
    HI_S32 s32ProposalCenterY = 0;
    HI_S32 s32PredW = 0;
    HI_S32 s32PredH = 0;
    HI_S32 s32PredCenterX = 0;
    HI_S32 s32PredCenterY = 0;
    HI_U32 u32DesBboxDeltaIndex = 0;
    HI_U32 u32DesScoreIndex = 0;
    HI_U32 u32RoiCount = 0;
    SAMPLE_SVP_NNIE_STACK_S* pstStack = NULL;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 c = 0;
    HI_U32 h = 0;
    HI_U32 w = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 p = 0;
    HI_U32 q = 0;
    HI_U32 z = 0;
    HI_U32 au32BaseAnchor[4] = {0, 0, (u32MinSize -1), (u32MinSize -1)};

    /*********************************** Faster RCNN *********************************************/
    /********* calculate the start pointer of each part in MemPool *********/
    pu32Ptr = (HI_U32*)pu32MemPool;
    ps32Anchors = (HI_S32*)pu32Ptr;
    u32NumAnchors = u32NumRatioAnchors * u32NumScaleAnchors * ( pu32ConvHeight[0] * pu32ConvWidth[0] );
    u32Size = SAMPLE_SVP_NNIE_COORDI_NUM * u32NumAnchors;
    pu32Ptr += u32Size;

    ps32BboxDelta = (HI_S32*)pu32Ptr;
    pu32Ptr += u32Size;

    ps32Proposals = (HI_S32*)pu32Ptr;
    u32Size = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32NumAnchors;
    pu32Ptr += u32Size;

    pf32RatioAnchors = (HI_FLOAT*)pu32Ptr;
    pf32Ptr = (HI_FLOAT*)pu32Ptr;
    u32Size = u32NumRatioAnchors * SAMPLE_SVP_NNIE_COORDI_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pf32ScaleAnchors = pf32Ptr;
    u32Size = u32NumScaleAnchors * u32NumRatioAnchors * SAMPLE_SVP_NNIE_COORDI_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pf32Scores = pf32Ptr;
    u32Size = u32NumAnchors * SAMPLE_SVP_NNIE_SCORE_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pstStack = (SAMPLE_SVP_NNIE_STACK_S*)pf32Ptr;

    /********************* Generate the base anchor ***********************/
    f32BaseW = (HI_FLOAT)(au32BaseAnchor[2] - au32BaseAnchor[0] + 1 );
    f32BaseH = (HI_FLOAT)(au32BaseAnchor[3] - au32BaseAnchor[1] + 1 );
    f32BaseXCtr = (HI_FLOAT)(au32BaseAnchor[0] + ( ( f32BaseW - 1 ) * 0.5 ) );
    f32BaseYCtr = (HI_FLOAT)(au32BaseAnchor[1] + ( ( f32BaseH - 1 ) * 0.5 ) );

    /*************** Generate Ratio Anchors for the base anchor ***********/
    pf32Ptr = pf32RatioAnchors;
    f32Size = f32BaseW * f32BaseH;
    for (i = 0; i < u32NumRatioAnchors; i++)
    {
        f32Ratios = (HI_FLOAT)au32Ratios[i]/SAMPLE_SVP_NNIE_QUANT_BASE;
        f32SizeRatios = f32Size / f32Ratios;
        f32BaseW = sqrt(f32SizeRatios);
        f32BaseW = (HI_FLOAT)(1.0 * ( (f32BaseW) >= 0 ? (HI_S32)(f32BaseW+SAMPLE_SVP_NNIE_HALF) : (HI_S32)(f32BaseW-SAMPLE_SVP_NNIE_HALF)));
        f32BaseH = f32BaseW * f32Ratios;
        f32BaseH = (HI_FLOAT)(1.0 * ( (f32BaseH) >= 0 ? (HI_S32)(f32BaseH+SAMPLE_SVP_NNIE_HALF) : (HI_S32)(f32BaseH-SAMPLE_SVP_NNIE_HALF)));

        *pf32Ptr++ = (HI_FLOAT)(f32BaseXCtr - ( ( f32BaseW - 1 ) * SAMPLE_SVP_NNIE_HALF ));
        *(pf32Ptr++) = (HI_FLOAT)(f32BaseYCtr - ( ( f32BaseH - 1 ) * SAMPLE_SVP_NNIE_HALF ));
        *(pf32Ptr++) = (HI_FLOAT)(f32BaseXCtr + ( ( f32BaseW - 1 ) * SAMPLE_SVP_NNIE_HALF ));
        *(pf32Ptr++) =  (HI_FLOAT)( f32BaseYCtr + ( ( f32BaseH - 1 ) * SAMPLE_SVP_NNIE_HALF ));
    }

    /********* Generate Scale Anchors for each Ratio Anchor **********/
    pf32Ptr = pf32RatioAnchors;
    pf32Ptr2 = pf32ScaleAnchors;
    /* Generate Scale Anchors for one pixel */
    for( i = 0; i < u32NumRatioAnchors; i++ )
    {
        for( j = 0; j < u32NumScaleAnchors; j++ )
        {
            f32BaseW = *( pf32Ptr + 2 ) - *( pf32Ptr ) + 1;
            f32BaseH = *( pf32Ptr + 3 ) - *( pf32Ptr + 1 ) + 1;
            f32BaseXCtr = (HI_FLOAT)( *( pf32Ptr ) + ( ( f32BaseW - 1 ) * SAMPLE_SVP_NNIE_HALF ));
            f32BaseYCtr = (HI_FLOAT)( *( pf32Ptr + 1 ) + ( ( f32BaseH - 1 ) * SAMPLE_SVP_NNIE_HALF ));

            *( pf32Ptr2++ ) = (HI_FLOAT) (f32BaseXCtr - ((f32BaseW * ((HI_FLOAT)au32Scales[j]/SAMPLE_SVP_NNIE_QUANT_BASE) - 1) * SAMPLE_SVP_NNIE_HALF));
            *( pf32Ptr2++ ) = (HI_FLOAT)(f32BaseYCtr - ((f32BaseH * ((HI_FLOAT)au32Scales[j]/SAMPLE_SVP_NNIE_QUANT_BASE) - 1) * SAMPLE_SVP_NNIE_HALF));
            *( pf32Ptr2++ ) = (HI_FLOAT)(f32BaseXCtr + ((f32BaseW * ((HI_FLOAT)au32Scales[j]/SAMPLE_SVP_NNIE_QUANT_BASE) - 1) * SAMPLE_SVP_NNIE_HALF));
            *( pf32Ptr2++ ) = (HI_FLOAT)(f32BaseYCtr + ((f32BaseH * ((HI_FLOAT)au32Scales[j]/SAMPLE_SVP_NNIE_QUANT_BASE) - 1) * SAMPLE_SVP_NNIE_HALF));
        }
            pf32Ptr += SAMPLE_SVP_NNIE_COORDI_NUM;
    }


    /******************* Copy the anchors to every pixel in the feature map ******************/
    ps32Ptr = ps32Anchors;
    u32PixelInterval = SAMPLE_SVP_NNIE_QUANT_BASE/ u32SpatialScale;

    for ( p = 0; p < pu32ConvHeight[0]; p++  )
    {
        for ( q = 0; q < pu32ConvWidth[0]; q++ )
        {
            pf32Ptr2 = pf32ScaleAnchors;
            for ( z = 0 ; z < u32NumScaleAnchors * u32NumRatioAnchors; z++ )
            {
                *(ps32Ptr++) = (HI_S32)(q * u32PixelInterval + *(pf32Ptr2++) );
                *(ps32Ptr++) = (HI_S32)( p * u32PixelInterval + *( pf32Ptr2++ ));
                *(ps32Ptr++) = (HI_S32)( q * u32PixelInterval + *( pf32Ptr2++ ));
                *(ps32Ptr++) = (HI_S32)( p * u32PixelInterval + *( pf32Ptr2++ ));
            }
        }
    }

    /********** do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) **********/
    u32MapSize = pu32ConvHeight[1] * u32ConvStride / sizeof(HI_U32);
    u32AnchorsPerPixel = u32NumRatioAnchors * u32NumScaleAnchors;
    u32BgBlobSize = u32AnchorsPerPixel * u32MapSize;
    u32LineSize = u32ConvStride / sizeof(HI_U32);
    u32SrcProbBias = 0;
    u32SrcBboxBias = 0;

    for ( c = 0; c < pu32ConvChannel[1]; c++ )
    {
        for ( h = 0; h < pu32ConvHeight[1]; h++ )
        {
            for ( w = 0; w < pu32ConvWidth[1]; w++ )
            {
                u32SrcBboxIndex = u32SrcBboxBias + c * u32MapSize + h * u32LineSize + w;
                u32SrcBgProbIndex = u32SrcProbBias + (c/SAMPLE_SVP_NNIE_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcFgProbIndex = u32BgBlobSize + u32SrcBgProbIndex;

                u32DesBox = ( u32AnchorsPerPixel ) * ( h * pu32ConvWidth[1] + w) + c/SAMPLE_SVP_NNIE_COORDI_NUM ;

                u32DesBboxDeltaIndex = SAMPLE_SVP_NNIE_COORDI_NUM * u32DesBox + c % SAMPLE_SVP_NNIE_COORDI_NUM;
                ps32BboxDelta[u32DesBboxDeltaIndex] = (HI_S32)pps32Src[1][u32SrcBboxIndex];

                u32DesScoreIndex = ( SAMPLE_SVP_NNIE_SCORE_NUM ) * u32DesBox;
                pf32Scores[u32DesScoreIndex] = (HI_FLOAT)((HI_S32)pps32Src[0][u32SrcBgProbIndex]) / SAMPLE_SVP_NNIE_QUANT_BASE;
                pf32Scores[u32DesScoreIndex + 1] = (HI_FLOAT)((HI_S32)pps32Src[0][u32SrcFgProbIndex]) / SAMPLE_SVP_NNIE_QUANT_BASE;
            }
        }
    }


    /************************* do softmax ****************************/
    pf32Ptr = pf32Scores;
    for( i = 0; i<u32NumAnchors; i++ )
    {
        s32Ret = SVP_NNIE_SoftMax(pf32Ptr, SAMPLE_SVP_NNIE_SCORE_NUM);
        pf32Ptr += SAMPLE_SVP_NNIE_SCORE_NUM;
    }


    /************************* BBox Transform *****************************/
    /* use parameters from Conv3 to adjust the coordinates of anchors */
    ps32Ptr = ps32Anchors;
    ps32Ptr2 = ps32Proposals;
    ps32Ptr3 = ps32BboxDelta;
    for( i = 0; i < u32NumAnchors; i++)
    {
        ps32Ptr = ps32Anchors;
        ps32Ptr = ps32Ptr + SAMPLE_SVP_NNIE_COORDI_NUM * i;
        ps32Ptr2 = ps32Proposals;
        ps32Ptr2 = ps32Ptr2 + SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
        ps32Ptr3 = ps32BboxDelta;
        ps32Ptr3 = ps32Ptr3 + SAMPLE_SVP_NNIE_COORDI_NUM * i;
        pf32Ptr = pf32Scores;
        pf32Ptr = pf32Ptr + i* ( SAMPLE_SVP_NNIE_SCORE_NUM );

        s32ProposalWidth = *(ps32Ptr+2) - *(ps32Ptr) + 1;
        s32ProposalHeight = *(ps32Ptr+3) - *(ps32Ptr+1) + 1;
        s32ProposalCenterX = *(ps32Ptr) + (HI_S32)( s32ProposalWidth * SAMPLE_SVP_NNIE_HALF );
        s32ProposalCenterY = *(ps32Ptr+1) + (HI_S32)( s32ProposalHeight * SAMPLE_SVP_NNIE_HALF);
        s32PredCenterX = (HI_S32)( ((HI_FLOAT)(*(ps32Ptr3))/SAMPLE_SVP_NNIE_QUANT_BASE) * s32ProposalWidth + s32ProposalCenterX );
        s32PredCenterY = (HI_S32)( ((HI_FLOAT)(*(ps32Ptr3 + 1))/SAMPLE_SVP_NNIE_QUANT_BASE) * s32ProposalHeight + s32ProposalCenterY);

        s32PredW = (HI_S32)(s32ProposalWidth * SVP_NNIE_QuickExp((HI_S32)( *(ps32Ptr3+2) )));
        s32PredH = (HI_S32)(s32ProposalHeight * SVP_NNIE_QuickExp((HI_S32)( *(ps32Ptr3+3) )));
        *(ps32Ptr2) = (HI_S32)( s32PredCenterX - SAMPLE_SVP_NNIE_HALF * s32PredW);
        *(ps32Ptr2+1) = (HI_S32)( s32PredCenterY - SAMPLE_SVP_NNIE_HALF * s32PredH );
        *(ps32Ptr2+2) = (HI_S32)( s32PredCenterX + SAMPLE_SVP_NNIE_HALF * s32PredW );
        *(ps32Ptr2+3) = (HI_S32)( s32PredCenterY + SAMPLE_SVP_NNIE_HALF * s32PredH );
        *(ps32Ptr2+4) = (HI_S32)(*(pf32Ptr+1) * SAMPLE_SVP_NNIE_QUANT_BASE);
        *(ps32Ptr2+5) = 0;
    }


    /************************ clip bbox *****************************/
    for( i = 0; i < u32NumAnchors; i++)
    {
        ps32Ptr = ps32Proposals;
        ps32Ptr = ps32Ptr + SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
        *ps32Ptr = SAMPLE_SVP_NNIE_MAX(SAMPLE_SVP_NNIE_MIN(*ps32Ptr, (HI_S32)u32OriImWidth-1),0 );
        *(ps32Ptr+1) = SAMPLE_SVP_NNIE_MAX(SAMPLE_SVP_NNIE_MIN(*(ps32Ptr+1), (HI_S32)u32OriImHeight-1),0 );
        *(ps32Ptr+2) = SAMPLE_SVP_NNIE_MAX(SAMPLE_SVP_NNIE_MIN(*(ps32Ptr+2), (HI_S32)u32OriImWidth-1),0 );
        *(ps32Ptr+3) = SAMPLE_SVP_NNIE_MAX(SAMPLE_SVP_NNIE_MIN(*(ps32Ptr+3), (HI_S32)u32OriImHeight-1),0 );
    }

    /************ remove the bboxes which are too small *************/
    for( i = 0; i< u32NumAnchors; i++)
    {
        ps32Ptr = ps32Proposals;
        ps32Ptr = ps32Ptr + SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
        s32ProposalWidth = *(ps32Ptr+2)-*(ps32Ptr) + 1;
        s32ProposalHeight = *(ps32Ptr+3)-*(ps32Ptr+1) + 1;
        if(s32ProposalWidth < (HI_S32)u32MinSize  || s32ProposalHeight < (HI_S32)u32MinSize  )
        {
            *(ps32Ptr+5) = 1;
        }
    }

    /********** remove low score bboxes ************/
    (void)SVP_NNIE_FilterLowScoreBbox( ps32Proposals, u32NumAnchors, u32FilterThresh, &u32NumAfterFilter );


    /********** sort ***********/
    (void)SVP_NNIE_NonRecursiveArgQuickSort( ps32Proposals, 0, u32NumAfterFilter - 1, pstStack ,u32NumBeforeNms);
    u32NumAfterFilter = ( u32NumAfterFilter < u32NumBeforeNms ) ? u32NumAfterFilter : u32NumBeforeNms;

    /* do nms to remove highly overlapped bbox */
    (void)SVP_NNIE_NonMaxSuppression(ps32Proposals, u32NumAfterFilter, u32NmsThresh, u32MaxRois);     /* function NMS */

    /************** write the final result to output ***************/
    u32RoiCount = 0;
    for( i = 0; i < u32NumAfterFilter; i++)
    {
        ps32Ptr = ps32Proposals;
        ps32Ptr = ps32Ptr + SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
        if( *(ps32Ptr+5) == 0)
        {
            /*In this sample,the output Roi coordinates will be input in hardware,
            so the type coordinates are convert to HI_S20Q12*/
            ps32ProposalResult[SAMPLE_SVP_NNIE_COORDI_NUM * u32RoiCount] = *ps32Ptr*SAMPLE_SVP_NNIE_QUANT_BASE;
            ps32ProposalResult[SAMPLE_SVP_NNIE_COORDI_NUM * u32RoiCount + 1] = *(ps32Ptr+1)*SAMPLE_SVP_NNIE_QUANT_BASE;
            ps32ProposalResult[SAMPLE_SVP_NNIE_COORDI_NUM * u32RoiCount + 2] = *(ps32Ptr+2)*SAMPLE_SVP_NNIE_QUANT_BASE;
            ps32ProposalResult[SAMPLE_SVP_NNIE_COORDI_NUM * u32RoiCount + 3] = *(ps32Ptr+3)*SAMPLE_SVP_NNIE_QUANT_BASE;
            u32RoiCount++;
        }
        if(u32RoiCount >= u32MaxRois)
        {
            break;
        }
    }

    *pu32NumRois = u32RoiCount;

    return s32Ret;

}

/*****************************************************************************
* Prototype :   SVP_NNIE_FasterRcnn_GetResult
* Description : this function is used to get FasterRcnn result
* Input :     HI_S32* ps32FcBbox             [IN] Bbox for Roi
*             HI_S32 *ps32FcScore            [IN] Score for roi
*             HI_S32 *ps32Proposals          [IN] proposal
*             HI_U32 u32RoiCnt               [IN] Roi num
*             HI_U32 *pu32ConfThresh         [IN] each class confidence thresh
*             HI_U32 u32NmsThresh            [IN] Nms thresh
*             HI_U32 u32MaxRoi               [IN] max roi
*             HI_U32 u32ClassNum             [IN] class num
*             HI_U32 u32OriImWidth           [IN] input image width
*             HI_U32 u32OriImHeight          [IN] input image height
*             HI_U32* pu32MemPool            [IN] assist buffer
*             HI_S32* ps32DstScore           [OUT] result of score
*             HI_S32* ps32DstRoi             [OUT] result of Bbox
*             HI_S32* ps32ClassRoiNum        [OUT] result of the roi num of each classs
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_FasterRcnn_GetResult(HI_S32*ps32FcBbox,HI_U32 u32BboxStride,
    HI_S32*ps32FcScore, HI_U32 u32ScoreStride,HI_S32* ps32Proposal,HI_U32 u32RoiCnt,
    HI_U32* pu32ConfThresh,HI_U32 u32NmsThresh,HI_U32 u32MaxRoi,HI_U32 u32ClassNum,
    HI_U32 u32OriImWidth,HI_U32 u32OriImHeight,HI_U32* pu32MemPool,HI_S32* ps32DstScore,
    HI_S32* ps32DstBbox,HI_S32* ps32ClassRoiNum)
{
    /************* define variables *****************/
    HI_U32 u32Size = 0;
    HI_U32 u32ClsScoreChannels = 0;
    HI_S32* ps32Proposals = NULL;
    HI_U32 u32FcScoreWidth = 0;
    HI_U32 u32FcBboxWidth = 0;
    HI_FLOAT f32ProposalWidth = 0.0;
    HI_FLOAT f32ProposalHeight = 0.0;
    HI_FLOAT f32ProposalCenterX = 0.0;
    HI_FLOAT f32ProposalCenterY = 0.0;
    HI_FLOAT f32PredW = 0.0;
    HI_FLOAT f32PredH = 0.0;
    HI_FLOAT f32PredCenterX = 0.0;
    HI_FLOAT f32PredCenterY = 0.0;
    HI_FLOAT* pf32FcScoresMemPool = NULL;
    HI_S32* ps32ProposalMemPool = NULL;
    HI_S32* ps32ProposalTmp = NULL;
    HI_U32 u32FcBboxIndex = 0;
    HI_U32 u32ProposalMemPoolIndex = 0;
    HI_FLOAT* pf32Ptr = NULL;
    HI_S32* ps32Ptr = NULL;
    HI_S32* ps32Score = NULL;
    HI_S32* ps32Bbox = NULL;
    HI_S32* ps32RoiCnt = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_U32 u32SrcIndex = 0;
    HI_U32 u32DstIndex = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 k = 0;
    SAMPLE_SVP_NNIE_STACK_S* pstStack=NULL;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32OffSet = 0;

  /******************* Get or calculate parameters **********************/
    u32ClsScoreChannels = u32ClassNum;   /*channel num is equal to class size, cls_score class*/
    u32FcScoreWidth = u32ScoreStride / sizeof(HI_U32);
    u32FcBboxWidth = u32BboxStride / sizeof(HI_U32);

    /*************** Get Start Pointer of MemPool ******************/
    pf32FcScoresMemPool = (HI_FLOAT*)pu32MemPool;
    pf32Ptr = pf32FcScoresMemPool;
    u32Size = u32MaxRoi * u32ClsScoreChannels;
    pf32Ptr += u32Size;

    ps32ProposalMemPool = (HI_S32*)pf32Ptr;
    ps32Ptr = ps32ProposalMemPool;
    u32Size = u32MaxRoi * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH ;
    ps32Ptr += u32Size;
    pstStack = (SAMPLE_SVP_NNIE_STACK_S* )ps32Ptr;

    u32DstIndex = 0;

    for( i = 0; i < u32RoiCnt; i++ )
    {
        for( k = 0; k < u32ClsScoreChannels; k++ )
        {
            u32SrcIndex = i * u32FcScoreWidth + k;
            pf32FcScoresMemPool[u32DstIndex++] = (HI_FLOAT)((HI_S32)ps32FcScore[u32SrcIndex]) / SAMPLE_SVP_NNIE_QUANT_BASE;
        }
    }
    ps32Proposals = (HI_S32*)ps32Proposal;

    /************** bbox tranform ************/
    for(j = 0; j < u32ClsScoreChannels; j++)
    {
        for(i = 0; i < u32RoiCnt; i++)
        {
            f32ProposalWidth   = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 2] - ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i] + 1);
            f32ProposalHeight  = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 3] - ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 1] + 1);
            f32ProposalCenterX = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i] + SAMPLE_SVP_NNIE_HALF * f32ProposalWidth);
            f32ProposalCenterY = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 1] + SAMPLE_SVP_NNIE_HALF * f32ProposalHeight);

            u32FcBboxIndex = u32FcBboxWidth * i + SAMPLE_SVP_NNIE_COORDI_NUM * j;
            f32PredCenterX = ((HI_FLOAT)ps32FcBbox[u32FcBboxIndex]/SAMPLE_SVP_NNIE_QUANT_BASE) * f32ProposalWidth  + f32ProposalCenterX;
            f32PredCenterY = ((HI_FLOAT)ps32FcBbox[u32FcBboxIndex + 1]/SAMPLE_SVP_NNIE_QUANT_BASE) * f32ProposalHeight + f32ProposalCenterY;
            f32PredW = f32ProposalWidth * SVP_NNIE_QuickExp((HI_S32)( ps32FcBbox[u32FcBboxIndex+2] ));
            f32PredH = f32ProposalHeight * SVP_NNIE_QuickExp((HI_S32)( ps32FcBbox[u32FcBboxIndex+3] ));

            u32ProposalMemPoolIndex =  SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = (HI_S32)(f32PredCenterX - SAMPLE_SVP_NNIE_HALF * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = (HI_S32)(f32PredCenterY - SAMPLE_SVP_NNIE_HALF * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = (HI_S32)(f32PredCenterX + SAMPLE_SVP_NNIE_HALF * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = (HI_S32)(f32PredCenterY + SAMPLE_SVP_NNIE_HALF * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] = (HI_S32)( pf32FcScoresMemPool[u32ClsScoreChannels*i+j] * SAMPLE_SVP_NNIE_QUANT_BASE );
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] = 0;   /* suprressed flag */

        }

        /* clip bbox */
       for(i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex])>((HI_S32)u32OriImWidth)? (u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex] ) ):0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1])>((HI_S32)u32OriImHeight)? (u32OriImHeight - 1):(ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] ) ):0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2])>((HI_S32)u32OriImWidth)? (u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] ) ):0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3])>((HI_S32)u32OriImHeight)? (u32OriImHeight - 1):(ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] ) ):0;
       }

        ps32ProposalTmp = ps32ProposalMemPool;

        (void)SVP_NNIE_NonRecursiveArgQuickSort( ps32ProposalTmp, 0, u32RoiCnt-1, pstStack,u32RoiCnt);

        (void)SVP_NNIE_NonMaxSuppression(ps32ProposalTmp, u32RoiCnt, u32NmsThresh, u32RoiCnt);

        ps32Score = (HI_S32*)ps32DstScore;
        ps32Bbox = (HI_S32*)ps32DstBbox;
        ps32RoiCnt = (HI_S32*)ps32ClassRoiNum;

        ps32Score += (HI_S32)(u32OffSet);
        ps32Bbox += (HI_S32)(SAMPLE_SVP_NNIE_COORDI_NUM * u32OffSet);

        u32RoiOutCnt = 0;
        for(i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            if( 0 == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5]  && ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] > (HI_S32)pu32ConfThresh[j] ) //Suppression = 0; CONF_THRESH == 0.8
            {
                ps32Score[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM ] = ps32ProposalMemPool[u32ProposalMemPoolIndex];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 1 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 2 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 3 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];
                u32RoiOutCnt++;
            }
            if(u32RoiOutCnt >= u32RoiCnt)break;
        }
        ps32RoiCnt[j] = (HI_S32)u32RoiOutCnt;
        u32OffSet += u32RoiOutCnt;

    }
    return s32Ret;
}



/*****************************************************************************
* Prototype :   SVP_NNIE_Pvanet_GetResult
* Description : this function is used to get FasterRcnn result
* Input :     HI_S32* ps32FcBbox             [IN] Bbox for Roi
*             HI_S32 *ps32FcScore            [IN] Score for roi
*             HI_S32 *ps32Proposals          [IN] proposal
*             HI_U32 u32RoiCnt               [IN] Roi num
*             HI_U32 *pu32ConfThresh         [IN] each class confidence thresh
*             HI_U32 u32NmsThresh            [IN] Nms thresh
*             HI_U32 u32MaxRoi               [IN] max roi
*             HI_U32 u32ClassNum             [IN] class num
*             HI_U32 u32OriImWidth           [IN] input image width
*             HI_U32 u32OriImHeight          [IN] input image height
*             HI_U32* pu32MemPool            [IN] assist buffer
*             HI_S32* ps32DstScore           [OUT] result of score
*             HI_S32* ps32DstRoi             [OUT] result of Bbox
*             HI_S32* ps32ClassRoiNum        [OUT] result of the roi num of each classs
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Pvanet_GetResult(HI_S32*ps32FcBbox,HI_U32 u32BboxStride,
    HI_S32*ps32FcScore, HI_U32 u32ScoreStride,HI_S32* ps32Proposal,HI_U32 u32RoiCnt,
    HI_U32* pu32ConfThresh,HI_U32 u32NmsThresh,HI_U32 u32MaxRoi,HI_U32 u32ClassNum,
    HI_U32 u32OriImWidth,HI_U32 u32OriImHeight,HI_U32* pu32MemPool,HI_S32* ps32DstScore,
    HI_S32* ps32DstBbox,HI_S32* ps32ClassRoiNum)
{
    /************* define variables *****************/
    HI_U32 u32Size = 0;
    HI_U32 u32ClsScoreChannels = 0;
    HI_S32* ps32Proposals = NULL;
    HI_U32 u32FcScoreWidth = 0;
    HI_U32 u32FcBboxWidth = 0;
    HI_FLOAT f32ProposalWidth = 0.0;
    HI_FLOAT f32ProposalHeight = 0.0;
    HI_FLOAT f32ProposalCenterX = 0.0;
    HI_FLOAT f32ProposalCenterY = 0.0;
    HI_FLOAT f32PredW = 0.0;
    HI_FLOAT f32PredH = 0.0;
    HI_FLOAT f32PredCenterX = 0.0;
    HI_FLOAT f32PredCenterY = 0.0;
    HI_FLOAT* pf32FcScoresMemPool = NULL;
    HI_S32* ps32ProposalMemPool = NULL;
    HI_S32* ps32ProposalTmp = NULL;
    HI_U32 u32FcBboxIndex = 0;
    HI_U32 u32ProposalMemPoolIndex = 0;
    HI_FLOAT* pf32Ptr = NULL;
    HI_S32* ps32Ptr = NULL;
    HI_S32* ps32Score = NULL;
    HI_S32* ps32Bbox = NULL;
    HI_S32* ps32RoiCnt = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_U32 u32SrcIndex = 0;
    HI_U32 u32DstIndex = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 k = 0;
    SAMPLE_SVP_NNIE_STACK_S* pstStack=NULL;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32OffSet = 0;

  /******************* Get or calculate parameters **********************/
    u32ClsScoreChannels = u32ClassNum;   /*channel num is equal to class size, cls_score class*/
    u32FcScoreWidth = u32ScoreStride / sizeof(HI_U32);
    u32FcBboxWidth = u32BboxStride / sizeof(HI_U32);

    /*************** Get Start Pointer of MemPool ******************/
    pf32FcScoresMemPool = (HI_FLOAT*)pu32MemPool;
    pf32Ptr = pf32FcScoresMemPool;
    u32Size = u32MaxRoi * u32ClsScoreChannels;
    pf32Ptr += u32Size;

    ps32ProposalMemPool = (HI_S32*)pf32Ptr;
    ps32Ptr = ps32ProposalMemPool;
    u32Size = u32MaxRoi * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH ;
    ps32Ptr += u32Size;
    pstStack = (SAMPLE_SVP_NNIE_STACK_S* )ps32Ptr;

    u32DstIndex = 0;

    for( i = 0; i < u32RoiCnt; i++ )
    {
        for( k = 0; k < u32ClsScoreChannels; k++ )
        {
            u32SrcIndex = i * u32FcScoreWidth + k;
            pf32FcScoresMemPool[u32DstIndex++] = (HI_FLOAT)((HI_S32)ps32FcScore[u32SrcIndex]) / SAMPLE_SVP_NNIE_QUANT_BASE;
        }
    }
    ps32Proposals = (HI_S32*)ps32Proposal;

    /************** bbox tranform ************/
    for(j = 0; j < u32ClsScoreChannels; j++)
    {
        for(i = 0; i < u32RoiCnt; i++)
        {
            f32ProposalWidth   = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 2] - ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i] + 1);
            f32ProposalHeight  = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 3] - ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 1] + 1);
            f32ProposalCenterX = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i] + SAMPLE_SVP_NNIE_HALF * f32ProposalWidth);
            f32ProposalCenterY = (HI_FLOAT)(ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 1] + SAMPLE_SVP_NNIE_HALF * f32ProposalHeight);

            u32FcBboxIndex = u32FcBboxWidth * i + SAMPLE_SVP_NNIE_COORDI_NUM * j;
            f32PredCenterX = ((HI_FLOAT)ps32FcBbox[u32FcBboxIndex]/SAMPLE_SVP_NNIE_QUANT_BASE) * f32ProposalWidth  + f32ProposalCenterX;
            f32PredCenterY = ((HI_FLOAT)ps32FcBbox[u32FcBboxIndex + 1]/SAMPLE_SVP_NNIE_QUANT_BASE) * f32ProposalHeight + f32ProposalCenterY;
            f32PredW = f32ProposalWidth * SVP_NNIE_QuickExp((HI_S32)( ps32FcBbox[u32FcBboxIndex+2] ));
            f32PredH = f32ProposalHeight * SVP_NNIE_QuickExp((HI_S32)( ps32FcBbox[u32FcBboxIndex+3] ));

            u32ProposalMemPoolIndex =  SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = (HI_S32)(f32PredCenterX - SAMPLE_SVP_NNIE_HALF * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = (HI_S32)(f32PredCenterY - SAMPLE_SVP_NNIE_HALF * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = (HI_S32)(f32PredCenterX + SAMPLE_SVP_NNIE_HALF * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = (HI_S32)(f32PredCenterY + SAMPLE_SVP_NNIE_HALF * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] = (HI_S32)( pf32FcScoresMemPool[u32ClsScoreChannels*i+j] * SAMPLE_SVP_NNIE_QUANT_BASE );
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] = 0;   /* suprressed flag */

        }

        /* clip bbox */
       for(i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex])>((HI_S32)u32OriImWidth)? (u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex] ) ):0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1])>((HI_S32)u32OriImHeight)? (u32OriImHeight - 1):(ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] ) ):0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2])>((HI_S32)u32OriImWidth)? (u32OriImWidth - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] ) ):0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = ( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1):( ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] ) )>0?( (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3])>((HI_S32)u32OriImHeight)? (u32OriImHeight - 1):(ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] ) ):0;
       }

        ps32ProposalTmp = ps32ProposalMemPool;

        (void)SVP_NNIE_NonRecursiveArgQuickSort( ps32ProposalTmp, 0, u32RoiCnt-1, pstStack,u32RoiCnt);

        (void)SVP_NNIE_NonMaxSuppression(ps32ProposalTmp, u32RoiCnt, u32NmsThresh, u32RoiCnt);

        ps32Score = (HI_S32*)ps32DstScore;
        ps32Bbox = (HI_S32*)ps32DstBbox;
        ps32RoiCnt = (HI_S32*)ps32ClassRoiNum;

        ps32Score += (HI_S32)(u32OffSet);
        ps32Bbox += (HI_S32)(SAMPLE_SVP_NNIE_COORDI_NUM * u32OffSet);

        u32RoiOutCnt = 0;
        for(i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            if( 0 == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5]  && ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] > (HI_S32)pu32ConfThresh[j] ) //Suppression = 0; CONF_THRESH == 0.8
            {
                ps32Score[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM ] = ps32ProposalMemPool[u32ProposalMemPoolIndex];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 1 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 2 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                ps32Bbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 3 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];
                u32RoiOutCnt++;
            }
            if(u32RoiOutCnt >= u32RoiCnt)break;
        }
        ps32RoiCnt[j] = (HI_S32)u32RoiOutCnt;
        u32OffSet += u32RoiOutCnt;

    }
    return s32Ret;
}



/*****************************************************************************
* Prototype :   SVP_NNIE_Rfcn_GetResult
* Description : this function is used to get RFCN result
* Input :     HI_S32* ps32FcBbox             [IN] Bbox for Roi
*             HI_U32 u32FcBboxStride         [IN] Bbox stride
*             HI_S32 *ps32FcScore            [IN] Score for roi
*             HI_U32 u32FcScoreStride        [IN] Score stride
*             HI_S32 *ps32Proposals          [IN] proposal
*             HI_U32 u32RoiCnt               [IN] Roi num
*             HI_U32 *pu32ConfThresh         [IN] each class confidence thresh
*             HI_U32 u32MaxRoi               [IN] max roi
*             HI_U32 u32ClassNum             [IN] class num
*             HI_U32 u32OriImWidth           [IN] input image width
*             HI_U32 u32OriImHeight          [IN] input image height
*             HI_U32 u32NmsThresh            [IN] num thresh
*             HI_U32* pu32MemPool            [IN] assist buffer
*             HI_S32* ps32DstScore           [OUT]result of score
*             HI_S32* ps32DstRoi             [OUT]result of Bbox
*             HI_S32* ps32ClassRoiNum        [OUT]result of the roi num of each classs
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32  SVP_NNIE_Rfcn_GetResult(HI_S32 *ps32FcScore,
    HI_U32 u32FcScoreStride,HI_S32* ps32FcBbox,HI_U32 u32FcBboxStride,
    HI_S32 *ps32Proposals, HI_U32 u32RoiCnt, HI_U32 *pu32ConfThresh,
    HI_U32 u32MaxRoi,HI_U32 u32ClassNum,HI_U32 u32OriImWidth,HI_U32 u32OriImHeight,
    HI_U32 u32NmsThresh, HI_U32* pu32MemPool,HI_S32 *ps32DstScores, HI_S32 *ps32DstRoi,
    HI_S32 *ps32ClassRoiNum)
{
    /************* define variables *****************/
    HI_U32 u32Size = 0;
    HI_U32 u32ClsScoreChannels = 0;
    HI_U32 u32FcScoreWidth = 0;
    HI_FLOAT f32ProposalWidth = 0.0;
    HI_FLOAT f32ProposalHeight = 0.0;
    HI_FLOAT f32ProposalCenterX = 0.0;
    HI_FLOAT f32ProposalCenterY = 0.0;
    HI_FLOAT f32PredW = 0.0;
    HI_FLOAT f32PredH = 0.0;
    HI_FLOAT f32PredCenterX = 0.0;
    HI_FLOAT f32PredCenterY = 0.0;
    HI_FLOAT* pf32FcScoresMemPool = NULL;
    HI_S32* ps32FcBboxMemPool = NULL;
    HI_S32* ps32ProposalMemPool = NULL;
    HI_S32* ps32ProposalTmp = NULL;
    HI_U32 u32FcBboxIndex = 0;
    HI_U32 u32ProposalMemPoolIndex = 0;
    HI_FLOAT* pf32Ptr = NULL;
    HI_S32* ps32Ptr = NULL;
    HI_S32* ps32DstScore = NULL;
    HI_S32* ps32DstBbox = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_U32 u32SrcIndex = 0;
    HI_U32 u32DstIndex = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 u32OffSet = 0;
    SAMPLE_SVP_NNIE_STACK_S* pstStack = NULL;
    HI_S32 s32Ret = HI_SUCCESS;

    /******************* Get or calculate parameters **********************/
    u32ClsScoreChannels = u32ClassNum;   /*channel num is equal to class size, cls_score class*/
    u32FcScoreWidth = u32ClsScoreChannels;

    /*************** Get Start Pointer of MemPool ******************/
    pf32FcScoresMemPool = (HI_FLOAT*)(pu32MemPool);
    pf32Ptr = pf32FcScoresMemPool;
    u32Size = u32MaxRoi * u32ClsScoreChannels;
    pf32Ptr += u32Size;

    ps32FcBboxMemPool = (HI_S32*)pf32Ptr;
    ps32Ptr = (HI_S32*)pf32Ptr;
    u32Size = u32MaxRoi * SAMPLE_SVP_NNIE_COORDI_NUM;
    ps32Ptr += u32Size;

    ps32ProposalMemPool = (HI_S32*)ps32Ptr;
    ps32Ptr = ps32ProposalMemPool;
    u32Size = u32MaxRoi * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH;
    ps32Ptr += u32Size;
    pstStack = (SAMPLE_SVP_NNIE_STACK_S*)ps32Ptr;

    // prepare input data
    for (i = 0; i < u32RoiCnt; i++)
    {
        for (j = 0; j < u32ClsScoreChannels; j++)
        {
            u32DstIndex = u32FcScoreWidth * i + j;
            u32SrcIndex = u32FcScoreStride/sizeof(HI_U32) * i + j;
            pf32FcScoresMemPool[u32DstIndex] = (HI_FLOAT)(ps32FcScore[u32SrcIndex]) / SAMPLE_SVP_NNIE_QUANT_BASE;
        }
    }

    for (i = 0; i < u32RoiCnt; i++)
    {
        for (j = 0; j < SAMPLE_SVP_NNIE_COORDI_NUM; j++)
        {
            u32SrcIndex = u32FcBboxStride/sizeof(HI_U32) * i + SAMPLE_SVP_NNIE_COORDI_NUM + j;
            u32DstIndex = SAMPLE_SVP_NNIE_COORDI_NUM * i + j;
            ps32FcBboxMemPool[u32DstIndex] = ps32FcBbox[u32SrcIndex];
        }
    }
    /************** bbox tranform ************
    change the fc output to Proposal temp MemPool.
    Each Line of the Proposal has 6 bits.
    The Format of the Proposal is:
    0-3: The four coordinate of the bbox, x1,y1,x2, y2
    4: The Confidence Score of the bbox
    5: The suprressed flag
    ******************************************/
    for (j = 0; j < u32ClsScoreChannels; j++)
    {
        for (i = 0; i < u32RoiCnt; i++)
        {
            f32ProposalWidth = ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 2] - ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i] + 1;
            f32ProposalHeight = ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 3] - ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 1] + 1;
            f32ProposalCenterX = ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i] + 0.5 * f32ProposalWidth;
            f32ProposalCenterY = ps32Proposals[SAMPLE_SVP_NNIE_COORDI_NUM*i + 1] + 0.5 * f32ProposalHeight;

            u32FcBboxIndex = SAMPLE_SVP_NNIE_COORDI_NUM * i;
            f32PredCenterX = ((HI_FLOAT)ps32FcBboxMemPool[u32FcBboxIndex] / 4096) * f32ProposalWidth + f32ProposalCenterX;
            f32PredCenterY = ((HI_FLOAT)ps32FcBboxMemPool[u32FcBboxIndex + 1] / 4096) * f32ProposalHeight + f32ProposalCenterY;
            f32PredW = f32ProposalWidth * SVP_NNIE_QuickExp(ps32FcBboxMemPool[u32FcBboxIndex + 2]);
            f32PredH = f32ProposalHeight * SVP_NNIE_QuickExp(ps32FcBboxMemPool[u32FcBboxIndex + 3]);

            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = (HI_S32)(f32PredCenterX - 0.5 * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = (HI_S32)(f32PredCenterY - 0.5 * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = (HI_S32)(f32PredCenterX + 0.5 * f32PredW);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = (HI_S32)(f32PredCenterY + 0.5 * f32PredH);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] = (HI_S32)(pf32FcScoresMemPool[u32ClsScoreChannels*i + j] * 4096);
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] = 0;   /* suprressed flag */
        }

        /* clip bbox */
        for (i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            ps32ProposalMemPool[u32ProposalMemPoolIndex] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex]) >((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex]))>0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex])>((HI_S32)u32OriImWidth) ? (u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex])) : 0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 1] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 1]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1]))>0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 1])>((HI_S32)u32OriImHeight) ? (u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 1])) : 0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 2] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 2]) > ((HI_S32)u32OriImWidth - 1) ? ((HI_S32)u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2]))>0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 2])>((HI_S32)u32OriImWidth) ? (u32OriImWidth - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 2])) : 0;
            ps32ProposalMemPool[u32ProposalMemPoolIndex + 3] = ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 3]) > ((HI_S32)u32OriImHeight - 1) ? ((HI_S32)u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3]))>0 ? ((ps32ProposalMemPool[u32ProposalMemPoolIndex + 3])>((HI_S32)u32OriImHeight) ? (u32OriImHeight - 1) : (ps32ProposalMemPool[u32ProposalMemPoolIndex + 3])) : 0;
        }

        ps32ProposalTmp = ps32ProposalMemPool;
        s32Ret = SVP_NNIE_NonRecursiveArgQuickSort(ps32ProposalTmp, 0, u32RoiCnt - 1, pstStack,u32RoiCnt);
        s32Ret = SVP_NNIE_NonMaxSuppression(ps32ProposalTmp, u32RoiCnt, u32NmsThresh, u32RoiCnt);
        u32RoiOutCnt = 0;

        ps32DstScore = (HI_S32*)ps32DstScores;
        ps32DstBbox = (HI_S32*)ps32DstRoi;

        ps32DstScore += (HI_S32)u32OffSet;
        ps32DstBbox += (HI_S32)(SAMPLE_SVP_NNIE_COORDI_NUM * u32OffSet);
        for (i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * i;
            if (0 == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] && ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] >(HI_S32)pu32ConfThresh[j]) //Suppression = 0; CONF_THRESH == 0.8
            {
                ps32DstScore[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM] = ps32ProposalMemPool[u32ProposalMemPoolIndex];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 1] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 2] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 3] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];
                u32RoiOutCnt++;
            }
            if (u32RoiOutCnt >= u32RoiCnt)
            {
                break;
            }
        }
        ps32ClassRoiNum[j] = (HI_S32)u32RoiOutCnt;
        u32OffSet = u32OffSet + u32RoiOutCnt;
    }

    return s32Ret;
}

/*****************************************************************************
* Prototype :   SVP_NNIE_Ssd_PriorBoxForward
* Description : this function is used to get SSD priorbox
* Input :     HI_U32 u32PriorBoxWidth            [IN] prior box width
*             HI_U32 u32PriorBoxHeight           [IN] prior box height
*             HI_U32 u32OriImWidth               [IN] input image width
*             HI_U32 u32OriImHeight              [IN] input image height
*             HI_U32 f32PriorBoxMinSize          [IN] prior box min size
*             HI_U32 u32MinSizeNum               [IN] min size num
*             HI_U32 f32PriorBoxMaxSize          [IN] prior box max size
*             HI_U32 u32MaxSizeNum               [IN] max size num
*             HI_BOOL bFlip                      [IN] whether do Flip
*             HI_BOOL bClip                      [IN] whether do Clip
*             HI_U32  u32InputAspectRatioNum     [IN] aspect ratio num
*             HI_FLOAT af32PriorBoxAspectRatio[] [IN] aspect ratio value
*             HI_FLOAT f32PriorBoxStepWidth      [IN] prior box step width
*             HI_FLOAT f32PriorBoxStepHeight     [IN] prior box step height
*             HI_FLOAT f32Offset                 [IN] offset value
*             HI_S32   as32PriorBoxVar[]         [IN] prior box variance
*             HI_S32*  ps32PriorboxOutputData    [OUT] output reslut
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Ssd_PriorBoxForward(HI_U32 u32PriorBoxWidth,
    HI_U32 u32PriorBoxHeight, HI_U32 u32OriImWidth, HI_U32 u32OriImHeight,
    HI_FLOAT* pf32PriorBoxMinSize, HI_U32 u32MinSizeNum, HI_FLOAT* pf32PriorBoxMaxSize,
    HI_U32 u32MaxSizeNum, HI_BOOL bFlip, HI_BOOL bClip, HI_U32 u32InputAspectRatioNum,
    HI_FLOAT af32PriorBoxAspectRatio[],HI_FLOAT f32PriorBoxStepWidth,
    HI_FLOAT f32PriorBoxStepHeight,HI_FLOAT f32Offset,HI_S32 as32PriorBoxVar[],
    HI_S32* ps32PriorboxOutputData)
{
    HI_U32 u32AspectRatioNum = 0;
    HI_U32 u32Index = 0;
    HI_FLOAT af32AspectRatio[SAMPLE_SVP_NNIE_SSD_ASPECT_RATIO_NUM] = { 0 };
    HI_U32 u32NumPrior = 0;
    HI_FLOAT f32CenterX = 0;
    HI_FLOAT f32CenterY = 0;
    HI_FLOAT f32BoxHeight = 0;
    HI_FLOAT f32BoxWidth = 0;
    HI_FLOAT f32MaxBoxWidth = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 n = 0;
    HI_U32 h = 0;
    HI_U32 w = 0;
    SAMPLE_SVP_CHECK_EXPR_RET((HI_TRUE == bFlip && u32InputAspectRatioNum >
        (SAMPLE_SVP_NNIE_SSD_ASPECT_RATIO_NUM-1)/2),HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,when bFlip is true, u32InputAspectRatioNum(%d) can't be greater than %d!\n",
        u32InputAspectRatioNum, (SAMPLE_SVP_NNIE_SSD_ASPECT_RATIO_NUM-1)/2);
    SAMPLE_SVP_CHECK_EXPR_RET((HI_FALSE == bFlip && u32InputAspectRatioNum >
        (SAMPLE_SVP_NNIE_SSD_ASPECT_RATIO_NUM-1)),HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,when bFlip is false, u32InputAspectRatioNum(%d) can't be greater than %d!\n",
        u32InputAspectRatioNum, (SAMPLE_SVP_NNIE_SSD_ASPECT_RATIO_NUM-1));

    // generate aspect_ratios
    u32AspectRatioNum = 0;
    af32AspectRatio[0] = 1;
    u32AspectRatioNum++;
    for (i = 0; i < u32InputAspectRatioNum; i++)
    {
        af32AspectRatio[u32AspectRatioNum++] = af32PriorBoxAspectRatio[i];
        if (bFlip)
        {
            af32AspectRatio[u32AspectRatioNum++] = 1.0f / af32PriorBoxAspectRatio[i];
        }
    }
    u32NumPrior = u32MinSizeNum * u32AspectRatioNum + u32MaxSizeNum;

    u32Index = 0;
    for (h = 0; h < u32PriorBoxHeight; h++)
    {
        for (w = 0; w < u32PriorBoxWidth; w++)
        {
            f32CenterX = (w + f32Offset) * f32PriorBoxStepWidth;
            f32CenterY = (h + f32Offset) * f32PriorBoxStepHeight;
            for (n = 0; n < u32MinSizeNum; n++)
            {
                /*** first prior ***/
                f32BoxHeight = pf32PriorBoxMinSize[n];
                f32BoxWidth = pf32PriorBoxMinSize[n];
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                /*** second prior ***/
                if(u32MaxSizeNum>0)
                {
                    f32MaxBoxWidth = sqrt(pf32PriorBoxMinSize[n] * pf32PriorBoxMaxSize[n]);
                    f32BoxHeight = f32MaxBoxWidth;
                    f32BoxWidth = f32MaxBoxWidth;
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                }
                /**** rest of priors, skip AspectRatio == 1 ****/
                for (i = 1; i < u32AspectRatioNum; i++)
                {
                    f32BoxWidth = (HI_FLOAT)(pf32PriorBoxMinSize[n] * sqrt( af32AspectRatio[i] ));
                    f32BoxHeight = (HI_FLOAT)(pf32PriorBoxMinSize[n]/sqrt( af32AspectRatio[i] ));
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                }
            }
        }
    }
    /************ clip the priors' coordidates, within [0, u32ImgWidth] & [0, u32ImgHeight] *************/
    if (bClip)
    {
        for (i = 0; i < (HI_U32)(u32PriorBoxWidth * u32PriorBoxHeight * SAMPLE_SVP_NNIE_COORDI_NUM*u32NumPrior / 2); i++)
        {
            ps32PriorboxOutputData[2 * i] = SAMPLE_SVP_NNIE_MIN((HI_U32)SAMPLE_SVP_NNIE_MAX(ps32PriorboxOutputData[2 * i], 0), u32OriImWidth);
            ps32PriorboxOutputData[2 * i + 1] = SAMPLE_SVP_NNIE_MIN((HI_U32)SAMPLE_SVP_NNIE_MAX(ps32PriorboxOutputData[2 * i + 1], 0), u32OriImHeight);
        }
    }
    /*********************** get var **********************/
    for (h = 0; h < u32PriorBoxHeight; h++)
    {
        for (w = 0; w < u32PriorBoxWidth; w++)
        {
            for (i = 0; i < u32NumPrior; i++)
            {
                for (j = 0; j < SAMPLE_SVP_NNIE_COORDI_NUM; j++)
                {
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)as32PriorBoxVar[j];
                }
            }
        }
    }
    return HI_SUCCESS;
}

/*****************************************************************************
* Prototype :   SVP_NNIE_Ssd_SoftmaxForward
* Description : this function is used to do SSD softmax
* Input :     HI_U32 u32SoftMaxInHeight          [IN] softmax input height
*             HI_U32 au32SoftMaxInChn[]          [IN] softmax input channel
*             HI_U32 u32ConcatNum                [IN] concat num
*             HI_U32 au32ConvStride[]            [IN] conv stride
*             HI_U32 u32SoftMaxOutWidth          [IN] softmax output width
*             HI_U32 u32SoftMaxOutHeight         [IN] softmax output height
*             HI_U32 u32SoftMaxOutChn            [IN] softmax output channel
*             HI_S32* aps32SoftMaxInputData[]    [IN] softmax input data
*             HI_S32* ps32SoftMaxOutputData      [OUT]softmax output data
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Ssd_SoftmaxForward(HI_U32 u32SoftMaxInHeight,
    HI_U32 au32SoftMaxInChn[], HI_U32 u32ConcatNum, HI_U32 au32ConvStride[],
    HI_U32 au32SoftMaxWidth[],HI_S32* aps32SoftMaxInputData[], HI_S32* ps32SoftMaxOutputData)
{
    HI_S32* ps32InputData = NULL;
    HI_S32* ps32OutputTmp = NULL;
    HI_U32 u32OuterNum = 0;
    HI_U32 u32InnerNum = 0;
    HI_U32 u32InputChannel = 0;
    HI_U32 i = 0;
    HI_U32 u32ConcatCnt = 0;
    HI_S32 s32Ret = 0;
    HI_U32 u32Stride = 0;
    HI_U32 u32Skip = 0;
    HI_U32 u32Left = 0;
    ps32OutputTmp = ps32SoftMaxOutputData;
    for (u32ConcatCnt = 0; u32ConcatCnt < u32ConcatNum; u32ConcatCnt++)
    {
        ps32InputData = aps32SoftMaxInputData[u32ConcatCnt];
        u32Stride = au32ConvStride[u32ConcatCnt];
        u32InputChannel = au32SoftMaxInChn[u32ConcatCnt];
        u32OuterNum = u32InputChannel / u32SoftMaxInHeight;
        u32InnerNum = u32SoftMaxInHeight;
        u32Skip = au32SoftMaxWidth[u32ConcatCnt] / u32InnerNum;
        u32Left = u32Stride - au32SoftMaxWidth[u32ConcatCnt];
        for (i = 0; i < u32OuterNum; i++)
        {
            s32Ret = SVP_NNIE_SSD_SoftMax(ps32InputData, (HI_S32)u32InnerNum,ps32OutputTmp);
            if ((i + 1) % u32Skip == 0)
            {
                ps32InputData += u32Left;
            }
            ps32InputData += u32InnerNum;
            ps32OutputTmp += u32InnerNum;
        }
    }
    return s32Ret;
}

/*****************************************************************************
* Prototype :   SVP_NNIE_Ssd_DetectionOutForward
* Description : this function is used to get detection result of SSD
* Input :     HI_U32 u32ConcatNum            [IN] SSD concat num
*             HI_U32 u32ConfThresh           [IN] confidence thresh
*             HI_U32 u32ClassNum             [IN] class num
*             HI_U32 u32TopK                 [IN] Topk value
*             HI_U32 u32KeepTopK             [IN] KeepTopK value
*             HI_U32 u32NmsThresh            [IN] NMS thresh
*             HI_U32 au32DetectInputChn[]    [IN] detection input channel
*             HI_S32* aps32AllLocPreds[]     [IN] Location prediction
*             HI_S32* aps32AllPriorBoxes[]   [IN] prior box
*             HI_S32* ps32ConfScores         [IN] confidence score
*             HI_S32* ps32AssistMemPool      [IN] assist buffer
*             HI_S32* ps32DstScoreSrc        [OUT] result of score
*             HI_S32* ps32DstBboxSrc         [OUT] result of Bbox
*             HI_S32* ps32RoiOutCntSrc       [OUT] result of the roi num of each class
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Ssd_DetectionOutForward(HI_U32 u32ConcatNum,
    HI_U32 u32ConfThresh,HI_U32 u32ClassNum, HI_U32 u32TopK, HI_U32 u32KeepTopK, HI_U32 u32NmsThresh,
    HI_U32 au32DetectInputChn[], HI_S32* aps32AllLocPreds[], HI_S32* aps32AllPriorBoxes[],
    HI_S32* ps32ConfScores, HI_S32* ps32AssistMemPool, HI_S32* ps32DstScoreSrc,
    HI_S32* ps32DstBboxSrc, HI_S32* ps32RoiOutCntSrc)
{
    /************* check input parameters ****************/
    /******** define variables **********/
    HI_S32* ps32LocPreds = NULL;
    HI_S32* ps32PriorBoxes = NULL;
    HI_S32* ps32PriorVar = NULL;
    HI_S32* ps32AllDecodeBoxes = NULL;
    HI_S32* ps32DstScore = NULL;
    HI_S32* ps32DstBbox = NULL;
    HI_S32* ps32ClassRoiNum = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_S32* ps32SingleProposal = NULL;
    HI_S32* ps32AfterTopK = NULL;
    SAMPLE_SVP_NNIE_STACK_S* pstStack = NULL;
    HI_U32 u32PriorNum = 0;
    HI_U32 u32NumPredsPerClass = 0;
    HI_FLOAT f32PriorWidth = 0;
    HI_FLOAT f32PriorHeight = 0;
    HI_FLOAT f32PriorCenterX = 0;
    HI_FLOAT f32PriorCenterY = 0;
    HI_FLOAT f32DecodeBoxCenterX = 0;
    HI_FLOAT f32DecodeBoxCenterY = 0;
    HI_FLOAT f32DecodeBoxWidth = 0;
    HI_FLOAT f32DecodeBoxHeight = 0;
    HI_U32 u32SrcIdx = 0;
    HI_U32 u32AfterFilter = 0;
    HI_U32 u32AfterTopK = 0;
    HI_U32 u32KeepCnt = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 u32Offset = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    u32PriorNum = 0;
    for (i = 0; i < u32ConcatNum; i++)
    {
        u32PriorNum += au32DetectInputChn[i] / SAMPLE_SVP_NNIE_COORDI_NUM;
    }
    //prepare for Assist MemPool
    ps32AllDecodeBoxes = ps32AssistMemPool;
    ps32SingleProposal = ps32AllDecodeBoxes + u32PriorNum * SAMPLE_SVP_NNIE_COORDI_NUM;
    ps32AfterTopK = ps32SingleProposal + SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32PriorNum;
    pstStack = (SAMPLE_SVP_NNIE_STACK_S*)(ps32AfterTopK + u32PriorNum * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH);
    u32SrcIdx = 0;
    for (i = 0; i < u32ConcatNum; i++)
    {
        /********** get loc predictions ************/
        ps32LocPreds = aps32AllLocPreds[i];
        u32NumPredsPerClass = au32DetectInputChn[i] / SAMPLE_SVP_NNIE_COORDI_NUM;
        /********** get Prior Bboxes ************/
        ps32PriorBoxes = aps32AllPriorBoxes[i];
        ps32PriorVar = ps32PriorBoxes + u32NumPredsPerClass*SAMPLE_SVP_NNIE_COORDI_NUM;
        for (j = 0; j < u32NumPredsPerClass; j++)
        {
            //printf("ps32PriorBoxes start***************\n");
            f32PriorWidth = (HI_FLOAT)(ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM+2] - ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM]);
            f32PriorHeight = (HI_FLOAT)(ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM+3] - ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM + 1]);
            f32PriorCenterX = (ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM+2] + ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM])*SAMPLE_SVP_NNIE_HALF;
            f32PriorCenterY = (ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM+3] + ps32PriorBoxes[j*SAMPLE_SVP_NNIE_COORDI_NUM+1])*SAMPLE_SVP_NNIE_HALF;

            f32DecodeBoxCenterX = ((HI_FLOAT)ps32PriorVar[j*SAMPLE_SVP_NNIE_COORDI_NUM]/SAMPLE_SVP_NNIE_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SAMPLE_SVP_NNIE_COORDI_NUM]/SAMPLE_SVP_NNIE_QUANT_BASE)*f32PriorWidth+f32PriorCenterX;

            f32DecodeBoxCenterY = ((HI_FLOAT)ps32PriorVar[j*SAMPLE_SVP_NNIE_COORDI_NUM+1]/SAMPLE_SVP_NNIE_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SAMPLE_SVP_NNIE_COORDI_NUM+1]/SAMPLE_SVP_NNIE_QUANT_BASE)*f32PriorHeight+f32PriorCenterY;

            f32DecodeBoxWidth = exp(((HI_FLOAT)ps32PriorVar[j*SAMPLE_SVP_NNIE_COORDI_NUM+2]/SAMPLE_SVP_NNIE_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SAMPLE_SVP_NNIE_COORDI_NUM+2]/SAMPLE_SVP_NNIE_QUANT_BASE))*f32PriorWidth;

            f32DecodeBoxHeight = exp(((HI_FLOAT)ps32PriorVar[j*SAMPLE_SVP_NNIE_COORDI_NUM+3]/SAMPLE_SVP_NNIE_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SAMPLE_SVP_NNIE_COORDI_NUM+3]/SAMPLE_SVP_NNIE_QUANT_BASE))*f32PriorHeight;

            //printf("ps32PriorBoxes end***************\n");

            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX - f32DecodeBoxWidth * SAMPLE_SVP_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY - f32DecodeBoxHeight * SAMPLE_SVP_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX + f32DecodeBoxWidth * SAMPLE_SVP_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY + f32DecodeBoxHeight * SAMPLE_SVP_NNIE_HALF);
        }
    }
    /********** do NMS for each class *************/
    u32AfterTopK = 0;
    for (i = 0; i < u32ClassNum; i++)
    {
        for (j = 0; j < u32PriorNum; j++)
        {
            ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH] = ps32AllDecodeBoxes[j * SAMPLE_SVP_NNIE_COORDI_NUM];
            ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 1] = ps32AllDecodeBoxes[j * SAMPLE_SVP_NNIE_COORDI_NUM + 1];
            ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 2] = ps32AllDecodeBoxes[j * SAMPLE_SVP_NNIE_COORDI_NUM + 2];
            ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 3] = ps32AllDecodeBoxes[j * SAMPLE_SVP_NNIE_COORDI_NUM + 3];
            ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 4] = ps32ConfScores[j*u32ClassNum + i];
            ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 5] = 0;
        }
        s32Ret = SVP_NNIE_NonRecursiveArgQuickSort(ps32SingleProposal, 0, u32PriorNum - 1, pstStack,u32TopK);
        u32AfterFilter = (u32PriorNum < u32TopK) ? u32PriorNum : u32TopK;
        s32Ret = SVP_NNIE_NonMaxSuppression(ps32SingleProposal, u32AfterFilter, u32NmsThresh, u32AfterFilter);
        u32RoiOutCnt = 0;
        ps32DstScore = (HI_S32*)ps32DstScoreSrc;
        ps32DstBbox = (HI_S32*)ps32DstBboxSrc;
        ps32ClassRoiNum = (HI_S32*)ps32RoiOutCntSrc;
        ps32DstScore += (HI_S32)u32AfterTopK;
        ps32DstBbox += (HI_S32)(u32AfterTopK * SAMPLE_SVP_NNIE_COORDI_NUM);
        for (j = 0; j < u32TopK; j++)
        {
            if (ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 5] == 0 &&
                ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 4] > (HI_S32)u32ConfThresh)
            {
                ps32DstScore[u32RoiOutCnt] = ps32SingleProposal[j * 6 + 4];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM] = ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 1] = ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 1];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 2] = ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 2];
                ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 3] = ps32SingleProposal[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 3];
                u32RoiOutCnt++;
            }
        }
        ps32ClassRoiNum[i] = (HI_S32)u32RoiOutCnt;
        u32AfterTopK += u32RoiOutCnt;
    }

    u32KeepCnt = 0;
    u32Offset = 0;
    if (u32AfterTopK > u32KeepTopK)
    {
        u32Offset = ps32ClassRoiNum[0];
        for (i = 1; i < u32ClassNum; i++)
        {
            ps32DstScore = (HI_S32*)ps32DstScoreSrc;
            ps32DstBbox = (HI_S32*)ps32DstBboxSrc;
            ps32ClassRoiNum = (HI_S32*)ps32RoiOutCntSrc;
            ps32DstScore += (HI_S32)(u32Offset);
            ps32DstBbox += (HI_S32)(u32Offset * SAMPLE_SVP_NNIE_COORDI_NUM);
            for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
            {
                ps32AfterTopK[u32KeepCnt * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH] = ps32DstBbox[j * SAMPLE_SVP_NNIE_COORDI_NUM];
                ps32AfterTopK[u32KeepCnt * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 1] = ps32DstBbox[j * SAMPLE_SVP_NNIE_COORDI_NUM + 1];
                ps32AfterTopK[u32KeepCnt * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 2] = ps32DstBbox[j * SAMPLE_SVP_NNIE_COORDI_NUM + 2];
                ps32AfterTopK[u32KeepCnt * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 3] = ps32DstBbox[j * SAMPLE_SVP_NNIE_COORDI_NUM + 3];
                ps32AfterTopK[u32KeepCnt * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 4] = ps32DstScore[j];
                ps32AfterTopK[u32KeepCnt * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 5] = i;
                u32KeepCnt++;
            }
            u32Offset = u32Offset + ps32ClassRoiNum[i];
        }
        s32Ret = SVP_NNIE_NonRecursiveArgQuickSort(ps32AfterTopK, 0, u32KeepCnt - 1, pstStack,u32KeepCnt);

        u32Offset = 0;
        u32Offset = ps32ClassRoiNum[0];
        for (i = 1; i < u32ClassNum; i++)
        {
            u32RoiOutCnt = 0;
            ps32DstScore = (HI_S32*)ps32DstScoreSrc;
            ps32DstBbox = (HI_S32*)ps32DstBboxSrc;
            ps32ClassRoiNum = (HI_S32*)ps32RoiOutCntSrc;
            ps32DstScore += (HI_S32)(u32Offset);
            ps32DstBbox += (HI_S32)(u32Offset * SAMPLE_SVP_NNIE_COORDI_NUM);
            for (j = 0; j < u32KeepTopK; j++)
            {
                if (ps32AfterTopK[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 5] == i)
                {
                    ps32DstScore[u32RoiOutCnt] = ps32AfterTopK[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 4];
                    ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM] = ps32AfterTopK[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH];
                    ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 1] = ps32AfterTopK[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 1];
                    ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 2] = ps32AfterTopK[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 2];
                    ps32DstBbox[u32RoiOutCnt * SAMPLE_SVP_NNIE_COORDI_NUM + 3] = ps32AfterTopK[j * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH + 3];
                    u32RoiOutCnt++;
                }
            }
            ps32ClassRoiNum[i] = (HI_S32)u32RoiOutCnt;
            u32Offset += u32RoiOutCnt;
        }
    }
    return s32Ret;
}

/*****************************************************************************
* Prototype :   SVP_NNIE_Yolov1_Iou
* Description : this function is used to calculate IOU
* Input :     HI_FLOAT *pf32Bbox         [IN] pointer to the Bbox memery
*             HI_U32   u32Idx1           [IN] first Bbox index
*             HI_U32   u32Idx2           [IN] second Bbox index
*
*
* Output :
* Return Value : HI_S32, the result of IOU.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolov1_Iou(HI_FLOAT *pf32Bbox, HI_U32 u32Idx1, HI_U32 u32Idx2)
{
    HI_FLOAT f32WidthDis = 0.0f, f32HeightDis = 0.0f;
    HI_FLOAT f32Intersection = 0.0f;
    HI_FLOAT f32Iou = 0.0f;
    f32WidthDis = SAMPLE_SVP_NNIE_MIN(pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM] +
        0.5f*pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+2],
        pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM] +
        0.5f*pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+2]) -
        SAMPLE_SVP_NNIE_MAX(pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM] -
        0.5f*pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+2],
        pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM] -
        0.5f*pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+2]);

    f32HeightDis = SAMPLE_SVP_NNIE_MIN(pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+1] +
        0.5f*pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+3],
        pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+1] +
        0.5f*pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+3]) -
        SAMPLE_SVP_NNIE_MAX(pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+1] -
        0.5f*pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+3],
        pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+1] -
        0.5f*pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+3]);

    if (f32WidthDis < 0 || f32HeightDis < 0)
    {
        f32Intersection = 0;
    }
    else
    {
        f32Intersection = f32WidthDis*f32HeightDis;
    }
    f32Iou = f32Intersection / (pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+2]*
        pf32Bbox[u32Idx1*SAMPLE_SVP_NNIE_COORDI_NUM+3] +
        pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+2] *
        pf32Bbox[u32Idx2*SAMPLE_SVP_NNIE_COORDI_NUM+3] - f32Intersection);

    return (HI_S32)(f32Iou*SAMPLE_SVP_NNIE_QUANT_BASE);
}

/*****************************************************************************
* Prototype :   SVP_NNIE_Yolov1_Argswap
* Description : this function is used to exchange data
* Input :     HI_S32*  ps32Src1           [IN] first input array
*             HI_S32*  ps32Src2           [IN] second input array
*             HI_U32  u32ArraySize        [IN] array size
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static void SVP_NNIE_Yolov1_Argswap(HI_S32* ps32Src1, HI_S32* ps32Src2,
    HI_U32 u32ArraySize)
{
    HI_U32 i = 0;
    HI_S32 s32Tmp = 0;
    for( i = 0; i < u32ArraySize; i++ )
    {
        s32Tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = s32Tmp;
    }
}

/*****************************************************************************
* Prototype :   SVP_NNIE_Yolov1_NonRecursiveArgQuickSort
* Description : this function is used to do quick sort
* Input :     HI_S32*  ps32Array          [IN] the array need to be sorted
*             HI_S32   s32Low             [IN] the start position of quick sort
*             HI_S32   s32High            [IN] the end position of quick sort
*             HI_U32   u32ArraySize       [IN] the element size of input array
*             HI_U32   u32ScoreIdx        [IN] the score index in array element
*             SAMPLE_SVP_NNIE_STACK_S *pstStack [IN] the buffer used to store start positions and end positions
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolo_NonRecursiveArgQuickSort(HI_S32* ps32Array,
    HI_S32 s32Low, HI_S32 s32High, HI_U32 u32ArraySize,HI_U32 u32ScoreIdx,
    SAMPLE_SVP_NNIE_STACK_S *pstStack)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;
    HI_S32 s32KeyConfidence = ps32Array[u32ArraySize * s32Low + u32ScoreIdx];
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    while(s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        s32KeyConfidence = ps32Array[u32ArraySize * s32Low + u32ScoreIdx];

        while(i < j)
        {
            while((i < j) && (s32KeyConfidence > ps32Array[j * u32ArraySize + u32ScoreIdx]))
            {
                j--;
            }
            if(i < j)
            {
                SVP_NNIE_Yolov1_Argswap(&ps32Array[i*u32ArraySize], &ps32Array[j*u32ArraySize],u32ArraySize);
                i++;
            }

            while((i < j) && (s32KeyConfidence < ps32Array[i*u32ArraySize + u32ScoreIdx]))
            {
                i++;
            }
            if(i < j)
            {
                SVP_NNIE_Yolov1_Argswap(&ps32Array[i*u32ArraySize], &ps32Array[j*u32ArraySize],u32ArraySize);
                j--;
            }
        }

        if(s32Low < i-1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = s32Low;
            pstStack[s32Top].s32Max = i-1;
        }

        if(s32High > i+1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = i+1;
            pstStack[s32Top].s32Max = s32High;
        }
    }
    return HI_SUCCESS;
}


/*****************************************************************************
* Prototype :   SVP_NNIE_Yolov1_Nms
* Description : this function is used to do NMS
* Input :     HI_S32*   ps32Score          [IN] class score of each bbox
*             HI_FLOAT* pf32Bbox           [IN] pointer to the Bbox memeory
*             HI_U32    u32ConfThresh      [IN] confidence thresh
*             HI_U32    u32NmsThresh       [IN] NMS thresh
*             HI_U32*   pu32TmpBuf         [IN] assist buffer
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolov1_Nms(HI_S32* ps32Score, HI_FLOAT* pf32Bbox,
    HI_U32 u32BboxNum,HI_U32 u32ConfThresh,HI_U32 u32NmsThresh,HI_U32* pu32TmpBuf)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32Idx1 = 0, u32Idx2 = 0;
    SAMPLE_SVP_NNIE_YOLOV1_SCORE_S *pstScore = (SAMPLE_SVP_NNIE_YOLOV1_SCORE_S*)pu32TmpBuf;
    SAMPLE_SVP_NNIE_STACK_S* pstAssitBuf = (SAMPLE_SVP_NNIE_STACK_S*)((HI_U8*)pu32TmpBuf+
        u32BboxNum*sizeof(SAMPLE_SVP_NNIE_YOLOV1_SCORE_S));
    for (i = 0; i < u32BboxNum; i++)
    {
        if (ps32Score[i] < (HI_S32)u32ConfThresh)
        {
            ps32Score[i] = 0;
        }
    }

    for (i = 0; i < u32BboxNum; ++i)
    {
        pstScore[i].u32Idx = i;
        pstScore[i].s32Score= (ps32Score[i]);
    }

    /*quick sort*/
    (void)SVP_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32*)pstScore,0,u32BboxNum-1,
        sizeof(SAMPLE_SVP_NNIE_YOLOV1_SCORE_S)/sizeof(HI_U32),1,pstAssitBuf);

    /*NMS*/
    for (i = 0; i < u32BboxNum; i++)
    {
        u32Idx1 = pstScore[i].u32Idx;
        if (0 == pstScore[i].s32Score)
        {
            continue;
        }
        for (j = i + 1; j < u32BboxNum; j++)
        {
            u32Idx2 = pstScore[j].u32Idx;
            if (0 == pstScore[j].s32Score)
            {
                continue;
            }
            if (SVP_NNIE_Yolov1_Iou(pf32Bbox, u32Idx1, u32Idx2) > (HI_S32)u32NmsThresh)
            {
                pstScore[j].s32Score = 0;
                ps32Score[pstScore[j].u32Idx] = 0;
            }
        }
    }

    return HI_SUCCESS;
}


/*****************************************************************************
* Prototype :   SVP_NNIE_Yolov1_ConvertPosition
* Description : this function is used to do convert position coordinates
* Input :     HI_FLOAT* pf32Bbox           [IN] pointer to the Bbox memeory
*             HI_U32    u32OriImgWidth     [IN] input image width
*             HI_U32    u32OriImagHeight   [IN] input image height
*             HI_FLOAT  af32Roi[]          [OUT] converted position coordinates
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
static void SVP_NNIE_Yolov1_ConvertPosition(HI_FLOAT*pf32Bbox,
    HI_U32 u32OriImgWidth, HI_U32 u32OriImagHeight, HI_FLOAT af32Roi[])
{
    HI_FLOAT f32Xmin, f32Ymin, f32Xmax, f32Ymax;
    f32Xmin = *pf32Bbox - *(pf32Bbox+2) * SAMPLE_SVP_NNIE_HALF;
    f32Xmin = f32Xmin > 0 ? f32Xmin : 0;
    f32Ymin = *(pf32Bbox+1) - *(pf32Bbox+3) * SAMPLE_SVP_NNIE_HALF;
    f32Ymin = f32Ymin > 0 ? f32Ymin : 0;
    f32Xmax = *pf32Bbox + *(pf32Bbox+2) * SAMPLE_SVP_NNIE_HALF;
    f32Xmax = f32Xmax > u32OriImgWidth ? u32OriImgWidth : f32Xmax;
    f32Ymax = *(pf32Bbox+1) + *(pf32Bbox+3) * SAMPLE_SVP_NNIE_HALF;
    f32Ymax = f32Ymax > u32OriImagHeight ? u32OriImagHeight : f32Ymax;

    af32Roi[0] = f32Xmin;
    af32Roi[1] = f32Ymin;
    af32Roi[2] = f32Xmax;
    af32Roi[3] = f32Ymax;
}

/*****************************************************************************
*   Prototype    : SVP_NNIE_Yolov1_Detection
*   Description  : Yolov1 detection
* Input :     HI_S32*   ps32Score       [IN]  bbox each class score
*              HI_FLOAT* pf32Bbox        [IN]  bbox
*              HI_U32    u32ClassNum     [IN]  Class num
*              HI_U32    u32GridNum      [IN]  grid num
*              HI_U32    u32BboxNum      [IN]  bbox num
*              HI_U32    u32ConfThresh   [IN]  confidence thresh
*              HI_U32    u32NmsThresh    [IN]  Nms thresh
*              HI_U32    u32OriImgWidth  [IN]  input image width
*              HI_U32    u32OriImgHeight [IN]  input image height
*              HI_U32*   pu32MemPool     [IN]  assist buffer
*              HI_S32    *ps32DstScores  [OUT]  dst score of ROI
*              HI_S32    *ps32DstRoi     [OUT]  dst Roi
*              HI_S32    *ps32ClassRoiNum[OUT]  dst roi num of each class
*
*   Output       :
*   Return Value : HI_SUCCESS: Success;Error codes: Failure.
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolov1_Detection(HI_S32* ps32Score, HI_FLOAT* pf32Bbox,
        HI_U32 u32ClassNum,HI_U32 u32GridNum,HI_U32 u32BboxNum,HI_U32 u32ConfThresh,
        HI_U32 u32NmsThresh,HI_U32 u32OriImgWidth, HI_U32 u32OriImgHeight,
        HI_U32* pu32MemPool,HI_S32 *ps32DstScores, HI_S32 *ps32DstRoi,
        HI_S32 *ps32ClassRoiNum)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32Idx = 0;
    HI_U32 u32RoiNum = 0;
    HI_S32* ps32EachClassScore = NULL;
    HI_FLOAT af32Roi[SAMPLE_SVP_NNIE_COORDI_NUM] = {0.0f};
    SAMPLE_SVP_NNIE_YOLOV1_SCORE_S *pstScore = NULL;
    *(ps32ClassRoiNum++) = 0;
    for (i = 0; i < u32ClassNum; i++)
    {
        ps32EachClassScore = ps32Score+u32BboxNum*i;
        (void)SVP_NNIE_Yolov1_Nms(ps32EachClassScore, pf32Bbox, u32BboxNum, u32ConfThresh,
            u32NmsThresh,pu32MemPool);

        pstScore = (SAMPLE_SVP_NNIE_YOLOV1_SCORE_S *)pu32MemPool;
        u32RoiNum = 0;
        for(j = 0; j < u32BboxNum; j++)
        {
            if(pstScore[j].s32Score!=0)
            {
                u32RoiNum++;
                *(ps32DstScores++)=pstScore[j].s32Score;
                u32Idx = pstScore[j].u32Idx;
                (void)SVP_NNIE_Yolov1_ConvertPosition((pf32Bbox+u32Idx*SAMPLE_SVP_NNIE_COORDI_NUM),
                    u32OriImgWidth,u32OriImgHeight,af32Roi);
                *(ps32DstRoi++) = (HI_S32)af32Roi[0];
                *(ps32DstRoi++) = (HI_S32)af32Roi[1];
                *(ps32DstRoi++) = (HI_S32)af32Roi[2];
                *(ps32DstRoi++) = (HI_S32)af32Roi[3];
            }
            else
            {
                continue;
            }
        }
        *(ps32ClassRoiNum++) = u32RoiNum;
    }
    return HI_SUCCESS;
}

/*****************************************************************************
*   Prototype    : SVP_NNIE_Yolov2_Iou
*   Description  : Yolov2 IOU
* Input :     SAMPLE_SVP_NNIE_YOLOV2_BBOX_S *pstBbox1 [IN]  first bbox
*              SAMPLE_SVP_NNIE_YOLOV2_BBOX_S *pstBbox2 [IN]  second bbox
*              HI_U32    u32ClassNum     [IN]  Class num
*              HI_U32    u32GridNum      [IN]  grid num
*              HI_U32    u32BboxNum      [IN]  bbox num
*              HI_U32    u32ConfThresh   [IN]  confidence thresh
*              HI_U32    u32NmsThresh    [IN]  Nms thresh
*              HI_U32    u32OriImgWidth  [IN]  input image width
*              HI_U32    u32OriImgHeight [IN]  input image height
*              HI_U32*   pu32MemPool     [IN]  assist buffer
*              HI_S32    *ps32DstScores  [OUT]  dst score of ROI
*              HI_S32    *ps32DstRoi     [OUT]  dst Roi
*              HI_S32    *ps32ClassRoiNum[OUT]  dst roi num of each class
*
*   Output       :
* Return Value : HI_DOUBLE: IOU result
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_DOUBLE SVP_NNIE_Yolov2_Iou(SAMPLE_SVP_NNIE_YOLOV2_BBOX_S *pstBbox1,
    SAMPLE_SVP_NNIE_YOLOV2_BBOX_S *pstBbox2)
{
    HI_FLOAT f32InterWidth = 0.0;
    HI_FLOAT f32InterHeight = 0.0;
    HI_DOUBLE f64InterArea = 0.0;
    HI_DOUBLE f64Box1Area = 0.0;
    HI_DOUBLE f64Box2Area = 0.0;
    HI_DOUBLE f64UnionArea = 0.0;

    f32InterWidth =  SAMPLE_SVP_NNIE_MIN(pstBbox1->f32Xmax, pstBbox2->f32Xmax) - SAMPLE_SVP_NNIE_MAX(pstBbox1->f32Xmin,pstBbox2->f32Xmin);
    f32InterHeight = SAMPLE_SVP_NNIE_MIN(pstBbox1->f32Ymax, pstBbox2->f32Ymax) - SAMPLE_SVP_NNIE_MAX(pstBbox1->f32Ymin,pstBbox2->f32Ymin);

    if(f32InterWidth <= 0 || f32InterHeight <= 0) return 0;

    f64InterArea = f32InterWidth * f32InterHeight;
    f64Box1Area = (pstBbox1->f32Xmax - pstBbox1->f32Xmin)* (pstBbox1->f32Ymax - pstBbox1->f32Ymin);
    f64Box2Area = (pstBbox2->f32Xmax - pstBbox2->f32Xmin)* (pstBbox2->f32Ymax - pstBbox2->f32Ymin);
    f64UnionArea = f64Box1Area + f64Box2Area - f64InterArea;

    return f64InterArea/f64UnionArea;
}


/*****************************************************************************
*   Prototype    : SVP_NNIE_Yolov2_NonMaxSuppression
*   Description  : Yolov2 NonMaxSuppression function
* Input :     SAMPLE_SVP_NNIE_YOLOV2_BBOX_S *pstBbox [IN]  input bbox
*              HI_U32    u32BoxNum       [IN]  Bbox num
*              HI_U32    u32ClassNum     [IN]  Class num
*              HI_U32    u32NmsThresh    [IN]  NMS thresh
*              HI_U32    u32BboxNum      [IN]  bbox num
*              HI_U32    u32MaxRoiNum    [IN]  max roi num
*
*   Output       :
*   Return Value : HI_SUCCESS: Success;Error codes: Failure.
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolov2_NonMaxSuppression( SAMPLE_SVP_NNIE_YOLOV2_BBOX_S* pstBbox,
    HI_U32 u32BboxNum, HI_U32 u32NmsThresh,HI_U32 u32MaxRoiNum)
{
    HI_U32 i,j;
    HI_U32 u32Num = 0;
    HI_DOUBLE f64Iou = 0.0;

    for (i = 0; i < u32BboxNum && u32Num < u32MaxRoiNum; i++)
    {
        if(pstBbox[i].u32Mask == 0 )
        {
            u32Num++;
            for(j= i+1;j< u32BboxNum; j++)
            {
                if( pstBbox[j].u32Mask == 0 )
                {
                    f64Iou = SVP_NNIE_Yolov2_Iou(&pstBbox[i],&pstBbox[j]);
                    if(f64Iou >= (HI_DOUBLE)u32NmsThresh/SAMPLE_SVP_NNIE_QUANT_BASE)
                    {
                        pstBbox[j].u32Mask = 1;
                    }
                }
            }
        }
    }

    return HI_SUCCESS;
}

/*****************************************************************************
*   Prototype    : SVP_NNIE_Yolov2_GetMaxVal
*   Description  : Yolov2 get max score value
* Input :     HI_FLOAT *pf32Val           [IN]  input score
*              HI_U32    u32Num            [IN]  score num
*              HI_U32 *  pu32MaxValueIndex [OUT] the class index of max score
*
*   Output       :
*   Return Value : HI_FLOAT: max score.
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_FLOAT SVP_NNIE_Yolov2_GetMaxVal(HI_FLOAT *pf32Val,HI_U32 u32Num,
    HI_U32 * pu32MaxValueIndex)
{
    HI_U32 i = 0;
    HI_FLOAT f32MaxTmp = 0;

    f32MaxTmp = pf32Val[0];
    *pu32MaxValueIndex = 0;
    for(i = 1;i < u32Num;i++)
    {
        if(pf32Val[i] > f32MaxTmp)
        {
            f32MaxTmp = pf32Val[i];
            *pu32MaxValueIndex = i;
        }
    }

    return f32MaxTmp;
}

/*****************************************************************************
*   Prototype    : SVP_NNIE_Yolov2_GetResult
*   Description  : Yolov2 GetResult function
* Input :     HI_S32    *ps32InputData    [IN]  pointer to the input data memory
*              HI_U32    u32GridNumWidth   [IN]  Grid num in width direction
*              HI_U32    u32GridNumHeight  [IN]  Grid num in height direction
*              HI_U32    u32EachGridBbox   [IN]  Bbox num of each gird
*              HI_U32    u32ClassNum       [IN]  class num
*              HI_U32    u32SrcWidth       [IN]  input image width
*              HI_U32    u32SrcHeight      [IN]  input image height
*              HI_U32    u32MaxRoiNum      [IN]  Max output roi num
*              HI_U32    u32NmsThresh      [IN]  NMS thresh
*              HI_U32*   pu32TmpBuf        [IN]  assist buffer
*              HI_S32    *ps32DstScores    [OUT] dst score
*              HI_S32    *ps32DstRoi       [OUT] dst roi
*              HI_S32    *ps32ClassRoiNum  [OUT] class roi num
*
*   Output       :
*   Return Value : HI_FLOAT: max score value.
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolov2_GetResult(HI_S32 *ps32InputData,HI_U32 u32GridNumWidth,
    HI_U32 u32GridNumHeight,HI_U32 u32EachGridBbox,HI_U32 u32ClassNum,HI_U32 u32SrcWidth,
    HI_U32 u32SrcHeight,HI_U32 u32MaxRoiNum,HI_U32 u32NmsThresh,HI_U32 u32ConfThresh,
    HI_FLOAT af32Bias[],HI_U32* pu32TmpBuf,HI_S32 *ps32DstScores, HI_S32 *ps32DstRoi,
    HI_S32 *ps32ClassRoiNum)
{
    HI_U32 u32GridNum = u32GridNumWidth*u32GridNumHeight;
    HI_U32 u32ParaNum = (SAMPLE_SVP_NNIE_COORDI_NUM+1+u32ClassNum);
    HI_U32 u32TotalBboxNum = u32GridNum*u32EachGridBbox;
    HI_U32 u32CStep = u32GridNum;
    HI_U32 u32HStep = u32GridNumWidth;
    HI_U32 u32BoxsNum = 0;
    HI_FLOAT *pf32BoxTmp = NULL;
    HI_FLOAT *f32InputData = NULL;
    HI_FLOAT f32ObjScore = 0.0;
    HI_FLOAT f32MaxScore = 0.0;
    HI_S32 s32Score = 0;
    HI_U32 u32MaxValueIndex = 0;
    HI_U32 h = 0,w = 0,n = 0;
    HI_U32 c = 0,k = 0,i=0;
    HI_U32 u32Index= 0;
    HI_FLOAT x,y,f32Width,f32Height;
    HI_U32 u32AssistBuffSize = u32TotalBboxNum * sizeof(SAMPLE_SVP_NNIE_STACK_S);
    HI_U32 u32BoxBuffSize = u32TotalBboxNum * sizeof(SAMPLE_SVP_NNIE_YOLOV2_BBOX_S);
    HI_U32 u32BoxResultNum = 0;
    SAMPLE_SVP_NNIE_STACK_S *pstAssistStack = NULL;
    SAMPLE_SVP_NNIE_YOLOV2_BBOX_S *pstBox = NULL;

    /*store float type data*/
    f32InputData = (HI_FLOAT*)pu32TmpBuf;
    /*assist buffer for sort*/
    pstAssistStack = (SAMPLE_SVP_NNIE_STACK_S*)(f32InputData+u32TotalBboxNum*u32ParaNum);
    /*assit box buffer*/
    pstBox = (SAMPLE_SVP_NNIE_YOLOV2_BBOX_S*)((HI_U8*)pstAssistStack+u32AssistBuffSize);
    /*box tmp buffer*/
    pf32BoxTmp = (HI_FLOAT*)((HI_U8*)pstBox + u32BoxBuffSize);

    for(i = 0;i < u32TotalBboxNum*u32ParaNum;i++)
    {
        f32InputData[i] = (HI_FLOAT)(ps32InputData[i])/SAMPLE_SVP_NNIE_QUANT_BASE;
    }

    //permute
    for(h = 0; h< u32GridNumHeight;h++)
    {
        for(w = 0;w < u32GridNumWidth;w++)
        {
            for(c = 0;c < u32EachGridBbox*u32ParaNum;c++)
            {
                pf32BoxTmp[n++] = f32InputData[c*u32CStep + h*u32HStep + w];
            }
        }
    }

    for(n = 0;n < u32GridNum ;n++)
    {
        //Grid
        w = n % u32GridNumWidth;
        h = n / u32GridNumWidth;
        for(k = 0;k <u32EachGridBbox;k++)
        {
            u32Index = (n * u32EachGridBbox + k) * u32ParaNum;
            x = (HI_FLOAT)((w + SAMPLE_SVP_NNIE_SIGMOID(pf32BoxTmp[u32Index + 0])) / u32GridNumWidth); // x
            y = (HI_FLOAT)((h + SAMPLE_SVP_NNIE_SIGMOID(pf32BoxTmp[u32Index + 1])) / u32GridNumHeight); // y
            f32Width = (HI_FLOAT)( (exp(pf32BoxTmp[u32Index + 2]) * af32Bias[2 * k]) / u32GridNumWidth); // w
            f32Height =(HI_FLOAT)((exp(pf32BoxTmp[u32Index + 3]) * af32Bias[2 * k + 1]) / u32GridNumHeight); // h

            f32ObjScore = SAMPLE_SVP_NNIE_SIGMOID(pf32BoxTmp[u32Index + 4]);
            SVP_NNIE_SoftMax(&pf32BoxTmp[u32Index + 5],u32ClassNum);

            f32MaxScore = SVP_NNIE_Yolov2_GetMaxVal(&pf32BoxTmp[u32Index + 5],u32ClassNum,&u32MaxValueIndex);

            s32Score = (HI_S32)(f32MaxScore * f32ObjScore*SAMPLE_SVP_NNIE_QUANT_BASE);
            if(s32Score > u32ConfThresh)
            {
                pstBox[u32BoxsNum].f32Xmin = (HI_FLOAT)(x - f32Width* SAMPLE_SVP_NNIE_HALF);
                pstBox[u32BoxsNum].f32Xmax = (HI_FLOAT)(x + f32Width* SAMPLE_SVP_NNIE_HALF);
                pstBox[u32BoxsNum].f32Ymin = (HI_FLOAT)(y - f32Height* SAMPLE_SVP_NNIE_HALF);
                pstBox[u32BoxsNum].f32Ymax = (HI_FLOAT)(y + f32Height* SAMPLE_SVP_NNIE_HALF);
                pstBox[u32BoxsNum].s32ClsScore = s32Score;
                pstBox[u32BoxsNum].u32ClassIdx = u32MaxValueIndex+1;
                pstBox[u32BoxsNum].u32Mask = 0;
                u32BoxsNum++;
            }
        }
    }
    //quick_sort
    if(u32BoxsNum > 1)
    {
        SVP_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32*)pstBox,0,u32BoxsNum-1,sizeof(SAMPLE_SVP_NNIE_YOLOV2_BBOX_S)/sizeof(HI_S32),
            4,pstAssistStack);
    }
    //Nms
    SVP_NNIE_Yolov2_NonMaxSuppression(pstBox,u32BoxsNum,u32NmsThresh,u32MaxRoiNum);
    //Get the result
    memset((void*)ps32ClassRoiNum,0,(u32ClassNum+1)*sizeof(HI_U32));
    for(i = 1; i < u32ClassNum+1; i++)
    {
        for(n = 0;n < u32BoxsNum && u32BoxResultNum < u32MaxRoiNum; n++)
        {
            if(0 == pstBox[n].u32Mask && i == pstBox[n].u32ClassIdx)
            {
                *(ps32DstRoi++) = (HI_S32)SAMPLE_SVP_NNIE_MAX(pstBox[n].f32Xmin * u32SrcWidth , 0);
                *(ps32DstRoi++) = (HI_S32)SAMPLE_SVP_NNIE_MAX(pstBox[n].f32Ymin * u32SrcHeight ,0);
                *(ps32DstRoi++) = (HI_S32)SAMPLE_SVP_NNIE_MIN(pstBox[n].f32Xmax * u32SrcWidth , u32SrcWidth);
                *(ps32DstRoi++) = (HI_S32)SAMPLE_SVP_NNIE_MIN(pstBox[n].f32Ymax * u32SrcHeight , u32SrcHeight);
                *(ps32DstScores++) = pstBox[n].s32ClsScore;
                *(ps32ClassRoiNum+pstBox[n].u32ClassIdx)=*(ps32ClassRoiNum+pstBox[n].u32ClassIdx)+1;
                u32BoxResultNum++;
            }
        }
    }
    return HI_SUCCESS;
}

/*****************************************************************************
*   Prototype    : SVP_NNIE_Yolov3_GetResult
*   Description  : Yolov3 GetResult function
* Input :      HI_S32    **pps32InputData     [IN]  pointer to the input data
*              HI_U32    au32GridNumWidth[]   [IN]  Grid num in width direction
*              HI_U32    au32GridNumHeight[]  [IN]  Grid num in height direction
*              HI_U32    au32Stride[]         [IN]  stride of input data
*              HI_U32    u32EachGridBbox      [IN]  Bbox num of each gird
*              HI_U32    u32ClassNum          [IN]  class num
*              HI_U32    u32SrcWidth          [IN]  input image width
*              HI_U32    u32SrcHeight         [IN]  input image height
*              HI_U32    u32MaxRoiNum         [IN]  Max output roi num
*              HI_U32    u32NmsThresh         [IN]  NMS thresh
*              HI_U32    u32ConfThresh        [IN]  conf thresh
*              HI_U32    af32Bias[][]         [IN]  bias
*              HI_U32*   pu32TmpBuf           [IN]  assist buffer
*              HI_S32    *ps32DstScores       [OUT] dst score
*              HI_S32    *ps32DstRoi          [OUT] dst roi
*              HI_S32    *ps32ClassRoiNum     [OUT] class roi num
*
*   Output       :
*   Return Value : HI_FLOAT: max score value.
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static HI_S32 SVP_NNIE_Yolov3_GetResult(HI_S32 **pps32InputData,HI_U32 au32GridNumWidth[],
    HI_U32 au32GridNumHeight[],HI_U32 au32Stride[],HI_U32 u32EachGridBbox,HI_U32 u32ClassNum,HI_U32 u32SrcWidth,
    HI_U32 u32SrcHeight,HI_U32 u32MaxRoiNum,HI_U32 u32NmsThresh,HI_U32 u32ConfThresh,
    HI_FLOAT af32Bias[SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM][SAMPLE_SVP_NNIE_YOLOV3_EACH_GRID_BIAS_NUM],
    HI_S32* ps32TmpBuf,HI_S32 *ps32DstScore, HI_S32 *ps32DstRoi, HI_S32 *ps32ClassRoiNum)
{
    HI_S32 *ps32InputBlob = NULL;
    HI_FLOAT *pf32Permute = NULL;
    SAMPLE_SVP_NNIE_YOLOV3_BBOX_S *pstBbox = NULL;
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

    for(i = 0; i < SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        u32BlobSize = au32GridNumWidth[i]*au32GridNumHeight[i]*sizeof(HI_U32)*
            SAMPLE_SVP_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM*u32EachGridBbox;
        if(u32MaxBlobSize < u32BlobSize)
        {
            u32MaxBlobSize = u32BlobSize;
        }
    }

    for(i = 0; i < SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        u32TotalBboxNum += au32GridNumWidth[i]*au32GridNumHeight[i]*u32EachGridBbox;
    }

    //get each tmpbuf addr
    pf32Permute = (HI_FLOAT*)ps32TmpBuf;
    pstBbox = (SAMPLE_SVP_NNIE_YOLOV3_BBOX_S*)(pf32Permute+u32MaxBlobSize/sizeof(HI_S32));
    ps32AssistBuf = (HI_S32*)(pstBbox+u32TotalBboxNum);

    for(i = 0; i < SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        //permute
        u32Offset = 0;
        ps32InputBlob = pps32InputData[i];
        u32ChnOffset = au32GridNumHeight[i]*au32Stride[i]/sizeof(HI_S32);
        u32HeightOffset = au32Stride[i]/sizeof(HI_S32);
        for (h = 0; h < au32GridNumHeight[i]; h++)
        {
            for (w = 0; w < au32GridNumWidth[i]; w++)
            {
                for (c = 0; c < SAMPLE_SVP_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM*u32EachGridBbox; c++)
                {
                    pf32Permute[u32Offset++] = (HI_FLOAT)(ps32InputBlob[c*u32ChnOffset+h*u32HeightOffset+w]) / SAMPLE_SVP_NNIE_QUANT_BASE;
                }
            }
        }

        //decode bbox and calculate score
        for(j = 0; j < au32GridNumWidth[i]*au32GridNumHeight[i]; j++)
        {
            u32GridXIdx = j % au32GridNumWidth[i];
            u32GridYIdx = j / au32GridNumWidth[i];
            for (k = 0; k < u32EachGridBbox; k++)
            {
                u32MaxValueIndex = 0;
                u32Offset = (j * u32EachGridBbox + k) * SAMPLE_SVP_NNIE_YOLOV3_EACH_BBOX_INFER_RESULT_NUM;
                //decode bbox
                f32StartX = ((HI_FLOAT)u32GridXIdx + SAMPLE_SVP_NNIE_SIGMOID(pf32Permute[u32Offset + 0])) / au32GridNumWidth[i];
                f32StartY = ((HI_FLOAT)u32GridYIdx + SAMPLE_SVP_NNIE_SIGMOID(pf32Permute[u32Offset + 1])) / au32GridNumHeight[i];
                f32Width = (HI_FLOAT)(exp(pf32Permute[u32Offset + 2]) * af32Bias[i][2*k]) / u32SrcWidth;
                f32Height = (HI_FLOAT)(exp(pf32Permute[u32Offset + 3]) * af32Bias[i][2*k + 1]) / u32SrcHeight;

                //calculate score
                f32ObjScore = SAMPLE_SVP_NNIE_SIGMOID(pf32Permute[u32Offset + 4]);
                (void)SVP_NNIE_SoftMax(&pf32Permute[u32Offset + 5], u32ClassNum);
                f32MaxScore = SVP_NNIE_Yolov2_GetMaxVal(&pf32Permute[u32Offset + 5], u32ClassNum, &u32MaxValueIndex);
                s32ClassScore = (HI_S32)(f32MaxScore * f32ObjScore*SAMPLE_SVP_NNIE_QUANT_BASE);

                //filter low score roi
                if (s32ClassScore > u32ConfThresh)
                {
                    pstBbox[u32BboxNum].f32Xmin= (HI_FLOAT)(f32StartX - f32Width * 0.5f);
                    pstBbox[u32BboxNum].f32Ymin= (HI_FLOAT)(f32StartY - f32Height * 0.5f);
                    pstBbox[u32BboxNum].f32Xmax= (HI_FLOAT)(f32StartX + f32Width * 0.5f);
                    pstBbox[u32BboxNum].f32Ymax= (HI_FLOAT)(f32StartY + f32Height * 0.5f);
                    pstBbox[u32BboxNum].s32ClsScore = s32ClassScore;
                    pstBbox[u32BboxNum].u32Mask= 0;
                    pstBbox[u32BboxNum].u32ClassIdx = (HI_S32)(u32MaxValueIndex+1);
                    u32BboxNum++;
                }
            }
        }
    }

    //quick sort
    (void)SVP_NNIE_Yolo_NonRecursiveArgQuickSort((HI_S32*)pstBbox, 0, u32BboxNum - 1,
        sizeof(SAMPLE_SVP_NNIE_YOLOV3_BBOX_S)/sizeof(HI_U32),4,(SAMPLE_SVP_NNIE_STACK_S*)ps32AssistBuf);

    //Yolov3 and Yolov2 have the same Nms operation
    (void)SVP_NNIE_Yolov2_NonMaxSuppression(pstBbox, u32BboxNum, u32NmsThresh, sizeof(SAMPLE_SVP_NNIE_YOLOV3_BBOX_S)/sizeof(HI_U32));

    //Get result
    for (i = 1; i < u32ClassNum; i++)
    {
        u32ClassRoiNum = 0;
        for(j = 0; j < u32BboxNum; j++)
        {
            if ((0 == pstBbox[j].u32Mask) && (i == pstBbox[j].u32ClassIdx) && (u32ClassRoiNum < u32MaxRoiNum))
            {
                *(ps32DstRoi++) = SAMPLE_SVP_NNIE_MAX((HI_S32)(pstBbox[j].f32Xmin*u32SrcWidth), 0);
                *(ps32DstRoi++) = SAMPLE_SVP_NNIE_MAX((HI_S32)(pstBbox[j].f32Ymin*u32SrcHeight), 0);
                *(ps32DstRoi++) = SAMPLE_SVP_NNIE_MIN((HI_S32)(pstBbox[j].f32Xmax*u32SrcWidth), u32SrcWidth);
                *(ps32DstRoi++) = SAMPLE_SVP_NNIE_MIN((HI_S32)(pstBbox[j].f32Ymax*u32SrcHeight), u32SrcHeight);
                *(ps32DstScore++) = pstBbox[j].s32ClsScore;
                u32ClassRoiNum++;
            }
        }
        *(ps32ClassRoiNum+i) = u32ClassRoiNum;
    }

    return HI_SUCCESS;
}


/*****************************************************************************
*   Prototype    : SAMPLE_COMM_SVP_NNIE_CnnGetTopN
*   Description  : Cnn GetTopN
* Input :     SAMPLE_SVP_NNIE_PARAM_S*              pstNnieParam     [IN]  the pointer to Cnn NNIE parameter
*              SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to Cnn software parameter
*
*
*   Output       :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-14
*           Author       :
*           Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Cnn_GetTopN(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = SVP_NNIE_Cnn_GetTopN(
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[0].astDst[0].u64VirAddr,
        pstNnieParam->astSegData[0].astDst[0].u32Stride, pstNnieParam->astSegData[0].astDst[0].unShape.stWhc.u32Width,
        pstNnieParam->astSegData[0].astDst[0].u32Num, pstSoftwareParam->u32TopN,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stAssistBuf.u64VirAddr, pstSoftwareParam->stGetTopN.u32Stride,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stGetTopN.u64VirAddr);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SVP_NNIE_Cnn_GetTopN failed!\n");
    return s32Ret;
}


/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_RpnTmpBufSize
* Description : this function is used to get RPN func's assist buffer size
* Input :     HI_U32 u32NumRatioAnchors     [IN]  ratio anchor num
*              HI_U32 u32NumScaleAnchors     [IN]  scale anchor num
*              HI_U32 u32ConvHeight          [IN]  convolution height
*              HI_U32 u32ConvWidth           [IN]  convolution width
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_RpnTmpBufSize(HI_U32 u32NumRatioAnchors,
    HI_U32 u32NumScaleAnchors, HI_U32 u32ConvHeight, HI_U32 u32ConvWidth)
{
    HI_U32 u32AnchorsNum = u32NumRatioAnchors * u32NumScaleAnchors * u32ConvHeight * u32ConvWidth;
    HI_U32 u32AnchorsSize = sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM * u32AnchorsNum;
    HI_U32 u32BboxDeltaSize = u32AnchorsSize;
    HI_U32 u32ProposalSize = sizeof(HI_U32) * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH * u32AnchorsNum;
    HI_U32 u32RatioAnchorsSize = sizeof(HI_FLOAT) * u32NumRatioAnchors * SAMPLE_SVP_NNIE_COORDI_NUM;
    HI_U32 u32ScaleAnchorsSize = sizeof(HI_FLOAT) * u32NumRatioAnchors * u32NumScaleAnchors * SAMPLE_SVP_NNIE_COORDI_NUM;
    HI_U32 u32ScoreSize = sizeof(HI_FLOAT) * u32AnchorsNum * 2;
    HI_U32 u32StackSize = sizeof( SAMPLE_SVP_NNIE_STACK_S ) * u32AnchorsNum;
    HI_U32 u32TotalSize = u32AnchorsSize + u32BboxDeltaSize + u32ProposalSize + u32RatioAnchorsSize + u32ScaleAnchorsSize + u32ScoreSize + u32StackSize;
    return u32TotalSize;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_FasterRcnn_Rpn
* Description : this function is used to do RPN
* Input :     SAMPLE_SVP_NNIE_PARAM_S*                     pstNnieParam     [IN]  the pointer to FasterRcnn NNIE parameter
*              SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to FasterRcnn software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_Rpn(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = SVP_NNIE_Rpn(
        pstSoftwareParam->aps32Conv, pstSoftwareParam->u32NumRatioAnchors, pstSoftwareParam->u32NumScaleAnchors,
        pstSoftwareParam->au32Scales, pstSoftwareParam->au32Ratios, pstSoftwareParam->u32OriImHeight,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->au32ConvHeight, pstSoftwareParam->au32ConvWidth,
        pstSoftwareParam->au32ConvChannel, pstSoftwareParam->u32ConvStride, pstSoftwareParam->u32MaxRoiNum,
        pstSoftwareParam->u32MinSize, pstSoftwareParam->u32SpatialScale, pstSoftwareParam->u32NmsThresh,
        pstSoftwareParam->u32FilterThresh, pstSoftwareParam->u32NumBeforeNms,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stRpnTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        &pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height);
    SAMPLE_COMM_SVP_FlushCache(
        pstSoftwareParam->stRpnBbox.u64PhyAddr, ( HI_VOID* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.u32Num * pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Chn *
            pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height * pstSoftwareParam->stRpnBbox.u32Stride);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SVP_NNIE_Rpn failed!\n");
    return s32Ret;
}


HI_S32 TNG_NNIE_OP_CPU_Proposal(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    return SAMPLE_SVP_NNIE_FasterRcnn_Rpn(pstNnieParam, pstSoftwareParam);
}


/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Pvanet_Rpn
* Description : this function is used to do RPN
* Input :     SAMPLE_SVP_NNIE_PARAM_S*                     pstNnieParam     [IN]  the pointer to FasterRcnn NNIE parameter
*              SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to FasterRcnn software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Pvanet_Rpn(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = SVP_NNIE_Rpn(
        pstSoftwareParam->aps32Conv, pstSoftwareParam->u32NumRatioAnchors, pstSoftwareParam->u32NumScaleAnchors,
        pstSoftwareParam->au32Scales, pstSoftwareParam->au32Ratios, pstSoftwareParam->u32OriImHeight,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->au32ConvHeight, pstSoftwareParam->au32ConvWidth,
        pstSoftwareParam->au32ConvChannel, pstSoftwareParam->u32ConvStride, pstSoftwareParam->u32MaxRoiNum,
        pstSoftwareParam->u32MinSize, pstSoftwareParam->u32SpatialScale, pstSoftwareParam->u32NmsThresh,
        pstSoftwareParam->u32FilterThresh, pstSoftwareParam->u32NumBeforeNms,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stRpnTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        &pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height);
    SAMPLE_COMM_SVP_FlushCache(
        pstSoftwareParam->stRpnBbox.u64PhyAddr, ( HI_VOID* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.u32Num * pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Chn *
            pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height * pstSoftwareParam->stRpnBbox.u32Stride);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SVP_NNIE_Rpn failed!\n");
    return s32Ret;
}



/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_FasterRcnn_GetResultTmpBufSize
* Description : this function is used to get tmp buffer size for FasterRcnn_GetResult func
* Input :     HI_U32 u32MaxRoiNum     [IN]  max roi num
*              HI_U32 u32ClassNum      [IN]  class num
*
*
*
*
* Output :
* Return Value : HI_U32: tmp buffer size
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_FasterRcnn_GetResultTmpBufSize(HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum)
{
    HI_U32 u32ScoreSize = sizeof(HI_FLOAT) * u32MaxRoiNum * u32ClassNum;
    HI_U32 u32ProposalSize = sizeof(HI_U32) * u32MaxRoiNum * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH;
    HI_U32 u32StackSize = sizeof(SAMPLE_SVP_NNIE_STACK_S) * u32MaxRoiNum;
    HI_U32 u32TotalSize = u32ScoreSize + u32ProposalSize + u32StackSize;
    return u32TotalSize;
}



/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Pvanet_GetResultTmpBufSize
* Description : this function is used to get tmp buffer size for FasterRcnn_GetResult func
* Input :     HI_U32 u32MaxRoiNum     [IN]  max roi num
*              HI_U32 u32ClassNum      [IN]  class num
*
*
*
*
* Output :
* Return Value : HI_U32: tmp buffer size
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_Pvanet_GetResultTmpBufSize(HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum)
{
    HI_U32 u32ScoreSize = sizeof(HI_FLOAT) * u32MaxRoiNum * u32ClassNum;
    HI_U32 u32ProposalSize = sizeof(HI_U32) * u32MaxRoiNum * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH;
    HI_U32 u32StackSize = sizeof(SAMPLE_SVP_NNIE_STACK_S) * u32MaxRoiNum;
    HI_U32 u32TotalSize = u32ScoreSize + u32ProposalSize + u32StackSize;
    return u32TotalSize;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_FasterRcnn_GetResult
* Description : this function is used to get FasterRcnn result
* Input :     SAMPLE_SVP_NNIE_PARAM_S*                     pstNnieParam     [IN]  the pointer to FasterRcnn NNIE parameter
*              SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to FasterRcnn software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_S32* ps32Proposal = ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr;
    SAMPLE_SVP_CHECK_EXPR_RET(0 == pstSoftwareParam->stRpnBbox.u64VirAddr,HI_INVALID_VALUE,
        SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,pstSoftwareParam->stRpnBbox.u64VirAddr can't be 0!\n");

    for(i = 0; i < pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height; i++)
    {
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+1) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+2) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+3) /= SAMPLE_SVP_NNIE_QUANT_BASE;
    }
    s32Ret = SVP_NNIE_FasterRcnn_GetResult(
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[1].astDst[0].u64VirAddr,
        pstNnieParam->astSegData[1].astDst[0].u32Stride,
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[1].astDst[1].u64VirAddr,
        pstNnieParam->astSegData[1].astDst[1].u32Stride, ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height, pstSoftwareParam->au32ConfThresh,
        pstSoftwareParam->u32ValidNmsThresh, pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32ClassNum,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32OriImHeight,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);

    return s32Ret;
}




/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Pvanet_GetResult
* Description : this function is used to get FasterRcnn result
* Input :     SAMPLE_SVP_NNIE_PARAM_S*                     pstNnieParam     [IN]  the pointer to FasterRcnn NNIE parameter
*              SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to FasterRcnn software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Pvanet_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_S32* ps32Proposal = ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr;
    SAMPLE_SVP_CHECK_EXPR_RET(0 == pstSoftwareParam->stRpnBbox.u64VirAddr,HI_INVALID_VALUE,
        SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,pstSoftwareParam->stRpnBbox.u64VirAddr can't be 0!\n");

    for(i = 0; i < pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height; i++)
    {
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+1) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+2) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+3) /= SAMPLE_SVP_NNIE_QUANT_BASE;
    }
    s32Ret = SVP_NNIE_Pvanet_GetResult(
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[1].astDst[0].u64VirAddr,
        pstNnieParam->astSegData[1].astDst[0].u32Stride,
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[1].astDst[1].u64VirAddr,
        pstNnieParam->astSegData[1].astDst[1].u32Stride, ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height, pstSoftwareParam->au32ConfThresh,
        pstSoftwareParam->u32ValidNmsThresh, pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32ClassNum,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32OriImHeight,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);

    return s32Ret;
}


/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Rfcn_GetResultTmpBuf
* Description : this function is used to get tmp buffer size for RFCN_GetResult func
* Input :     HI_U32 u32MaxRoiNum     [IN]  Max Roi num
*              HI_U32 u32ClassNum      [IN]  class num
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_Rfcn_GetResultTmpBuf(HI_U32 u32MaxRoiNum, HI_U32 u32ClassNum)
{
    HI_U32 u32ScoreSize = sizeof(HI_FLOAT) * u32MaxRoiNum * u32ClassNum;
    HI_U32 u32ProposalSize = sizeof(HI_U32) * u32MaxRoiNum * SAMPLE_SVP_NNIE_PROPOSAL_WIDTH;
    HI_U32 u32BboxSize = sizeof(HI_U32) * u32MaxRoiNum * SAMPLE_SVP_NNIE_COORDI_NUM;
    HI_U32 u32StackSize = sizeof(SAMPLE_SVP_NNIE_STACK_S) * u32MaxRoiNum;
    HI_U32 u32TotalSize = u32ScoreSize + u32ProposalSize + u32BboxSize+u32StackSize;
    return u32TotalSize;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Rfcn_Rpn
* Description : this function is used to do rpn
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to RFCN NNIE parameter
*              SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to RFCN software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Rfcn_Rpn(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = SVP_NNIE_Rpn(
        pstSoftwareParam->aps32Conv, pstSoftwareParam->u32NumRatioAnchors, pstSoftwareParam->u32NumScaleAnchors,
        pstSoftwareParam->au32Scales, pstSoftwareParam->au32Ratios, pstSoftwareParam->u32OriImHeight,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->au32ConvHeight, pstSoftwareParam->au32ConvWidth,
        pstSoftwareParam->au32ConvChannel, pstSoftwareParam->u32ConvStride, pstSoftwareParam->u32MaxRoiNum,
        pstSoftwareParam->u32MinSize, pstSoftwareParam->u32SpatialScale, pstSoftwareParam->u32NmsThresh,
        pstSoftwareParam->u32FilterThresh, pstSoftwareParam->u32NumBeforeNms,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stRpnTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        &pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height);
    SAMPLE_COMM_SVP_FlushCache(
        pstSoftwareParam->stRpnBbox.u64PhyAddr, ( HI_VOID* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.u32Num * pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Chn *
            pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height * pstSoftwareParam->stRpnBbox.u32Stride);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SVP_NNIE_Rpn failed!\n");
    return s32Ret;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Rfcn_GetResult
* Description : this function is used to Get RFCN Result
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to RFCN NNIE parameter
*              SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S*  pstSoftwareParam [IN]  the pointer to RFCN software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Rfcn_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_S32* ps32Proposal = ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr;

    SAMPLE_SVP_CHECK_EXPR_RET(0 == pstSoftwareParam->stRpnBbox.u64VirAddr,HI_INVALID_VALUE,
        SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,pstSoftwareParam->stRpnBbox.u64VirAddr can't be 0!\n");
    for(i = 0; i < pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height; i++)
    {
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+1) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+2) /= SAMPLE_SVP_NNIE_QUANT_BASE;
        *(ps32Proposal+SAMPLE_SVP_NNIE_COORDI_NUM*i+3) /= SAMPLE_SVP_NNIE_QUANT_BASE;
    }
    s32Ret = SVP_NNIE_Rfcn_GetResult(
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[1].astDst[0].u64VirAddr,
        pstNnieParam->astSegData[1].astDst[0].u32Stride,
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[2].astDst[0].u64VirAddr,
        pstNnieParam->astSegData[2].astDst[0].u32Stride, ( HI_S32* )( HI_UL )pstSoftwareParam->stRpnBbox.u64VirAddr,
        pstSoftwareParam->stRpnBbox.unShape.stWhc.u32Height, pstSoftwareParam->au32ConfThresh,
        pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32ClassNum, pstSoftwareParam->u32OriImWidth,
        pstSoftwareParam->u32OriImHeight, pstSoftwareParam->u32ValidNmsThresh,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SVP_NNIE_Rfcn_GetResult failed!\n");
    return s32Ret;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Ssd_GetResultTmpBuf
* Description : this function is used to Get SSD GetResult tmp buffer size
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to SSD NNIE parameter
*              SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to SSD software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_Ssd_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_U32 u32PriorBoxSize = 0;
    HI_U32 u32SoftMaxSize = 0;
    HI_U32 u32DetectionSize = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32PriorNum = 0;
    HI_U32 i = 0;

    /*priorbox size*/
    for(i = 0; i < pstNnieParam->pstModel->astSeg[0].u16DstNum/2; i++)
    {
        u32PriorBoxSize += pstSoftwareParam->au32PriorBoxHeight[i]*pstSoftwareParam->au32PriorBoxWidth[i]*
            SAMPLE_SVP_NNIE_COORDI_NUM*2*(pstSoftwareParam->u32MaxSizeNum+pstSoftwareParam->u32MinSizeNum+
            pstSoftwareParam->au32InputAspectRatioNum[i]*2*pstSoftwareParam->u32MinSizeNum)*sizeof(HI_U32);
    }
    pstSoftwareParam->stPriorBoxTmpBuf.u32Size = u32PriorBoxSize;
    u32TotalSize+=u32PriorBoxSize;

    /*softmax size*/
    for(i = 0; i < pstSoftwareParam->u32ConcatNum; i++)
    {
        u32SoftMaxSize += pstSoftwareParam->au32SoftMaxInChn[i]*sizeof(HI_U32);
    }
    pstSoftwareParam->stSoftMaxTmpBuf.u32Size = u32SoftMaxSize;
    u32TotalSize+=u32SoftMaxSize;

    /*detection size*/
    for(i = 0; i < pstSoftwareParam->u32ConcatNum; i++)
    {
        u32PriorNum+=pstSoftwareParam->au32DetectInputChn[i]/SAMPLE_SVP_NNIE_COORDI_NUM;
    }
    u32DetectionSize+=u32PriorNum*SAMPLE_SVP_NNIE_COORDI_NUM*sizeof(HI_U32);
    u32DetectionSize+=u32PriorNum*SAMPLE_SVP_NNIE_PROPOSAL_WIDTH*sizeof(HI_U32)*2;
    u32DetectionSize+=u32PriorNum*2*sizeof(HI_U32);
    pstSoftwareParam->stGetResultTmpBuf.u32Size = u32DetectionSize;
    u32TotalSize+=u32DetectionSize;

    return u32TotalSize;
}


/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Ssd_GetResult
* Description : this function is used to Get SSD result
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to SSD NNIE parameter
*              SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to SSD software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Ssd_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_S32* aps32PermuteResult[SAMPLE_SVP_NNIE_SSD_REPORT_NODE_NUM];
    HI_S32* aps32PriorboxOutputData[SAMPLE_SVP_NNIE_SSD_PRIORBOX_NUM];
    HI_S32* aps32SoftMaxInputData[SAMPLE_SVP_NNIE_SSD_SOFTMAX_NUM];
    HI_S32* aps32DetectionLocData[SAMPLE_SVP_NNIE_SSD_SOFTMAX_NUM];
    HI_S32* ps32SoftMaxOutputData = NULL;
    HI_S32* ps32DetectionOutTmpBuf = NULL;
    HI_U32  au32SoftMaxWidth[SAMPLE_SVP_NNIE_SSD_SOFTMAX_NUM];
    HI_U32 u32Size = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;

    /*get permut result*/
    for(i = 0; i < SAMPLE_SVP_NNIE_SSD_REPORT_NODE_NUM; i++)
    {
        aps32PermuteResult[i] = ( HI_S32* )( HI_UL )pstNnieParam->astSegData[0].astDst[i].u64VirAddr;
    }

    /*priorbox*/
    aps32PriorboxOutputData[0] = ( HI_S32* )( HI_UL )pstSoftwareParam->stPriorBoxTmpBuf.u64VirAddr;
    for (i = 1; i < SAMPLE_SVP_NNIE_SSD_PRIORBOX_NUM; i++)
    {
        u32Size = pstSoftwareParam->au32PriorBoxHeight[i-1]*pstSoftwareParam->au32PriorBoxWidth[i-1]*
            SAMPLE_SVP_NNIE_COORDI_NUM*2*(pstSoftwareParam->u32MaxSizeNum+pstSoftwareParam->u32MinSizeNum+
            pstSoftwareParam->au32InputAspectRatioNum[i-1]*2*pstSoftwareParam->u32MinSizeNum);
        aps32PriorboxOutputData[i] = aps32PriorboxOutputData[i - 1] + u32Size;
    }

    for (i = 0; i < SAMPLE_SVP_NNIE_SSD_PRIORBOX_NUM; i++)
    {
        s32Ret = SVP_NNIE_Ssd_PriorBoxForward(pstSoftwareParam->au32PriorBoxWidth[i],
            pstSoftwareParam->au32PriorBoxHeight[i], pstSoftwareParam->u32OriImWidth,
            pstSoftwareParam->u32OriImHeight, pstSoftwareParam->af32PriorBoxMinSize[i],
            pstSoftwareParam->u32MinSizeNum,pstSoftwareParam->af32PriorBoxMaxSize[i],
            pstSoftwareParam->u32MaxSizeNum, pstSoftwareParam->bFlip, pstSoftwareParam->bClip,
            pstSoftwareParam->au32InputAspectRatioNum[i],pstSoftwareParam->af32PriorBoxAspectRatio[i],
            pstSoftwareParam->af32PriorBoxStepWidth[i],pstSoftwareParam->af32PriorBoxStepHeight[i],
            pstSoftwareParam->f32Offset,pstSoftwareParam->as32PriorBoxVar,
            aps32PriorboxOutputData[i]);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SVP_NNIE_Ssd_PriorBoxForward failed!\n");
    }

    /*softmax*/
    ps32SoftMaxOutputData = ( HI_S32* )( HI_UL )pstSoftwareParam->stSoftMaxTmpBuf.u64VirAddr;
    for(i = 0; i < SAMPLE_SVP_NNIE_SSD_SOFTMAX_NUM; i++)
    {
        aps32SoftMaxInputData[i] = aps32PermuteResult[i*2+1];
        au32SoftMaxWidth[i] = pstSoftwareParam->au32ConvChannel[i*2+1];
    }

    (void)SVP_NNIE_Ssd_SoftmaxForward(pstSoftwareParam->u32SoftMaxInHeight,
        pstSoftwareParam->au32SoftMaxInChn, pstSoftwareParam->u32ConcatNum,
        pstSoftwareParam->au32ConvStride, au32SoftMaxWidth,
        aps32SoftMaxInputData, ps32SoftMaxOutputData);

    /*detection*/
    ps32DetectionOutTmpBuf = ( HI_S32* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr;
    for(i = 0; i < SAMPLE_SVP_NNIE_SSD_PRIORBOX_NUM; i++)
    {
        aps32DetectionLocData[i] = aps32PermuteResult[i*2];
    }

    ( void )SVP_NNIE_Ssd_DetectionOutForward(
        pstSoftwareParam->u32ConcatNum, pstSoftwareParam->u32ConfThresh, pstSoftwareParam->u32ClassNum,
        pstSoftwareParam->u32TopK, pstSoftwareParam->u32KeepTopK, pstSoftwareParam->u32NmsThresh,
        pstSoftwareParam->au32DetectInputChn, aps32DetectionLocData, aps32PriorboxOutputData, ps32SoftMaxOutputData,
        ps32DetectionOutTmpBuf, ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);

    return s32Ret;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Yolov1_GetResultTmpBuf
* Description : this function is used to Get Yolov1 GetResult tmp buffer size
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to YOLOV1 NNIE parameter
*              SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to YOLOV1 software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_Yolov1_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_U32 u32TotalGridNum = pstSoftwareParam->u32GridNumHeight*pstSoftwareParam->u32GridNumWidth;
    HI_U32 u32ClassNum = pstSoftwareParam->u32ClassNum;
    HI_U32 u32EachGridBboxNum = pstSoftwareParam->u32BboxNumEachGrid;
    HI_U32 u32TotalBboxNum = u32TotalGridNum*u32EachGridBboxNum;
    HI_U32 u32TransSize = (u32ClassNum+u32EachGridBboxNum*(SAMPLE_SVP_NNIE_COORDI_NUM+1))*
        u32TotalGridNum*sizeof(HI_U32);
    HI_U32 u32Probsize = u32ClassNum*u32TotalBboxNum*sizeof(HI_U32);
    HI_U32 u32ScoreSize = u32TotalBboxNum*sizeof(SAMPLE_SVP_NNIE_YOLOV1_SCORE_S);
    HI_U32 u32StackSize = u32TotalBboxNum*sizeof(SAMPLE_SVP_NNIE_STACK_S);
    HI_U32 u32TotalSize = u32TransSize+u32Probsize+u32ScoreSize+u32StackSize;
    return u32TotalSize;
}


/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Yolov1_GetResult
* Description : this function is used to Get Yolov1 result
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to YOLOV1 NNIE parameter
*              SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to YOLOV1 software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Yolov1_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_FLOAT *pf32ClassProb = NULL;
    HI_FLOAT *pf32Confidence = NULL;
    HI_FLOAT *pf32Bbox = NULL;
    HI_S32 *ps32Score = NULL;
    HI_U32 *pu32AssistBuf = NULL;
    HI_U32 u32Offset = 0;
    HI_U32 u32Index = 0;
    HI_U32 u32GridNum = pstSoftwareParam->u32GridNumHeight*pstSoftwareParam->u32GridNumWidth;
    HI_U32 i = 0, j = 0, k = 0;
    HI_U8* pu8Tmp = ( HI_U8* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr;
    HI_FLOAT f32Score = 0.0f;
    u32Offset = u32GridNum*(pstSoftwareParam->u32BboxNumEachGrid*5+pstSoftwareParam->u32ClassNum);
    pf32ClassProb = (HI_FLOAT*)pu8Tmp;
    pf32Confidence = pf32ClassProb + u32GridNum*pstSoftwareParam->u32ClassNum;
    pf32Bbox = pf32Confidence + u32GridNum*pstSoftwareParam->u32BboxNumEachGrid;

    ps32Score = (HI_S32*)(pf32ClassProb+u32Offset);
    pu32AssistBuf = (HI_U32*)(ps32Score+u32GridNum*pstSoftwareParam->u32BboxNumEachGrid*
        pstSoftwareParam->u32ClassNum);

    for(i = 0; i < u32Offset; i++)
    {
        (( HI_FLOAT* )pu8Tmp)[i] = (( HI_S32* )( HI_UL )pstNnieParam->astSegData[0].astDst[0].u64VirAddr)[i] /
                                   (( HI_FLOAT )SAMPLE_SVP_NNIE_QUANT_BASE);
    }
    for(i = 0; i < u32GridNum; i++)
    {
        for(j = 0; j < pstSoftwareParam->u32BboxNumEachGrid; j++)
        {
            for(k = 0; k < pstSoftwareParam->u32ClassNum; k++)
            {
                u32Offset = k*u32GridNum*pstSoftwareParam->u32BboxNumEachGrid;
                f32Score = *(pf32ClassProb+i*pstSoftwareParam->u32ClassNum+k)**(pf32Confidence+i*pstSoftwareParam->u32BboxNumEachGrid+j);
                *(ps32Score+u32Offset+u32Index) = (HI_S32)(f32Score*SAMPLE_SVP_NNIE_QUANT_BASE);
            }
            u32Index++;
        }
    }

    for(i= 0; i < u32GridNum; i++)
    {
        for(j = 0; j < pstSoftwareParam->u32BboxNumEachGrid; j++)
        {
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 0] =
                (pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 0] +
                i %  pstSoftwareParam->u32GridNumWidth) / pstSoftwareParam->u32GridNumWidth * pstSoftwareParam->u32OriImWidth;
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 1] =
                (pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 1] +
                i / pstSoftwareParam->u32GridNumWidth) / pstSoftwareParam->u32GridNumHeight * pstSoftwareParam->u32OriImHeight;
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 2] =
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 2] *
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 2] * pstSoftwareParam->u32OriImWidth;
            pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 3] =
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 3] *
                pf32Bbox[(i * pstSoftwareParam->u32BboxNumEachGrid + j) * SAMPLE_SVP_NNIE_COORDI_NUM + 3] * pstSoftwareParam->u32OriImHeight;
        }
    }

    ( void )SVP_NNIE_Yolov1_Detection(ps32Score, pf32Bbox, pstSoftwareParam->u32ClassNum, u32GridNum,
                                      u32GridNum * pstSoftwareParam->u32BboxNumEachGrid,
                                      pstSoftwareParam->u32ConfThresh, pstSoftwareParam->u32NmsThresh,
                                      pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32OriImHeight, pu32AssistBuf,
                                      ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
                                      ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
                                      ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);
    return HI_SUCCESS;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf
* Description : this function is used to Get Yolov2 GetResult tmp buffer size
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to YOLOV2 NNIE parameter
*              SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to YOLOV2 software parameter
*
*
*
*
* Output :
* Return Value : HI_U32: tmp buffer size.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_U32 u32TotalGridNum = pstSoftwareParam->u32GridNumHeight*pstSoftwareParam->u32GridNumWidth;
    HI_U32 u32ParaLength = pstSoftwareParam->u32BboxNumEachGrid*(SAMPLE_SVP_NNIE_COORDI_NUM+1+pstSoftwareParam->u32ClassNum);
    HI_U32 u32TotalBboxNum = u32TotalGridNum*pstSoftwareParam->u32BboxNumEachGrid;
    HI_U32 u32TransSize = u32TotalGridNum*u32ParaLength*sizeof(HI_U32);
    HI_U32 u32BboxAssistBufSize = u32TotalBboxNum*sizeof(SAMPLE_SVP_NNIE_STACK_S);
    HI_U32 u32BboxBufSize = u32TotalBboxNum*sizeof(SAMPLE_SVP_NNIE_YOLOV2_BBOX_S);
    HI_U32 u32BboxTmpBufSize = u32TotalGridNum*u32ParaLength*sizeof(HI_FLOAT);
    HI_U32 u32TotalSize = u32TransSize+u32BboxAssistBufSize+u32BboxBufSize+u32BboxTmpBufSize;
    return u32TotalSize;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Yolov2_GetResult
* Description : this function is used to Get Yolov2 result
* Input :     SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to YOLOV2 NNIE parameter
*              SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to YOLOV2 software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Yolov2_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    return SVP_NNIE_Yolov2_GetResult(
        ( HI_S32* )( HI_UL )pstNnieParam->astSegData[0].astDst[0].u64VirAddr, pstSoftwareParam->u32GridNumWidth,
        pstSoftwareParam->u32GridNumHeight, pstSoftwareParam->u32BboxNumEachGrid, pstSoftwareParam->u32ClassNum,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32OriImHeight, pstSoftwareParam->u32MaxRoiNum,
        pstSoftwareParam->u32NmsThresh, pstSoftwareParam->u32ConfThresh, pstSoftwareParam->af32Bias,
        ( HI_U32* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Yolov3_GetResultTmpBuf
* Description : this function is used to Get Yolov3 GetResult tmp buffer size
* Input :      SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to YOLOV3 NNIE parameter
*              SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to YOLOV3 software parameter
*
*
*
*
* Output :
* Return Value : HI_U32: tmp buffer size.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_U32 SAMPLE_SVP_NNIE_Yolov3_GetResultTmpBuf(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_U32 u32TotalSize = 0;
    HI_U32 u32AssistStackSize = 0;
    HI_U32 u32TotalBboxNum = 0;
    HI_U32 u32TotalBboxSize = 0;
    HI_U32 u32DstBlobSize = 0;
    HI_U32 u32MaxBlobSize = 0;
    HI_U32 i = 0;

    for(i = 0; i < pstNnieParam->pstModel->astSeg[0].u16DstNum; i++)
    {
        u32DstBlobSize = pstNnieParam->pstModel->astSeg[0].astDstNode[i].unShape.stWhc.u32Width*sizeof(HI_U32)*
            pstNnieParam->pstModel->astSeg[0].astDstNode[i].unShape.stWhc.u32Height*
            pstNnieParam->pstModel->astSeg[0].astDstNode[i].unShape.stWhc.u32Chn;
        if(u32MaxBlobSize < u32DstBlobSize)
        {
            u32MaxBlobSize = u32DstBlobSize;
        }
        u32TotalBboxNum += pstSoftwareParam->au32GridNumWidth[i]*pstSoftwareParam->au32GridNumHeight[i]*
            pstSoftwareParam->u32BboxNumEachGrid;
    }
    u32AssistStackSize = u32TotalBboxNum*sizeof(SAMPLE_SVP_NNIE_STACK_S);
    u32TotalBboxSize = u32TotalBboxNum*sizeof(SAMPLE_SVP_NNIE_YOLOV3_BBOX_S);
    u32TotalSize += (u32MaxBlobSize+u32AssistStackSize+u32TotalBboxSize);

    return u32TotalSize;
}

/*****************************************************************************
* Prototype :   SAMPLE_SVP_NNIE_Yolov3_GetResult
* Description : this function is used to Get Yolov3 result
* Input :      SAMPLE_SVP_NNIE_PARAM_S*               pstNnieParam     [IN]  the pointer to YOLOV3 NNIE parameter
*              SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S*   pstSoftwareParam [IN]  the pointer to YOLOV3 software parameter
*
*
*
*
* Output :
* Return Value : HI_SUCCESS: Success;Error codes: Failure.
* Spec :
* Calls :
* Called By :
* History:
*
* 1. Date : 2017-11-10
* Author :
* Modification : Create
*
*****************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_Yolov3_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftwareParam)
{
    HI_U32 i = 0;
    HI_S32 *aps32InputBlob[SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM] = {0};
    HI_U32 au32Stride[SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM] = {0};

    for(i = 0; i < SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM; i++)
    {
        aps32InputBlob[i] = ( HI_S32* )( HI_UL )pstNnieParam->astSegData[0].astDst[i].u64VirAddr;
        au32Stride[i] = pstNnieParam->astSegData[0].astDst[i].u32Stride;
    }
    return SVP_NNIE_Yolov3_GetResult(
        aps32InputBlob, pstSoftwareParam->au32GridNumWidth, pstSoftwareParam->au32GridNumHeight, au32Stride,
        pstSoftwareParam->u32BboxNumEachGrid, pstSoftwareParam->u32ClassNum, pstSoftwareParam->u32OriImWidth,
        pstSoftwareParam->u32OriImWidth, pstSoftwareParam->u32MaxRoiNum, pstSoftwareParam->u32NmsThresh,
        pstSoftwareParam->u32ConfThresh, pstSoftwareParam->af32Bias,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stGetResultTmpBuf.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstScore.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stDstRoi.u64VirAddr,
        ( HI_S32* )( HI_UL )pstSoftwareParam->stClassRoiNum.u64VirAddr);
}

#ifdef __cplusplus
}
#endif
