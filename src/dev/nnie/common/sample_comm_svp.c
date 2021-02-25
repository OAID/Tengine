#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#include "hi_common.h"
#include "hi_comm_video.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "sample_comm_svp.h"

static HI_BOOL s_bSampleSvpInit = HI_FALSE;

/*
*System init
*/
static HI_S32 SAMPLE_COMM_SVP_SysInit(HI_VOID)
{
    HI_S32 s32Ret = HI_FAILURE;
    VB_CONFIG_S struVbConf;

    HI_MPI_SYS_Exit();
    HI_MPI_VB_Exit();

    memset(&struVbConf, 0, sizeof(VB_CONFIG_S));

    struVbConf.u32MaxPoolCnt = 2;
    struVbConf.astCommPool[1].u64BlkSize = 768 * 576 * 2;
    struVbConf.astCommPool[1].u32BlkCnt = 1;

    s32Ret = HI_MPI_VB_SetConfig((const VB_CONFIG_S *)&struVbConf);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_VB_SetConf failed!\n", s32Ret);

    s32Ret = HI_MPI_VB_Init();
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_VB_Init failed!\n", s32Ret);

    s32Ret = HI_MPI_SYS_Init();
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_Init failed!\n", s32Ret);

    return s32Ret;
}
/*
*System exit
*/
static HI_S32 SAMPLE_COMM_SVP_SysExit(HI_VOID)
{
    HI_S32 s32Ret = HI_FAILURE;

    s32Ret = HI_MPI_SYS_Exit();
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_Exit failed!\n", s32Ret);

    s32Ret = HI_MPI_VB_Exit();
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_VB_Exit failed!\n", s32Ret);

    return HI_SUCCESS;
}
/*
*System init
*/
HI_VOID SAMPLE_COMM_SVP_CheckSysInit(HI_VOID)
{
    if (HI_FALSE == s_bSampleSvpInit)
    {
        if (SAMPLE_COMM_SVP_SysInit())
        {
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_ERROR, "Svp mpi init failed!\n");
            exit(-1);
        }
        s_bSampleSvpInit = HI_TRUE;
    }
}
/*
*System exit
*/
HI_VOID SAMPLE_COMM_SVP_CheckSysExit(HI_VOID)
{
    if (s_bSampleSvpInit)
    {
        SAMPLE_COMM_SVP_SysExit();
        s_bSampleSvpInit = HI_FALSE;
    }
}

/*
*Align
*/
HI_U32 SAMPLE_COMM_SVP_Align(HI_U32 u32Size, HI_U16 u16Align)
{
    HI_U32 u32Stride = u32Size + (u16Align - u32Size % u16Align) % u16Align;
    return u32Stride;
}

/*
*Create Image memory
*/
HI_S32 SAMPLE_COMM_SVP_CreateImage(SVP_IMAGE_S *pstImg, SVP_IMAGE_TYPE_E enType, HI_U32 u32Width,
                                   HI_U32 u32Height, HI_U32 u32AddrOffset)
{
    HI_U32 u32Size = 0;
    HI_S32 s32Ret;

    pstImg->enType = enType;
    pstImg->u32Width = u32Width;
    pstImg->u32Height = u32Height;
    pstImg->au32Stride[0] = SAMPLE_COMM_SVP_Align(pstImg->u32Width, SAMPLE_SVP_ALIGN_16);

    switch (enType)
    {
    case SVP_IMAGE_TYPE_U8C1:
    case SVP_IMAGE_TYPE_S8C1:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);
        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
    }
    break;
    case SVP_IMAGE_TYPE_YUV420SP:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 3 / 2 + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);
        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au32Stride[1] = pstImg->au32Stride[0];
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
    }
    break;
    case SVP_IMAGE_TYPE_YUV422SP:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 2 + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);
        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au32Stride[1] = pstImg->au32Stride[0];
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
    }
    break;
    case SVP_IMAGE_TYPE_YUV420P:
    {
        pstImg->au32Stride[1] = SAMPLE_COMM_SVP_Align(pstImg->u32Width / 2, SAMPLE_SVP_ALIGN_16);
        pstImg->au32Stride[2] = pstImg->au32Stride[1];

        u32Size = pstImg->au32Stride[0] * pstImg->u32Height + pstImg->au32Stride[1] * pstImg->u32Height + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64PhyAddr[2] = pstImg->au64PhyAddr[1] + pstImg->au32Stride[1] * pstImg->u32Height / 2;
        pstImg->au64VirAddr[2] = pstImg->au64VirAddr[1] + pstImg->au32Stride[1] * pstImg->u32Height / 2;
    }
    break;
    case SVP_IMAGE_TYPE_YUV422P:
    {

        pstImg->au32Stride[1] = SAMPLE_COMM_SVP_Align(pstImg->u32Width / 2, SAMPLE_SVP_ALIGN_16);
        pstImg->au32Stride[2] = pstImg->au32Stride[1];
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height + pstImg->au32Stride[1] * pstImg->u32Height * 2 + u32AddrOffset;

        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64PhyAddr[2] = pstImg->au64PhyAddr[1] + pstImg->au32Stride[1] * pstImg->u32Height;
        pstImg->au64VirAddr[2] = pstImg->au64VirAddr[1] + pstImg->au32Stride[1] * pstImg->u32Height;
    }
    break;
    case SVP_IMAGE_TYPE_S8C2_PACKAGE:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 2 + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au32Stride[1] = pstImg->au32Stride[0];
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + 1;
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + 1;
    }
    break;
    case SVP_IMAGE_TYPE_S8C2_PLANAR:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 2 + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au32Stride[1] = pstImg->au32Stride[0];
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * pstImg->u32Height;
    }
    break;
    case SVP_IMAGE_TYPE_S16C1:
    case SVP_IMAGE_TYPE_U16C1:
    {

        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * sizeof(HI_U16) + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);
        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
    }
    break;
    case SVP_IMAGE_TYPE_U8C3_PACKAGE:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 3 + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au32Stride[1] = pstImg->au32Stride[0];
        pstImg->au32Stride[2] = pstImg->au32Stride[0];
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + 1;
        pstImg->au64VirAddr[2] = pstImg->au64VirAddr[1] + 1;
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + 1;
        pstImg->au64PhyAddr[2] = pstImg->au64PhyAddr[1] + 1;
    }
    break;
    case SVP_IMAGE_TYPE_U8C3_PLANAR:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 3 + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
        pstImg->au32Stride[1] = pstImg->au32Stride[0];
        pstImg->au32Stride[2] = pstImg->au32Stride[0];
        pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + (HI_U64)pstImg->au32Stride[0] * (HI_U64)pstImg->u32Height;
        pstImg->au64VirAddr[2] = pstImg->au64VirAddr[1] + (HI_U64)pstImg->au32Stride[1] * (HI_U64)pstImg->u32Height;
        pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + (HI_U64)pstImg->au32Stride[0] * (HI_U64)pstImg->u32Height;
        pstImg->au64PhyAddr[2] = pstImg->au64PhyAddr[1] + (HI_U64)pstImg->au32Stride[1] * (HI_U64)pstImg->u32Height;
    }
    break;
    case SVP_IMAGE_TYPE_S32C1:
    case SVP_IMAGE_TYPE_U32C1:
    {
        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * sizeof(HI_U32) + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);
        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
    }
    break;
    case SVP_IMAGE_TYPE_S64C1:
    case SVP_IMAGE_TYPE_U64C1:
    {

        u32Size = pstImg->au32Stride[0] * pstImg->u32Height * sizeof(HI_U64) + u32AddrOffset;
        s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (void **)&pstImg->au64VirAddr[0], NULL, HI_NULL, u32Size);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

        pstImg->au64PhyAddr[0] += u32AddrOffset;
        pstImg->au64VirAddr[0] += u32AddrOffset;
    }
    break;
    default:
        break;
    }

    return HI_SUCCESS;
}
/*
*Destory image memory
*/
HI_VOID SAMPLE_COMM_SVP_DestroyImage(SVP_IMAGE_S *pstImg, HI_U32 u32AddrOffset)
{
    if (NULL != pstImg)
    {
        if ((0 != pstImg->au64VirAddr[0]) && (0 != pstImg->au64PhyAddr[0]))
        {
            (HI_VOID) HI_MPI_SYS_MmzFree(pstImg->au64PhyAddr[0] - u32AddrOffset,
                                         (void *)(HI_UL)(pstImg->au64VirAddr[0] - u32AddrOffset));
        }
        memset(pstImg, 0, sizeof(*pstImg));
    }
}

/*
*Create mem info
*/
HI_S32 SAMPLE_COMM_SVP_CreateMemInfo(SVP_MEM_INFO_S *pstMemInfo, HI_U32 u32Size, HI_U32 u32AddrOffset)
{
    HI_S32 s32Ret;
    HI_U32 u32SizeTmp;

    u32SizeTmp = u32Size + u32AddrOffset;
    pstMemInfo->u32Size = u32Size;
    s32Ret = HI_MPI_SYS_MmzAlloc(&pstMemInfo->u64PhyAddr, (void **)&pstMemInfo->u64VirAddr, NULL, HI_NULL, u32SizeTmp);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error(%#x):HI_MPI_SYS_MmzAlloc failed!\n", s32Ret);

    pstMemInfo->u64PhyAddr += u32AddrOffset;
    pstMemInfo->u64VirAddr += u32AddrOffset;

    return s32Ret;
}

/*
Destory mem info
*/
HI_VOID SAMPLE_COMM_SVP_DestroyMemInfo(SVP_MEM_INFO_S *pstMemInfo, HI_U32 u32AddrOffset)
{
    if (NULL != pstMemInfo)
    {
        if ((0 != pstMemInfo->u64VirAddr) && (0 != pstMemInfo->u64PhyAddr))
        {
            (HI_VOID) HI_MPI_SYS_MmzFree(pstMemInfo->u64PhyAddr - u32AddrOffset,
                                         (void *)(HI_UL)(pstMemInfo->u64VirAddr - u32AddrOffset));
        }
        memset(pstMemInfo, 0, sizeof(*pstMemInfo));
    }
}
/*
*Malloc memory
*/
HI_S32 SAMPLE_COMM_SVP_MallocMem(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;

    s32Ret = HI_MPI_SYS_MmzAlloc(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);

    return s32Ret;
}

/*
*Malloc memory with cached
*/
HI_S32 SAMPLE_COMM_SVP_MallocCached(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);

    return s32Ret;
}

/*
*Fulsh cached
*/
HI_S32 SAMPLE_COMM_SVP_FlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = HI_MPI_SYS_MmzFlushCache(u64PhyAddr, pvVirAddr, u32Size);
    return s32Ret;
}

/*
*Gen rand data
*/
HI_S32 SAMPLE_COMM_SVP_GenRandS32(HI_S32 s32Max, HI_S32 s32Min)
{
    HI_S32 s32Ret = 0;

    if (s32Min >= 0)
    {
        s32Ret = s32Min + ((HI_U32)rand()) % (s32Max - s32Min + 1);
    }
    else
    {
        s32Ret = ((HI_U32)rand()) % (s32Max - s32Min + 1);
        s32Ret = s32Ret > s32Max ? s32Max - s32Ret : s32Ret;
    }

    return s32Ret;
}

/*
*Gen image
*/
HI_VOID SAMPLE_COMM_SVP_GenImage(HI_U64 au64Buff[3], HI_U32 au32Stride[3], SVP_IMAGE_TYPE_E enType, HI_U32 u32Width, HI_U32 u32Height)
{
    HI_U32 i, j, k;
    HI_U8 *pu8;
    HI_S8 *ps8;
    HI_U16 *pu16;
    HI_S16 *ps16;
    HI_U32 *pu32;
    HI_S32 *ps32;
    HI_U64 *pu64;
    HI_S64 *ps64;
    HI_U8 *apu8Buff1[3] = {NULL, NULL, NULL};

    switch (enType)
    {
    case SVP_IMAGE_TYPE_U8C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];

        pu8 = apu8Buff1[0];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_S8C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];

        ps8 = (HI_S8 *)apu8Buff1[0];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                ps8[j] = SAMPLE_COMM_SVP_GenRandS32(127, -128);
            }
            ps8 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_YUV420SP:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];

        apu8Buff1[1] = (HI_U8 *)(HI_UL)au64Buff[1];

        pu8 = apu8Buff1[0];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[0];
        }

        pu8 = apu8Buff1[1];
        u32Height /= 2;
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[1];
        }
    }
    break;
    case SVP_IMAGE_TYPE_YUV422SP:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        apu8Buff1[1] = (HI_U8 *)(HI_UL)au64Buff[1];
        pu8 = apu8Buff1[0];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[0];
        }
        pu8 = apu8Buff1[1];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[1];
        }
    }
    break;
    case SVP_IMAGE_TYPE_YUV420P:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        apu8Buff1[1] = (HI_U8 *)(HI_UL)au64Buff[1];
        apu8Buff1[2] = (HI_U8 *)(HI_UL)au64Buff[2];

        pu8 = apu8Buff1[0];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[0];
        }

        pu8 = apu8Buff1[1];
        u32Height /= 2;
        u32Width /= 2;
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[1];
        }

        pu8 = apu8Buff1[2];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[2];
        }
    }
    break;
    case SVP_IMAGE_TYPE_YUV422P:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        apu8Buff1[1] = (HI_U8 *)(HI_UL)au64Buff[1];
        apu8Buff1[2] = (HI_U8 *)(HI_UL)au64Buff[2];

        pu8 = apu8Buff1[0];
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[0];
        }

        pu8 = apu8Buff1[1];
        u32Width /= 2;
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[1];
        }
        pu8 = apu8Buff1[2];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[2];
        }
    }
    break;
    case SVP_IMAGE_TYPE_S8C2_PACKAGE:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        ps8 = (HI_S8 *)apu8Buff1[0];
        u32Width += u32Width;
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                ps8[j] = SAMPLE_COMM_SVP_GenRandS32(127, -128);
            }
            ps8 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_S8C2_PLANAR:
    {
        for (k = 0; k < 2; k++)
        {
            apu8Buff1[k] = (HI_U8 *)(HI_UL)au64Buff[k];
            ps8 = (HI_S8 *)apu8Buff1[k];

            for (i = 0; i < u32Height; i++)
            {
                for (j = 0; j < u32Width; j++)
                {
                    ps8[j] = SAMPLE_COMM_SVP_GenRandS32(127, -128);
                }
                ps8 += au32Stride[k];
            }
        }
    }
    break;
    case SVP_IMAGE_TYPE_S16C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        ps16 = (HI_S16 *)apu8Buff1[0];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {

                ps16[j] = (HI_S16)SAMPLE_COMM_SVP_GenRandS32(32767, -32768);
            }
            ps16 = (HI_S16 *)((HI_U8 *)ps16 + au32Stride[0]);
        }
    }
    break;
    case SVP_IMAGE_TYPE_U16C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        pu16 = (HI_U16 *)apu8Buff1[0];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {

                pu16[j] = SAMPLE_COMM_SVP_GenRandS32(65535, 0);
            }
            pu16 = (HI_U16 *)((HI_U8 *)pu16 + au32Stride[0]);
        }
    }
    break;
    case SVP_IMAGE_TYPE_U8C3_PACKAGE:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        pu8 = apu8Buff1[0];
        u32Width *= 3;
        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
            }
            pu8 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_U8C3_PLANAR:
    {

        for (k = 0; k < 3; k++)
        {
            apu8Buff1[k] = (HI_U8 *)(HI_UL)au64Buff[k];

            pu8 = apu8Buff1[k];
            for (i = 0; i < u32Height; i++)
            {
                for (j = 0; j < u32Width; j++)
                {
                    pu8[j] = SAMPLE_COMM_SVP_GenRandS32(255, 0);
                }
                pu8 += au32Stride[k];
            }
        }
    }
    break;
    case SVP_IMAGE_TYPE_S32C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        ps32 = (HI_S32 *)apu8Buff1[0];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                ps32[j] = SAMPLE_COMM_SVP_GenRandS32(2147483646, -2147483647);
            }
            ps32 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_U32C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        pu32 = (HI_U32 *)apu8Buff1[0];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu32[j] = SAMPLE_COMM_SVP_GenRandS32(65535, 0) * SAMPLE_COMM_SVP_GenRandS32(65535, 0);
            }
            pu32 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_S64C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        ps64 = (HI_S64 *)apu8Buff1[0];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                ps64[j] = SAMPLE_COMM_SVP_GenRandS32(2147483646, -2147483647) * SAMPLE_COMM_SVP_GenRandS32(65535, 0) * SAMPLE_COMM_SVP_GenRandS32(65535, 0);
            }
            ps64 += au32Stride[0];
        }
    }
    break;
    case SVP_IMAGE_TYPE_U64C1:
    {
        apu8Buff1[0] = (HI_U8 *)(HI_UL)au64Buff[0];
        pu64 = (HI_U64 *)apu8Buff1[0];

        for (i = 0; i < u32Height; i++)
        {
            for (j = 0; j < u32Width; j++)
            {
                pu64[j] = (HI_U64)SAMPLE_COMM_SVP_GenRandS32(65535, 0) * (HI_U64)SAMPLE_COMM_SVP_GenRandS32(65535, 0) * (HI_U64)SAMPLE_COMM_SVP_GenRandS32(65535, 0) * (HI_U64)SAMPLE_COMM_SVP_GenRandS32(65535, 0);
            }
            pu64 += au32Stride[0];
        }
    }
    break;
    default:
        break;
    }
}
