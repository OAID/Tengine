#ifndef __SAMPLE_COMM_SVP_H__
#define __SAMPLE_COMM_SVP_H__

#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif /* __cplusplus */
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

#include "hi_comm_svp.h"
#include "hi_common.h"
#include "hi_debug.h"
#include "mpi_sys.h"

    typedef enum hiSAMPLE_SVP_ERR_LEVEL_E
    {
        SAMPLE_SVP_ERR_LEVEL_DEBUG = 0x0,   /* debug-level								   */
        SAMPLE_SVP_ERR_LEVEL_INFO = 0x1,	/* informational 							   */
        SAMPLE_SVP_ERR_LEVEL_NOTICE = 0x2,  /* normal but significant condition			   */
        SAMPLE_SVP_ERR_LEVEL_WARNING = 0x3, /* warning conditions						   */
        SAMPLE_SVP_ERR_LEVEL_ERROR = 0x4,   /* error conditions							   */
        SAMPLE_SVP_ERR_LEVEL_CRIT = 0x5,	/* critical conditions						   */
        SAMPLE_SVP_ERR_LEVEL_ALERT = 0x6,   /* action must be taken immediately			   */
        SAMPLE_SVP_ERR_LEVEL_FATAL = 0x7,   /* just for compatibility with previous version */

        SAMPLE_SVP_ERR_LEVEL_BUTT
    } SAMPLE_SVP_ERR_LEVEL_E;

#define SAMPLE_SVP_PRINTF(LevelStr, Msg, ...)                                                                           \
    do                                                                                                                  \
    {                                                                                                                   \
        fprintf(stderr, "[Level]:%s,[Func]:%s [Line]:%d [Info]:" Msg, LevelStr, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    } while (0)
#define SAMPLE_SVP_PRINTF_RED(LevelStr, Msg, ...)                                                                                                 \
    do                                                                                                                                            \
    {                                                                                                                                             \
        fprintf(stderr, "\033[0;31m [Level]:%s,[Func]:%s [Line]:%d [Info]:" Msg "\033[0;39m\n", LevelStr, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    } while (0)
/* system is unusable	*/
#define SAMPLE_SVP_TRACE_FATAL(Msg, ...) SAMPLE_SVP_PRINTF_RED("Fatal", Msg, ##__VA_ARGS__)
/* action must be taken immediately */
#define SAMPLE_SVP_TRACE_ALERT(Msg, ...) SAMPLE_SVP_PRINTF_RED("Alert", Msg, ##__VA_ARGS__)
/* critical conditions */
#define SAMPLE_SVP_TRACE_CRIT(Msg, ...) SAMPLE_SVP_PRINTF_RED("Critical", Msg, ##__VA_ARGS__)
/* error conditions */
#define SAMPLE_SVP_TRACE_ERR(Msg, ...) SAMPLE_SVP_PRINTF_RED("Error", Msg, ##__VA_ARGS__)
/* warning conditions */
#define SAMPLE_SVP_TRACE_WARN(Msg, ...) SAMPLE_SVP_PRINTF("Warning", Msg, ##__VA_ARGS__)
/* normal but significant condition  */
#define SAMPLE_SVP_TRACE_NOTICE(Msg, ...) SAMPLE_SVP_PRINTF("Notice", Msg, ##__VA_ARGS__)
/* informational */
#define SAMPLE_SVP_TRACE_INFO(Msg, ...) SAMPLE_SVP_PRINTF("Info", Msg, ##__VA_ARGS__)
/* debug-level messages  */
#define SAMPLE_SVP_TRACE_DEBUG(Msg, ...) SAMPLE_SVP_PRINTF("Debug", Msg, ##__VA_ARGS__)

#define SAMPLE_SVP_TRACE(Level, Msg, ...)                \
    do                                                   \
    {                                                    \
        switch (Level)                                   \
        {                                                \
        case SAMPLE_SVP_ERR_LEVEL_DEBUG:                 \
            SAMPLE_SVP_TRACE_DEBUG(Msg, ##__VA_ARGS__);  \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_INFO:                  \
            SAMPLE_SVP_TRACE_INFO(Msg, ##__VA_ARGS__);   \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_NOTICE:                \
            SAMPLE_SVP_TRACE_NOTICE(Msg, ##__VA_ARGS__); \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_WARNING:               \
            SAMPLE_SVP_TRACE_WARN(Msg, ##__VA_ARGS__);   \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_ERROR:                 \
            SAMPLE_SVP_TRACE_ERR(Msg, ##__VA_ARGS__);    \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_CRIT:                  \
            SAMPLE_SVP_TRACE_CRIT(Msg, ##__VA_ARGS__);   \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_ALERT:                 \
            SAMPLE_SVP_TRACE_ALERT(Msg, ##__VA_ARGS__);  \
            break;                                       \
        case SAMPLE_SVP_ERR_LEVEL_FATAL:                 \
            SAMPLE_SVP_TRACE_FATAL(Msg, ##__VA_ARGS__);  \
            break;                                       \
        default:                                         \
            break;                                       \
        }                                                \
    } while (0)
/****
*Expr is true,goto
*/
#define SAMPLE_SVP_CHECK_EXPR_GOTO(Expr, Label, Level, Msg, ...) \
    do                                                           \
    {                                                            \
        if (Expr)                                                \
        {                                                        \
            SAMPLE_SVP_TRACE(Level, Msg, ##__VA_ARGS__);         \
            goto Label;                                          \
        }                                                        \
    } while (0)
/****
*Expr is true,return void
*/
#define SAMPLE_SVP_CHECK_EXPR_RET_VOID(Expr, Level, Msg, ...) \
    do                                                        \
    {                                                         \
        if (Expr)                                             \
        {                                                     \
            SAMPLE_SVP_TRACE(Level, Msg, ##__VA_ARGS__);      \
            return;                                           \
        }                                                     \
    } while (0)
/****
*Expr is true,return Ret
*/
#define SAMPLE_SVP_CHECK_EXPR_RET(Expr, Ret, Level, Msg, ...) \
    do                                                        \
    {                                                         \
        if (Expr)                                             \
        {                                                     \
            SAMPLE_SVP_TRACE(Level, Msg, ##__VA_ARGS__);      \
            return Ret;                                       \
        }                                                     \
    } while (0)
/****
*Expr is true,trace
*/
#define SAMPLE_SVP_CHECK_EXPR_TRACE(Expr, Level, Msg, ...) \
    do                                                     \
    {                                                      \
        if (Expr)                                          \
        {                                                  \
            SAMPLE_SVP_TRACE(Level, Msg, ##__VA_ARGS__);   \
        }                                                  \
    } while (0)

#define SAMPLE_SVP_ALIGN_16 16
#define SAMPLE_SVP_ALIGN_32 32
#define SAMPLE_SVP_D1_PAL_HEIGHT 576
#define SAMPLE_SVP_D1_PAL_WIDTH 704

#define SAMPLE_SVP_NNIE_ALIGN_16 16
#ifndef SAMPLE_SVP_NNIE_ALIGN16
#define SAMPLE_SVP_NNIE_ALIGN16(u32Num) ((u32Num + SAMPLE_SVP_NNIE_ALIGN_16 - 1) / SAMPLE_SVP_NNIE_ALIGN_16 * SAMPLE_SVP_NNIE_ALIGN_16)
#endif
//free mmz
#define SAMPLE_SVP_MMZ_FREE(phy, vir)                        \
    do                                                       \
    {                                                        \
        if ((0 != (phy)) && (0 != (vir)))                    \
        {                                                    \
            HI_MPI_SYS_MmzFree((phy), (void *)(HI_UL)(vir)); \
            (phy) = 0;                                       \
            (vir) = 0;                                       \
        }                                                    \
    } while (0)

#define SAMPLE_SVP_CLOSE_FILE(fp) \
    do                            \
    {                             \
        if (NULL != (fp))         \
        {                         \
            fclose((fp));         \
            (fp) = NULL;          \
        }                         \
    } while (0)

    /*
         *System init
         */
    void
    SAMPLE_COMM_SVP_CheckSysInit(void);
    /*
*System exit
*/
    void SAMPLE_COMM_SVP_CheckSysExit(void);
    /*
*Align
*/
    HI_U32 SAMPLE_COMM_SVP_Align(HI_U32 u32Size, HI_U16 u16Align);

    /*
*Create Image memory
*/
    HI_S32 SAMPLE_COMM_SVP_CreateImage(SVP_IMAGE_S *pstImg, SVP_IMAGE_TYPE_E enType, HI_U32 u32Width,
                                       HI_U32 u32Height, HI_U32 u32AddrOffset);
    /*
*Destory image memory
*/
    HI_VOID SAMPLE_COMM_SVP_DestroyImage(SVP_IMAGE_S *pstImg, HI_U32 u32AddrOffset);

    /*
*Create mem info
*/
    HI_S32 SAMPLE_COMM_SVP_CreateMemInfo(SVP_MEM_INFO_S *pstMemInfo, HI_U32 u32Size, HI_U32 u32AddrOffset);

    /*
Destory mem info
*/
    HI_VOID SAMPLE_COMM_SVP_DestroyMemInfo(SVP_MEM_INFO_S *pstMemInfo, HI_U32 u32AddrOffset);
    /*
*Malloc memory
*/
    HI_S32 SAMPLE_COMM_SVP_MallocMem(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size);
    /*
*Malloc memory with cached
*/
    HI_S32 SAMPLE_COMM_SVP_MallocCached(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size);
    /*
*Fulsh cached
*/
    HI_S32 SAMPLE_COMM_SVP_FlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size);
    /*
*Gen rand data
*/
    HI_S32 SAMPLE_COMM_SVP_GenRandS32(HI_S32 s32Max, HI_S32 s32Min);
    /*
*Gen image
*/
    HI_VOID SAMPLE_COMM_SVP_GenImage(HI_U64 au64Buff[3], HI_U32 au32Stride[3], SVP_IMAGE_TYPE_E enType, HI_U32 u32Width, HI_U32 u32Height);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __SAMPLE_COMM_SVP_H__ */
