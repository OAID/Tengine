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
 * Author: bsun@openailab.com
 */

#include <string.h>
#include <sys/time.h>
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif
#include "tengine_log.h"

#ifdef __cplusplus
}
#endif

#include "nnie_device.h"
#include "nnie_param.h"
#include "nnie_opmaps.h"
#include "sample_comm_svp.h"
#include "sample_svp_nnie_software.h"

#ifdef CONFIG_TIME_LIMIT
const static int PLUGIN_TIME_LIMIT_SEC = CONFIG_TIME_LIMIT;
#else
const static int PLUGIN_TIME_LIMIT_SEC = 0;
#endif

#include "sample_comm_nnie.h"
#include <unordered_map>
#include <vector>
#define NNIE_DEVICE "nnie"

struct NnieGraph
{
    std::unordered_map<std::string, ir_tensor *> input_tensors;
    std::unordered_map<std::string, ir_tensor *> output_tensors;
    std::vector<ir_node *> nnie_nodes;
    std::vector<ir_tensor *> input_nnie_tensors;
    std::vector<ir_tensor *> output_nnie_tensors;
};

static HI_S32 TNG_NNIE_OP_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                  SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstIputDataIdx,
                                  SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
                               (HI_VOID *)pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr,
                               pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);
    // LOG_ERROR() << "u64PhyAddr: " << pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr <<
    // "\n"; LOG_ERROR() << "u64VirAddr: " << pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr
    // << "\n"; LOG_ERROR() << "u32Size: " << pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size <<
    // "\n";
    /*NNIE_Forward*/
    s32Ret = HI_MPI_SVP_NNIE_Forward(
        &hSvpNnieHandle, &pstNnieParam->astSegData[pstIputDataIdx->u32SegIdx].astSrc[pstIputDataIdx->u32NodeIdx],
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if (bInstant)
    {
        /*Wait NNIE finish*/
        while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT ==
               (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
                                               hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO, "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }

    bFinish = HI_FALSE;
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
    {
        for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
        {
            u32TotalStepNum +=
                *((HI_U8 *)(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr) + j);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                   (HI_VOID *)pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                                   u32TotalStepNum *
                                       pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        // printf("[%s][%d] addr:%p\n", __FUNCTION__, __LINE__,
        //        ( HI_VOID* )pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr);
    }
    else
    {
        for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
        {
            SAMPLE_COMM_SVP_FlushCache(
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                (HI_VOID *)pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    return s32Ret;
}

/******************************************************************************
 * function : NNIE ForwardWithBbox
 ******************************************************************************/
static HI_S32 TNG_NNIE_OP_ForwardWithBbox(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                          SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx,
                                          SVP_SRC_BLOB_S astBbox[],
                                          SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;
    HI_U32 i, j;

    SAMPLE_COMM_SVP_FlushCache(
        pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        (HI_VOID *)pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr,
        pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    /*NNIE_ForwardWithBbox*/
    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(
        &hSvpNnieHandle, &pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[pstInputDataIdx->u32NodeIdx],
        astBbox, pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "Error,HI_MPI_SVP_NNIE_ForwardWithBbox failed!\n");

    if (bInstant)
    {
        /*Wait NNIE finish*/
        while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT ==
               (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
                                               hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO, "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }

    bFinish = HI_FALSE;

    for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum +=
                    *((HI_U8 *)(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr) + j);
            }
            SAMPLE_COMM_SVP_FlushCache(
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                (HI_VOID *)pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                (HI_VOID *)pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    return s32Ret;
}

static void copy_nnie_data(struct ir_tensor *tensor, SVP_DST_BLOB_S *pstDst)
{
    HI_U32 n = 0, i = 0, j = 0;
    HI_U32 u32Num = 0, u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0;
    HI_U32 u32VarSize = 0;
    HI_U8 *pu8PicAddr = NULL;
    HI_U8 *tensor_buffer = (HI_U8 *)tensor->data;
    // int tensor_mem_size = get_tensor_mem_size(tensor);
    // int tensor_buffer_len = 0;
    /*get data size*/
    if (SVP_BLOB_TYPE_U8 <= pstDst->enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstDst->enType)
    {
        u32VarSize = sizeof(HI_U8);
    }
    else
    {
        u32VarSize = sizeof(HI_U32);
    }

    u32Num = pstDst->u32Num;
    u32Chn = pstDst->unShape.stWhc.u32Chn;
    u32Height = pstDst->unShape.stWhc.u32Height;
    u32Width = pstDst->unShape.stWhc.u32Width;
    u32Stride = pstDst->u32Stride;

    int copy_data_len = 0;
    pu8PicAddr = (HI_U8 *)(pstDst->u64VirAddr);
    for (n = 0; n < u32Num; n++)
    {
        for (i = 0; i < u32Chn; i++)
        {
            for (j = 0; j < u32Height; j++)
            {
                memcpy(tensor_buffer, (void *)pu8PicAddr, u32Width * u32VarSize);
                tensor_buffer += (u32Width * u32VarSize);
                copy_data_len += u32Width * u32VarSize;
                pu8PicAddr += u32Stride; ///sizeof(HI_U32);
            }
        }
    }
    TLOG_DEBUG(" u32Num:%u u32Chn:%u u32Height:%u u32Width:%u u32VarSize:%u u32Stride:%u copy_data_len:%u %u\n",
        u32Num, u32Chn, u32Height, u32Width, u32VarSize, u32Stride, copy_data_len, u32Num * u32Chn * u32Height * u32Width * u32VarSize);
    return;
}

static HI_S32 SAMPLE_SVP_NNIE_FillDataFromTensor(struct ir_tensor *tensor, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                                 SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx)
{
    HI_U32 i = 0, j = 0, n = 0;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
    HI_U32 u32VarSize = 0;
    HI_U8 *pu8PicAddr = NULL;
    HI_U32 *pu32StepAddr = NULL;
    HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
    HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
    HI_U32 u32TotalStepNum = 0;

    /*tensor date*/
    HI_U64 tensor_data_addr = (HI_U64)tensor->data;

    /*get data size*/
    if (SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32VarSize = sizeof(HI_U8);
    }
    else
    {
        u32VarSize = sizeof(HI_U32);
    }

    /*fill src data*/
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32Dim = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u32Dim;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu32StepAddr = (HI_U32 *)(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
        pu8PicAddr = (HI_U8 *)(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
        {
            for (i = 0; i < *(pu32StepAddr + n); i++)
            {
                memcpy(pu8PicAddr, (void *)tensor_data_addr, u32Dim * u32VarSize);
                tensor_data_addr += u32Dim * u32VarSize;
                pu8PicAddr += u32Stride;
            }
            u32TotalStepNum += *(pu32StepAddr + n);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
                                   (HI_VOID *)pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr,
                                   u32TotalStepNum * u32Stride);
    }
    else
    {
        u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu8PicAddr = (HI_U8 *)(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
        {
            for (i = 0; i < u32Chn; i++)
            {
                for (j = 0; j < u32Height; j++)
                {
                    memcpy(pu8PicAddr, (void *)tensor_data_addr, u32Width * u32VarSize);
                    tensor_data_addr += u32Width * u32VarSize;
                    pu8PicAddr += u32Stride;
                }
            }
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
                                   (HI_VOID *)pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr,
                                   pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num * u32Chn * u32Height *
                                       u32Stride);
    }

    return HI_SUCCESS;
}

static NnieGraph *CreateNnieGraph(struct subgraph* subgraph)
{
    struct NnieGraph* nnie_graph = new NnieGraph();
    for (unsigned int i = 0; i < subgraph->graph->input_num; i++)
    {
        struct ir_node* node = subgraph->graph->node_list[subgraph->graph->input_nodes[i]];
        if(!strncmp(node->name, "InputOp/node_", 13))
        {
            struct ir_tensor* tensor = subgraph->graph->tensor_list[node->output_tensors[0]];
            nnie_graph->input_tensors[tensor->name] = tensor;
        }
    }

    for (unsigned int i = 0; i < subgraph->graph->node_num; i++)
    {
        struct ir_node* node = subgraph->graph->node_list[i];
        if(node->attr_num > 0){
            struct ir_attr* p_attr = node->attr_mem;
            if ( !strcmp(p_attr->attr_name, NNIE_OP_FORWARD) || !strcmp(p_attr->attr_name, NNIE_OP_FORWARD_WITH_BBOX) ||
            !strcmp(p_attr->attr_name, NNIE_OP_CPU_PROPOSAL))
            {
                nnie_graph->nnie_nodes.push_back(node);
            }
        }
    }

    for (unsigned int i = 0; i < nnie_graph->nnie_nodes[0]->input_num; i++)
    {
        struct ir_tensor* tensor = subgraph->graph->tensor_list[
                                    nnie_graph->nnie_nodes[0]->input_tensors[i]];
        nnie_graph->input_nnie_tensors.push_back(tensor);
    }

    //???
    for (unsigned int i = 0; i < subgraph->graph->output_num; i++)
    {
        struct ir_node *node = subgraph->graph->node_list[subgraph->graph->output_nodes[i]];
        if(node->attr_num > 0){
            struct ir_attr* p_attr = node->attr_mem;
            if ( strcmp(p_attr->attr_name, NNIE_OP_FORWARD) && strcmp(p_attr->attr_name, NNIE_OP_FORWARD_WITH_BBOX) &&
                strcmp(p_attr->attr_name, NNIE_OP_CPU_PROPOSAL))
            {
                TLOG_ERR("Output node is not an nnie node.\n");
                return nullptr;
            }

            TLOG_INFO( "%s OutputNum:%d\n", node->name, node->output_num);
            for (unsigned int j = 0; j < node->output_num; j++)
            {
                struct ir_tensor *out_tensor = subgraph->graph->tensor_list[node->output_tensors[j]];
                //???
                if (out_tensor->data == nullptr)
                {
                    int mem_size = (out_tensor->elem_size) * (out_tensor->elem_num);
                    out_tensor->data = (void*)sys_malloc(mem_size);
                    TLOG_INFO( "mem_addr:%u mem_size:%d\n", out_tensor->data, mem_size);
                    //XLOG_DEBUG() << "mem_addr:" << mem_addr << " mem_size:" << mem_size << "\n";
                }
                else
                {
                    TLOG_INFO( "mem not null, size:%d\n", (out_tensor->elem_size) * (out_tensor->elem_num));
                    //XLOG_DEBUG() << "mem not null, size:" << get_tensor_mem_size(out_tensor) << "\n";
                }

                nnie_graph->output_tensors[out_tensor->name] = out_tensor;
                nnie_graph->output_nnie_tensors.push_back(out_tensor);
            }
        }
        else
        {
            TLOG_ERR("Output node is not an nnie node.\n");
            return nullptr;
        }
        for(unsigned int i = 0; i < nnie_graph->output_nnie_tensors.size(); i++)
        {
            std::cout<< nnie_graph->output_nnie_tensors[i]<< " "<< i;
        }
        std::cout<<std::endl;
    }

    return nnie_graph;
}

static int prerun(struct nn_device* dev, struct subgraph* subgraph, int num_thread, int cpu_affinity, int mode)
{
    if (subgraph == nullptr)
    {
        TLOG_ERR("Get graph handler failed.\n");
        return -1;
    }

    struct NnieGraph *nnie_graph = CreateNnieGraph(subgraph);
    if (nnie_graph == nullptr)
    {
        TLOG_ERR("Create aipu graph failed.\n");
        return -1;
    }
    struct nnie_device* nnie_dev = (struct nnie_device*)dev;
    nnie_dev->nnie_graph = nnie_graph;

    return 0;
}

static int run(struct nn_device* dev, struct subgraph* subgraph)
{
    struct nnie_device* nnie_dev = (struct nnie_device*) dev;
    NnieGraph *nnie_graph = nnie_dev->nnie_graph;
    if (nnie_graph == nullptr)
    {
        TLOG_ERR("Get aipu graph handler failed.\n");
        return -1;
    }

    struct ir_node *node = nnie_graph->nnie_nodes[0];
    nnie_param_t* nnieparam = (nnie_param_t* )node->op.param_mem;
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnNnieParam = (SAMPLE_SVP_NNIE_PARAM_S *)nnieparam->nnie_node;
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstFasterRcnnSoftwareParam = (SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *)nnieparam->software_param;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;

    // copy input data
    for (unsigned int i = 0; i < nnie_graph->input_nnie_tensors.size(); i++)
    {
        struct ir_tensor *tensor = nnie_graph->input_nnie_tensors[i];
        if (tensor == nullptr)
        {
            TLOG_ERR("Get input tensor failed.\n");
            return -1;
        }
        // check tensor memory size
        int tensor_size = (tensor->elem_size) * (tensor->elem_num);
        TLOG_INFO("addr:%u tensor_size:%d\n", tensor->data, tensor_size );
        SAMPLE_SVP_NNIE_FillDataFromTensor(tensor, pstCnnNnieParam, &stInputDataIdx);
    }

    // run all the nodes
    TLOG_INFO( "nnie_nodes.size():%u\n", nnie_graph->nnie_nodes.size());
    for (unsigned int i = 0; i < nnie_graph->nnie_nodes.size(); i++)
    {
        // struct timeval t0, t1;
        // gettimeofday(&t0, NULL);
        struct ir_node *nnie_node = nnie_graph->nnie_nodes[i];
        if(nnie_node->attr_num > 0)
        {
            char* opname = nnie_node->attr_mem->attr_name;
            SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
            stProcSegIdx.u32SegIdx = i;
            if (!strcmp(opname, NNIE_OP_FORWARD))
            {
                TLOG_INFO( "TNG_NNIE_OP_Forward\n");
                int s32Ret = TNG_NNIE_OP_Forward(pstCnnNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
                SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                        "Error, TNG_NNIE_OP_Forward failed!\n");
            }
            else if ( !strcmp(opname, NNIE_OP_FORWARD_WITH_BBOX) )
            {
                TLOG_INFO( "TNG_NNIE_OP_ForwardWithBbox\n");
                stProcSegIdx.u32SegIdx = 1;
                int s32Ret = TNG_NNIE_OP_ForwardWithBbox(
                    pstCnnNnieParam, &stInputDataIdx, &pstFasterRcnnSoftwareParam->stRpnBbox, &stProcSegIdx, HI_TRUE);
                SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                        "Error, TNG_NNIE_OP_ForwardWithBbox failed!\n");
            }
            else
            {
                TLOG_INFO( "%s\n", opname);
                Tengine::NnieOpMaps *ops = Tengine::NnieOpMaps::getInstance();
                custom_op *customOp = ops->getCustomOpByName(opname);
                customOp->param = pstFasterRcnnSoftwareParam;
                TLOG_INFO( "%s\n", customOp->name);
                customOp->run(customOp, nullptr, 0, nullptr, 0);
                TLOG_INFO( "customOp->run\n");
            }
        }
    }

    // copy output data
    for (unsigned int i = 0; i < nnie_graph->output_nnie_tensors.size(); i++)
    {
        struct ir_tensor *tensor = nnie_graph->output_nnie_tensors[i];
        if (tensor == nullptr)
        {
            TLOG_ERR("Get output tensor failed.\n");
            return -1;
        }
        // check tensor memory size ???
        void *addr = tensor->data;
        HI_U32 memSize = (HI_U32)(tensor->elem_size) * (tensor->elem_num);
        TLOG_INFO( "addr:%X memSize:%d\n", addr, memSize);
        int index = pstCnnNnieParam->pstModel->u32NetSegNum - 1;
        TLOG_INFO( "index:%d", index );
        copy_nnie_data(tensor, &pstCnnNnieParam->astSegData[index].astDst[i]);
    }

    return 0;
}

static int postrun(struct nn_device* dev, struct subgraph* subgraph)
{
    struct nnie_device* nnie_dev = (struct nnie_device*)dev;
    NnieGraph *nnie_graph = nnie_dev->nnie_graph;
    if (nnie_graph == nullptr)
    {
        TLOG_ERR("Get aipu graph handler failed.\n");
        return -1;
    }

    std::unordered_map<std::string, struct ir_tensor *>::iterator iter;
    for (iter = nnie_graph->output_tensors.begin(); iter != nnie_graph->output_tensors.end(); iter++)
    {
        struct ir_tensor *tensor = iter->second;
        if(tensor->data != NULL)
        {
            sys_free(tensor->data);
        }
    }

    delete nnie_graph;

    return 0;
}

static int init_nnie_device (struct nn_device* dev)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    struct nnie_device* nnie_dev = (struct nnie_device*) dev;
    nnie_dev->tv_start = tv.tv_sec;
    return 0;
}

struct nnie_device nnie_device = {
    .base = {.name = (char *)NNIE_DEVICE,
             .init = init_nnie_device,
             .prerun = prerun,
             .run = run,
             .postrun = postrun,
             .async_run = NULL,
             .async_wait = NULL,
             .release = NULL,
             .release_exec_graph = NULL
             },
    .nnie_graph = NULL,
    .tv_start = 0,
};

extern "C" int register_nnie_device(void)
{
    TLOG_INFO("Tengine plugin device %s is registered.\n", nnie_device.base.name);
    return register_nn_device(&nnie_device.base);
}

#ifndef STANDLONE_MODE
REGISTER_NN_DEVICE(&nnie_device.base);
#endif
