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

#include <iostream>
#include <assert.h>
#include <unordered_map>
#include <set>
#include <stdarg.h>

#include <string>
extern "C" {
#include "sys_port.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "compiler.h"
}

#include "sample_svp_nnie_software.h"
#include "nnie_serializer.h"
#include "nnie_param.h"
#include "nnie_model_cfg.hpp"

#define NNIE_CORE_NUM 1

namespace Tengine
{

static SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S s_stFastrcnnSoftwareParam = {0};
SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *ex_pstFastrcnnSoftwareParam = &s_stFastrcnnSoftwareParam;

static HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_SoftwareInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg,
                                                      SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RpnTmpBufSize = 0;
    HI_U32 u32RpnBboxBufSize = 0;
    HI_U32 u32GetResultTmpBufSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    /*RPN parameter init*/
    pstSoftWareParam->u32MaxRoiNum = pstCfg->u32MaxRoiNum;
    {
        pstSoftWareParam->u32ClassNum = 2;
        pstSoftWareParam->u32NumRatioAnchors = 1;
        pstSoftWareParam->u32NumScaleAnchors = 9;
        pstSoftWareParam->au32Scales[0] = 1.5 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[1] = 2.1 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[2] = 2.9 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[3] = 4.1 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[4] = 5.8 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[5] = 8.0 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[6] = 11.3 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[7] = 15.8 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Scales[8] = 22.1 * SAMPLE_SVP_QUANT_BASE;
        pstSoftWareParam->au32Ratios[0] = 2.44 * SAMPLE_SVP_QUANT_BASE;
    }

    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32MinSize = 16;
    pstSoftWareParam->u32FilterThresh = 16;
    pstSoftWareParam->u32SpatialScale = (HI_U32)(0.0625 * SAMPLE_SVP_QUANT_BASE);
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.7 * SAMPLE_SVP_QUANT_BASE);
    pstSoftWareParam->u32FilterThresh = 0;
    pstSoftWareParam->u32NumBeforeNms = 6000;

    char str1[] = "rpn_cls_score";
    char str2[] = "rpn_bbox_pred";
    pstSoftWareParam->apcRpnDataLayerName[0] = str1;
    pstSoftWareParam->apcRpnDataLayerName[1] = str2;
    for (i = 0; i < pstSoftWareParam->u32ClassNum; i++)
    {
        pstSoftWareParam->au32ConfThresh[i] = 1;
    }
    pstSoftWareParam->u32ValidNmsThresh = (HI_U32)(0.3 * SAMPLE_SVP_QUANT_BASE);
    pstSoftWareParam->stRpnBbox.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Height = pstCfg->u32MaxRoiNum;
    pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Width = SAMPLE_SVP_COORDI_NUM;
    pstSoftWareParam->stRpnBbox.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(SAMPLE_SVP_COORDI_NUM * sizeof(HI_U32));
    pstSoftWareParam->stRpnBbox.u32Num = 1;
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < pstNnieParam->pstModel->astSeg[0].u16DstNum; j++)
        {
            printf("[%s][%d]szName:%s\n", __FUNCTION__, __LINE__, pstNnieParam->pstModel->astSeg[0].astDstNode[j].szName);
            if (0 == strncmp(pstNnieParam->pstModel->astSeg[0].astDstNode[j].szName,
                             pstSoftWareParam->apcRpnDataLayerName[i],
                             SVP_NNIE_NODE_NAME_LEN))
            {
                pstSoftWareParam->aps32Conv[i] = (HI_S32 *)pstNnieParam->astSegData[0].astDst[j].u64VirAddr;
                pstSoftWareParam->au32ConvHeight[i] = pstNnieParam->pstModel->astSeg[0].astDstNode[j].unShape.stWhc.u32Height;
                pstSoftWareParam->au32ConvWidth[i] = pstNnieParam->pstModel->astSeg[0].astDstNode[j].unShape.stWhc.u32Width;
                pstSoftWareParam->au32ConvChannel[i] = pstNnieParam->pstModel->astSeg[0].astDstNode[j].unShape.stWhc.u32Chn;
                break;
            }
        }
        SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[0].u16DstNum),
                                  HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,failed to find report node %s!\n",
                                  pstSoftWareParam->apcRpnDataLayerName[i]);
        if (0 == i)
        {
            pstSoftWareParam->u32ConvStride = pstNnieParam->astSegData[0].astDst[j].u32Stride;
        }
    }

    /*calculate software mem size*/
    u32ClassNum = pstSoftWareParam->u32ClassNum;
    u32RpnTmpBufSize = SAMPLE_SVP_NNIE_RpnTmpBufSize(pstSoftWareParam->u32NumRatioAnchors,
                                                     pstSoftWareParam->u32NumScaleAnchors, pstSoftWareParam->au32ConvHeight[0],
                                                     pstSoftWareParam->au32ConvWidth[0]);
    u32RpnTmpBufSize = SAMPLE_SVP_NNIE_ALIGN16(u32RpnTmpBufSize);
    u32RpnBboxBufSize = pstSoftWareParam->stRpnBbox.u32Num *
                        pstSoftWareParam->stRpnBbox.unShape.stWhc.u32Height * pstSoftWareParam->stRpnBbox.u32Stride;
    u32GetResultTmpBufSize = SAMPLE_SVP_NNIE_FasterRcnn_GetResultTmpBufSize(pstCfg->u32MaxRoiNum, u32ClassNum);
    u32GetResultTmpBufSize = SAMPLE_SVP_NNIE_ALIGN16(u32GetResultTmpBufSize);
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstCfg->u32MaxRoiNum * sizeof(HI_U32) * SAMPLE_SVP_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstCfg->u32MaxRoiNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32RpnTmpBufSize + u32RpnBboxBufSize + u32GetResultTmpBufSize + u32DstRoiSize +
                   u32DstScoreSize + u32ClassRoiNumSize;

    /*malloc mem*/
    std::string name = "SAMPLE_RCNN_INIT";
    s32Ret = SAMPLE_COMM_SVP_MallocCached((char *)name.c_str(), NULL, (HI_U64 *)&u64PhyAddr,
                                          (void **)&pu8VirAddr, u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "Error,Malloc memory failed!\n");
    memset(pu8VirAddr, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /*set addr*/
    pstSoftWareParam->stRpnTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stRpnTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr);
    pstSoftWareParam->stRpnTmpBuf.u32Size = u32RpnTmpBufSize;

    pstSoftWareParam->stRpnBbox.u64PhyAddr = u64PhyAddr + u32RpnTmpBufSize;
    pstSoftWareParam->stRpnBbox.u64VirAddr = (HI_U64)(pu8VirAddr) + u32RpnTmpBufSize;

    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr + u32RpnBboxBufSize + u32RpnTmpBufSize;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnBboxBufSize + u32RpnTmpBufSize);
    pstSoftWareParam->stGetResultTmpBuf.u32Size = u32GetResultTmpBufSize;

    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32RpnBboxBufSize + u32RpnTmpBufSize + u32GetResultTmpBufSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnBboxBufSize + u32RpnTmpBufSize + u32GetResultTmpBufSize);
    pstSoftWareParam->stDstRoi.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) * SAMPLE_SVP_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum * SAMPLE_SVP_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32RpnBboxBufSize + u32RpnTmpBufSize + u32GetResultTmpBufSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnBboxBufSize + u32RpnTmpBufSize + u32GetResultTmpBufSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32RpnBboxBufSize + u32RpnTmpBufSize + u32GetResultTmpBufSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)(pu8VirAddr + u32RpnBboxBufSize + u32RpnTmpBufSize + u32GetResultTmpBufSize + u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}


const char * get_nnie_tensor_layout(SVP_NNIE_NODE_S* pstNode)
{
    switch(pstNode->unShape.u32Dim)
    {
        case 4:
            return "NCHW";
        case 3:
            return "TND";
        case 2:
            return "NC";
        default:
            return "";
    }
}

static int get_nnie_tensor_datatype(SVP_NNIE_NODE_S *pstNode)
{
    TLOG_INFO("pstNode->enType:%d\n",  pstNode->enType);
    switch (pstNode->enType)
    {
    case SVP_BLOB_TYPE_S32:
    case SVP_BLOB_TYPE_VEC_S32:
    case SVP_BLOB_TYPE_SEQ_S32:
        return TENGINE_DT_FP32;
    case SVP_BLOB_TYPE_U8:
        return TENGINE_DT_INT8;
    case SVP_BLOB_TYPE_YVU420SP:
    case SVP_BLOB_TYPE_YVU422SP:
    default:
        TLOG_ERR("Unknow enType: %d\n", pstNode->enType);
        return -1;
    }
}

static void get_nnie_tensor_dims(struct ir_tensor* tensor,SVP_NNIE_NODE_S *pstNode, int maxroiNum)
{
    TLOG_INFO("\npstNode name:%s\n", pstNode->szName);
    TLOG_INFO("pstNode enType:%d\n", pstNode->enType);
    TLOG_INFO("pstNode u32Dim:%u\n", pstNode->unShape.u32Dim);
    TLOG_INFO("pstNode u32Chn:%u\n", pstNode->unShape.stWhc.u32Chn);
    TLOG_INFO("pstNode u32Height:%u\n", pstNode->unShape.stWhc.u32Height);
    TLOG_INFO("pstNode u32Width:%u\n", pstNode->unShape.stWhc.u32Width);
    TLOG_INFO("maxroiNum:%d\n\n", maxroiNum);
    int i = 0;
    int dims[MAX_SHAPE_DIM_NUM * 2];
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNode->enType)
    {
        dims[i] = 1;
        i++;
    }
    else
    {
        if (maxroiNum)
        {
            dims[i] = maxroiNum;
            i++;
        }
        else
        {
            dims[i] = 1;
            i++;
        }
        dims[i] = pstNode->unShape.stWhc.u32Chn;
        i++;
        dims[i] = pstNode->unShape.stWhc.u32Height;
        i++;
        dims[i] = pstNode->unShape.stWhc.u32Width;
        i++;
    }
    set_ir_tensor_shape(tensor, dims, i);
}

int get_nnie_tensor_size(SVP_NNIE_NODE_S *pstDstNode)
{
    int tensor_size = pstDstNode->unShape.stWhc.u32Chn * pstDstNode->unShape.stWhc.u32Height * pstDstNode->unShape.stWhc.u32Width;
    printf("[%s][%d] tensor_size:%d  c:%d  h:%d  w:%d\n", __FUNCTION__, __LINE__, tensor_size,
           pstDstNode->unShape.stWhc.u32Chn, pstDstNode->unShape.stWhc.u32Height, pstDstNode->unShape.stWhc.u32Width);
    return tensor_size;
}

static int staticGraphreleaseFunc(struct serializer* s, struct ir_graph* graph, void* s_priv, void* dev_priv)
{
    TE_NNIE_CONTEXT_S *stTengineContxt = (TE_NNIE_CONTEXT_S *)(dev_priv);
    int s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(&stTengineContxt->stCnnModel);
    if (s32Ret != 0)
    {
        TLOG_ERR("SAMPLE_COMM_SVP_NNIE_UnloadModel fail\n");
    }
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(stTengineContxt->stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    if (s32Ret != 0)
    {
        TLOG_ERR("HI_MPI_SVP_NNIE_RemoveTskBuf fail\n");
    }
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(&stTengineContxt->stCnnNnieParam);
    if (s32Ret != 0)
    {
        TLOG_ERR("SAMPLE_COMM_SVP_NNIE_ParamDeinit fail\n");
    }
    delete stTengineContxt;
    return 0;
}

static int release_nnie_serializer(struct serializer* s)
{
    const char *b_init = std::getenv("NNIE_PLUGIN_INIT_SYS");
    int b_init_flag = 1;
    if (b_init)
        b_init_flag = std::strtoul(b_init, NULL, 10);

    if (b_init_flag)
        SAMPLE_COMM_SVP_CheckSysExit();
    TLOG_INFO("release_nnie_serializer\n");
    return 0;
}

static int init_serializer(struct serializer* s)
{
    TLOG_INFO("init_serializer\n");
    const char *b_init = std::getenv("NNIE_PLUGIN_INIT_SYS");
    int b_init_flag = 1;
    struct nnie_serializer* serializer = (struct nnie_serializer*)s;
    serializer->nnieContextCount = 0;
    if (b_init)
        b_init_flag = std::strtoul(b_init, NULL, 10);

    if (b_init_flag)
        SAMPLE_COMM_SVP_CheckSysInit();
    return 0;
}

std::string NNieNetTypetoString(SVP_NNIE_NET_TYPE_E enNetType)
{
    switch (enNetType)
    {
    case SVP_NNIE_NET_TYPE_CNN:
        return NNIE_OP_FORWARD;
    case SVP_NNIE_NET_TYPE_ROI:
        return NNIE_OP_FORWARD_WITH_BBOX;
    case SVP_NNIE_NET_TYPE_RECURRENT:
        return NNIE_OP_FORWARD;
    default:
        return nullptr;
    }
}

static int NNieNetTypetoOpType(SVP_NNIE_NET_TYPE_E enNetType)
{
    switch (enNetType)
    {
    case SVP_NNIE_NET_TYPE_CNN:
        return OP_NNIE_FORWARD;
    case SVP_NNIE_NET_TYPE_ROI:
        return OP_NNIE_FORWARD_WITHBBOX;
    case SVP_NNIE_NET_TYPE_RECURRENT:
        return OP_NNIE_FORWARD;
    default:
        return OP_INPUT;
    }
}

static void CreateInputNode(struct nnie_serializer* serializer, struct ir_graph* graph, SVP_NNIE_MODEL_S *model)
{
    int inputs_num = model->astSeg[0].u16SrcNum;
    int16_t* input_nodes = (int16_t*) sys_malloc(inputs_num * sizeof(int16_t));
    for (int i = 0; i < inputs_num; i++)
    {
        SVP_NNIE_NODE_S *pstSrcNode = &model->astSeg[0].astSrcNode[i];
        std::string name = "InputOp/node_" + std::to_string(pstSrcNode->u32NodeId);
        struct ir_node *node = create_ir_node(graph, name.c_str(), OP_INPUT, 1);
        input_nodes[i] = node->idx;
        //struct ir_node* node = CreateStaticNode(graph, name);
        add_node_attr(node, "InputOp", NULL, 0);
        // StaticOp *op = CreateStaticOp(graph, "InputOp");
        // SetNodeOp(node, op);

        struct ir_tensor* tensor = create_ir_tensor(graph, name.c_str(), get_nnie_tensor_datatype(pstSrcNode));
        // StaticTensor *tensor = CreateStaticTensor(graph, name);
        TLOG_INFO("tesnro name:%s\n", name.c_str());
        //SetTensorDataLayout(tensor, get_nnie_tensor_layout(pstSrcNode));
        get_nnie_tensor_dims(tensor, pstSrcNode, 0);
        //SetTensorDim(tensor, get_nnie_tensor_dims(pstSrcNode, 0));
        set_ir_node_output_tensor(node, node->output_num, tensor);
        //AddNodeOutputTensor(node, tensor);?

        //AddGraphInputNode(graph, node);
    }
    set_ir_graph_input_node(graph, input_nodes, inputs_num);
    sys_free(input_nodes);
}

static bool LoadNnieCpuNode(struct nnie_serializer* serializer, struct ir_graph* graph, struct ir_node* node, NnieCpuNode *cpunode)
{
    // set the inputs of the static node
    int inputs_num = cpunode->getInputNum();
    for (int i = 0; i < inputs_num; i++)
    {
        std::string name = cpunode->getInputTensorName(i);
        int idx = get_tensor_idx_from_name(graph, name.c_str());
        if (idx == -1)
        {
            TLOG_ERR("Input tensor not found: %s\n", name.c_str());
            return false;
        }
        else
        {
            set_ir_node_input_tensor(node, node->input_num, graph->tensor_list[idx]);
        }
    }

    // set the outputs of the static node
    int outputs_num = cpunode->getOutputNum();
    for (int i = 0; i < outputs_num; i++)
    {
        std::string name = cpunode->getOutputTensorName(i);
        TLOG_INFO("name:%s\n", name.c_str());
        int idx = get_tensor_idx_from_name(graph, name.c_str());
        //StaticTensor *tensor = FindTensor(graph, name);
        if (idx == -1)
        {
            TLOG_ERR("output tensor not found: %s\n", name.c_str());
            return false;
        }
        set_ir_node_output_tensor(node, node->output_num, graph->tensor_list[idx]);
    }

    // set the op of the static node
    TE_NNIE_CONTEXT_S *stTengineNnieContxt = (TE_NNIE_CONTEXT_S *)graph->dev_priv;
    TLOG_INFO("NnieOpName name:%s\n", node->name);
    add_node_attr(node, node->name, NULL, 0);
    //StaticOp *op = CreateStaticOp(graph, NnieOpName);
    nnie_param_t*  param = ( nnie_param_t* )node->op.param_mem;
    //NnieParam param = any_cast<NnieParam>(OpManager::GetOpDefParam(NnieOpName));
    param->nnie_node = &stTengineNnieContxt->stCnnNnieParam;
    param->software_param = &s_stFastrcnnSoftwareParam;
    TLOG_INFO("param software_param:%u\n", param->software_param);
    // SetOperatorParam(op, param);
    // SetNodeOp(node, op);

    //???
    // set the proposal of the static node?
    // DevProposal prop;
    // prop.dev_id = NNIE_DEVICE;
    // prop.level = DEV_PROPOSAL_STATIC;

    // Attribute &attrs = node->attrs;
    // attrs.SetAttr(DEV_PROPOSAL_ATTR, prop);

    return true;
}

static bool LoadNnieNode(struct nnie_serializer* serializer, struct ir_graph* graph, struct ir_node* node, SVP_NNIE_SEG_S *seg, std::string inputNodeName)
{
    // set the inputs of the static node
    int inputs_num = seg->u16SrcNum;
    for (int i = 0; i < inputs_num; i++)
    {
        SVP_NNIE_NODE_S *pstSrcNode = &seg->astSrcNode[i];
        std::string name = inputNodeName + "/node_" + std::to_string(pstSrcNode->u32NodeId);
        int idx = get_tensor_idx_from_name(graph, name.c_str());
        if (idx == -1)
        {
            TLOG_ERR("Input tensor not found: %s\n", name.c_str());
            std::string newname = std::string(node->name) + "/node_" + std::to_string(pstSrcNode->u32NodeId);
            struct ir_tensor* tensor = create_ir_tensor(graph, newname.c_str(), get_nnie_tensor_datatype(pstSrcNode));
            get_nnie_tensor_dims(tensor, pstSrcNode, 0);
            //SetTensorSize(tensor, get_nnie_tensor_size(pstSrcNode));
            set_ir_node_input_tensor(node, node->input_num, tensor);
            //AddNodeInputTensor(node, tensor);
        }
        else
        {
            set_ir_node_input_tensor(node, node->input_num, graph->tensor_list[idx]);
            //AddNodeInputTensor(node, tensor);
        }
    }

    // no need: set const tensor
    TE_NNIE_CONTEXT_S *stTengineNnieContxt = (TE_NNIE_CONTEXT_S *)graph->dev_priv;
    // set the outputs of the static node
    int outputs_num = seg->u16DstNum;
    for (int i = 0; i < outputs_num; i++)
    {
        SVP_NNIE_NODE_S *pstDstNode = &seg->astDstNode[i];
        std::string name = std::string(node->name) + "/" + pstDstNode->szName;

        //StaticTensor *tensor = CreateStaticTensor(graph, name);
        struct ir_tensor* tensor = create_ir_tensor(graph, name.c_str(), get_nnie_tensor_datatype(pstDstNode));
        TLOG_INFO("output tensor name:%s\n", name.c_str());
        //SetTensorDataType(tensor, get_nnie_tensor_datatype(pstDstNode));
        get_nnie_tensor_dims(tensor, pstDstNode, stTengineNnieContxt->stNnieCfg.u32MaxRoiNum);
        //SetTensorDim(tensor, get_nnie_tensor_dims(pstDstNode, stTengineNnieContxt->stNnieCfg.u32MaxRoiNum));
        //SetTensorSize(tensor, get_nnie_tensor_size(pstDstNode));?
        set_ir_node_output_tensor(node, node->output_num, tensor);
        //AddNodeOutputTensor(node, tensor);
    }

    // set the op of the static node
    std::string NnieOpName = NNieNetTypetoString(seg->enNetType);
    add_node_attr(node, NnieOpName.c_str(), NULL, 0);
    //StaticOp *op = CreateStaticOp(graph, NnieOpName);
    nnie_param_t*  param = ( nnie_param_t* )node->op.param_mem;
    param->nnie_node = &stTengineNnieContxt->stCnnNnieParam;
    param->software_param = &s_stFastrcnnSoftwareParam;

    //???
    // set the proposal of the static node
    // DevProposal prop;
    // prop.dev_id = NNIE_DEVICE;
    // prop.level = DEV_PROPOSAL_STATIC;

    // Attribute &attrs = node->attrs;
    // attrs.SetAttr(DEV_PROPOSAL_ATTR, prop);

    return true;
}

static void CreateOutputNode(struct nnie_serializer* serializer, struct ir_graph* graph)
{
    //key tensor idx value node idx
    std::unordered_map<uint16_t, uint16_t> input_tensor_map;
    //key tensor idx value node idx
    std::unordered_map<uint16_t, uint16_t> output_tensor_map;
    std::set<int16_t> output_node_set;

    for(int i = 0; i < graph->node_num; i++)
    {
        struct ir_node* node = graph->node_list[i];
        for(int j = 0; j < node->input_num; j++)
        {
            input_tensor_map[node->input_tensors[j]] = node->idx;
        }

        for(int k =0; k < node->output_num; k++)
        {
            output_tensor_map[node->output_tensors[k]] = node->idx;
        }
    }

    for(auto iter = output_tensor_map.begin(); iter != output_tensor_map.end(); iter++)
    {
        if( input_tensor_map.find(iter->first) == input_tensor_map.end())
        {
            output_node_set.insert(iter->second);
        }
    }
    int16_t* output_nodes = (int16_t*) sys_malloc( (output_node_set.size()) * sizeof(int16_t) );
    int output_nodes_num = 0;
    for(auto iter = output_node_set.begin(); iter != output_node_set.end(); iter++, output_nodes_num++)
    {
        output_nodes[output_nodes_num] = *iter;
    }
    set_ir_graph_output_node(graph, output_nodes, output_nodes_num);
    sys_free(output_nodes);
}

static bool LoadGraph(struct nnie_serializer* serializer, struct ir_graph* graph, SAMPLE_SVP_NNIE_MODEL_S *mode, NnieCpuNodes *CpuNodes)
{
    // set the domain, name and version of the static graph
    //SetGraphIdentity(graph, "nnie", graph->model_name, "v0.0.1");

    // set the input node list of the static graph
    CreateInputNode(serializer, graph, &mode->stModel);

    // load the nnie nodes
    int nodes_cnt = mode->stModel.u32NetSegNum;
    for (int i = 0; i < nodes_cnt; i++)
    {
        std::string nodeName = NNieNetTypetoString(mode->stModel.astSeg[i].enNetType);
        nodeName += "_" + std::to_string(i);
        int op_type = NNieNetTypetoOpType(mode->stModel.astSeg[i].enNetType);
        struct ir_node *node = create_ir_node(graph, nodeName.c_str(), op_type, OP_VERSION);
        std::string inputnodename = std::string("");
        if (i == 0)
            inputnodename = "InputOp";
        if (!LoadNnieNode(serializer, graph, node, &mode->stModel.astSeg[i], inputnodename))
        {
            TLOG_ERR("Load nnie nodes into static graph failed.\n");
            return false;
        }
    }

    if (CpuNodes)
    {
        for (int i = 0; i < CpuNodes->getnode_size(); i++)
        {
            NnieCpuNode cpunode = CpuNodes->getnode(i);
            std::string nodeName = cpunode.getName();
            //???
            struct ir_node *node = create_ir_node(graph, nodeName.c_str(), OP_NNIE_CPU_PROPOSAL, OP_VERSION);
            if (!LoadNnieCpuNode(serializer, graph, node, &cpunode))
            {
                TLOG_ERR("Load nnie cpu nodes into static graph failed.\n");
                return false;
            }
        }
    }

    CreateOutputNode(serializer, graph);
    return true;
}

static unsigned int GetFileNum()
{
    return 2;
}

static bool LoadModel(struct nnie_serializer* serializer, const std::vector<std::string> &file_list, struct ir_graph* graph)
{
    // check file number
    if (file_list.size() != GetFileNum())
    {
        TLOG_ERR("File number error.\n");
    }

    // call nnie api to parse the model file
    TE_NNIE_CONTEXT_S *stTengineNnieContxt = new TE_NNIE_CONTEXT_S();
    graph->dev_priv = stTengineNnieContxt;
    memset(&stTengineNnieContxt->stCnnModel, 0, sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    TLOG_INFO("nnie plugin memset:%d\n", serializer->nnieContextCount);
    int parse_ret = SAMPLE_COMM_SVP_NNIE_LoadModel(const_cast<char *>(file_list[0].c_str()), &stTengineNnieContxt->stCnnModel);
    if (parse_ret < 0)
    {
        TLOG_ERR("Parse nnie model failed.\n");
        return false;
    }

    //???
    // SetGraphSource(static_graph, file_list[0]);
    // SetGraphSourceFormat(static_graph, "nnie");
    // SetGraphConstTensorFile(static_graph, file_list[0]);

    TLOG_INFO("file_list[1]:%s\n",file_list[1].c_str());
    bool bCpuConfig = false;
    if (!file_list[1].empty() && file_list[1].find("noconfig") == std::string::npos)
    {
        bCpuConfig = true;
    }

    //int nniecfg
    SVP_NNIE_ID_E nnie_id = SVP_NNIE_ID_E((serializer->nnieContextCount) % NNIE_CORE_NUM);
    serializer->nnieContextCount++;
    stTengineNnieContxt->stNnieCfg.pszPic = const_cast<char *>(file_list[0].c_str());
    stTengineNnieContxt->stNnieCfg.u32MaxInputNum = 1; // max input image num in each batch
    if (stTengineNnieContxt->stCnnModel.stModel.u32NetSegNum == 2)
    {
        stTengineNnieContxt->stNnieCfg.u32MaxRoiNum = 300;
        serializer->segNet = true;
    }
    else
    {
        stTengineNnieContxt->stNnieCfg.u32MaxRoiNum = 0;
        serializer->segNet = false;
    }
    stTengineNnieContxt->stNnieCfg.aenNnieCoreId[0] = nnie_id; // set NNIE core
    stTengineNnieContxt->stNnieCfg.aenNnieCoreId[1] = nnie_id; // set NNIE core
    TLOG_INFO("nnie_id:%d u32MaxRoiNum:%d\n", nnie_id, stTengineNnieContxt->stNnieCfg.u32MaxRoiNum);

    //load the content of nnie nodes into the static graph
    if (bCpuConfig)
    {
        NnieCpuNodes myNnieCpuNodes(file_list[1]);
        bool load_ret = LoadGraph(serializer, graph, &stTengineNnieContxt->stCnnModel, &myNnieCpuNodes);
        if (!load_ret)
        {
            TLOG_ERR("Load into static graph failed.\n");
            return false;
        }
    }
    else
    {
        bool load_ret = LoadGraph(serializer, graph, &stTengineNnieContxt->stCnnModel, nullptr);
        if (!load_ret)
        {
            TLOG_ERR("Load into static graph failed.\n");
            return false;
        }
    }

    stTengineNnieContxt->stCnnNnieParam.pstModel = &stTengineNnieContxt->stCnnModel.stModel;

    int s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(&stTengineNnieContxt->stNnieCfg, &stTengineNnieContxt->stCnnNnieParam);
    if (s32Ret != 0)
    {
        TLOG_ERR("SAMPLE_COMM_SVP_NNIE_ParamInit fail\n");
        return false;
    }
    if (bCpuConfig)
    {
        s32Ret = SAMPLE_SVP_NNIE_FasterRcnn_SoftwareInit(&stTengineNnieContxt->stNnieCfg, &stTengineNnieContxt->stCnnNnieParam, &s_stFastrcnnSoftwareParam);
        if (s32Ret != 0)
        {
            TLOG_ERR("SAMPLE_SVP_NNIE_FasterRcnn_SoftwareInit fail\n");
            return false;
        }
    }

    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(stTengineNnieContxt->stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    if (s32Ret != 0)
    {
        TLOG_ERR("HI_MPI_SVP_NNIE_AddTskBuf(Phy Addr:%p, Vir Addr: %p, size: %u) fail\n",
                 stTengineNnieContxt->stCnnNnieParam.astForwardCtrl[0].stTskBuf.u64PhyAddr,
                 stTengineNnieContxt->stCnnNnieParam.astForwardCtrl[0].stTskBuf.u64VirAddr,
                 stTengineNnieContxt->stCnnNnieParam.astForwardCtrl[0].stTskBuf.u32Size);
        return false;
    }

    return true;
}

static bool LoadModel(struct nnie_serializer* serializer, const std::vector<const void *> &addr_list, const std::vector<int> &size_list,
                               struct ir_graph* graph)
{
    TLOG_ERR("LoadModel from mem\n");
    // check file number
    if (addr_list.size() != 1)
    {
        TLOG_ERR("File number %d error.\n", addr_list.size());
    }

    // call nnie api to parse the model file
    TE_NNIE_CONTEXT_S *stTengineNnieContxt = new TE_NNIE_CONTEXT_S();
    graph->dev_priv = stTengineNnieContxt;
    memset(&stTengineNnieContxt->stCnnModel, 0, sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    int parse_ret = SAMPLE_COMM_SVP_NNIE_LoadModel_Mem((HI_U8 *)addr_list[0], size_list[0], &stTengineNnieContxt->stCnnModel);
    if (parse_ret < 0)
    {
        TLOG_ERR("Parse nnie model failed.\n");
        return false;
    }

    //???
    // SetGraphSource(static_graph, "in_mem");
    // SetGraphSourceFormat(static_graph, "nnie");

    // bool bCpuConfig = false;

    //int nniecfg
    SVP_NNIE_ID_E nnie_id = SVP_NNIE_ID_E((serializer->nnieContextCount) % NNIE_CORE_NUM);
    serializer->nnieContextCount++;
    // stTengineNnieContxt->stNnieCfg.pszPic = "in_mem";
    stTengineNnieContxt->stNnieCfg.u32MaxInputNum = 1; // max input image num in each batch
    if (stTengineNnieContxt->stCnnModel.stModel.u32NetSegNum == 2)
    {
        stTengineNnieContxt->stNnieCfg.u32MaxRoiNum = 300;
        serializer->segNet = true;
    }
    else
    {
        stTengineNnieContxt->stNnieCfg.u32MaxRoiNum = 0;
        serializer->segNet = false;
    }
    stTengineNnieContxt->stNnieCfg.aenNnieCoreId[0] = nnie_id; // set NNIE core
    stTengineNnieContxt->stNnieCfg.aenNnieCoreId[1] = nnie_id; // set NNIE core
    TLOG_INFO("nnie_id:%d u32MaxRoiNum:%u\n", nnie_id, stTengineNnieContxt->stNnieCfg.u32MaxRoiNum);

    //load the content of nnie nodes into the static graph
    bool load_ret = LoadGraph(serializer, graph, &stTengineNnieContxt->stCnnModel, nullptr);
    if (!load_ret)
    {
        TLOG_ERR("Load into static graph failed.\n");
        return false;
    }

    stTengineNnieContxt->stCnnNnieParam.pstModel = &stTengineNnieContxt->stCnnModel.stModel;

    int s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(&stTengineNnieContxt->stNnieCfg, &stTengineNnieContxt->stCnnNnieParam);
    if (s32Ret != 0)
    {
        TLOG_ERR("SAMPLE_COMM_SVP_NNIE_ParamInit fail\n");
        return false;
    }
    // if (bCpuConfig)
    // {
    //     s32Ret = SAMPLE_SVP_NNIE_FasterRcnn_SoftwareInit(&stTengineNnieContxt->stNnieCfg, &stTengineNnieContxt->stCnnNnieParam, &s_stFastrcnnSoftwareParam);
    //     if (s32Ret != 0)
    //     {
    //         LOG_ERROR() << "SAMPLE_SVP_NNIE_FasterRcnn_SoftwareInit fail\n";
    //         return false;
    //     }
    // }

    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(stTengineNnieContxt->stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    if (s32Ret != 0)
    {
        TLOG_ERR("HI_MPI_SVP_NNIE_AddTskBuf fail\n");
        return false;
    }

    return true;
}

extern "C" int nnie_load_model(struct serializer* s, struct ir_graph* graph, const char* fname, va_list ap)
{
    std::vector<std::string> file_list;
    file_list.emplace_back(fname);
    const char* config = va_arg(ap, const char*);
    file_list.emplace_back(config);
    LoadModel((struct nnie_serializer*) s, file_list, graph);
    graph->serializer = s;
    return 0;
}

extern "C" int nnie_load_mem (struct serializer* s, struct ir_graph* graph, const void* addr, int size, va_list ap)
{
    std::vector<const void*> addr_list;
    std::vector<int> size_list;

    addr_list.push_back(addr);
    size_list.push_back(size);

    for(unsigned int i = 1; i < GetFileNum(); i++)
    {
        addr_list.emplace_back(va_arg(ap, const void*));
        size_list.emplace_back(va_arg(ap, int));
    }
    LoadModel((struct nnie_serializer*) s, addr_list, size_list, graph);
    graph->serializer = s;
    return 0;
}

static const char* get_name(struct serializer* s)
{
    return "nnie";
}

static struct nnie_serializer nnie_serializer = {
    .base =
        {
            .get_name = get_name,
            .load_model = nnie_load_model,
            .load_mem = nnie_load_mem,
            .unload_graph = staticGraphreleaseFunc,
            .register_op_loader = NULL,
            .unregister_op_loader = NULL,
            .init = init_serializer,
            .release = release_nnie_serializer,
        },
    .nnieContextCount = 0,
};

static int reg_nnie_serializer(void)
{
    return register_serializer(( struct serializer* )&nnie_serializer);
}

static int unreg_nnie_serializer(void* arg)
{
    return unregister_serializer(( struct serializer* )&nnie_serializer);
}



extern "C"
{
#ifndef STANDLONE_MODE
static void init_nnie_serializer(void)
#else
void init_nnie_serializer(void)
#endif
{
    reg_nnie_serializer();
    register_module_exit(MOD_DEVICE_LEVEL, "unreg_nnie_serializer", unreg_nnie_serializer, nullptr);
}

#ifndef STANDLONE_MODE
DECLARE_AUTO_INIT_FUNC(init_nnie_serializer);
#endif
}

} // namespace Tengine
