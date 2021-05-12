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
 * Copyright (c) 2021, Open AI Lab
 * Author: hhchen@openailab.com
 */

#include "acl_executor.hpp"

#include "acl_define.h"

extern "C"
{
#include "operator/op.h"
#include "utility/sys_port.h"
#include "utility/utils.h"

#include "convolution_param.h"
}


#define USE_CPU_CONVERT
//#define ACL_EXTENSTION
#ifdef __ANDROID__
#define dynamic_cast static_cast
#endif

static inline void copy_fp32_to_fp16(__fp16* f16, const float* f32, const int f32_size)
{
    for(unsigned int i = 0; i < f32_size / sizeof(float); i++)
        f16[i] = f32[i];
}

static inline void copy_fp16_to_fp32(float* f32, const __fp16* f16, const int f16_size)
{
    for(unsigned int i = 0; i < f16_size / sizeof(__fp16); i++)
        f32[i] = f16[i];
}

void copy_buffer(void* dest, const void* src, const int src_len, DataType dest_type, DataType src_type)
{
    if(dest_type == src_type)
        memcpy(dest, src, src_len);
    else if(dest_type == DataType::F16 && src_type == DataType::F32)
        copy_fp32_to_fp16(( __fp16* )dest, ( const float* )src, src_len);
    else if(dest_type == DataType::F32 && src_type == DataType::F16)
        copy_fp16_to_fp32(( float* )dest, ( const __fp16* )src, src_len);
    else
        fprintf(stderr, "copy_buffer may failed!!!");
}


template <typename T>
inline void _PermuteDataLayoutNCHWToNHWCInter(T* pvData, int n, int c, int h, int w, T* pvOutputData)
{
    T* pDataInputBuf = pvData;
    T* pDataOutputBuf = pvOutputData;
    int s32Cnt = 0;
    for(int z = 0; z < n; z++)
    {
        for(int i = 0; i < h; i++)
        {
            const T* pRowStartAddr = pDataInputBuf + w * i + z * w * h * c;
            for(int j = 0; j < w; j++)
            {
                for(int k = 0; k < c; k++)
                {
                    const T* pCkData = pRowStartAddr + k * (w * h) + j;
                    pDataOutputBuf[s32Cnt] = *pCkData;
                    s32Cnt++;
                }
            }
        }
    }
}

void _PermuteDataLayoutNCHWToNHWC(void* pvData, int n, int c, int h, int w, void* pvOutputData, int DataEleSize)
{
    assert(pvData != NULL);
    assert(pvOutputData != NULL);
    assert(DataEleSize == 1 || DataEleSize == 2 || DataEleSize == 4);
    if(DataEleSize == 4)
    {
        _PermuteDataLayoutNCHWToNHWCInter((int *) pvData, n, c, h, w, (int *) pvOutputData);
    }
    else if(DataEleSize == 2)
    {
        _PermuteDataLayoutNCHWToNHWCInter((short *) pvData, n, c, h, w, (short *) pvOutputData);
    }
    else
    {
        _PermuteDataLayoutNCHWToNHWCInter((char *) pvData, n, c, h, w, (char *) pvOutputData);
    }
}

template <typename T>
inline void _PermuteDatalayoutNHWCToNCHWInter(T* pvData, int n, int c, int h, int w, T* pvOutputData)
{
    T* pDataInputBuf = pvData;
    T* pDataOutputBuf = pvOutputData;
    int s32Cnt = 0;
    for(int z = 0; z < n; z++)
    {
        for(int i = 0; i < h; i++)
        {
            T* pRowStartAddr = pDataOutputBuf + w * i + z * w * h * c;
            for(int j = 0; j < w; j++)
            {
                for(int k = 0; k < c; k++)
                {
                    T* pCkData = pRowStartAddr + k * (w * h) + j;
                    *pCkData = pDataInputBuf[s32Cnt];
                    s32Cnt++;
                }
            }
        }
    }
}

inline void _PermuteDatalayoutNHWCToNCHW(void* pvData, int n, int c, int h, int w, void* pvOutputData, int DataEleSize)
{
    assert(pvData != NULL);
    assert(pvOutputData != NULL);
    assert(DataEleSize == 1 || DataEleSize == 2 || DataEleSize == 4);

    if(DataEleSize == 4)
    {
        _PermuteDatalayoutNHWCToNCHWInter(( int* )pvData, n, c, h, w, ( int* )pvOutputData);
    }
    else if(DataEleSize == 2)
    {
        _PermuteDatalayoutNHWCToNCHWInter(( short* )pvData, n, c, h, w, ( short* )pvOutputData);
    }
    else
    {
        _PermuteDatalayoutNHWCToNCHWInter(( char* )pvData, n, c, h, w, ( char* )pvOutputData);
    }
}

static void copy_itensor(CLTensor* cl_tensor, void* buf, int buf_size, bool to_tensor, DataType data_type)
{
    auto* cl_info = cl_tensor->info();

    const size_t slice_num = cl_info->tensor_shape().total_size_upper(2);
    const Strides strides = cl_info->strides_in_bytes();
    const PaddingSize padding = cl_info->padding();

    int slice_w = cl_info->dimension(0) + padding.left + padding.right;
    int slice_h = cl_info->dimension(1) + padding.bottom + padding.top;

    uint8_t* slice_ptr = cl_tensor->buffer();
    uint8_t* buf_ptr = ( uint8_t* )buf;

    for(unsigned int i = 0; i < slice_num; i++)
    {
        uint8_t* data_ptr = slice_ptr + padding.top * strides[1] + padding.left * strides[0];
        for(unsigned int h = 0; h < cl_info->dimension(1); h++)
        {
            int data_len = cl_info->dimension(0) * strides[0];
            int buf_len = data_len;

            if(data_type == DataType::F16)
                buf_len = data_len << 1;

            if(to_tensor)
            {
                copy_buffer(data_ptr, buf_ptr, buf_len, data_type, DataType::F32);
            }
            else
            {
                copy_buffer(buf_ptr, data_ptr, data_len, DataType::F32, data_type);
            }

            buf_ptr = buf_ptr + buf_len;

            data_ptr += slice_w * strides[0];
        }

        slice_ptr += slice_h * slice_w * strides[0];
    }
}

static void copy_to_itensor(CLTensor* cl_tensor, const void* buf, int buf_size, DataType tensor_dt)
{
    copy_itensor(cl_tensor, ( void* )buf, buf_size, true, tensor_dt);
}
void copy_from_itensor(const CLTensor* cl_tensor, void* buf, int buf_size, DataType tensor_dt)
{
    copy_itensor(( CLTensor* )cl_tensor, buf, buf_size, false, tensor_dt);
}

void copy_from_itensor_with_permuteNHWCTONCHW(CLTensor* cl_tensor, void* buf, int buf_size, DataType data_type)
{
    auto* cl_info = cl_tensor->info();

    // const size_t slice_num = cl_info->tensor_shape().total_size_upper(2);
    const Strides strides = cl_info->strides_in_bytes();
    const PaddingSize padding = cl_info->padding();

    int slice_w = cl_info->dimension(0) + padding.left + padding.right;
    int slice_h = cl_info->dimension(1) + padding.bottom + padding.top;

    uint8_t* slice_ptr = cl_tensor->buffer();
    // uint8_t* buf_ptr = ( uint8_t* )buf;

    float* pf32DataOutputBuf = ( float* )buf;
    // float *pf32DataInputRowBuf;
    uint8_t* pu8RowInputData;

    uint8_t* cur_slice_ptr = slice_ptr;

    int n = cl_info->dimension(3);
    int c = cl_info->dimension(0);
    int h = cl_info->dimension(2);
    int w = cl_info->dimension(1);

    int hw = (w * h);
    int offsetSize = padding.top * strides[1] + padding.left * strides[0];

    assert(n * h * w * c * 4 == buf_size);

    if(data_type == DataType::F32)
    {
        float* pf32DataInput;

        for(int z = 0; z < n; z++)
        {
            uint8_t* pu8SliceAddr = cur_slice_ptr + slice_h * slice_w * h * z * strides[0];
            float* pf32OutStartAddr0 = pf32DataOutputBuf + z * w * h * c;
            for(int i = 0; i < h; i++)
            {
                float* pf32OutStartAddr1 = pf32OutStartAddr0 + w * i;
                uint8_t* pu8SliceAddr_h_ele = pu8SliceAddr + i * slice_h * slice_w * strides[0];
                for(int j = 0; j < w; j++)
                {
                    pu8RowInputData = pu8SliceAddr_h_ele + offsetSize + j * strides[1];
                    pf32DataInput = ( float* )pu8RowInputData;

                    float* pf32RowStartAddr = pf32OutStartAddr1 + j;
                    for(int k = 0; k < c; k++)
                    {
                        float* pf32CkData = pf32RowStartAddr + k * hw;

                        *pf32CkData = pf32DataInput[k];
                    }
                }
            }
        }
    }
    else
    {
        assert(data_type == DataType::F16);

        __fp16* pf16DataInput;

        for(int z = 0; z < n; z++)
        {
            uint8_t* pu8SliceAddr = cur_slice_ptr + slice_h * slice_w * h * z * strides[0];
            float* pf32OutStartAddr0 = pf32DataOutputBuf + z * w * h * c;
            for(int i = 0; i < h; i++)
            {
                float* pf32OutStartAddr1 = pf32OutStartAddr0 + w * i;
                uint8_t* pu8SliceAddr_h_ele = pu8SliceAddr + i * slice_h * slice_w * strides[0];
                for(int j = 0; j < w; j++)
                {
                    pu8RowInputData = pu8SliceAddr_h_ele + offsetSize + j * strides[1];

                    pf16DataInput = ( __fp16* )pu8RowInputData;
                    float* pf32RowStartAddr = pf32OutStartAddr1 + j;
                    for(int k = 0; k < c; k++)
                    {
                        float* pf32CkData = pf32RowStartAddr + k * hw;

                        *pf32CkData = pf16DataInput[k];
                    }
                }
            }
        }
    }
}

CLGraph::CLGraph()
{
    bForcedNHWCMode_ = false;
    pcScratchMem_ = new char[8];
    l32ScratchMemSize_ = 0;
    l32AclNHWCOptimizeFlag_ = false;
};

CLGraph::~CLGraph()
{
    delete[] pcScratchMem_;
}

void CLGraph::init(std::string name, DataType type)
{
    name_ = name;
    data_type_ = type;
};

bool CLGraph::CreateACLGraph(struct subgraph* subgraph, DataType type, bool bDataLayoutOpFlag)
{
    CLScheduler::get().default_init();
    this->init("acl_graph", type); // tengine-lite's subgraph has not name.

    /* 1  Check Data Layout Work Mode*/
    this->bForcedNHWCMode_ = bDataLayoutOpFlag;

    /* first, process input nodes' input tensor */
    struct graph* ir_graph = subgraph->graph;
    int input_size = subgraph->input_num;
    for(int i = 0; i < input_size; i++)
    {
        struct tensor* tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);

        if(tensor->tensor_type != TENSOR_TYPE_CONST)
        {
            CLTensor* itensor = new CLTensor();
            int* dims = tensor->dims;
            const std::string& name = tensor->name;

            int dim_size = tensor->dim_num;

            for (int j=0;j<dim_size;j++)
            {
                if (dims[j] == 0)
                    dims[j] = 1;
            }

            if(dim_size == 4)
            {
                TensorInfo i_info =
                        TensorInfo(TensorShape(dims[3], dims[2], dims[1], dims[0]), 1, type);    // lxm add

                DataLayout aclDataLayout;
                aclDataLayout = (tensor->layout == TENGINE_LAYOUT_NCHW) ? DataLayout::NCHW : DataLayout::NHWC;
                i_info.set_data_layout(aclDataLayout);

                itensor->allocator()->init(i_info);
            }
            else if(dim_size == 3)
            {
                itensor->allocator()->init(TensorInfo(TensorShape(dims[2], dims[1], dims[0], 1), 1, type));
            }
            else if(dim_size == 2)
            {
                itensor->allocator()->init(TensorInfo(TensorShape(dims[1], dims[0], 1, 1), 1, type));
            }
            else if(dim_size == 1)
            {
                itensor->allocator()->init(TensorInfo(TensorShape(dims[0], 1, 1, 1), 1, type));
            }
            else
            {
                TLOG_ERR("Bad shape dim: %d\n", dim_size);
            }

            this->tensors_map_[name] = itensor;
        }
    }

    /* now, let's scan all nodes! */
    int node_size = subgraph->node_num;
    for(int i = 0; i < node_size; i++)
    {
        bool ret = false;
        struct node* node = get_ir_graph_node(ir_graph, subgraph->node_list[i]);
        uint16_t op_type = node->op.type;
        if(op_type == OP_CONST)
            continue;

        switch (op_type)
        {
            case OP_BATCHNORM:
            {
                struct node* node_next = get_ir_graph_node(ir_graph, subgraph->node_list[++i]);
                if(node_next->op.type != OP_SCALE)
                    ret = false;
                else
                    ret = this->AddBNLayer(node, node_next);
                break;
            }
            case OP_CAST:
                ret = this->AddCastLayer(node);
                break;
            case OP_CONCAT:
            {
                if (node->input_num < 2)
                    ret = this->AddDropoutLayer(node);
                else
                    ret = this->AddConcatLayer(node);
                break;
            }
            case OP_CLIP:
                ret = this->AddReLu6Layer(node);
                break;
            case OP_CONV:
                ret = this->AddConvolutionLayer(node);
                break;
            case OP_CROP:
                ret = this->AddCropLayer(node);
                break;
            case OP_DECONV:
            {
                /* deconv upsample */
                ret = this->AddInterpLayer(node);
                break;
            }
            case OP_DROPOUT:
                ret = this->AddDropoutLayer(node);
                break;
            case OP_ELTWISE:
                ret = this->AddEltwiseLayer(node);
                break;
            case OP_FC:
                ret = this->AddFCLayer(node);
                break;
            case OP_INPUT:
                ret = this->AddInputLayer(node);
                break;
            case OP_POOL:
                ret = this->AddPoolingLayer(node);
                break;
            case OP_RELU:
                ret = this->AddReLuLayer(node);
                break;
            case OP_RESHAPE:
                ret = this->AddReshapeLayer(node);
                break;
            case OP_RESIZE:
                ret = this->AddResizeLayer(node);
                break;
            case OP_SOFTMAX:
                ret = this->AddSoftmaxLayer(node);
                break;
            case OP_INTERP:
                ret = this->AddInterpLayer(node);
                break;
            default:
                fprintf(stderr,"Fail to support this op(%d)!!!\n",i);
                return false;
        }

        if(!ret)
        {
            fprintf(stderr,"Create ACL for Op %s failed! \n", get_op_name_from_type(op_type));
            return false;
        }
    }

    return true;
}

int CLGraph::prerun(struct subgraph *subgraph, struct acl_option* option)
{
    fprintf(stderr, "ACL initialized\n");

    DataType data_type = DataType::F32;

    if (nullptr != option)
    {
        switch(option->precision)
        {
            case TENGINE_DT_FP32:
                data_type = DataType::F32;
                fprintf(stderr, "ACL Backend set precision Float32\n");
                break;
            case TENGINE_DT_FP16:
                data_type = DataType::F16;
                fprintf(stderr, "ACL Backend set precision Float16\n");
                break;
            default:
                fprintf(stderr, "ACL Backend not support this %d data mode\n", option->precision);
                return -1;
        }
    }

    l32AclNHWCOptimizeFlag_ = true;
    this->CreateACLGraph(subgraph, data_type, l32AclNHWCOptimizeFlag_);

    auto ir_start = this->tensors_map_.begin();
    auto ir_end = this->tensors_map_.end();

    for(auto ir = ir_start; ir != ir_end; ir++)
    {
        CLTensor* tensor = ir->second;
        if(tensor->allocator()->info().is_resizable())
            tensor->allocator()->allocate();
    }

    struct graph* ir_graph = subgraph->graph;
    int output_node_size = subgraph->output_num;
    for (int i = 0; i < output_node_size; i++)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, subgraph->output_tensor_list[i]);
        void* mem_addr = output_tensor->data;
        if(mem_addr)
            continue;
        else
            output_tensor->data = (void*)sys_malloc(output_tensor->elem_size * output_tensor->elem_num);
    }

    return 0;
}

int CLGraph::run(struct subgraph *subgraph)
{
    struct graph* ir_graph = subgraph->graph;
    int input_number = subgraph->input_num;
    DataType data_type_ = this->data_type_;
    int l32ScratchMemSize_ = this->l32ScratchMemSize_;
    void* scratch_mem = nullptr;

    for(int i = 0; i < input_number; i++)
    {
        struct tensor* tensor_input = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);
        uint8_t tensor_type = tensor_input->tensor_type;
        if(tensor_type == TENSOR_TYPE_INPUT)
        {
            bool bDataPermute = false;
            if(l32AclNHWCOptimizeFlag_ == 1)
            {
                int DataLayoutType = tensor_input->layout;
                if(DataLayoutType == TENGINE_LAYOUT_NCHW)
                {
                    // need to permute data layout type to nhwc
                    int* Dim = tensor_input->dims;
                    int tengine_data_type = tensor_input->data_type;
                    int DataEleSize = gs32TengineDataElemetSize[tengine_data_type];

                    int l32InputDataSize = tensor_input->elem_size * tensor_input->elem_num;
//                    assert(l32InputDataSize == Dim[1] * Dim[2] * Dim[3] * 4 * Dim[0]);

                    scratch_mem = sys_malloc(l32InputDataSize);
                    assert(scratch_mem != NULL);
                    void* pvTensorDataMem = tensor_input->data;

                    // need to permute data to nhwc
                    _PermuteDataLayoutNCHWToNHWC(pvTensorDataMem, Dim[0], Dim[1], Dim[2], Dim[3], scratch_mem,
                                                 DataEleSize);

                    bDataPermute = true;
                }
            }

            CLTensor* acl_input = this->GetCLTensor(tensor_input->name);
            void* buf = (bDataPermute == true) ? scratch_mem : tensor_input->data;
            int size = tensor_input->elem_size * tensor_input->elem_num;
            acl_input->map();
            copy_to_itensor(acl_input, buf, size, data_type_);
            acl_input->unmap();
        }
        else
        {
            /* normal Input Node */
            bool bDataPermute = false;
            if(l32AclNHWCOptimizeFlag_ == 1)
            {
                int DataLayoutType = tensor_input->layout;
                if( DataLayoutType == TENGINE_LAYOUT_NCHW)
                {
                    // need to permute data layout type  to nhwc
                    int* Dim = tensor_input->dims;
                    int tengine_data_type = tensor_input->data_type;
                    int DataEleSize = gs32TengineDataElemetSize[tengine_data_type];

                    int l32InputDataSize = tensor_input->elem_size * tensor_input->elem_num;

                    scratch_mem = sys_malloc(l32InputDataSize);
                    assert(scratch_mem != NULL);
                    void* pvTensorDataMem = tensor_input->data;

                    // need to permute data to nhwc
                    _PermuteDataLayoutNCHWToNHWC(pvTensorDataMem, Dim[0], Dim[1], Dim[2], Dim[3], scratch_mem,
                                                 DataEleSize);

                    bDataPermute = true;
                }
            }
            CLTensor* acl_input = this->GetCLTensor(tensor_input->name);
            void* buf = (bDataPermute == true) ? scratch_mem : tensor_input->data;
            int size = tensor_input->elem_size * tensor_input->elem_num;
            acl_input->map();
            copy_to_itensor(acl_input, buf, size, data_type_);
            acl_input->unmap();
        }
    }

    if(!scratch_mem)
        sys_free(scratch_mem);

    int size = functions_map_.size();
    for(int i = 0; i < size; i++)
    {
        functions_map_[i]->run();
    }

    int output_num = subgraph->output_num;
    for(int i = 0; i < output_num; i++)
    {
        struct tensor* output = get_ir_graph_tensor(ir_graph, subgraph->output_tensor_list[i]);

        std::string output_name = output->name;
        CLTensor* cltensor = this->GetCLTensor(output_name);
        TensorInfo* ptTensorInfo = cltensor->info();
        int DataLayoutType = output->layout;
        DataLayout AclDataLayout = ptTensorInfo->data_layout();
        int AclDataLayoutforTengine = (AclDataLayout == DataLayout::NHWC) ? TENGINE_LAYOUT_NHWC : TENGINE_LAYOUT_NCHW;

        void* output_buf = output->data;
        int out_size = output->elem_size * output->elem_num;

        // if we enable ACL_OP flag, we need to permute output data back
        if(DataLayoutType != AclDataLayoutforTengine)
        {
            if(AclDataLayoutforTengine == TENGINE_LAYOUT_NHWC)
            {
                cltensor->map();
                copy_from_itensor_with_permuteNHWCTONCHW(cltensor, output_buf, out_size, data_type_);
                cltensor->unmap();
            }
            else
            {
                cltensor->map();
                copy_from_itensor(cltensor, output_buf, out_size, data_type_);
                cltensor->unmap();
            }
        }
        else
        {
            cltensor->map();
            copy_from_itensor(cltensor, output_buf, out_size, data_type_);
            cltensor->unmap();
        }
    }

    return 0;
}

int CLGraph::postrun(struct subgraph *subgraph)
{
    for (auto& var : tensors_map_)
    {
        if (nullptr != var.second)
        {
            delete var.second;
            var.second = nullptr;
        }
    }

    tensors_map_.clear();
    functions_map_.clear();

    return 0;
}

CLTensor* CLGraph::GetCLTensor(std::string name)
{
    return tensors_map_[name];
}
