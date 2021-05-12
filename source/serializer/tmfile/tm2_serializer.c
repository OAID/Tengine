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
 * Author: haitao@openailab.com
 */

#include "tm2_serializer.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <fcntl.h>

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "device/device.h"
#include "serializer/serializer.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

#include <string.h>


struct op_loader_entry
{
    int op_type;
    int op_version;
    tm2_op_loader_t loader;
    tm2_map_t op_map;
    tm2_map_t ver_map;
};

struct tm2_serializer
{
    struct serializer base;

    struct vector* loader_list;
};

static const char* tm2_name = "tengine";

static int unload_graph(struct serializer* s, struct graph* graph, void* s_priv, void* dev_priv);

static char* strdup_name(char* buf, int size)
{
    char* p = sys_malloc(size + 1);
    memcpy(p, buf, size);
    p[size] = 0x0;

    return p;
}

static inline const TM2_Header* get_tm_file_header(const char* base)
{
    return ( const TM2_Header* )(base);
}

static inline const TM2_Model* get_tm_file_model(const char* base, const TM2_Header* header)
{
    return ( const TM2_Model* )(base + header->offset_root);
}

static inline const TM2_Subgraph* get_tm_file_subgraph(const char* base, const TM2_Model* model)
{
    const TM2_Vector_offsets* v_graphs = ( TM2_Vector_offsets* )(base + model->offset_vo_subgraphs);
    const TM2_Subgraph* tm_graph = ( TM2_Subgraph* )(base + v_graphs->offsets[0]);

    return tm_graph;
}

static struct op_loader_entry* find_op_loader(struct tm2_serializer* s, int op_type, int op_version)
{
    int loader_num = get_vector_num(s->loader_list);

    for (int i = 0; i < loader_num; i++)
    {
        struct op_loader_entry* e = ( struct op_loader_entry* )get_vector_data(s->loader_list, i);

        if (e->op_type == op_type)
            return e;
    }

    return NULL;
}

static int register_tm2_op_loader(struct tm2_serializer* s, int op_type, int op_version, tm2_op_loader_t op_loader,
                                  tm2_map_t op_map, tm2_map_t ver_map)
{
    if (find_op_loader(s, op_type, op_version) != NULL)
    {
        // TLOG_ERR("serializer: op: %d version %d has loader already\n", op_type, op_version);
        TLOG_DEBUG("serializer: op: %d version %d has loader already\n", op_type, op_version);
        return -1;
    }

    struct op_loader_entry e;

    e.op_type = op_type;
    e.op_version = op_version;
    e.loader = op_loader;
    e.op_map = op_map;
    e.ver_map = ver_map;

    push_vector_data(s->loader_list, &e);

    return 0;
}

static int unregister_tm2_op_loader(struct tm2_serializer* s, int op_type, int op_version, tm2_op_loader_t op_loader)
{
    int n = get_vector_num(s->loader_list);

    for (int i = 0; i < n; i++)
    {
        struct op_loader_entry* e = ( struct op_loader_entry* )get_vector_data(s->loader_list, i);

        if (e->op_type == op_type && e->loader == op_loader)
        {
            remove_vector_via_pointer(s->loader_list, e);
            return 0;
        }
    }

    return -1;
}

static int load_graph_tensors(struct tm2_serializer* tm2_s, struct graph* graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Subgraph* tm_graph = priv->subgraph;

    const TM2_Vector_offsets* v_tensors = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_tensors);
    const TM2_Vector_offsets* v_buffers = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_buffers);

    graph->graph_layout = tm_graph->graph_layout;
    graph->model_layout = tm_graph->model_layout;

    // premute layout from NHWC to HCHW
    if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        graph->graph_layout = TENGINE_LAYOUT_NCHW;
        TLOG_DEBUG("premute layout: graph_layout from nhwc to nchw\n");
    }

    for (int i = 0; i < v_tensors->v_num; i++)
    {
        const TM2_Tensor* tm_tensor = ( TM2_Tensor* )(mem_base + v_tensors->offsets[i]);
        int flag_permute = 0;    // flag the tensor has to be permute
        int dims_org[8] = {0};

        /* TODO: check type definition */
        struct tensor* ir_tensor = create_ir_tensor(graph, NULL, tm_tensor->data_type);

        if (ir_tensor == NULL)
        {
            return -1;
        }

        ir_tensor->tensor_type = tm_tensor->type;

        /* name */
        if (tm_tensor->offset_s_tname != TM2_NOT_SET)
        {
            // TODO: using update the TM2 model
            const TM2_String* tm_str = ( TM2_String* )(mem_base + tm_tensor->offset_s_tname);
            ir_tensor->name = strdup_name(mem_base + tm_str->offset_data, tm_str->size);
        }

        /* shape */
        if (tm_tensor->offset_vd_dims != TM2_NOT_SET)
        {
            const TM2_Vector_dims* v_dims = ( TM2_Vector_dims* )(mem_base + tm_tensor->offset_vd_dims);

            if (tm_graph->model_layout == TENGINE_LAYOUT_NCHW)
            {
                set_ir_tensor_shape(ir_tensor, v_dims->dims, v_dims->v_num);
            }
            else
            {
                if (v_dims->v_num == 4)
                {
                    int dims[8] = {0};

                    dims_org[0] = v_dims->dims[0];
                    dims_org[1] = v_dims->dims[1];
                    dims_org[2] = v_dims->dims[2];
                    dims_org[3] = v_dims->dims[3];

                    dims[0] = v_dims->dims[0];    // c_out
                    dims[1] = v_dims->dims[3];    // c_in
                    dims[2] = v_dims->dims[1];    // h
                    dims[3] = v_dims->dims[2];    // w

                    set_ir_tensor_shape(ir_tensor, dims, v_dims->v_num);

                    flag_permute = 1;
                }
                else
                {
                    set_ir_tensor_shape(ir_tensor, v_dims->dims, v_dims->v_num);
                }
            }
        }

        /* load const type of tensor, such as the weight or bias for convolution node */
        if (ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            const TM2_Buffer* tm_buf = ( TM2_Buffer* )(mem_base + v_buffers->offsets[tm_tensor->buffer_id]);

            /* fill temp data buffer to benchmark */
            if (tm_buf->offset_data == TM2_NOT_SET)
            {
                ir_tensor->data = sys_malloc(ir_tensor->elem_num * ir_tensor->elem_size);
                memset(ir_tensor->data, 0, ir_tensor->elem_num * ir_tensor->elem_size);
                ir_tensor->free_host_mem = 1;
            }
            else
            {
                ir_tensor->data = mem_base + tm_buf->offset_data;

                if (ir_tensor->elem_size * ir_tensor->elem_num > tm_buf->size)
                {
                    TLOG_ERR("serializer: const tensor size in model is too small\n");
                    return -1;
                }

                /* permute the data of tensor from nhwc to nchw */
                if (flag_permute)
                {
                    int size = ir_tensor->elem_num;
                    int type = ir_tensor->data_type;

                    if (type == TENGINE_DT_FP32)
                    {
                        float* tensor_data_org = (float*)sys_malloc(size * sizeof(float));
                        float* original_date = ir_tensor->data;

                        for (int n = 0; n < size; n++)
                        {
                            tensor_data_org[n] = original_date[n];
                        }

                        int dims[4];
                        dims[0] = ir_tensor->dims[0];
                        dims[1] = ir_tensor->dims[1];
                        dims[2] = ir_tensor->dims[2];
                        dims[3] = ir_tensor->dims[3];

                        /* nhwc to nchw */
                        //                    fprintf(stderr, "%s:\n", ir_tensor->name);
                        //                    fprintf(stderr, "original %d, %d, %d, %d\n", dims_org[0], dims_org[1], dims_org[2],
                        //                    dims_org[3]); fprintf(stderr, "permute  %d, %d, %d, %d\n", dims[0], dims[1], dims[2],
                        //                    dims[3]);

                        float* input = tensor_data_org;
                        float* output = ir_tensor->data;

                        int cout = dims[0];
                        int cin = dims[1];
                        int h = dims[2];
                        int w = dims[3];

                        if (cin == 1)
                        {
                            for (int co = 0; co < cout; co++)
                            {
                                for (int hi = 0; hi < h; hi++)
                                {
                                    for (int wi = 0; wi < w; wi++)
                                    {
                                        int offset_org = 0;
                                        offset_org += co;
                                        offset_org += hi * w * cout;
                                        offset_org += wi * cout;

                                        int offset = 0;
                                        offset += co * h * w;
                                        offset += hi * w;
                                        offset += wi;

                                        output[offset] = input[offset_org];
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int co = 0; co < cout; co++)
                            {
                                for (int ci = 0; ci < cin; ci++)
                                {
                                    for (int hi = 0; hi < h; hi++)
                                    {
                                        for (int wi = 0; wi < w; wi++)
                                        {
                                            int offset_org = 0;
                                            offset_org += co * cin * h * w;
                                            offset_org += ci;
                                            offset_org += hi * w * cin;
                                            offset_org += wi * cin;

                                            int offset = 0;
                                            offset += co * cin * h * w;
                                            offset += ci * h * w;
                                            offset += hi * w;
                                            offset += wi;

                                            output[offset] = input[offset_org];
                                        }
                                    }
                                }
                            }
                        }

                        sys_free(tensor_data_org);
                    }

                    if (type == TENGINE_DT_UINT8 || type == TENGINE_DT_INT8)
                    {
                        unsigned char* tensor_data_org = ( unsigned char* )sys_malloc(size * sizeof(unsigned char));
                        unsigned char* original_date = ir_tensor->data;

                        for (int n = 0; n < size; n++)
                        {
                            tensor_data_org[n] = original_date[n];
                        }

                        int dims[4];
                        dims[0] = ir_tensor->dims[0];
                        dims[1] = ir_tensor->dims[1];
                        dims[2] = ir_tensor->dims[2];
                        dims[3] = ir_tensor->dims[3];

                        /* nhwc to nchw */
//                        fprintf(stderr, "%s:\n", ir_tensor->name);
//                        fprintf(stderr, "original %d, %d, %d, %d\n", dims_org[0], dims_org[1], dims_org[2], dims_org[3]);
//                        fprintf(stderr, "permute  %d, %d, %d, %d\n", dims[0], dims[1], dims[2], dims[3]);

                        unsigned char* input = tensor_data_org;
                        unsigned char* output = ir_tensor->data;

                        int cout = dims[0];
                        int cin = dims[1];
                        int h = dims[2];
                        int w = dims[3];

                        if (cin == 1)
                        {
                            for (int co = 0; co < cout; co++)
                            {
                                for (int hi = 0; hi < h; hi++)
                                {
                                    for (int wi = 0; wi < w; wi++)
                                    {
                                        int offset_org = 0;
                                        offset_org += co;
                                        offset_org += hi * w * cout;
                                        offset_org += wi * cout;

                                        int offset = 0;
                                        offset += co * h * w;
                                        offset += hi * w;
                                        offset += wi;

                                        output[offset] = input[offset_org];
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int co = 0; co < cout; co++)
                            {
                                for (int ci = 0; ci < cin; ci++)
                                {
                                    for (int hi = 0; hi < h; hi++)
                                    {
                                        for (int wi = 0; wi < w; wi++)
                                        {
                                            int offset_org = 0;
                                            offset_org += co * cin * h * w;
                                            offset_org += ci;
                                            offset_org += hi * w * cin;
                                            offset_org += wi * cin;

                                            int offset = 0;
                                            offset += co * cin * h * w;
                                            offset += ci * h * w;
                                            offset += hi * w;
                                            offset += wi;

                                            output[offset] = input[offset_org];
                                        }
                                    }
                                }
                            }
                        }

                        sys_free(tensor_data_org);
                    }
                }
            }
        }

        /* load vector type of tensor */
        if (tm_tensor->offect_vo_quantparams != TM2_NOT_SET)
        {
            const TM2_Vector_offsets* v_quantparams =
                    ( TM2_Vector_offsets* )(mem_base + tm_tensor->offect_vo_quantparams);

            /* currently only support one quant param */
            ir_tensor->quant_param_num = v_quantparams->v_num;
            if (v_quantparams->v_num == 1)
            {
                const TM2_QuantParam* tm_qtparam = ( TM2_QuantParam* )(mem_base + v_quantparams->offsets[0]);
                ir_tensor->scale = tm_qtparam->scale;
                ir_tensor->zero_point = tm_qtparam->zero_point;

//                printf("name %s, scale %f, zero %d\n", ir_tensor->name, ir_tensor->scale, ir_tensor->zero_point);
            }
            else if (v_quantparams->v_num > 1)
            {
                // to do : need to be updated
                ir_tensor->scale_list = ( float* )sys_malloc(sizeof(float) * v_quantparams->v_num);
                ir_tensor->zp_list = ( int* )sys_malloc(sizeof(int) * v_quantparams->v_num);

                for (int j = 0; j < v_quantparams->v_num; j++)
                {
                    const TM2_QuantParam* tm_qtparam = ( TM2_QuantParam* )(mem_base + v_quantparams->offsets[j]);
                    ir_tensor->scale_list[j] = tm_qtparam->scale;
                    ir_tensor->zp_list[j] = tm_qtparam->zero_point;
                }
            }
        }
    }
    return 0;
}

static int load_graph_nodes(struct tm2_serializer* tm2_s, struct graph* ir_graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Subgraph* tm_graph = priv->subgraph;
    const TM2_Vector_offsets* v_nodes = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_seq_nodes);

    unsigned int i;

    for (i = 0; i < v_nodes->v_num; i++)
    {
        const TM2_Node* tm_node = ( TM2_Node* )(mem_base + v_nodes->offsets[i]);
        const TM2_Operator* tm_operator = ( TM2_Operator* )(mem_base + tm_node->offset_t_operator);
        int op_type = tm_operator->operator_type;
        int op_version = tm_operator->op_ver;

        struct op_loader_entry* e = find_op_loader(tm2_s, op_type, op_version);

        if (e == NULL)
        {
            TLOG_ERR("serializer: cannot find op loader for op: %d version: %d\n", op_type, op_version);
            break;
        }

        int op_type_mapped = op_type;

        if (e->op_map)
            op_type_mapped = e->op_map(op_type);

        int op_ver_mapped = op_version;

        if (e->ver_map)
            op_ver_mapped = e->ver_map(op_version);

        struct node* ir_node = create_ir_node(ir_graph, NULL, op_type_mapped, op_ver_mapped);

        if (ir_node == NULL)
        {
            break;
        }

        if (tm_node->offset_s_nname != TM2_NOT_SET)
        {
            const TM2_String* str = ( TM2_String* )(mem_base + tm_node->offset_s_nname);
            // TODO: update with new tm2
            ir_node->name = strdup_name(mem_base + str->offset_data, str->size);
        }

        /* node inputs */
        if (tm_node->offset_vi_input_tensors != TM2_NOT_SET)
        {
            const TM2_Vector_indices* v_input_tensors =
                    ( TM2_Vector_indices* )(mem_base + tm_node->offset_vi_input_tensors);

            for (int j = 0; j < v_input_tensors->v_num; j++)
            {
                int tensor_idx = v_input_tensors->indices[j];

                if (tensor_idx < 0 || tensor_idx >= ir_graph->tensor_num)
                {
                    TLOG_ERR("invalid input tensor slot: %d idx: %d for node: %d\n", j, tensor_idx, ir_node->index);
                    break;
                }

                struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, tensor_idx);

                set_ir_node_input_tensor(ir_node, j, ir_tensor);
            }
        }

        if (tm_node->offset_vi_output_tensors == TM2_NOT_SET)
        {
            TLOG_ERR("node: %d has no output\n", ir_node->index);
            break;
        }

        const TM2_Vector_indices* v_output_tensors =
                ( TM2_Vector_indices* )(mem_base + tm_node->offset_vi_output_tensors);

        for (int j = 0; j < v_output_tensors->v_num; j++)
        {
            int tensor_idx = v_output_tensors->indices[j];

            if (tensor_idx < 0 || tensor_idx >= ir_graph->tensor_num)
            {
                TLOG_ERR("invalid output tensor slot: %d idx: %d for node: %d\n", j, tensor_idx, ir_node->index);
                break;
            }

            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, tensor_idx);

            set_ir_node_output_tensor(ir_node, j, ir_tensor);
        }

        /* axis param from nhwc to nchw */
        if (tm_graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            if (op_type == TM2_OPTYPE_SOFTMAX)
            {
                TM2_SoftmaxParam* tm_param = ( TM2_SoftmaxParam* )(mem_base + tm_operator->offset_t_param);

                if (tm_param->axis == 3)
                    tm_param->axis = 1;
                else
                    fprintf(stderr, "softmax, axis is not 3, need todo something\n");
            }

            //            if (op_type == TM2_OPTYPE_CONCAT)
            //            {
            //                TM2_ConcatParam* tm_param = (TM2_ConcatParam*)(mem_base + tm_operator->offset_t_param);
            //
            //                if (tm_param->axis == 3)
            //                    tm_param->axis = 1;
            //                else if (tm_param->axis == 1)
            //                    tm_param->axis = 2;
            //                else
            //                    fprintf(stderr, "concat, axis is not 3 or 1, need todo something\n");
            //            }

            if (op_type == TM2_OPTYPE_REDUCTION)
            {
                TM2_ReductionParam* tm_param = ( TM2_ReductionParam* )(mem_base + tm_operator->offset_t_param);

                if (tm_param->dim_0 == 1 && tm_param->dim_1 == 2)
                {
                    tm_param->dim_0 = 2;
                    tm_param->dim_1 = 3;
                }
                else if(tm_param->dim_0 == -1)
                {
                    tm_param->dim_0 = 4;
                }
                else
                {
                    fprintf(stderr, "reduction, nhwc to nchw, need todo something\n");
                }
            }

            if (op_type == TM2_OPTYPE_PAD)
            {
                TM2_PadParam* tm_param = ( TM2_PadParam* )(mem_base + tm_operator->offset_t_param);

                int pads[8] = {0};
                pads[0] = tm_param->pad_n_0;    // n
                pads[1] = tm_param->pad_n_1;

                pads[2] = tm_param->pad_c_0;    // h
                pads[3] = tm_param->pad_c_1;

                pads[4] = tm_param->pad_h_0;    // w
                pads[5] = tm_param->pad_h_1;

                pads[6] = tm_param->pad_w_0;    // c
                pads[7] = tm_param->pad_w_1;

                /* nhwc to nchw */
                tm_param->pad_c_0 = pads[6];    // c
                tm_param->pad_c_1 = pads[7];

                tm_param->pad_h_0 = pads[2];    // h
                tm_param->pad_h_1 = pads[3];

                tm_param->pad_w_0 = pads[4];    // w
                tm_param->pad_w_1 = pads[5];
            }

            if (op_type == TM2_OPTYPE_STRIDEDSLICE)
            {
                TM2_StridedSliceParam* tm_param = ( TM2_StridedSliceParam* )(mem_base + tm_operator->offset_t_param);

                int begin[4] = {0};
                int end[4] = {0};
                int stride[4] = {0};

                begin[0] = tm_param->begin_n;
                begin[1] = tm_param->begin_w;
                begin[2] = tm_param->begin_c;
                begin[3] = tm_param->begin_h;

                end[0] = tm_param->end_n;
                end[1] = tm_param->end_w;
                end[2] = tm_param->end_c;
                end[3] = tm_param->end_h;

                stride[0] = tm_param->stride_n;
                stride[1] = tm_param->stride_w;
                stride[2] = tm_param->stride_c;
                stride[3] = tm_param->stride_h;

                tm_param->begin_n = begin[0];
                tm_param->begin_c = begin[1];
                tm_param->begin_h = begin[2];
                tm_param->begin_w = begin[3];

                tm_param->end_n = end[0];
                tm_param->end_c = end[1];
                tm_param->end_h = end[2];
                tm_param->end_w = end[3];

                tm_param->stride_n = stride[0];
                tm_param->stride_c = stride[1];
                tm_param->stride_h = stride[2];
                tm_param->stride_w = stride[3];
            }

            if (op_type == TM2_OPTYPE_RESHAPE)
            {
                TM2_ReshapeParam* tm_param = ( TM2_ReshapeParam* )(mem_base + tm_operator->offset_t_param);
                TM2_Vector_dims* v_reshape = ( TM2_Vector_dims* )(mem_base + tm_param->offset_re_shape);

                if (tm_param->offset_re_shape != TM2_NOT_SET)
                {
                    int dims[MAX_SHAPE_DIM_NUM * 2] = {0};
                    int reshape_num = v_reshape->v_num;

                    if (reshape_num > MAX_SHAPE_DIM_NUM * 2)
                    {
                        fprintf(stderr, "the reshape dims is to bigger than 8\n");
                        return -1;
                    }

                    for (int j = 0; j < reshape_num; j++)
                        dims[j] = v_reshape->dims[j];

                    if (reshape_num == 3)
                    {
                        v_reshape->dims[0] = dims[0];
                        v_reshape->dims[1] = dims[2];
                        v_reshape->dims[2] = dims[1];
                    }

                    if (reshape_num == 4)
                    {
                        v_reshape->dims[0] = dims[0];
                        v_reshape->dims[1] = dims[3];
                        v_reshape->dims[2] = dims[1];
                        v_reshape->dims[3] = dims[2];
                    }

                    if (reshape_num >= 5)
                    {
                        fprintf(stderr, "nhwc reshape num >= 5, todo support\n");
                        return -1;
                    }
                }
                else
                {
                    fprintf(stderr, "the version of reshape is older, todo support\n");
                    return -1;
                }
            }
        }

        /* load the op parameters */
        if (e->loader != NULL_TM2_OP_LOADER && e->loader(ir_graph, ir_node, tm_node, tm_operator) < 0)
        {
            TLOG_ERR("failed to load op: %d version: %d for node: %d\n", op_type, op_version, ir_node->index);
            break;
        }
    }

    if (i < v_nodes->v_num)
    {
        return -1;
    }

    return 0;
}

static int set_graph_io_nodes(struct tm2_serializer* tm2_s, struct graph* ir_graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Subgraph* tm_graph = priv->subgraph;
    const TM2_Vector_indices* v_input_nodes = ( TM2_Vector_indices* )(mem_base + tm_graph->offset_vi_input_indices);
    const TM2_Vector_indices* v_output_nodes = ( TM2_Vector_indices* )(mem_base + tm_graph->offset_vi_output_indices);

    int16_t* node_idx = ( int16_t* )sys_malloc(sizeof(int16_t) * v_input_nodes->v_num);

    if (node_idx == NULL)
    {
        return -1;
    }

    for (unsigned int i = 0; i < v_input_nodes->v_num; i++)
    {
        node_idx[i] = v_input_nodes->indices[i];
    }

    set_ir_graph_input_node(ir_graph, node_idx, v_input_nodes->v_num);

    sys_free(node_idx);

    node_idx = ( int16_t* )sys_malloc(sizeof(int16_t) * v_output_nodes->v_num);

    for (unsigned int i = 0; i < v_output_nodes->v_num; i++)
    {
        node_idx[i] = v_output_nodes->indices[i];
    }

    set_ir_graph_output_node(ir_graph, node_idx, v_output_nodes->v_num);

    sys_free(node_idx);

    return 0;
}

static int load_graph_sub_info(struct tm2_serializer* s, struct graph* graph, struct tm2_priv* priv)
{
    char* mem_base = ( char* )priv->base;
    const TM2_Vector_offsets* v_graphs = ( TM2_Vector_offsets* )(mem_base + priv->model->offset_vo_subgraphs);
    const TM2_Subgraph* tm_graph = priv->subgraph;
    const TM2_Vector_offsets* v_sub_info = ( TM2_Vector_offsets* )(mem_base + tm_graph->offset_vo_sub_info);

    if (v_sub_info == TM2_NOT_SET || v_graphs->v_num == 1)
    {
        return 0;
    }

    fprintf(stderr, "---load graph sub info!---\n");

    int sub_graph_num = v_sub_info->v_num;
    for (int i = 0; i < sub_graph_num; i++)
    {
        struct subgraph* subgraph = ( struct subgraph* )sys_malloc(sizeof(struct subgraph));
        init_ir_subgraph(graph, subgraph, i);

        TM2_Sub_Info* sub_info = ( TM2_Sub_Info* )(mem_base + v_sub_info->offsets[i]);
        subgraph->index = sub_info->subgraph_id;
        subgraph->input_wait_count = sub_info->input_wait_count;

        // to do. data type? device type?
        subgraph->device = find_default_device();
        TM2_String* device_name = (TM2_String*)(mem_base + sub_info->offset_s_device_name);
        // subgraph->nn_dev->name = strdup_name(mem_base + device_name->offset_data, device_name->size);
        char* name = (char*)(mem_base + device_name->offset_data);

        TM2_Vector_indices* v_node_list = ( TM2_Vector_indices* )(mem_base + sub_info->offset_vi_node_list);
        subgraph->node_num = v_node_list->v_num;
        subgraph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * subgraph->node_num);
        for (int j = 0; j < v_node_list->v_num; j++)
        {
            subgraph->node_list[j] = v_node_list->indices[j];
        }

        TM2_Vector_indices* v_input_tensor = ( TM2_Vector_indices* )(mem_base + sub_info->offset_vi_input_tensor);
        subgraph->input_num = v_input_tensor->v_num;
        subgraph->input_tensor_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * subgraph->input_num);
        for (int j = 0; j < v_input_tensor->v_num; j++)
        {
            subgraph->input_tensor_list[j] = v_input_tensor->indices[j];
        }

        TM2_Vector_indices* v_output_tensor = ( TM2_Vector_indices* )(mem_base + sub_info->offset_vi_output_tensor);
        subgraph->output_num = v_output_tensor->v_num;
        subgraph->output_tensor_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * subgraph->output_num);
        for (int j = 0; j < v_output_tensor->v_num; j++)
        {
            subgraph->output_tensor_list[j] = v_output_tensor->indices[j];
        }

        // dump sub info
        fprintf(stderr, "sub id:%d, node num: %d, input num:%d, output num:%d, input wait count:%d, device name:%s\n",
                subgraph->index, subgraph->node_num, subgraph->input_num, subgraph->output_num, subgraph->input_wait_count, subgraph->device->name);

        push_vector_data(graph->subgraph_list, &subgraph);
    }

    return 0;
}

static int load_graph(struct serializer* s, struct graph* graph, struct tm2_priv* priv)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;

    /* version check */
    if (priv->header->ver_main != TM2_FILE_VER_MAIN)
    {
        TLOG_ERR("model is not version 2\n");
        return -1;
    }

    if (load_graph_tensors(tm2_s, graph, priv) < 0)
        goto error;

    if (load_graph_nodes(tm2_s, graph, priv) < 0)
        goto error;

    if (set_graph_io_nodes(tm2_s, graph, priv) < 0)
        goto error;

    if (load_graph_sub_info(tm2_s, graph, priv) < 0)
        goto error;

    return 0;

    error:
    unload_graph(s, graph, priv, NULL);
    return -1;
}

static int load_model(struct serializer* s, struct graph* graph, const char* fname, va_list ap)
{
    struct stat stat;

    int fd = open(fname, O_RDONLY);

    if (fd < 0)
    {
        TLOG_ERR("cannot open file %s\n", fname);
        return -1;
    }

    fstat(fd, &stat);

    int file_len = stat.st_size;

    //    void* mem_base = mmap(NULL, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
    //
    //    if(mem_base == MAP_FAILED)
    //    {
    //        set_tengine_errno(errno);
    //        close(fd);
    //        return -1;
    //    }

    void* mem_base = ( void* )sys_malloc(file_len);
    int ret = read(fd, mem_base, file_len);

    struct tm2_priv* priv = ( struct tm2_priv* )sys_malloc(sizeof(struct tm2_priv));

    if (priv == NULL)
    {
        close(fd);
        return -1;
    }

    priv->fd = fd;
    priv->mem_len = file_len;
    priv->base = mem_base;
    priv->header = get_tm_file_header(mem_base);
    priv->model = get_tm_file_model(mem_base, priv->header);
    priv->subgraph = get_tm_file_subgraph(mem_base, priv->model);

    graph->serializer = s;
    graph->serializer_privacy = priv;
    graph->device_privacy = NULL;

    return load_graph(s, graph, priv);
}

static int load_mem(struct serializer* s, struct graph* graph, const void* addr, int size, va_list ap)
{
    struct tm2_priv* priv = ( struct tm2_priv* )sys_malloc(sizeof(struct tm2_priv));

    if (priv == NULL)
    {
        return -1;
    }

    priv->fd = -1;
    priv->mem_len = size;
    priv->base = addr;
    priv->header = get_tm_file_header(addr);
    priv->model = get_tm_file_model(addr, priv->header);
    priv->subgraph = get_tm_file_subgraph(addr, priv->model);

    graph->serializer = s;
    graph->serializer_privacy = priv;
    graph->device_privacy = NULL;

    return load_graph(s, graph, priv);
}

static int unload_graph(struct serializer* s, struct graph* graph, void* s_priv, void* dev_priv)
{
    struct tm2_priv* priv = ( struct tm2_priv* )s_priv;

    if (priv->fd >= 0)
    {
        // munmap(( void* )priv->base, priv->mem_len);
        close(priv->fd);
        priv->fd = -1;
    }

    if (priv->base)
    {
        sys_free(( void* )priv->base);
        priv->base = NULL;
    }

    graph->serializer = NULL;
    graph->serializer_privacy = NULL;

    sys_free(priv);
    priv = NULL;

    return 0;
}

/* a simple wrapper for type convsion */
static int register_op_loader(struct serializer* s, int op_type, int op_ver, void* op_load_func, void* op_map_func,
                              void* ver_map_func)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;
    tm2_op_loader_t op_load = op_load_func;
    tm2_map_t op_map = op_map_func;
    tm2_map_t ver_map = ver_map_func;

    return register_tm2_op_loader(tm2_s, op_type, op_ver, op_load, op_map, ver_map);
}

static int unregister_op_loader(struct serializer* s, int op_type, int op_ver, void* op_load_func)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;
    tm2_op_loader_t op_load = op_load_func;

    return unregister_tm2_op_loader(tm2_s, op_type, op_ver, op_load);
}

static const char* get_name(struct serializer* s)
{
    return tm2_name;
}

static int const_op_map(int op)
{
    return OP_CONST;
}

static int input_op_map(int op)
{
    return OP_INPUT;
}

static int init_tm2_serializer(struct serializer* s)
{
    struct tm2_serializer* tm2_s = ( struct tm2_serializer* )s;

    tm2_s->loader_list = create_vector(sizeof(struct op_loader_entry), NULL);

    if (tm2_s->loader_list == NULL)
        return -1;

    s->register_op_loader(s, TM2_OPTYPE_INPUTOP, 1, NULL_TM2_OP_LOADER, input_op_map, NULL);
    s->register_op_loader(s, TM2_OPTYPE_CONST, 1, NULL_TM2_OP_LOADER, const_op_map, NULL);

    return 0;
}

static int release_tm2_serializer(struct serializer* s)
{
    struct tm2_serializer* tm2_s = (struct tm2_serializer*)s;

    s->unregister_op_loader(s, TM2_OPTYPE_INPUTOP, 1, NULL_TM2_OP_LOADER);
    s->unregister_op_loader(s, TM2_OPTYPE_CONST, 1, NULL_TM2_OP_LOADER);

    release_vector(tm2_s->loader_list);

    return 0;
}

static struct tm2_serializer tm2_serializer = {
        .base =
                {
                        .get_name = get_name,
                        .load_model = load_model,
                        .load_mem = load_mem,
                        .unload_graph = unload_graph,
                        .register_op_loader = register_op_loader,
                        .unregister_op_loader = unregister_op_loader,
                        .init = init_tm2_serializer,
                        .release = release_tm2_serializer,
                },
        .loader_list = NULL,
};


int register_tm2_serializer()
{
    return register_serializer((struct serializer*)&tm2_serializer);
}


int unregister_tm2_serializer()
{
    return unregister_serializer((struct serializer*)&tm2_serializer);
}
