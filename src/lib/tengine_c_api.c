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
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

#include "compiler.h"
#include "sys_port.h"
#include "lock.h"
#include "module.h"
#include "cpu.h"

#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_c_api.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_exec.h"
#include "nn_device.h"
#include "tengine_utils.h"
#include "tengine_serializer.h"
#include "exec_graph.h"

#ifdef ENABLE_ONLINE_REPORT
#include "tenginereportmgr.h"
#endif

char* get_hcl_version(void);
char* gsHclVersion = NULL;

typedef const char* const_char_t;
typedef void* void_ptr_t;

#define STR_VERSION2(a) #a
#define STR_VERSION(a) STR_VERSION2(a)

#ifdef TENGINE_LITE_VERSION
static const char* tengine_lite_version = STR_VERSION(TENGINE_LITE_VERSION);
#else
static const char* tengine_lite_version = "1.2";
#endif

#ifdef TENGINE_VERSION_POSTFIX
static const char* ver_postfix = STR_VERSION(TENGINE_VERSION_POSTFIX);
#else
static const char* ver_postfix = "dev";
#endif

void (*enable_intern_allocator)(void) = NULL;
void (*enable_mem_stat)(void) = NULL;

void (*disable_intern_allocator)(void) = NULL;
void (*disable_mem_stat)(void) = NULL;

int init_tengine(void)
{
    // if (enable_intern_allocator)
    //     enable_intern_allocator();

    set_log_level(LOG_ERR);

    if (enable_mem_stat)
        enable_mem_stat();

    int ret = 0;

    ret = init_op_name_map();
    if (0 != ret)
    {
        TLOG_ERR("init map of operator names failed: %d\n", ret);
        return ret;
    }

    ret = init_op_registry();
    if (0 != ret)
    {
        TLOG_ERR("register operators failed: %d\n", ret);
        return ret;
    }

    ret = init_nn_dev_registry();
    if (0 != ret)
    {
        TLOG_ERR("register device failed: %d\n", ret);
        return ret;
    }

    ret = init_serializer_registry();
    if (0 != ret)
    {
        TLOG_ERR("register serializer failed: %d\n", ret);
        return ret;
    }

#ifdef ENABLE_ONLINE_REPORT
    gsHclVersion = get_hcl_version();
    init_tengine_report_mgr();
    do_tengine_report(ACTION_INIT);
#endif

    ret = exec_module_init(0);
    if (0 != ret)
    {
        TLOG_ERR("init exec module failed: %d\n", ret);
        return ret;
    }

    return 0;
}

void release_tengine(void)
{
    int ret = 0;

    ret = exec_module_exit(0);
    if (0 != ret)
    {
        TLOG_ERR("release exec module failed: %d\n", ret);
    }

    release_serializer_registry();
    release_nn_dev_registry();
    release_op_registry();
    release_op_name_map();

    if (disable_mem_stat)
        disable_mem_stat();

    // if (disable_intern_allocator)
    //     disable_intern_allocator();
#ifdef ENABLE_ONLINE_REPORT
    // do_tengine_report(ACTION_RELEASE);
    release_tengine_report_mgr();
#endif
}

const_char_t get_tengine_version(void)
{
    static char buf[128];

    snprintf(buf, 128, "%s-%s", tengine_lite_version, ver_postfix);

    buf[127] = 0x0; /* save moving */

    return buf;
}

int request_tengine_version(const char* version)
{
    return 1;
}

graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)
{
    int priv_context = 0;

    if (context == NULL)
    {
        context = create_context(NULL, 1);
        priv_context = 1;
    }

    struct ir_graph* ir_graph = create_ir_graph(( struct exec_context* )context);

    if (ir_graph == NULL)
    {
        if (priv_context)
            destroy_context(context);

        return NULL;
    }

    ir_graph->exec_attr->priv_context = priv_context;

    if (model_format != NULL)
    {
        int ret = 0;
        struct serializer* loader = find_serializer(model_format);

        if (loader == NULL)
        {
            TLOG_ERR("no serializer found for %s\n", model_format);
            goto error;
        }

        va_list ap;
        va_start(ap, fname);

        char* p = strchr(model_format, ':');

        /* load from file or memory */
        if (p == NULL)
            ret = loader->load_model(loader, ir_graph, fname, ap);
        else
        {
            if (p[1] != 'm')
            {
                TLOG_ERR("invalid postfix for model format: should 'm' only\n");
                set_tengine_errno(EINVAL);
                goto error;
            }

            if (!loader->load_mem)
            {
                TLOG_ERR("%s serializer does not support load from memory\n", loader->get_name(loader));
                set_tengine_errno(ENOTSUP);
                goto error;
            }

            int size = va_arg(ap, int);

            ret = loader->load_mem(loader, ir_graph, ( void* )fname, size, ap);
        }

        va_end(ap);

        if (ret < 0)
            goto error;

        ir_graph->nn_dev = get_default_nn_device();
    }

    return ir_graph;

error:

    destroy_graph(ir_graph);

    return NULL;
}

int set_graph_layout(graph_t graph, int layout_type)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if ((layout_type != TENGINE_LAYOUT_NCHW) && (layout_type != TENGINE_LAYOUT_NHWC))
    {
        set_tengine_errno(EINVAL);
        return -1;
    }

    ir_graph->graph_layout = layout_type;

    return 0;
}

int set_graph_input_node(graph_t graph, const char* input_nodes[], int input_number)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    int16_t* node_idxs;

    node_idxs = ( int16_t* )sys_malloc(sizeof(int16_t) * input_number);

    if (node_idxs == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    for (int i = 0; i < input_number; i++)
    {
        int node_idx = get_node_idx_from_name(ir_graph, input_nodes[i]);

        if (node_idx < 0)
        {
            set_tengine_errno(ENOENT);
            sys_free(node_idxs);
            return -1;
        }

        node_idxs[i] = node_idx;
    }

    int ret = set_ir_graph_input_node(ir_graph, node_idxs, input_number);

    sys_free(node_idxs);

    return ret;
}

int set_graph_output_node(graph_t graph, const char* output_nodes[], int output_number)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    int16_t* node_idxs;

    node_idxs = ( int16_t* )sys_malloc(sizeof(int16_t) * output_number);

    if (node_idxs == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    for (int i = 0; i < output_number; i++)
    {
        int node_idx = get_node_idx_from_name(ir_graph, output_nodes[i]);

        if (node_idx < 0)
        {
            set_tengine_errno(EINVAL);
            sys_free(node_idxs);
            return -1;
        }

        node_idxs[i] = node_idx;
    }

    int ret = set_ir_graph_output_node(ir_graph, node_idxs, output_number);

    sys_free(node_idxs);

    return ret;
}

int destroy_graph(graph_t graph)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (ir_graph->exec_attr->priv_context)
        destroy_context(ir_graph->exec_attr->exec_context);

    destroy_ir_graph(ir_graph);

    return 0;
}

int get_graph_input_node_number(graph_t graph)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    return ir_graph->input_num;
}

node_t get_graph_input_node(graph_t graph, int idx)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (idx < 0 || idx >= ir_graph->input_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return get_ir_graph_node(ir_graph, ir_graph->input_nodes[idx]);
}

int get_graph_output_node_number(graph_t graph)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    return ir_graph->output_num;
}

node_t get_graph_output_node(graph_t graph, int idx)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (idx < 0 || idx >= ir_graph->output_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return get_ir_graph_node(ir_graph, ir_graph->output_nodes[idx]);
}

tensor_t get_graph_input_tensor(graph_t graph, int input_idx, int tensor_idx)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (input_idx < 0 || input_idx >= ir_graph->input_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    int input_node_idx = ir_graph->input_nodes[input_idx];

    struct ir_node* ir_node = ir_graph->node_list[input_node_idx];

    if (tensor_idx < 0 || tensor_idx >= ir_node->output_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return get_ir_graph_tensor(ir_node->graph, ir_node->output_tensors[tensor_idx]);
}

tensor_t get_graph_output_tensor(graph_t graph, int output_idx, int tensor_idx)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (output_idx < 0 || output_idx >= ir_graph->output_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    int output_node_idx = ir_graph->output_nodes[output_idx];

    struct ir_node* ir_node = ir_graph->node_list[output_node_idx];

    if (tensor_idx < 0 || tensor_idx >= ir_node->output_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return get_ir_graph_tensor(ir_node->graph, ir_node->output_tensors[tensor_idx]);
}

node_t create_graph_node(graph_t graph, const char* node_name, const char* op_name)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    int node_idx = get_node_idx_from_name(ir_graph, node_name);

    if (node_idx >= 0)
    {
        set_tengine_errno(EEXIST);
        return NULL;
    }

    int op_type = get_op_type(op_name);

    if (op_type < 0)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return create_ir_node(ir_graph, node_name, op_type, 1);
}

node_t get_graph_node(graph_t graph, const char* node_name)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    int node_idx = get_node_idx_from_name(ir_graph, node_name);

    if (node_idx < 0)
    {
        set_tengine_errno(ENOENT);
        return NULL;
    }

    return ir_graph->node_list[node_idx];
}

node_t get_graph_node_by_idx(graph_t graph, int idx)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (idx < 0 || idx >= ir_graph->node_num)
        return NULL;

    return ir_graph->node_list[idx];
}

int get_graph_node_num(graph_t graph)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    return ir_graph->node_num;
}

const_char_t get_node_name(node_t node)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    if (ir_node->name)
        return ir_node->name;

    ir_node->name = create_node_name_from_idx(ir_node->idx);

    return ir_node->name;
}

const_char_t get_node_op(node_t node)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    int op_type = ir_node->op.op_type;

    return get_op_name(op_type);
}

void release_graph_node(node_t node)
{
    ( void )node;
    // NOTHING NEEDS TO DO
}

tensor_t get_node_input_tensor(node_t node, int input_idx)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    if (input_idx < 0 || input_idx >= ir_node->input_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return get_ir_graph_tensor(ir_node->graph, ir_node->input_tensors[input_idx]);
}

tensor_t get_node_output_tensor(node_t node, int output_idx)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    if (output_idx < 0 || output_idx >= ir_node->output_num)
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return get_ir_graph_tensor(ir_node->graph, ir_node->output_tensors[output_idx]);
}

int set_custom_kernel(node_t node, const char* dev_name, struct custom_kernel_ops* kernel_ops)
{
    // TODO: set custom kernel
    return -1;
}

int remove_custom_kernel(node_t node, const char* dev_name)
{
    // TODO: remove custom kernel
    return -1;
}

int set_node_input_tensor(node_t node, int input_idx, tensor_t tensor)
{
    struct ir_node* ir_node = ( struct ir_node* )node;
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return set_ir_node_input_tensor(ir_node, input_idx, ir_tensor);
}

int set_node_output_tensor(node_t node, int output_idx, tensor_t tensor, int tensor_type)
{
    struct ir_node* ir_node = ( struct ir_node* )node;
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    ir_tensor->tensor_type = tensor_type;

    return set_ir_node_output_tensor(ir_node, output_idx, ir_tensor);
}

int get_node_output_number(node_t node)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    return ir_node->output_num;
}

int get_node_input_number(node_t node)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    return ir_node->input_num;
}

int add_node_attr(node_t node, const char* attr_name, const char* type_name, int size)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    int attr_num = ir_node->attr_num;
    void* attr_mem = ir_node->attr_mem;

    void* new_attr_mem = add_new_attr(attr_mem, attr_num, attr_name, type_name, size);

    if (new_attr_mem == NULL)
        return -1;

    if (attr_mem)
        sys_free(attr_mem);

    ir_node->attr_num++;
    ir_node->attr_mem = new_attr_mem;

    return 0;
}

int get_node_attr_int(node_t node, const char* attr_name, int* attr_val)
{
    return get_node_attr_generic(node, attr_name, data_type_typeinfo_name(TENGINE_DT_INT32), attr_val, sizeof(int));
}

int get_node_attr_float(node_t node, const char* attr_name, float* attr_val)
{
    return get_node_attr_generic(node, attr_name, data_type_typeinfo_name(TENGINE_DT_FP32), attr_val, sizeof(float));
}

int get_node_attr_pointer(node_t node, const char* attr_name, void* attr_val)
{
    return get_node_attr_generic(node, attr_name, NULL, attr_val, sizeof(void*));
}

int get_node_attr_generic(node_t node, const char* attr_name, const char* type_name, void* buf, int size)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    /* first to check if the attr_name is an op param */
    struct op_method* m = find_op_method(ir_node->op.op_type, ir_node->op.op_version);

    if (m && m->access_param_entry)
    {
        int pe_type = param_entry_type_mapping(type_name);

        if (m->access_param_entry(ir_node->op.param_mem, attr_name, pe_type, buf, size, 0) == 0)
            return 0;
    }

    struct ir_attr* attr_mem = ir_node->attr_mem;
    int attr_num = ir_node->attr_num;

    return get_attr_val(attr_mem, attr_num, attr_name, type_name, buf, size);
}

int set_node_attr_int(node_t node, const char* attr_name, const int* attr_val)
{
    return set_node_attr_generic(node, attr_name, data_type_typeinfo_name(TENGINE_DT_INT32), attr_val, sizeof(int));
}

int set_node_attr_float(node_t node, const char* attr_name, const float* attr_val)
{
    return set_node_attr_generic(node, attr_name, data_type_typeinfo_name(TENGINE_DT_FP32), attr_val, sizeof(float));
}

int set_node_attr_pointer(node_t node, const char* attr_name, const void* attr_val)
{
    return set_node_attr_generic(node, attr_name, NULL, attr_val, sizeof(void*));
}

int set_node_attr_generic(node_t node, const char* attr_name, const char* type_name, const void* buf,
                                    int size)
{
    struct ir_node* ir_node = ( struct ir_node* )node;

    /* first to check if the attr_name is an op param */
    struct op_method* m = find_op_method(ir_node->op.op_type, ir_node->op.op_version);

    if (m && m->access_param_entry)
    {
        int pe_type = param_entry_type_mapping(type_name);

        if (m->access_param_entry(ir_node->op.param_mem, attr_name, pe_type, ( void* )buf, size, 1) == 0)
            return 0;
    }

    struct ir_attr* attr_mem = ir_node->attr_mem;
    int attr_num = ir_node->attr_num;

    return set_attr_val(attr_mem, attr_num, attr_name, type_name, buf, size);
}

tensor_t create_graph_tensor(graph_t graph, const char* tensor_name, int data_type)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    return create_ir_tensor(ir_graph, tensor_name, data_type);
}

tensor_t get_graph_tensor(graph_t graph, const char* tensor_name)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(graph, i);
        if (NULL == ir_node)
        {
            continue;
        }
        else
        {
            for (int j = 0; j < ir_node->input_num; j++)
            {
                struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_node->graph, ir_node->input_tensors[j]);
                if (ir_tensor && ir_tensor->name && !strcmp(ir_tensor->name, tensor_name))
                    return ( tensor_t )ir_tensor;
            }

            for (int j = 0; j < ir_node->output_num; j++)
            {
                struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_node->graph, ir_node->output_tensors[j]);
                if (ir_tensor && ir_tensor->name && !strcmp(ir_tensor->name, tensor_name))
                    return ( tensor_t )ir_tensor;
            }
        }
    }

    return NULL;
}

const_char_t get_tensor_name(tensor_t tensor)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    if (ir_tensor->name == NULL)
        ir_tensor->name = create_tensor_name_from_idx(ir_tensor->idx);

    return ir_tensor->name;
}

void release_graph_tensor(tensor_t tensor)
{
    // NOTHING NEEDS TO DO
}

int set_tensor_shape(tensor_t tensor, const int dims[], int dim_number)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return set_ir_tensor_shape(ir_tensor, dims, dim_number);
}

int get_tensor_shape(tensor_t tensor, int dims[], int dim_number)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    if (dim_number < ir_tensor->dim_num)
    {
        set_tengine_errno(EINVAL);
        return -1;
    }

    for (int i = 0; i < ir_tensor->dim_num; i++)
        dims[i] = ir_tensor->dims[i];

    return ir_tensor->dim_num;
}

int get_tensor_buffer_size(tensor_t tensor)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return ir_tensor->elem_size * ir_tensor->elem_num;
}

void_ptr_t get_tensor_buffer(tensor_t tensor)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    // TODO: take dev mem into consideration

    return ir_tensor->data;
}

int set_tensor_buffer(tensor_t tensor, void* buffer, int buffer_size)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;
    int tensor_size = get_tensor_buffer_size(tensor);

    if (tensor_size != buffer_size)
    {
        fprintf(stderr, "tensor_size != buffer_size, tensor_size: %d, buffer_size: %d\n", tensor_size, buffer_size);
        set_tengine_errno(EINVAL);
        return -1;
    }

    if (ir_tensor->data && ir_tensor->free_host_mem)
        sys_free(ir_tensor->data);

    ir_tensor->free_host_mem = 0;
    ir_tensor->internal_allocated = 0;
    ir_tensor->data = buffer;

    return 0;
}

int get_tensor_data(tensor_t tensor, void* output_data, int data_size)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;
    int tensor_size = get_tensor_buffer_size(tensor);

    if (data_size < tensor_size)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    if (ir_tensor->data)
    {
        memcpy(output_data, ir_tensor->data, tensor_size);
        return 0;
    }

    if (ir_tensor->dev_mem == NULL)
    {
        set_tengine_errno(ENODATA);
        return -1;
    }

    // TODO: handle dev_mem case

    return -1;
}

int set_tensor_data(tensor_t tensor, const void* input_data, int data_size)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;
    int tensor_size = get_tensor_buffer_size(tensor);

    if (data_size > tensor_size)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    if (ir_tensor->data)
    {
        memcpy(ir_tensor->data, input_data, tensor_size);
        return 0;
    }

    // TODO: handle dev_mem case
    return -1;
}

int get_tensor_data_type(tensor_t tensor)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return ir_tensor->data_type;
}

int set_tensor_data_type(tensor_t tensor, int data_type)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    ir_tensor->data_type = data_type;

    return 0;
}

int get_tensor_layout(tensor_t tensor)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return ir_tensor->layout;
}

int set_tensor_layout(tensor_t tensor, int layout)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    ir_tensor->layout = layout;

    return 0;
}

int set_tensor_quant_param(tensor_t tensor, const float* scale, const int* zero_point, int number)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return set_ir_tensor_quant_param(ir_tensor, scale, zero_point, number);
}

int get_tensor_quant_param(tensor_t tensor, float* scale, int* zero_point, int number)
{
    struct ir_tensor* ir_tensor = ( struct ir_tensor* )tensor;

    return get_ir_tensor_quant_param(ir_tensor, scale, zero_point, number);
}

int set_graph_device(graph_t graph, const char* dev_name)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct nn_device* nn_dev = get_nn_device_by_name(dev_name);

    if (nn_dev == NULL)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    ir_graph->nn_dev = nn_dev;
    return 0;
}

int set_graph_attr(graph_t graph, const char* attr_name, const void* buf, int size)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct ir_attr* attr_mem = ir_graph->attr_mem;
    int attr_num = ir_graph->attr_num;

    struct ir_attr* new_attr_mem = add_new_attr(attr_mem, attr_num, attr_name, NULL, size);

    if (new_attr_mem)
    {
        ir_graph->attr_mem = new_attr_mem;
        ir_graph->attr_num++;
        attr_mem = new_attr_mem;
        attr_num++;
    }

    set_attr_val(attr_mem, attr_num, attr_name, NULL, buf, size);

    return 0;
}

int get_graph_attr(graph_t graph, const char* attr_name, void* buf, int size)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct ir_attr* attr_mem = ir_graph->attr_mem;
    int attr_num = ir_graph->attr_num;

    return get_attr_val(attr_mem, attr_num, attr_name, NULL, buf, size);
}

size_t get_cluster_affinity_mask(int cluster)
{
    check_cpu();
    return get_cluster_mask(cluster);
}

int set_graph_thread(graph_t graph, int cluster, int threads)
{
    check_cpu();
    size_t mask = get_cluster_mask(cluster);
    int count = get_mask_count(mask);

    if (count > threads)
        count = threads;

    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct exec_context* context = get_ir_graph_context(ir_graph);
    struct exec_scheduler* scheduler = context->scheduler;

    if (scheduler->prerun(scheduler, ir_graph, count, cluster, TENGINE_MODE_FP32) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "scheduler->prerun failed\n");
        return -1;
    }
    set_cpu_affine(mask);

    return 0;
}

int set_graph_thread_mask(graph_t graph, size_t cpu_mask)
{
    check_cpu();
    size_t all_mask = get_cluster_mask(TENGINE_CLUSTER_ALL);
    size_t mask = all_mask & cpu_mask;

    set_cpu_affine(mask);

    return 0;
}

int prerun_graph(graph_t graph)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;

    if (infer_shape_graph(ir_graph) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "infer_shape_graph failed\n");
        return -1;
    }

    struct exec_context* context = get_ir_graph_context(ir_graph);

    if (0 != split_graph(ir_graph))
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "split graph failed\n");
        return -1;
    }

    if (0 != optimize_graph(ir_graph, TENGINE_MODE_FP32))
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "optimize graph failed\n");
        return -1;
    }

    struct exec_scheduler* scheduler = context->scheduler;

    if (scheduler->prerun(scheduler, ir_graph, 1, TENGINE_CLUSTER_BIG, TENGINE_MODE_FP32) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "scheduler->prerun failed\n");
        return -1;
    }

    ir_graph->status = GRAPH_STAT_READY;

    return 0;
}

int prerun_graph_multithread(graph_t graph, struct options opt)
{
    check_cpu();
    size_t mask = get_cluster_mask(opt.cluster);
    int count = get_mask_count(mask);

    if (count > opt.num_thread)
        count = opt.num_thread;

    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    if (infer_shape_graph(ir_graph) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "infer_shape_graph failed\n");
        return -1;
    }

    struct exec_context* context = get_ir_graph_context(ir_graph);

    if (0 != split_graph(ir_graph))
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "split graph failed\n");
        return -1;
    }

    if (0 != optimize_graph(ir_graph, opt.precision))
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "optimize graph failed\n");
        return -1;
    }

    struct exec_scheduler* scheduler = context->scheduler;
    if (scheduler->prerun(scheduler, ir_graph, count, opt.cluster, opt.precision) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        fprintf(stderr, "scheduler->prerun failed\n");
        return -1;
    }

    ir_graph->status = GRAPH_STAT_READY;

    if (0 != opt.affinity && 0 != (opt.affinity & mask))
    {
        set_cpu_affine(opt.affinity);
    }
    else
    {
        set_cpu_affine(mask);
    }

    return 0;
}

int run_graph(graph_t graph, int block)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct exec_context* context = get_ir_graph_context(ir_graph);
    struct exec_scheduler* scheduler = context->scheduler;

    ir_graph->status = GRAPH_STAT_RUNNING;

    if (scheduler->run(scheduler, ir_graph, block) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        return -1;
    }
    else
    {
        if (block)
            ir_graph->status = GRAPH_STAT_READY;
    }

    return 0;
}

int wait_graph(graph_t graph, int try_wait)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct exec_context* context = get_ir_graph_context(ir_graph);
    struct exec_scheduler* scheduler = context->scheduler;

    if (ir_graph->status != GRAPH_STAT_RUNNING || ir_graph->status != GRAPH_STAT_READY)
    {
        set_tengine_errno(EINVAL);
        return -1;
    }

    if (ir_graph->status == GRAPH_STAT_READY)
        return 0;

    if (!try_wait)
    {
        set_tengine_errno(EAGAIN);
        return -1;
    }

    return scheduler->wait(scheduler, ir_graph);
}

int postrun_graph(graph_t graph)
{
    struct ir_graph* ir_graph = ( struct ir_graph* )graph;
    struct exec_context* context = get_ir_graph_context(ir_graph);
    struct exec_scheduler* scheduler = context->scheduler;

    if (scheduler->postrun(scheduler, ir_graph) < 0)
    {
        ir_graph->status = GRAPH_STAT_ERROR;
        return -1;
    }

    ir_graph->status = GRAPH_STAT_DONE;

    return 0;
}

void dump_graph(graph_t graph)
{
    dump_ir_graph(graph);
}

const_char_t get_node_device(node_t node)
{
    struct ir_node* ir_node = ( struct ir_node* )node;
    struct ir_graph* graph = ir_node->graph;

    int subgraph_count = get_vector_num(graph->subgraph_list);
    if (subgraph_count > 0)
    {
        if (0 > ir_node->subgraph_idx)
            return NULL;
        else
        {
            struct subgraph* subgraph = get_ir_graph_subgraph(graph, ir_node->subgraph_idx);
            if (subgraph->nn_dev)
                return subgraph->nn_dev->name;
        }
    }
    else
    {
        return graph->nn_dev->name;
    }
}

context_t create_context(const char* context_name, int empty_context)
{
    struct exec_context* context = ( struct exec_context* )sys_malloc(sizeof(struct exec_context));

    if (context == NULL)
        return NULL;

    if (context_name)
        context->name = strdup(context_name);
    else
        context->name = NULL;

    context->scheduler = get_default_scheduler();
    context->dev_allocator = get_default_dev_allocator();
    context->def_dev = get_default_nn_device();
    context->dev_list = create_vector(sizeof(struct nn_device*), NULL);

    if (!empty_context)
    {
        int dev_num = get_nn_device_number();

        for (int i = 0; i < dev_num; i++)
        {
            struct nn_device* dev = get_nn_device(i);
            push_vector_data(context->dev_list, &dev);
        }
    }

    return context;
}

void destroy_context(context_t context)
{
    struct exec_context* exec_context = ( struct exec_context* )context;

    release_vector(exec_context->dev_list);

    if (exec_context->name)
        sys_free(exec_context->name);

    sys_free(exec_context);
}

int get_context_device_number(context_t context)
{
    struct exec_context* exec_context = ( struct exec_context* )context;

    return get_vector_num(exec_context->dev_list);
}

const_char_t get_context_device(context_t context, int idx)
{
    struct exec_context* exec_context = ( struct exec_context* )context;

    if (idx >= get_vector_num(exec_context->dev_list))
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    struct nn_device* dev = ( struct nn_device* )get_vector_data(exec_context->dev_list, idx);

    return dev->name;
}

int add_context_device(context_t context, const char* dev_name)
{
    struct nn_device* dev = get_nn_device_by_name(dev_name);

    if (dev == NULL)
    {
        TLOG_ERR("not found device");
        set_tengine_errno(ENOENT);
        return -1;
    }

    struct exec_context* exec_context = ( struct exec_context* )context;

    int dev_num = get_vector_num(exec_context->dev_list);
    int dev_idx = -1;

    for (int i = 0; i < dev_num; i++)
    {
        struct nn_device* dev0 = ( struct nn_device* )get_vector_data(exec_context->dev_list, i);

        if (!strcmp(dev0->name, dev_name))
        {
            dev_idx = i;
            break;
        }
    }

    if (dev_idx >= 0)
    {
        set_tengine_errno(EEXIST);
        return -1;
    }

    push_vector_data(exec_context->dev_list, &dev);

    struct dev_allocator* dev_allocator = get_dev_allocator(dev_name);

    if (NULL != dev_allocator)
    {
        TLOG_INFO("add dev allocator\n");
        exec_context->dev_allocator = dev_allocator;
    }
    else
    {
        TLOG_WARNING("dev allocator not found\n");
    }

    return 0;
}

int remove_context_device(context_t context, const char* dev_name)
{
    struct exec_context* exec_context = ( struct exec_context* )context;

    int dev_num = get_vector_num(exec_context->dev_list);
    int dev_idx = -1;

    for (int i = 0; i < dev_num; i++)
    {
        struct nn_device* dev = ( struct nn_device* )get_vector_data(exec_context->dev_list, i);

        if (!strcmp(dev->name, dev_name))
        {
            dev_idx = i;
            break;
        }
    }

    if (dev_idx < 0)
        return -1;

    remove_vector_by_idx(exec_context->dev_list, dev_idx);

    return 0;
}

int set_context_attr(context_t context, const char* attr_name, const void* val, int val_size)
{
    set_tengine_errno(ENOTSUP);
    return -1;
}

int get_context_attr(context_t context, const char* attr_name, void* val, int val_size)
{
    set_tengine_errno(ENOTSUP);
    return -1;
}

int clr_tengine_errno()
{
    int ret = get_tengine_errno();
    set_tengine_errno(0);
    return ret;
}

void set_log_level(enum log_level level)
{
    SET_LOG_LEVEL(level);
}

void set_log_output(log_print_t func)
{
    SET_LOG_OUTPUT(func);
}

const char* get_tengine_hcl_version()
{
    return gsHclVersion;
}