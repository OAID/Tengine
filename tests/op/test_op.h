#ifndef __TEST_COMMON_H__
#define __TEST_COMMON_H__

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#include "float.h"
#include "compiler_fp16.h"
#include "tengine/c_api.h"

#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"

#define TENSOR_SHOW_LEADING_BLANK "    "
#define TENSOR_FLOAT_EPSILON 0.0001f

typedef int (*common_test)(graph_t, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w);


void dump_tensor_line(void* data_ptr, int offset, int data_type, int w)
{
    if (0 >= w)
    {
        fprintf(stderr, "Tensor.width = %d, not match width > 0.\n", w);
        return;
    }

    printf("[ ");

    switch(data_type)
    {
        case TENGINE_DT_FP32:
        {
            float* p = ( float* )data_ptr;

            for(int i = 0; i < w - 1; i++)
            {
                printf("%0.2f, ", p[offset + i]);
            }
            printf("%0.2f ", p[offset + w - 1]);

            break;
        }
        case TENGINE_DT_FP16:
        {
            __fp16* p = ( __fp16* )data_ptr;

#ifdef __ARM_ARCH
            for(int i = 0; i < w - 1; i++)
            {
                printf("%f, ", (float)p[offset + i]);
            }
            printf("%f ", (float)p[offset + w - 1]);
#else
            for(int i = 0; i < w - 1; i++)
            {
                printf("%f, ", fp16_to_fp32(p[offset + i]));
            }
            printf("%f ", fp16_to_fp32(p[offset + w - 1]));
#endif
            break;
        }
        case TENGINE_DT_INT8:
        case TENGINE_DT_UINT8:
        {
            if(data_type == TENGINE_DT_INT8)
            {
                int8_t* p = ( int8_t* )data_ptr;

                for(int i = 0; i < w - 1; i++)
                {
                    printf("%d, ", (int)p[offset + i]);
                }
                printf("%d ", (int)p[offset + w - 1]);
            }
            else
            {
                uint8_t* p = ( uint8_t* )data_ptr;

                for(int i = 0; i < w - 1; i++)
                {
                    printf("%d, ", (int)p[offset + i]);
                }
                printf("%d ", (int)p[offset + w - 1]);
            }

            break;
        }
        default:
            // not deal with TENGINE_DT_INT16 and TENGINE_DT_INT32
            fprintf(stderr, "Unsupported data type for now. ");
    }

    printf("]");
}


void dump_tensor(tensor_t tensor, const char* message)
{
    int data_type = get_tensor_data_type(tensor);
    void* data_ptr = get_tensor_buffer(tensor);

    int dim_array[MAX_SHAPE_DIM_NUM] = { 0 };
    int dim_count = get_tensor_shape(tensor, dim_array, MAX_SHAPE_DIM_NUM);
    if (0 >= dim_count)
        fprintf(stderr, "Cannot get tensor shape.");

    int line_count = 1;
    for (int i = 0; i < dim_count - 1; i++)
        line_count *= dim_array[i];

    int n = 0, c = 0, h = 0, w = 0;

    switch (dim_count)
    {
        case 4:
        {
            n = dim_array[0];
            c = dim_array[1];
            h = dim_array[2];
            w = dim_array[3];
            break;
        }
        case 3:
        {
            c = dim_array[0];
            h = dim_array[1];
            w = dim_array[2];
            break;
        }
        case 2:
        {
            h = dim_array[0];
            w = dim_array[1];
            break;
        }
        case 1:
        {
            w = dim_array[0];
            break;
        }
        default:
            fprintf(stderr, "Cannot found the type of tensor.\n");
    }

    // print leader
    printf("%s is { n, c, h, w } = { %d, %d, %d, %d }:\n", message, n, c, h, w);
    printf("[\n");

    for (int line = 0; line < line_count; line++)
    {
        if (2 <= dim_count && 0 == line % h)
            printf(TENSOR_SHOW_LEADING_BLANK "[\n");

        // print each line
        {
            for (int i = 0; i < dim_count - 2; i++)
                printf(TENSOR_SHOW_LEADING_BLANK);

            dump_tensor_line(data_ptr, line * w, data_type, w);

            if (0 != (line + 1) % h)
                printf(";\n");
            else
                printf("\n");
        }

        if (2 <= dim_count && 0 == (line + 1) % h)
        {
            if (line_count != line + 1)
                printf(TENSOR_SHOW_LEADING_BLANK "];\n");
            else
                printf(TENSOR_SHOW_LEADING_BLANK "]\n");
        }
    }
    printf("].\n");
}


void dump_node_input(node_t test_node, int index)
{
    tensor_t tensor = get_node_input_tensor(test_node, index);
    if(NULL == tensor)
    {
        fprintf(stderr, "Get input tensor(%d) from the node failed.\n", index);
        return;
    }

    char name[16] = {0};
    sprintf(name, "In%d", index);

    dump_tensor(tensor, name);

    release_graph_tensor(tensor);
}


void dump_node_output(node_t test_node, int index)
{
    tensor_t tensor = get_node_output_tensor(test_node, index);
    if(NULL == tensor)
    {
        fprintf(stderr, "Get output tensor from the node failed.\n");
        return;
    }

    char name[16] = {0};
    sprintf(name, "Out%d", index);

    dump_tensor(tensor, name);

    release_graph_tensor(tensor);
}


int create_node(graph_t graph, const char* node_name, int n, int c, int h, int w, int data_type, int layout)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    if (NULL == node)
    {
        fprintf(stderr, "Create node(%s) with shape [n c h w] = [%d %d %d %d] failed.\n", node_name, n, c, h, w);
        return -1;
    }

    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);
    if(NULL == tensor)
    {
        release_graph_node(node);

        fprintf(stderr, "Create tensor from node(%s) with shape [n c h w] = [%d %d %d %d] failed.\n", node_name, n, c, h, w);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    if(TENGINE_LAYOUT_NCHW == layout)
    {
        int dims[4] = {n, c, h, w};
        set_tensor_shape(tensor, dims, 4);
    }

    if(TENGINE_LAYOUT_NHWC == layout)
    {
        int dims[4] = {n, h, w, c};
        set_tensor_shape(tensor, dims, 4);
    }

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}


int create_input_node(graph_t graph, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    int dims_count = 4;
    if (0 == n) dims_count = 3;
    if (0 == c) dims_count = 2;
    if (0 == h) dims_count = 1;
    if (0 == w)
    {
        fprintf(stderr, "Dim of input node is not allowed. { n, c, h, w } = {%d, %d, %d, %d}.\n", n, c, h, w);
        return -1;
    }

    node_t node = create_graph_node(graph, node_name, "InputOp");
    if (NULL == node)
    {
        fprintf(stderr, "Create %d dims node(%s) failed. ", dims_count, node_name);
        return -1;
    }

    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);
    if(NULL == tensor)
    {
        release_graph_node(node);

        fprintf(stderr, "Create %d dims tensor for node(%s) failed. ", dims_count, node_name);

        return -1;
    }

    int ret = set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);
    if(0 != ret)
    {
        release_graph_tensor(tensor);
        release_graph_node(node);

        fprintf(stderr, "Set %d dims output tensor for node(%s) failed. ", dims_count, node_name);

        return -1;
    }

    switch(dims_count)
    {
        case 1:
        {
            int dims_array[1] = { w };
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
        case 2:
        {
            int dims_array[2] = { h, w };
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
        case 3:
        {
            if (TENGINE_LAYOUT_NCHW == layout)
            {
                int dims_array[3] = { c, h, w };
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }

            if (TENGINE_LAYOUT_NHWC == layout)
            {
                int dims_array[3] = { h, w, c };
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }
        }
        case 4:
        {
            if (TENGINE_LAYOUT_NCHW == layout)
            {
                int dims_array[4] = { n, c, h, w };
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }

            if (TENGINE_LAYOUT_NHWC == layout)
            {
                int dims_array[4] = { n, h, w, c };
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }
        }
        default:
            fprintf(stderr, "Cannot support %d dims tensor.\n", dims_count);
    }

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}


int fill_fp32_tensor(tensor_t tensor, float value)
{
    int dims[MAX_SHAPE_DIM_NUM];
    int dims_count = get_tensor_shape(tensor, dims, MAX_SHAPE_DIM_NUM);

    int type = get_tensor_data_type(tensor);

    if (TENGINE_DT_FP32 != type)
        return -1;

    int element_count = 1;
    for (int i = 0; i < dims_count; i++)
        element_count *= dims[i];

    if (0 == element_count)
        return -1;

    float* data_ptr = (float*)get_tensor_buffer(tensor);
    for (int i = 0; i < element_count; i++)
        data_ptr[i] = value;

    return 0;
}


int fill_uint8_tensor(tensor_t tensor, float value)
{
    int dims[MAX_SHAPE_DIM_NUM];
    int dims_count = get_tensor_shape(tensor, dims, MAX_SHAPE_DIM_NUM);

    int type = get_tensor_data_type(tensor);

    if (TENGINE_DT_UINT8 != type)
        return -1;

    int element_count = 1;
    for (int i = 0; i < dims_count; i++)
        element_count *= dims[i];

    if (0 == element_count)
        return -1;

    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(tensor, &input_scale, &input_zero_point, 1);

    uint8_t * data_ptr = (uint8_t *)get_tensor_buffer(tensor);
    for (int i = 0; i < element_count; i++)
    {
        int udata = (round)(value / input_scale + (float)input_zero_point);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        data_ptr[i] = udata;
    }

    return 0;
}


void fill_input_float_tensor_by_index(graph_t graph, int input_node_index, int tensor_index, float value)
{
    tensor_t tensor = get_graph_input_tensor(graph, input_node_index, tensor_index);
    if(NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node index(%d).\n", tensor_index, input_node_index);

    int buf_size = get_tensor_buffer_size(tensor);
    float* data = (float* )malloc(buf_size);

//    for(int i = 0; i < buf_size/sizeof(float); i++)
//        data[i] = value;

    int ret = set_tensor_buffer(tensor, (void* )data, buf_size);
    if(0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_fp32_tensor(tensor, value);
    if(0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}


void fill_input_uint8_tensor_by_index(graph_t graph, int input_node_index, int tensor_index, float value)
{
    tensor_t tensor = get_graph_input_tensor(graph, input_node_index, tensor_index);
    if(NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node index(%d).\n", tensor_index, input_node_index);

    int buf_size = get_tensor_buffer_size(tensor);
    uint8_t* data = (uint8_t* )malloc(buf_size);

    int ret = set_tensor_buffer(tensor, (void* )data, buf_size);
    if(0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_uint8_tensor(tensor, value);
    if(0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}


void fill_input_float_tensor_by_name(graph_t graph, const char* node_name, int tensor_index, float value)
{
    node_t node = get_graph_node(graph, node_name);
    if(NULL == node)
        fprintf(stderr, "Cannot get node via node name(%s).\n", node_name);

    tensor_t tensor = get_node_input_tensor(node, tensor_index);
    if(NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node name(%s)\n", tensor_index, node_name);

    int buf_size = get_tensor_buffer_size(tensor);
    float* data = (float* )malloc(buf_size);

//    for(unsigned int i = 0; i < buf_size/sizeof(float) ; i++)
//        data[i] = value;

    int ret = set_tensor_buffer(tensor, (void* )data, buf_size);
    if(0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_fp32_tensor(tensor, value);
    if(0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}


void fill_input_float_buffer_tensor_by_name(graph_t graph, const char* node_name, int tensor_index, void* value, int buf_size)
{
    node_t node = get_graph_node(graph, node_name);
    if(NULL == node)
        fprintf(stderr, "Cannot get node via node name(%s).\n", node_name);

    tensor_t tensor = get_node_input_tensor(node, tensor_index);
    if(NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node name(%s).\n", tensor_index, node_name);

    int ret = set_tensor_buffer(tensor, value, buf_size);
    if(0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");
}


void fill_input_integer_tensor_by_name(graph_t graph, const char* node_name, int tensor_index, int value)
{
    node_t node = get_graph_node(graph, node_name);
    if(NULL == node)
    {
        fprintf(stderr, "Cannot get node via node name(%s).\n", node_name);
        return;
    }

    tensor_t tensor = get_node_input_tensor(node, tensor_index);
    if(NULL == tensor)
    {
        fprintf(stderr, "Cannot find the %dth tensor via node name(%s).\n", tensor_index, node_name);
        return;
    }

    int buf_size = get_tensor_buffer_size(tensor);
    int* data = (int* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size/sizeof(int) ; i++)
        data[i] = value;

    int ret = set_tensor_buffer(tensor, (void* )data, buf_size);
    if(0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");
}


int test_graph_init()
{
    // now init tengine will mask critical filed and return an error
    // TODO: fix this fatal issue
    init_tengine();

    return 0;
}


int test_graph_run(graph_t graph)
{
    if(prerun_graph(graph) < 0)
    {
        fprintf(stderr, "Pre-run graph failed.\n");
        return -1;
    }

    dump_graph(graph);

    if (0 != run_graph(graph, 1))
    {
        fprintf(stderr, "Run graph error.\n");
        return -1;
    }

    return 0;
}


void test_graph_release(graph_t graph)
{
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}


graph_t create_common_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func)
{
    graph_t graph = create_graph(NULL, NULL, NULL);
    if(NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if(set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if(create_input_node(graph, input_name, data_type, layout, n, c, h, w) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if(test_func(graph, input_name, test_node_name, data_type, layout, n, c, h ,w) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}


graph_t create_timvx_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func)
{
    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = add_context_device(timvx_context, "TIMVX");
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return NULL;
    }

    graph_t graph = create_graph(timvx_context, NULL, NULL);
    if(NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if(set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if(create_input_node(graph, input_name, data_type, layout, n, c, h, w) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if(test_func(graph, input_name, test_node_name, data_type, layout, n, c, h ,w) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}


int compare_tensor(tensor_t a, tensor_t b)
{
    int a_dim[MAX_SHAPE_DIM_NUM], b_dim[MAX_SHAPE_DIM_NUM];
    int a_dim_count = get_tensor_shape(a, a_dim, MAX_SHAPE_DIM_NUM);
    int b_dim_count = get_tensor_shape(b, b_dim, MAX_SHAPE_DIM_NUM);

    if (a_dim_count <= 0 || a_dim_count != b_dim_count)
        return -1;

    for (int i = 0; i < a_dim_count; i++)
        if (a_dim[i] != b_dim[i])
            return -1;

    int a_type = get_tensor_data_type(a);
    int b_type = get_tensor_data_type(b);

    if (a_type != b_type)
        return -1;

    int element_size = 1;
    for (int i = 0; i < a_dim_count; i++)
        element_size *= a_dim[i];

    if (element_size <= 0)
    {
        fprintf(stderr, "One of dims is 0. Zero is not allowed.\n");
        return -1;
    }

    switch (a_type)
    {
        case TENGINE_DT_FP32:
        {
            float* a_data_ptr = (float*)get_tensor_buffer(a);
            float* b_data_ptr = (float*)get_tensor_buffer(b);

            for (int i = 0; i < element_size; i++)
                if (fabsf(a_data_ptr[i] - b_data_ptr[i]) < TENSOR_FLOAT_EPSILON)
                    return -1;

            break;
        }
        case TENGINE_DT_FP16:
        {
            __fp16* a_data_ptr = (__fp16*)get_tensor_buffer(a);
            __fp16* b_data_ptr = (__fp16*)get_tensor_buffer(b);

            for (int i = 0; i < element_size; i++)
            {
                if (fabsf((float)fp16_to_fp32(a_data_ptr[i]) - (float)fp16_to_fp32(b_data_ptr[i])) < TENSOR_FLOAT_EPSILON)
                    return -1;
            }

            break;
        }
        case TENGINE_DT_INT32:
        {
            int32_t* a_data_ptr = (int32_t*)get_tensor_buffer(a);
            int32_t* b_data_ptr = (int32_t*)get_tensor_buffer(b);

            for (int i = 0; i < element_size; i++)
                if (a_data_ptr[i] != b_data_ptr[i])
                    return -1;

            break;
        }
        case TENGINE_DT_INT16:
        {
            int16_t* a_data_ptr = (int16_t*)get_tensor_buffer(a);
            int16_t* b_data_ptr = (int16_t*)get_tensor_buffer(b);

            for (int i = 0; i < element_size; i++)
                if (a_data_ptr[i] != b_data_ptr[i])
                    return -1;

            break;
        }
        case TENGINE_DT_UINT8:
        case TENGINE_DT_INT8:
        {
            int8_t* a_data_ptr = (int8_t*)get_tensor_buffer(a);
            int8_t* b_data_ptr = (int8_t*)get_tensor_buffer(b);

            for (int i = 0; i < element_size; i++)
                if (a_data_ptr[i] != b_data_ptr[i])
                    return -1;

            break;
        }
        default:
        {
            fprintf(stderr, "The type of tensor was not supported.\n");
            return -1;
        }
    }

    return 0;
}


static inline unsigned long get_current_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

#endif
