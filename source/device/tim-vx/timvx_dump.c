
#include "timvx_dump.h"

#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "utility/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif

int print_tensor_data_value_timvx(FILE* file, const struct tensor* tensor, int offset)
{
    switch (tensor->data_type)
    {
    case TENGINE_DT_FP32:
    {
        float* base_ptr = (float*)tensor->data;
        float val = base_ptr[offset];
        if (val < 0)
            fprintf(file, "%.4f ", val);
        else
            fprintf(file, " %.4f ", val);
        break;
    }
        //        case TENGINE_DT_FP16:
        //        {
        //            fp16_t* base_ptr = (fp16_t*)tensor->data;
        //            fp16_t val = base_ptr[offset];
        //
        //            float val_fp32 = fp16_to_fp32(val);
        //
        //            if (val_fp32 < 0)
        //                fprintf(file, "%.4f ", val_fp32);
        //            else
        //                fprintf(file, " %.4f ", val_fp32);
        //            break;
        //        }
    case TENGINE_DT_UINT8:
    {
        uint8_t* base_ptr = (uint8_t*)tensor->data;
        uint8_t val = base_ptr[offset];

        float scale = tensor->scale;
        int32_t zero_point = tensor->zero_point;

        float val_fp32 = (float)((int)val - (int)zero_point) * scale;
        if (val_fp32 < 0)
            fprintf(file, "%.4f ", val_fp32);
        else
            fprintf(file, " %.4f ", val_fp32);
        break;
    }
    case TENGINE_DT_INT8:
    {
        int8_t* base_ptr = (int8_t*)tensor->data;
        int8_t val = base_ptr[offset];

        float scale = tensor->scale;

        float val_fp32 = (float)val * scale;
        if (val_fp32 < 0)
            fprintf(file, "%.4f ", val_fp32);
        else
            fprintf(file, " %.4f ", val_fp32);
    }
    case TENGINE_DT_INT32:
    {
        int32_t* base_ptr = (int32_t*)tensor->data;
        int8_t val = base_ptr[offset];

        float scale = tensor->scale;
        float val_fp32 = (float)val * scale;

        if (val_fp32 < 0)
            fprintf(file, "%.6f ", val_fp32);
        else
            fprintf(file, " %.6f ", val_fp32);
    }
    }

    return 0;
}

const char* get_tensor_data_type_string_timvx(int data_type)
{
    switch (data_type)
    {
    case TENGINE_DT_FP32:
        return "fp32";
    case TENGINE_DT_FP16:
        return "fp16";
    case TENGINE_DT_INT8:
        return "int8";
    case TENGINE_DT_UINT8:
        return "uint8";
    case TENGINE_DT_INT32:
        return "int32";
    case TENGINE_DT_INT16:
        return "int16";
    default:
        return "unknown";
    }
}

void print_tensor_data_to_file_timvx(FILE* file, const struct tensor* tensor)
{
    switch (tensor->dim_num)
    {
    case 5:
    {
        int dim5 = tensor->dims[0], batch = tensor->dims[1], channel = 0, height = 0, width = 0;

        if (TENGINE_LAYOUT_NCHW == tensor->layout)
        {
            channel = tensor->dims[2];
            height = tensor->dims[3];
            width = tensor->dims[4];
        }
        if (TENGINE_LAYOUT_NHWC == tensor->layout)
        {
            height = tensor->dims[2];
            width = tensor->dims[3];
            channel = tensor->dims[4];
        }

        if (TENGINE_DT_FP32 == tensor->data_type)
        {
            fprintf(file, "Shape is {%d %d %d %d %d}, data type is fp32\n", dim5, batch, channel, height, width);
        }
        else
        {
            if (TENGINE_DT_FP16 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d %d %d %d}, data type is fp16, cast to fp32\n", dim5, batch, channel, height, width);
            }
            else
            {
                const char* type_name = get_tensor_data_type_string_timvx(tensor->data_type);
                fprintf(file, "Shape is {%d %d %d %d %d}, data type is %s, inverse quantization to fp32\n", dim5, batch, channel, height, width, type_name);
            }
        }

        for (int d5 = 0; d5 < dim5; d5++)
        {
            fprintf(file, "Dim5 %d:\n", d5);

            for (int n = 0; n < batch; n++)
            {
                fprintf(file, "\tBatch %d:\n", n);

                for (int ch = 0; ch < channel; ch++)
                {
                    fprintf(file, "\t\tChannel %d:\n", ch);

                    for (int h = 0; h < height; h++)
                    {
                        fprintf(file, "\t\t\t");

                        for (int w = 0; w < width; w++)
                        {
                            int offset = 0;

                            if (TENGINE_LAYOUT_NCHW == tensor->layout)
                            {
                                offset += d5 * batch * channel * height * width;
                                offset += n * channel * height * width;
                                offset += ch * height * width;
                                offset += h * width;
                                offset += w;
                            }
                            if (TENGINE_LAYOUT_NHWC == tensor->layout)
                            {
                                offset += d5 * batch * channel * height * width;
                                offset += n * channel * height * width;
                                offset += ch;
                                offset += h * width * channel;
                                offset += w * channel;
                            }

                            print_tensor_data_value_timvx(file, tensor, offset);
                        }
                        fprintf(file, "\n");
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }

        break;
    }
    case 4:
    {
        int batch = tensor->dims[0], channel = 0, height = 0, width = 0;

        if (TENGINE_LAYOUT_NCHW == tensor->layout)
        {
            channel = tensor->dims[1];
            height = tensor->dims[2];
            width = tensor->dims[3];
        }
        if (TENGINE_LAYOUT_NHWC == tensor->layout)
        {
            height = tensor->dims[1];
            width = tensor->dims[2];
            channel = tensor->dims[3];
        }

        if (TENGINE_DT_FP32 == tensor->data_type)
        {
            fprintf(file, "Shape is {%d %d %d %d}, data type is fp32\n", batch, channel, height, width);
        }
        else
        {
            if (TENGINE_DT_FP16 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d %d %d}, data type is fp16, cast to fp32\n", batch, channel, height, width);
            }
            else
            {
                const char* type_name = get_tensor_data_type_string_timvx(tensor->data_type);
                fprintf(file, "Shape is {%d %d %d %d}, data type is %s, inverse quantization to fp32\n", batch, channel, height, width, type_name);
            }
        }

        for (int n = 0; n < batch; n++)
        {
            fprintf(file, "Batch %d:\n", n);

            for (int ch = 0; ch < channel; ch++)
            {
                fprintf(file, "\tChannel %d:\n", ch);

                for (int h = 0; h < height; h++)
                {
                    fprintf(file, "\t\t");

                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        if (TENGINE_LAYOUT_NCHW == tensor->layout)
                        {
                            offset += n * channel * height * width;
                            offset += ch * height * width;
                            offset += h * width;
                            offset += w;
                        }
                        if (TENGINE_LAYOUT_NHWC == tensor->layout)
                        {
                            offset += n * channel * height * width;
                            offset += ch;
                            offset += h * width * channel;
                            offset += w * channel;
                        }

                        print_tensor_data_value_timvx(file, tensor, offset);
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }

        break;
    }
    case 3:
    {
        int batch = 0, height = 0, width = 0;

        if (TENGINE_LAYOUT_NCHW == tensor->layout)
        {
            batch = tensor->dims[0];
            height = tensor->dims[1];
            width = tensor->dims[2];
        }
        if (TENGINE_LAYOUT_NHWC == tensor->layout)
        {
            height = tensor->dims[0];
            width = tensor->dims[1];
            batch = tensor->dims[2];
        }

        if (TENGINE_DT_FP32 == tensor->data_type)
        {
            fprintf(file, "Shape is {%d %d %d}, data type is fp32\n", batch, height, width);
        }
        else
        {
            if (TENGINE_DT_FP16 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d %d}, data type is fp16, cast to fp32\n", batch, height, width);
            }
            else
            {
                const char* type_name = get_tensor_data_type_string_timvx(tensor->data_type);
                fprintf(file, "Shape is {%d %d %d}, data type is %s, inverse quantization to fp32\n", batch, height, width, type_name);
            }
        }

        for (int n = 0; n < batch; n++)
        {
            for (int h = 0; h < height; h++)
            {
                fprintf(file, "Channel %d:\n", h);
                fprintf(file, "\t");

                for (int w = 0; w < width; w++)
                {
                    int offset = 0;

                    if (TENGINE_LAYOUT_NCHW == tensor->layout)
                    {
                        offset += n * height * width;
                        offset += h * width;
                        offset += w;
                    }
                    if (TENGINE_LAYOUT_NHWC == tensor->layout)
                    {
                        offset += h;
                        offset += n * width * height;
                        offset += w * height;
                    }

                    print_tensor_data_value_timvx(file, tensor, offset);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }

        break;
    }
    case 2:
    {
        int batch = 0, width = 0;

        if (TENGINE_LAYOUT_NCHW == tensor->layout)
        {
            batch = tensor->dims[0];
            width = tensor->dims[1];
        }
        if (TENGINE_LAYOUT_NHWC == tensor->layout)
        {
            batch = tensor->dims[0];
            width = tensor->dims[1];
        }

        if (TENGINE_DT_FP32 == tensor->data_type)
        {
            fprintf(file, "Shape is {%d %d}, data type is fp32\n", batch, width);
        }
        else
        {
            if (TENGINE_DT_FP16 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d}, data type is fp16, cast to fp32\n", batch, width);
            }
            else
            {
                const char* type_name = get_tensor_data_type_string_timvx(tensor->data_type);
                fprintf(file, "Shape is {%d %d}, data type is %s, inverse quantization to fp32\n", batch, width, type_name);
            }
        }

        for (int n = 0; n < batch; n++)
        {
            for (int w = 0; w < width; w++)
            {
                int offset = 0;

                offset += n * width;
                offset += w;

                print_tensor_data_value_timvx(file, tensor, offset);
            }
            fprintf(file, "\n");
        }

        break;
    }
    case 1:
    {
        int width = tensor->dims[0];

        fprintf(file, "Shape is {%d}, data type is fp32\n", width);

        for (int w = 0; w < width; w++)
        {
            print_tensor_data_value_timvx(file, tensor, w);
        }

        break;
    }
    default:
        printf("Input dimension %d not to be supported.\n", tensor->dim_num);
    }
}

char* replace_string_character_timvx(char* src_str, char* dst_str, char* target_char, char* replaced_char)
{
    char* p;
    char* _out = dst_str;
    char* _str = src_str;
    char* _src = target_char;
    char* _dst = replaced_char;
    size_t src_size = strlen(_src);
    size_t dst_size = strlen(_dst);
    size_t len = 0;

    do
    {
        p = strstr(_str, _src);
        if (p == 0)
        {
            strcpy(_out, _str);
            return dst_str;
        }
        len = p - _str;
        memcpy(_out, _str, len);
        memcpy(_out + len, _dst, dst_size);
        _str = p + src_size;
        _out = _out + len + dst_size;
    } while (p);

    return dst_str;
}

void extract_feature_from_tensor_timvx(const char* comment, const char* layer_name, const struct tensor* tensor)
{
    // 1. deal with saving path
    char save_dir[256] = {'0'};

    const char* env_path = getenv(TENGINE_DUMP_DIR);

    if (NULL != env_path && '\0' != env_path[0] && (256 - 2) > strlen(env_path))
    {
        strcpy(save_dir, env_path);

        if ('/' != save_dir[strlen(env_path) - 1] && '\\' != save_dir[strlen(env_path) - 1])
        {
#ifdef _MSC_VER
            save_dir[strlen(env_path)] = '\\';
            save_dir[strlen(env_path) + 1] = 0;
#else
            save_dir[strlen(env_path)] = '/';
            save_dir[strlen(env_path) + 1] = 0;
#endif
        }
    }
    else
    {
        //        TLOG_WARNING("Tengine: Env var \"TENGINE_DUMP_DIR\" is too long(%d vs. 254). Using default path.\n", strlen(env_path));
        sprintf(save_dir, "./output/");
#ifdef _MSC_VER
        CreateDirectoryA(save_dir, NULL);
#else
        int ret = mkdir(save_dir, S_IRWXU | S_IRGRP | S_IWGRP | S_IROTH);
//        if (0 != ret)
//        {
//            TLOG_WARNING("Tengine: Create saving folder failed(%d), skip dump.\n", ret);
//            return;
//        }
#endif
    }

    // 2. deal with layer name
    char layer_short_name[64], layer_legal_name[64];

    if (64 < strlen(layer_name))
    {
        memcpy(layer_short_name, layer_name, 64 - 1);
        layer_short_name[64 - 1] = 0;
    }
    else
    {
        strcpy(layer_short_name, layer_name);
    }

    replace_string_character_timvx(layer_short_name, layer_legal_name, "/", "-");

    // 3. join path
    char output_file_path[512] = {'0'};

    if (strlen(layer_legal_name) + strlen(save_dir) + strlen(comment) > 256 - 16)
    {
        TLOG_WARNING("Tengine: Name of saving file is too long(%d vs. %d), skip dump.\n", strlen(layer_legal_name) + strlen(save_dir) + strlen(comment), 256 - 16);
        return;
    }

    sprintf(output_file_path, "%s%s_%s_blob_data.txt", save_dir, layer_legal_name, comment);

    FILE* file = fopen(output_file_path, "w");
    if (NULL == file)
    {
        fprintf(stderr, "Tengine: Open file(%s) failed, skip dump\n", output_file_path);
        return;
    }

    print_tensor_data_to_file_timvx(file, tensor);

    // close file
    fclose(file);
    file = NULL;
}

void dump_sub_graph_timvx(struct subgraph* sub_graph)
{
    TLOG_INFO("Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->index, sub_graph->device->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
    TLOG_INFO("\tSub nodes: [ ");

    for (int j = 0; j < sub_graph->node_num - 1; j++)
    {
        int node_id = sub_graph->node_list[j];
        TLOG_INFO("%d, ", node_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);

    TLOG_INFO("\tSub input tensors: [ ");
    for (int j = 0; j < sub_graph->input_num - 1; j++)
    {
        int tensor_id = sub_graph->input_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);

    TLOG_INFO("\tSub output tensors: [ ");
    for (int j = 0; j < sub_graph->output_num - 1; j++)
    {
        int tensor_id = sub_graph->output_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
}