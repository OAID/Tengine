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
 * Author: bzhang@openailab.com
 */

#include "ncnn2tengine.hpp"

/*
*   SELF DEFINE VARIABLE
*   FOR NCNN SERIALIZER
*/
const int OP_VERSION = 1;

/*
*   ASSIST FUNCTIONS FOR NCNN SERIALIZER START
*/
bool vstr_is_float(const char vstr[16])
{
    for (int j = 0; j < 16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

static float vstr_to_float(const char vstr[16])
{
    double v = 0.0;

    const char* p = vstr;

    // sign
    bool sign = *p != '-';
    if (*p == '+' || *p == '-')
    {
        p++;
    }

    // digits before decimal point or exponent
    unsigned int v1 = 0;
    while (isdigit(*p))
    {
        v1 = v1 * 10 + (*p - '0');
        p++;
    }

    v = (double)v1;

    // digits after decimal point
    if (*p == '.')
    {
        p++;

        unsigned int pow10 = 1;
        unsigned int v2 = 0;

        while (isdigit(*p))
        {
            v2 = v2 * 10 + (*p - '0');
            pow10 *= 10;
            p++;
        }

        v += v2 / (double)pow10;
    }

    // exponent
    if (*p == 'e' || *p == 'E')
    {
        p++;

        // sign of exponent
        bool fact = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits of exponent
        unsigned int expon = 0;
        while (isdigit(*p))
        {
            expon = expon * 10 + (*p - '0');
            p++;
        }

        double scale = 1.0;
        while (expon >= 8)
        {
            scale *= 1e8;
            expon -= 8;
        }
        while (expon > 0)
        {
            scale *= 10.0;
            expon -= 1;
        }

        v = fact ? v * scale : v / scale;
    }

    return sign ? (float)v : (float)-v;
}

int ncnn_serializer::read(void* buf, int size)
{
    return fread(buf, 1, size, fp);
}
void remove_ncnn_split(std::vector<NcnnNode>& nodelist)
{
    for (auto& curr_node : nodelist)
    {
        if (curr_node.op == "Split")
        {
            for (auto& in_node : nodelist)
            {
                if (in_node.output_name[0] == curr_node.inputs_name[0])
                {
                    auto out_name = in_node.output_name[0];
                    for (auto& out_node : nodelist)
                    {
                        for (auto& out_node_inbound_name : out_node.inputs_name)
                        {
                            for (auto& curr_node_outbound_name : curr_node.output_name)
                            {
                                if (out_node_inbound_name == curr_node_outbound_name)
                                {
                                    out_node.inputs_name.erase(std::remove(
                                                                   out_node.inputs_name.begin(),
                                                                   out_node.inputs_name.end(),
                                                                   out_node_inbound_name),
                                                               out_node.inputs_name.end());
                                    out_node.inputs_name.push_back(in_node.output_name[0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    nodelist.erase(std::remove_if(nodelist.begin(), nodelist.end(), [&](NcnnNode& n) { return n.op == "Split"; }), nodelist.end());
}
int ncnn_serializer::load_model_file(const char* fname, std::vector<NcnnNode>& nodelist)
{
    fp = fopen(fname, "rb");
    if (!fp)
    {
        TLOG_ERR("Cannot open the param file: %s \n", fname);
        return -1;
    }
    // parse each key=value pair
    int id = 0;
    int res = 0;
    int magic = 0;
    res = fscanf(fp, "%d=", &magic);
    fprintf(stderr, "%s magic: %d \n", fname, magic);
    if (magic != 7767517)
    {
        TLOG_ERR("param is too old, please regenerate \n");
    }
    int layer_count = 0;
    int blob_count = 0;
    res = fscanf(fp, "%d=", &layer_count);
    res = fscanf(fp, "%d=", &blob_count);
    // printf("layer_count: %d , blob_count: %d \n", layer_count, blob_count);

    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;

        res = fscanf(fp, "%255s=", layer_type);
        res = fscanf(fp, "%255s=", layer_name);
        res = fscanf(fp, "%d=", &bottom_count);
        res = fscanf(fp, "%d=", &top_count);
        NcnnNode node;
        node.op = layer_type;
        node.optimized = 0;
        node.name = layer_name;

        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[256];
            res = fscanf(fp, "%255s=", bottom_name);
            node.inputs_name.push_back(bottom_name);
        }

        for (int j = 0; j < top_count; j++)
        {
            char top_name[256];
            res = fscanf(fp, "%255s=", top_name);
            node.output_name.push_back(top_name);
        }

        if (res < 0)
        {
            TLOG_ERR("Read Param file data failed\n");
            return false;
        }
        while (fscanf(fp, "%d=", &id) == 1)
        {
            bool array_selection = id <= -23300;

            if (node.op == "Input" && array_selection)
            {
                node.optimized = 1;
            }
            if (array_selection)
            {
                id = -id - 23300;
            }
            if (node.optimized == 1 && array_selection)
            {
                int len = 0;
                int nscan = fscanf(fp, "%d", &len);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array length failed\n");
                    return false;
                }

                params[id].f_data_array = (float*)malloc(sizeof(float) * len);
                params[id].i_data_array = (int*)malloc(sizeof(int) * len);
                // std::vector<std::string> opt_str;
                std::string str = "";
                for (int j = 0; j < len; j++)
                {
                    char vstr[16];
                    nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict read array opt element failed\n");
                        return false;
                    }
                    if (str == "")
                    {
                        str = vstr;
                    }
                    else
                    {
                        str = str + "," + vstr;
                    }
                    bool is_float = vstr_is_float(vstr);
                    if (is_float)
                    {
                        float* ptr = params[id].f_data_array;
                        ptr[j] = vstr_to_float(vstr);
                    }
                    else
                    {
                        int* ptr = params[id].i_data_array;
                        nscan = sscanf(vstr, "%d", &ptr[j]);
                    }
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict parse array opt element failed\n");
                        return false;
                    }
                }
                free(params[id].f_data_array);
                free(params[id].i_data_array);
                node.attrs.insert(std::pair<int, std::string>(id, str));
            }
            else
            {
                if (array_selection)
                {
                    int len = 0;
                    int nscan = fscanf(fp, "%d", &len);
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict read array length failed\n");
                        return false;
                    }
                    std::string str = "";
                    params[id].f_data_array = (float*)malloc(sizeof(float) * len);
                    params[id].i_data_array = (int*)malloc(sizeof(int) * len);
                    for (int j = 0; j < len; j++)
                    {
                        char vstr[16];
                        nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                        if (nscan != 1)
                        {
                            fprintf(stderr, "ParamDict read array normal element failed\n");
                            return false;
                        }
                        if (str == "")
                        {
                            str = vstr;
                        }
                        else
                        {
                            str = str + "," + vstr;
                        }
                        bool is_float = vstr_is_float(vstr);
                        if (is_float)
                        {
                            float* ptr = params[id].f_data_array;
                            ptr[j] = vstr_to_float(vstr);
                        }
                        else
                        {
                            int* ptr = params[id].i_data_array;
                            nscan = sscanf(vstr, "%d", &ptr[j]);
                        }
                        if (nscan != 1)
                        {
                            fprintf(stderr, "ParamDict parse array normal element failed\n");
                            return false;
                        }
                    }
                    node.attrs.insert(std::pair<int, std::string>(id, str));
                    free(params[id].f_data_array);
                    free(params[id].i_data_array);
                }
                else
                {
                    char vstr[16];

                    int nscan = fscanf(fp, "%15s", vstr);
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict read value failed\n");
                        return false;
                    }
                    bool is_float = vstr_is_float(vstr);
                    // printf("string value: %s \n", vstr);
                    node.attrs.insert(std::pair<int, std::string>(id, vstr));
                    float f_data;
                    int i_data;
                    if (is_float)
                    {
                        f_data = vstr_to_float(vstr);
                    }
                    else
                    {
                        nscan = sscanf(vstr, "%d", &i_data);
                    }
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict parse value failed\n");
                        return false;
                    }
                }
            }
            params[id].loaded = 1;
        }

        nodelist.push_back(node);
    }
    remove_ncnn_split(nodelist);
    return 0;
}

int ncnn_serializer::load_binary_file(const char* fname, std::vector<NcnnParam>& paramlist, std::vector<NcnnNode>& nodelist)
{
    fp = fopen(fname, "rb");
    if (!fp)
    {
        TLOG_ERR("Cannot open the bin file: %d\n ");
        return false;
    }

    float magic = 0;
    int nscan = 0;
    for (int i = 0; i < (int)nodelist.size(); i++)
    {
        if (nodelist[i].op == "Convolution" || nodelist[i].op == "DeconvolutionDepthWise" || nodelist[i].op == "Deconvolution" || nodelist[i].op == "ConvolutionDepthWise")
        {
            NcnnParam weight;
            nscan = read(&magic, sizeof(float));
            weight.name = nodelist[i].name + "_w";

            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(6);
            weight.data_len = std::atoi(iter->second.c_str());
            // printf("%s %d ",nodelist[i].name.c_str(), weight.data_len);
            iter = nodelist[i].attrs.find(0);
            int output_channel = std::atoi(iter->second.c_str());

            weight.data = (float*)malloc(sizeof(float) * weight.data_len);
            read(weight.data, sizeof(float) * weight.data_len);
            // printf("%f %f \n", weight.data, weight.data);
            iter = nodelist[i].attrs.find(1);
            int kernel_size = std::atoi(iter->second.c_str());
            int c = weight.data_len / (output_channel * kernel_size * kernel_size);
            weight.dims.push_back(output_channel);
            weight.dims.push_back(c);
            weight.dims.push_back(kernel_size);
            weight.dims.push_back(kernel_size);
            iter = nodelist[i].attrs.find(5);
            int biasTerm = 0;

            if (!iter->second.empty())
                biasTerm = std::atoi(iter->second.c_str());

            paramlist.push_back(weight);
            if (biasTerm == 1)
            {
                NcnnParam bias;
                bias.name = nodelist[i].name + "_b";
                bias.data_len = output_channel;
                bias.data = (float*)malloc(sizeof(float) * output_channel);
                read(bias.data, sizeof(float) * output_channel);
                bias.dims.push_back(output_channel);
                paramlist.push_back(bias);
            }
        }
        else if (nodelist[i].op == "BatchNorm")
        {
            NcnnParam slope, mean, variance, bias;
            slope.name = nodelist[i].name + "_s";
            mean.name = nodelist[i].name + "_m";
            variance.name = nodelist[i].name + "_v";
            bias.name = nodelist[i].name + "_b";

            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);

            slope.data_len = std::atoi(iter->second.c_str());
            mean.data_len = std::atoi(iter->second.c_str());
            variance.data_len = std::atoi(iter->second.c_str());
            bias.data_len = std::atoi(iter->second.c_str());

            bias.data = (float*)malloc(sizeof(float) * slope.data_len);
            variance.data = (float*)malloc(sizeof(float) * slope.data_len);
            slope.data = (float*)malloc(sizeof(float) * slope.data_len);
            mean.data = (float*)malloc(sizeof(float) * slope.data_len);

            read(slope.data, sizeof(float) * slope.data_len);
            read(mean.data, sizeof(float) * slope.data_len);
            read(variance.data, sizeof(float) * slope.data_len);
            read(bias.data, sizeof(float) * slope.data_len);

            slope.dims.push_back(slope.data_len);
            mean.dims.push_back(slope.data_len);
            variance.dims.push_back(slope.data_len);
            bias.dims.push_back(slope.data_len);

            paramlist.push_back(slope);
            paramlist.push_back(mean);
            paramlist.push_back(variance);
            paramlist.push_back(bias);
        }
        else if (nodelist[i].op == "Embed")
        {
            NcnnParam weight, bias;
            nscan = read(&magic, sizeof(float));
            weight.name = nodelist[i].name + "_w";
            bias.name = nodelist[i].name + "_b";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(3);
            weight.data_len = std::atoi(iter->second.c_str());
            iter = nodelist[i].attrs.find(0);
            bias.data_len = std::atoi(iter->second.c_str());

            weight.data = (float*)malloc(sizeof(float) * weight.data_len);
            bias.data = (float*)malloc(sizeof(float) * bias.data_len);
            read(weight.data, sizeof(float) * weight.data_len);
            read(bias.data, sizeof(float) * bias.data_len);
            weight.dims.push_back(weight.data_len);
            bias.dims.push_back(bias.data_len);
            paramlist.push_back(weight);
            paramlist.push_back(bias);
        }
        else if (nodelist[i].op == "InnerProduct")
        {
            NcnnParam weight;
            nscan = read(&magic, sizeof(float));
            weight.name = nodelist[i].name + "_w";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);
            int output_num = std::atoi(iter->second.c_str());
            iter = nodelist[i].attrs.find(2);
            weight.data_len = std::atoi(iter->second.c_str());

            weight.data = (float*)malloc(sizeof(float) * weight.data_len);
            read(weight.data, sizeof(float) * weight.data_len);
            weight.dims.push_back(output_num);
            weight.dims.push_back(weight.data_len / output_num);
            paramlist.push_back(weight);
            iter = nodelist[i].attrs.find(1);
            int biasTerm = std::atoi(iter->second.c_str());
            if (biasTerm == 1)
            {
                NcnnParam bias;
                bias.name = nodelist[i].name + "_b";
                bias.data_len = output_num;
                bias.data = (float*)malloc(sizeof(float) * output_num);
                read(bias.data, sizeof(float) * output_num);
                bias.dims.push_back(output_num);
                paramlist.push_back(bias);
            }
        }
        else if (nodelist[i].op == "Normalize")
        {
            NcnnParam scale;
            nscan = read(&magic, sizeof(float));
            scale.name = nodelist[i].name + "_s";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(3);
            scale.data_len = std::atoi(iter->second.c_str());
            scale.data = (float*)malloc(sizeof(float) * scale.data_len);
            read(scale.data, sizeof(float) * scale.data_len);
            scale.dims.push_back(scale.data_len);
            paramlist.push_back(scale);
        }
        else if (nodelist[i].op == "PReLU")
        {
            NcnnParam slope;
            nscan = read(&magic, sizeof(float));
            slope.name = nodelist[i].name + "_s";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);
            slope.data_len = std::atoi(iter->second.c_str());
            slope.data = (float*)malloc(sizeof(float) * slope.data_len);
            read(slope.data, sizeof(float) * slope.data_len);
            slope.dims.push_back(slope.data_len);
            paramlist.push_back(slope);
        }
        else if (nodelist[i].op == "Scale")
        {
            NcnnParam scale;
            nscan = read(&magic, sizeof(float));
            scale.name = nodelist[i].name + "_s";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);
            scale.data_len = std::atoi(iter->second.c_str());
            scale.data = (float*)malloc(sizeof(float) * scale.data_len);
            read(scale.data, sizeof(float) * scale.data_len);
            scale.dims.push_back(scale.data_len);
            paramlist.push_back(scale);

            iter = nodelist[i].attrs.find(1);
            int biasTerm = std::atoi(iter->second.c_str());
            if (biasTerm == 1)
            {
                NcnnParam bias;
                bias.name = nodelist[i].name + "_b";
                bias.data_len = scale.data_len;
                bias.data = (float*)malloc(sizeof(float) * scale.data_len);
                read(bias.data, sizeof(float) * scale.data_len);
                bias.dims.push_back(scale.data_len);
                paramlist.push_back(bias);
            }
        }
        else if (nodelist[i].op == "MemoryData")
        {
            NcnnParam const_data;
            std::map<int, std::string>::iterator iter;
            int data_len = 1;
            int size = (int)nodelist[i].attrs.size();
            std::vector<int> dims(size);
            for (iter = nodelist[i].attrs.begin(); iter != nodelist[i].attrs.end(); iter++)
            {
                std::pair<int, std::string> pair = *iter;
                data_len *= atoi(pair.second.c_str());
                dims[pair.first] = atoi(pair.second.c_str());
            }
            const_data.name = nodelist[i].name;
            const_data.dim_size = (int)dims.size();
            const_data.dims = dims;
            const_data.data_len = data_len;
            const_data.data = (float*)malloc(sizeof(float) * data_len);
            read(const_data.data, sizeof(float) * data_len);
            paramlist.push_back(const_data);
        }
    }
    if (nscan < 0)
    {
        TLOG_ERR("Cannot read the binary file: %s \n ", fname);
    }
#if 0
    printf("total size: %d \n", totalSize);
    float* data = (float*)malloc(sizeof(float)*totalSize);
    fread(data, 1, sizeof(float)*totalSize, fp);
    for(int j = 0; j < totalSize; j++){
        if(j % 12 == 0){
            printf("\n");
        }
        printf("%f ", data[j]);
    }
#endif
    return 0;
}

int ncnn_serializer::load_constant_tensor(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist)
{
    int const_tensor_number = paramlist.size();
    std::set<std::string> node_name_set;

#ifdef DEBUG
    std::cout << "Load Const Tensor:" << std::endl;
#endif
    for (int i = 0; i < const_tensor_number; i++)
    {
        const NcnnParam& ncnn_tensor = paramlist.at(i);
#if 0
        std::cout << "    name: " << ncnn_tensor.name << std::endl;
#endif
        std::vector<int> dims = ncnn_tensor.dims;
        ir_tensor_t* ir_tensor = create_ir_tensor(graph, ncnn_tensor.name.c_str(), TENGINE_DT_FP32);
        int* tensor_dims = new int[(int)dims.size()];
        for (int j = 0; j < (int)dims.size(); j++)
        {
            tensor_dims[j] = ncnn_tensor.dims[j];
        }
        set_ir_tensor_shape(ir_tensor, tensor_dims, dims.size());
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
        int tensor_size = ncnn_tensor.data_len * sizeof(float);
        ir_tensor->data = (float*)malloc(tensor_size);

        float* mem_buf = (float*)ir_tensor->data;
        float* raw_data = (float*)ncnn_tensor.data;
        /* load data */
        for (int k = 0; k < ncnn_tensor.data_len; k++)
        {
            mem_buf[k] = raw_data[k];
        }
        ir_node_t* ir_node = create_ir_node(graph, ncnn_tensor.name.c_str(), OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
    }

    return 0;
}

static bool GetParam(const std::string name, const std::vector<NcnnParam>& paramlist, NcnnParam& param)
{
    for (unsigned int i = 0; i < paramlist.size(); i++)
    {
        if (paramlist.at(i).name == name)
        {
            param = paramlist.at(i);
            return true;
        }
    }
    return false;
}

float ParseNumber(const char* s, float d)
{
    bool bNegtiveBase, bNegtiveExp;
    int nPreZero = 0;
    const char* p;
    int sum_i = 0;
    float sum_f = 0.0;
    int sum_exp = 0;
    float sum = 0.0;
    bNegtiveBase = bNegtiveExp = false;
    if (!s)
        return false;
    if ('-' == *s)
    {
        bNegtiveBase = true;
        s++;
    }
    for (; '0' == *s; nPreZero++, s++)
        ;
    for (; *s != '.' && *s != 'e' && *s != 'E' && *s != '\0'; s++)
    {
        if (*s < '0' || *s > '9')
        {
            return false;
        }
        sum_i = sum_i * 10 + *s - '0';
    }
    if (0 == sum_i && 0 == nPreZero)
        return false;
    if ('.' == *s)
    {
        for (p = s; *p != 'e' && *p != 'E' && *p != '\0'; p++)
            ;
        if (s == p - 1)
            return false;
        s = p;
        p--;
        for (; *p != '.'; p--)
        {
            if (*p < '0' || *p > '9')
                return false;
            sum_f = sum_f * 0.1 + 0.1 * (*p - '0');
        }
    }
    if ('e' == *s || 'E' == *s)
    {
        s++;
        if ('-' == *s)
        {
            bNegtiveExp = true;
            s++;
        }
        else if ('+' == *s)
        {
            bNegtiveExp = false;
            s++;
        }
        nPreZero = 0;
        for (; *s != '\0'; s++)
        {
            if (*s < '0' || *s > '9')
            {
                return false;
            }
            sum_exp = sum_exp * 10 + *s - '0';
            nPreZero++;
        }
        if (0 == sum_exp && 0 == nPreZero)
            return false;
    }
    sum = sum_i + sum_f;
    if (bNegtiveExp)
    {
        while (sum_exp > 0)
        {
            sum /= 10;
            sum_exp--;
        }
    }
    else
    {
        while (sum_exp > 0)
        {
            sum *= 10;
            sum_exp--;
        }
    }
    if (bNegtiveBase)
        sum = -sum;
    d = sum;
    // printf("%f \n", d);
    return d;
}
std::vector<float>& split(const std::string& str, char delim, std::vector<float>& elems, bool skip_empty = true)
{
    std::istringstream iss(str);
    for (std::string item; getline(iss, item, delim);)
        if (skip_empty && item.empty())
            continue;
        else
        {
            float d = ParseNumber(item.c_str(), d);
            elems.push_back(d);
        }
    return elems;
}
void ParseAttr_n(const std::string str, std::vector<float>& result)
{
    split(str, ',', result);
}

int ncnn_serializer::set_graph_input(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist)
{
#ifdef DEBUG
    std::cout << "Create Input Node:" << std::endl;
#endif
    std::vector<int16_t> input_nodes;
    for (unsigned int i = 0; i < nodelist.size(); i++)
    {
        const NcnnNode& ncnn_node = nodelist.at(i);
        if (ncnn_node.op == "Input")
        {
            std::string input_name = ncnn_node.name;

            ir_tensor_t* ir_tensor = create_ir_tensor(graph, input_name.c_str(), TENGINE_DT_FP32);

            NcnnParam param;
            if (GetParam(input_name, paramlist, param))
            {
                std::vector<int> ir_dims = param.dims;
                int* tensor_dims = new int[ir_dims.size()];
                for (int j = 0; j < ir_dims.size(); j++)
                {
                    tensor_dims[j] = ir_dims[j];
                }
                if (ir_dims.size() > 0)
                    ;
                set_ir_tensor_shape(ir_tensor, tensor_dims, ir_dims.size());
            }
            ir_node_t* node = create_ir_node(graph, input_name.c_str(), OP_INPUT, OP_VERSION);
            set_ir_node_output_tensor(node, 0, ir_tensor);
            input_nodes.push_back(node->index);
#ifdef DEBUG
            std::cout << "    create an input node for " << ncnn_node.name << std::endl;
#endif
        }
    }
    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * input_nodes.size());
    for (int i = 0; i < input_nodes.size(); i++)
    {
        node_idx[i] = input_nodes[i];
    }
    set_ir_graph_input_node(graph, node_idx, input_nodes.size());
    return 0;
}

int ncnn_serializer::set_graph_output(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist)
{
    std::vector<int16_t> output_nodes;
    for (unsigned int i = 0; i < nodelist.size(); i++)
    {
        const NcnnNode& ncnn_node = nodelist[i];
        std::string input_name = ncnn_node.output_name[0];
        int tensor_id = get_ir_tensor_index_from_name(graph, input_name.c_str());
        ir_tensor_t* ir_tensor = get_ir_graph_tensor(graph, tensor_id);

        if (ir_tensor->consumer_num == 0)
        {
            NcnnParam param;
            if (GetParam(input_name, paramlist, param))
            {
                std::vector<int> ir_dims = param.dims;
                int* tensor_dims = new int[ir_dims.size()];
                for (int j = 0; j < ir_dims.size(); j++)
                {
                    tensor_dims[j] = ir_dims[j];
                }
                if (ir_dims.size() > 0)
                    ;
                set_ir_tensor_shape(ir_tensor, tensor_dims, ir_dims.size());
            }

            ir_node_t* node = create_ir_node(graph, input_name.c_str(), OP_INPUT, OP_VERSION);
            set_ir_node_output_tensor(node, 0, ir_tensor);
            output_nodes.push_back(node->index);
        }
    }
    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * output_nodes.size());
    for (int i = 0; i < output_nodes.size(); i++)
    {
        node_idx[i] = output_nodes[i];
    }
    set_ir_graph_output_node(graph, node_idx, output_nodes.size());
    return 0;
}

bool ncnn_serializer::find_op_load_method(const std::string& op_name)
{
    if (op_load_map.count(op_name))
        return true;

    return false;
}
ir_tensor_t* ncnn_serializer::find_tensor(ir_graph_t* graph, const std::string& tensor_name)
{
    for (uint16_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->name == tensor_name)
            return tensor;
    }

    return nullptr;
}
int ncnn_serializer::load_graph_node(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist)
{
    std::vector<int> node_to_remove;
    unsigned int i;
    std::vector<std::string> no_supported_op;
    for (i = 0; i < nodelist.size(); i++)
    {
        NcnnNode ncnn_node = nodelist.at(i);
        if (ncnn_node.op == "Noop" && ncnn_node.output_name.size() == 0)
            node_to_remove.push_back(i);
        if (!find_op_load_method(ncnn_node.op))
        {
            auto it = find(no_supported_op.begin(), no_supported_op.end(), ncnn_node.op);
            if (it != no_supported_op.end())
                no_supported_op.push_back(ncnn_node.op);
        }
    }
    for (i = 0; i < nodelist.size(); i++)
    {
        NcnnNode ncnn_node = nodelist.at(i);
        if (ncnn_node.op == "Input" || ncnn_node.op == "MemoryData")
            continue;

        if (ncnn_node.op == "Noop" && ncnn_node.output_name.size() == 0)
            continue;

        ir_node_t* ir_node = nullptr;

        ir_node = create_ir_node(graph, ncnn_node.name.c_str(), op_load_map[ncnn_node.op].first, OP_VERSION);
        if (ir_node == NULL)
        {
            return -1;
        }
        int input_number = ncnn_node.inputs_name.size();
        int size = (int)nodelist.size();
        int in_num = 0;
        for (in_num = 0; in_num < input_number; in_num++)
        {
            std::string input_name = ncnn_node.inputs_name[in_num];

            int tensor_id = get_ir_tensor_index_from_name(graph, input_name.c_str());
            ir_tensor_t* ir_tensor = get_ir_graph_tensor(graph, tensor_id);
            if (ir_tensor == NULL)
            {
                fprintf(stderr, "Can not find tensor : %s \n", input_name.c_str());
            }
            set_ir_node_input_tensor(ir_node, in_num, ir_tensor);
            size = (int)paramlist.size();
            int tensor_idx = 0;
            for (int j = 0; j < size; j++)
            {
                std::string input_name = paramlist[j].name;
                std::string name = input_name.substr(0, input_name.length() - 2);
                if (name == ncnn_node.name)
                {
                    tensor_idx++;
                    ir_tensor_t* tensor = find_tensor(graph, paramlist[j].name);
                    set_ir_node_input_tensor(ir_node, tensor_idx, tensor);
                }
            }
        }

        int out_size = (int)ncnn_node.output_name.size();
        for (int j = 0; j < out_size; j++)
        {
            const std::string& output_name = ncnn_node.output_name[j];
            ir_tensor_t* tensor = create_ir_tensor(graph, output_name.c_str(), TENGINE_DT_FP32);
            set_ir_node_output_tensor(ir_node, j, tensor);
        }
        op_load_t loader = op_load_map[ncnn_node.op].second;

        if (loader(graph, ir_node, ncnn_node) < 0)
        {
            TLOG_ERR("load op %s func failed in node %s .\n", ncnn_node.op.c_str(), ncnn_node.name.c_str());
            return -1;
        }
    }
    if (i < nodelist.size())
    {
        return false;
    }

#if DEBUG
    std::cout << "Successfully load all nodes" << std::endl;
#endif

    return 0;
}

int ncnn_serializer::load_model(ir_graph_t* graph, std::string bin_file, std::string params_file)
{
    register_op_load();

    std::vector<NcnnNode> nodelist;
    std::vector<NcnnParam> paramlist;

    if (load_model_file(params_file.c_str(), nodelist) < 0)
        return -1;
    fprintf(stderr, "Process 1: Finish load model file \n");
    if (load_binary_file(bin_file.c_str(), paramlist, nodelist) < 0)
        return -1;
    fprintf(stderr, "Process 2: Finish load binary file \n");
    if (load_constant_tensor(graph, nodelist, paramlist) < 0)
        return -1;
    fprintf(stderr, "Process 3: Finish load tensor data \n");
    if (set_graph_input(graph, nodelist, paramlist) < 0)
        return -1;
    fprintf(stderr, "Process 4: Finish load graph input node \n");
    if (load_graph_node(graph, nodelist, paramlist) < 0)
        return -1;
    fprintf(stderr, "Process 5: Finish load graph node \n");
    if (set_graph_output(graph, nodelist, paramlist) < 0)
        return -1;
    fprintf(stderr, "Process 6: Finish load graph output node \n");

    return 0;
}
graph_t ncnn_serializer::ncnn2tengine(std::string model_file, std::string proto_file)
{
    fprintf(stderr, "----------ncnn2tengine begin----------\n");

    context_t context = create_context(NULL, 1);
    ir_graph_t* ir_graph = create_ir_graph((struct context*)context);
    if (ir_graph == NULL)
    {
        destroy_context(context);
        return NULL;
    }
    ir_graph->attribute->private_context = 1; // new context

    int ret = load_model(ir_graph, model_file, proto_file);
    if (0 != ret)
    {
        destroy_graph(ir_graph);
        return NULL;
    }
    ir_graph->device = find_default_device();

    fprintf(stderr, "----------ncnn2tengine done.----------\n");
    return ir_graph;
}

typedef std::map<int, std::string>::const_iterator const_iterator;
int load_conv(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct conv_param* param = (struct conv_param*)node->op.param_mem;

    const_iterator iter;

    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
    {
        param->output_channel = std::atoi(iter->second.c_str());
        if (ncnn_node.op == "ConvolutionDepthWise")
        {
            param->group = std::atoi(iter->second.c_str());
        }
    }

    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
    {
        param->kernel_h = std::atoi(iter->second.c_str());
        param->kernel_w = std::atoi(iter->second.c_str());
    }
    else
    {
        param->kernel_h = 0;
        param->kernel_w = 0;
    }

    iter = ncnn_node.attrs.find(2);
    if (iter != ncnn_node.attrs.end())
    {
        param->dilation_h = std::atoi(iter->second.c_str());
        param->dilation_w = std::atoi(iter->second.c_str());
    }
    else
    {
        param->dilation_h = 1;
        param->dilation_w = 1;
    }

    iter = ncnn_node.attrs.find(3);
    if (iter != ncnn_node.attrs.end())
    {
        param->stride_h = std::atoi(iter->second.c_str());
        param->stride_w = std::atoi(iter->second.c_str());
    }
    else
    {
        param->stride_h = 1;
        param->stride_w = 1;
    }

    iter = ncnn_node.attrs.find(4);
    if (iter != ncnn_node.attrs.end())
    {
        param->pad_w0 = std::atoi(iter->second.c_str());
        param->pad_w1 = std::atoi(iter->second.c_str());
        param->pad_h0 = std::atoi(iter->second.c_str());
        param->pad_h1 = std::atoi(iter->second.c_str());
    }

    iter = ncnn_node.attrs.find(9);
    if (iter != ncnn_node.attrs.end())
    {
        param->activation = 0;
    }
    return 0;
}
int load_pool(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct pool_param* param = (struct pool_param*)node->op.param_mem;
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
    {
        int type = std::atoi(iter->second.c_str());
        if (type == 0)
        {
            param->pool_method = 0;
        }
        else if (type == 1)
        {
            param->pool_method = 1;
        }
    }

    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
    {
        param->kernel_h = std::atoi(iter->second.c_str());
        param->kernel_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(2);
    if (iter != ncnn_node.attrs.end())
    {
        param->stride_h = std::atoi(iter->second.c_str());
        param->stride_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(3);
    if (iter != ncnn_node.attrs.end())
    {
        param->pad_h0 = std::atoi(iter->second.c_str());
        param->pad_h1 = std::atoi(iter->second.c_str());
        param->pad_w0 = std::atoi(iter->second.c_str());
        param->pad_w1 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(4);
    if (iter != ncnn_node.attrs.end())
    {
        param->global = std::atoi(iter->second.c_str());
    }
    param->caffe_flavor = 1;
    return 0;
}
int load_relu(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct relu_param* relu_param = (struct relu_param*)node->op.param_mem;

    return 0;
}
int load_concat(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct concat_param* param = (struct concat_param*)node->op.param_mem;
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
        param->axis = std::atoi(iter->second.c_str()) + 1;

    return 0;
}
int load_softmax(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct softmax_param* param = (struct softmax_param*)node->op.param_mem;
    const_iterator iter;

    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
        param->axis = std::atoi(iter->second.c_str());

    // param.axis = 1;
    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
        param->axis = param->axis + std::atoi(iter->second.c_str());

    return 0;
}
int load_no_param(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    return 0;
}
int load_bn(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct batchnorm_param* param = (struct batchnorm_param*)node->op.param_mem;

    const_iterator iter;

    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
        param->eps = std::atoi(iter->second.c_str());
    param->caffe_flavor = 1;
    return 0;
}
int load_scale(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct scale_param* param = (struct scale_param*)node->op.param_mem;

    const_iterator iter;

    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
        param->bias_term = std::atoi(iter->second.c_str());

    return 0;
}
int load_clip(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct clip_param* param = (struct clip_param*)node->op.param_mem;
    const_iterator iter;

    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
        param->max = std::atoi(iter->second.c_str());

    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
        param->min = std::atoi(iter->second.c_str());

    return 0;
}
int load_fc(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct fc_param* param = (struct fc_param*)node->op.param_mem;

    const_iterator iter;

    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
        param->num_output = std::atoi(iter->second.c_str());

    return 0;
}
int load_flatten(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct flatten_param* param = (struct flatten_param*)node->op.param_mem;

    param->axis = 1;

    return 0;
}
int load_reshape(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct reshape_param* param = (struct reshape_param*)node->op.param_mem;
    std::vector<int> dim_shape;
    const_iterator iter;
    iter = ncnn_node.attrs.find(3);
    if (iter != ncnn_node.attrs.end())
    {
        if (-233 != std::atoi(iter->second.c_str()))
        {
            dim_shape.push_back(std::atoi(iter->second.c_str()));
        }
    }
    else
    {
        dim_shape.push_back(0);
    }
    iter = ncnn_node.attrs.find(2);
    if (iter != ncnn_node.attrs.end())
    {
        if (-233 != std::atoi(iter->second.c_str()))
        {
            dim_shape.push_back(std::atoi(iter->second.c_str()));
        }
    }
    else
    {
        dim_shape.push_back(0);
    }
    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
    {
        if (-233 != std::atoi(iter->second.c_str()))
        {
            dim_shape.push_back(std::atoi(iter->second.c_str()));
        }
    }
    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
    {
        if (-233 != std::atoi(iter->second.c_str()))
        {
            dim_shape.push_back(std::atoi(iter->second.c_str()));
        }
    }

    int size = (int)dim_shape.size();
    param->re_shape = (int*)sys_malloc(sizeof(int) * size);
    param->dim_size = size;
    for (int i = 0; i < size; i++)
    {
        param->re_shape[i] = dim_shape[i];
    }

    return 0;
}
int load_eltwise(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct eltwise_param* param = (struct eltwise_param*)node->op.param_mem;
    const_iterator iter;

    std::vector<float> coef;

    iter = ncnn_node.attrs.find(0);
    int opType = -1;
    if (iter != ncnn_node.attrs.end())
        opType = std::atoi(iter->second.c_str());

    if (opType == 0)
    {
        param->type = ELT_PROD;
    }
    else if (opType == 1)
    {
        param->type = ELT_SUM;
    }
    else if (opType == 2)
    {
        param->type = ELT_MAX;
    }
    else
    {
        param->type = ELT_SUM;
    }
    param->caffe_flavor = 0;
    return 0;
}
int load_resize(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct interp_param* param = (struct interp_param*)node->op.param_mem;

    std::vector<float> v1, v2;
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
    {
        param->resize_type = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
    {
        ParseAttr_n(iter->second, v1);
        param->width_scale = v1.at(0);
    }
    else
    {
        param->width_scale = 0;
    }
    iter = ncnn_node.attrs.find(2);
    if (iter != ncnn_node.attrs.end())
    {
        ParseAttr_n(iter->second, v2);
        param->height_scale = v2.at(0);
    }
    else
    {
        param->height_scale = 0;
    }
    iter = ncnn_node.attrs.find(3);
    if (iter != ncnn_node.attrs.end())
    {
        ParseAttr_n(iter->second, v2);
        param->output_width = v2.at(0);
    }
    iter = ncnn_node.attrs.find(4);
    if (iter != ncnn_node.attrs.end())
    {
        ParseAttr_n(iter->second, v2);
        param->output_height = v2.at(0);
    }
    return 0;
}
int load_slice(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct slice_param* param = (struct slice_param*)node->op.param_mem;
    // param->isncnn= true;
    param->iscaffe = false;
    param->ismxnet = false;
    param->isonnx = false;
    // param->slice_point_.clear();
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    std::vector<float> v1;
    if (iter != ncnn_node.attrs.end())
    {
        ParseAttr_n(iter->second, v1);
        std::vector<int> slice_shape;
        for (int i = 0; i < (int)v1.size(); i++)
        {
            // param->slice_point_.push_back((int)v1.at(i));
        }
    }
    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
    {
        param->axis = std::atoi(iter->second.c_str()) + 1;
    }
    return 0;
}

int load_unary(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct unary_param* param = (struct unary_param*)node->op.param_mem;
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
        param->type = std::atoi(iter->second.c_str());

    return 0;
}
int load_deconv(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node)
{
    struct deconv_param* param = (struct deconv_param*)node->op.param_mem;
    const_iterator iter;
    std::vector<float> v1;
    iter = ncnn_node.attrs.find(0);
    if (iter != ncnn_node.attrs.end())
    {
        param->num_output = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(1);
    if (iter != ncnn_node.attrs.end())
    {
        param->kernel_w = std::atoi(iter->second.c_str());
        param->kernel_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(11);
    if (iter != ncnn_node.attrs.end())
    {
        param->kernel_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(2);
    if (iter != ncnn_node.attrs.end())
    {
        param->dilation_w = std::atoi(iter->second.c_str());
        param->dilation_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(12);
    if (iter != ncnn_node.attrs.end())
    {
        param->dilation_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(3);
    if (iter != ncnn_node.attrs.end())
    {
        param->stride_h = std::atoi(iter->second.c_str());
        param->stride_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(13);
    if (iter != ncnn_node.attrs.end())
    {
        param->stride_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(4);
    if (iter != ncnn_node.attrs.end())
    {
        param->pad_w0 = std::atoi(iter->second.c_str());
        param->pad_w1 = std::atoi(iter->second.c_str());
        param->pad_h0 = std::atoi(iter->second.c_str());
        param->pad_h1 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(15);
    if (iter != ncnn_node.attrs.end())
    {
        param->pad_w1 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(16);
    if (iter != ncnn_node.attrs.end())
    {
        param->pad_h0 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(17);
    if (iter != ncnn_node.attrs.end())
    {
        param->pad_h1 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(7);
    if (iter != ncnn_node.attrs.end())
    {
        param->group = std::atoi(iter->second.c_str());
    }
    return 0;
}

/*
*   OPERAOTR REGISTER FUNCTION DEFINE FOR NCNN SERIALIZER START
*/
void ncnn_serializer::register_op_load()
{
    op_load_map["Convolution"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["ConvolutionDepthWise"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["Pooling"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["ReLU"] = std::pair<int, op_load_t>(OP_RELU, load_relu);
    op_load_map["Concat"] = std::pair<int, op_load_t>(OP_CONCAT, load_concat);
    op_load_map["Softmax"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["Dropout"] = std::pair<int, op_load_t>(OP_DROPOUT, load_no_param);
    op_load_map["BatchNorm"] = std::pair<int, op_load_t>(OP_BATCHNORM, load_bn);
    op_load_map["Scale"] = std::pair<int, op_load_t>(OP_SCALE, load_scale);
    op_load_map["Clip"] = std::pair<int, op_load_t>(OP_CLIP, load_clip);
    op_load_map["InnerProduct"] = std::pair<int, op_load_t>(OP_FC, load_fc);
    // op_load_map["PriorBox"]                          = std::pair<int, op_load_t>();
    op_load_map["Flatten"] = std::pair<int, op_load_t>(OP_FLATTEN, load_flatten);
    op_load_map["Reshape"] = std::pair<int, op_load_t>(OP_RESHAPE, load_reshape);
    op_load_map["Eltwise"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Interp"] = std::pair<int, op_load_t>(OP_INTERP, load_resize);
    op_load_map["Slice"] = std::pair<int, op_load_t>(OP_SLICE, load_slice);
    op_load_map["Sigmoid"] = std::pair<int, op_load_t>(OP_SIGMOID, load_no_param);
    op_load_map["UnaryOp"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["Deconvolution"] = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
    op_load_map["DeconvolutionDepthWise"] = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
}
/*
*   OPERATOR REGISTER FUNCTION DEFINE FOR NCNN SERIALIZER END
*/
