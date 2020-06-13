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
 * Copyright (c) 2019, Open AI Lab
 * Author: bzhang@openailab.com
 */

#include "ncnn_serializer.hpp"
#include <set>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "type_name.hpp"
#include "data_type.hpp"
#include "tengine_errno.hpp"
#include "static_graph.hpp"
#include "operator_manager.hpp"

#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/split_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/scale_param.hpp"
#include "operator/clip_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/permute_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/priorbox_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/detection_output_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/interp_param.hpp"
#include "operator/crop_param.hpp"
#include "operator/slice_param.hpp"
#include "operator/deconv_param.hpp"
#include "operator/yolov3detectionoutput_param.hpp"
#include "operator/unary_param.hpp"
namespace TEngine{

typedef std::map<int, std::string>::const_iterator const_iterator;
using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)>;

bool NcnnSerializer::vstr_is_float(const char vstr[16])
{
    for (int j=0; j<16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

int  NcnnSerializer::read(void* buf, int size)
{
    //printf("start read.\n");
    return fread(buf, 1, size, fp);
}

bool NcnnSerializer::LoadBinaryFile(const char* fname, std::vector<NcnnParam>& paramlist, std::vector<NcnnNode>& nodelist){

    fp = fopen(fname, "rb");
    if(!fp)
    {
        LOG_ERROR() << "Cannot open the bin file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }
   
    float magic = 0; 
    int nscan = 0;
    for(int i = 0; i < (int)nodelist.size(); i++){
        //printf("%d of %d %s type %s.\n", i, (int)nodelist.size(), nodelist[i].name.c_str(), nodelist[i].op.c_str());  
        if(nodelist[i].op == "Convolution" || nodelist[i].op == "DeconvolutionDepthWise" 
        || nodelist[i].op == "Deconvolution" || nodelist[i].op == "ConvolutionDepthWise"){
            NcnnParam weight;
            nscan = read(&magic, sizeof(float));
            weight.name = nodelist[i].name+"_w";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(6);
            //printf("%s\n", iter->second.c_str());
            weight.data_len = std::atoi(iter->second.c_str());
            //printf("%s %d ",nodelist[i].name.c_str(), weight.data_len);
            iter = nodelist[i].attrs.find(0);
            //printf("%d\n", std::atoi(iter->second.c_str()));
            int output_channel = std::atoi(iter->second.c_str());            

            weight.data = (float*)malloc(sizeof(float)*weight.data_len);            
            //printf("%f %f ", weight.data, weight.data);
            read(weight.data, sizeof(float)*weight.data_len);
            //printf("%s read weigth done.",nodelist[i].name.c_str());
            iter = nodelist[i].attrs.find(1);
            //printf("%d\n", std::atoi(iter->second.c_str()));
            int kernel_size = std::atoi(iter->second.c_str());
            int c = weight.data_len/(output_channel*kernel_size*kernel_size);
            weight.dims.push_back(output_channel);
            weight.dims.push_back(c);
            weight.dims.push_back(kernel_size);
            weight.dims.push_back(kernel_size);
            iter = nodelist[i].attrs.find(5);
            int biasTerm = 0;
            
            if(!iter->second.empty())
                biasTerm = std::atoi(iter->second.c_str());

            paramlist.push_back(weight);
            if(biasTerm == 1){
                NcnnParam bias;
                bias.name = nodelist[i].name+"_b";
                bias.data_len = output_channel;
                bias.data = (float*)malloc(sizeof(float)*output_channel);
                read(bias.data, sizeof(float)*output_channel);
                bias.dims.push_back(output_channel);
                paramlist.push_back(bias);
                //printf("biased\n");
            }            
        }
        else if(nodelist[i].op == "BatchNorm"){
            NcnnParam slope, mean, variance, bias;
            slope.name = nodelist[i].name+"_s";
            mean.name = nodelist[i].name+"_m";
            variance.name = nodelist[i].name+"_v";
            bias.name = nodelist[i].name+"_b";

            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);

            slope.data_len = std::atoi(iter->second.c_str());
            mean.data_len =  std::atoi(iter->second.c_str());
            variance.data_len = std::atoi(iter->second.c_str());
            bias.data_len = std::atoi(iter->second.c_str());

            bias.data = (float*)malloc(sizeof(float)*slope.data_len);
            variance.data = (float*)malloc(sizeof(float)*slope.data_len);
            slope.data = (float*)malloc(sizeof(float)*slope.data_len);
            mean.data = (float*)malloc(sizeof(float)*slope.data_len);

            read(slope.data, sizeof(float)*slope.data_len);
            read(mean.data, sizeof(float)*slope.data_len);
            read(variance.data, sizeof(float)*slope.data_len);
            read(bias.data, sizeof(float)*slope.data_len);

            slope.dims.push_back(slope.data_len);
            mean.dims.push_back(slope.data_len);
            variance.dims.push_back(slope.data_len);
            bias.dims.push_back(slope.data_len);

            paramlist.push_back(slope);
            paramlist.push_back(mean);
            paramlist.push_back(variance);
            paramlist.push_back(bias);
        }
        else if(nodelist[i].op == "Embed"){
            NcnnParam weight, bias;
            nscan = read(&magic, sizeof(float));
            weight.name = nodelist[i].name+"_w";
            bias.name = nodelist[i].name+"_b";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(3);
            weight.data_len = std::atoi(iter->second.c_str());
            iter = nodelist[i].attrs.find(0);
            bias.data_len = std::atoi(iter->second.c_str());
            
            weight.data = (float*)malloc(sizeof(float)*weight.data_len);
            bias.data = (float*)malloc(sizeof(float)*bias.data_len);
            read(weight.data, sizeof(float)*weight.data_len);
            read(bias.data, sizeof(float)*bias.data_len);
            weight.dims.push_back(weight.data_len);
            bias.dims.push_back(bias.data_len);
            paramlist.push_back(weight);
            paramlist.push_back(bias);
        }
        else if(nodelist[i].op == "InnerProduct"){
            NcnnParam weight, bias;
            nscan = read(&magic, sizeof(float));
            weight.name = nodelist[i].name+"_w";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);
            int output_num = std::atoi(iter->second.c_str());            
            iter = nodelist[i].attrs.find(2);
            weight.data_len = std::atoi(iter->second.c_str());

            weight.data = (float*)malloc(sizeof(float)*weight.data_len);
            read(weight.data, sizeof(float)*weight.data_len);
            weight.dims.push_back(output_num);
            weight.dims.push_back(weight.data_len/output_num);
            paramlist.push_back(weight);
            iter = nodelist[i].attrs.find(1);
            int biasTerm = std::atoi(iter->second.c_str());
            if(biasTerm == 1){
                NcnnParam bias;
                bias.name = nodelist[i].name+"_b";
                bias.data_len = output_num;
                bias.data = (float*)malloc(sizeof(float)*output_num);
                read(bias.data, sizeof(float)*output_num);
                bias.dims.push_back(output_num);
                paramlist.push_back(bias);
            }     
        }
        else if(nodelist[i].op == "Normalize"){
            NcnnParam scale;
            nscan = read(&magic, sizeof(float));
            scale.name = nodelist[i].name+"_s";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(3);
            scale.data_len = std::atoi(iter->second.c_str());
            scale.data = (float*)malloc(sizeof(float)*scale.data_len);
            read(scale.data, sizeof(float)*scale.data_len);
            scale.dims.push_back(scale.data_len);
            paramlist.push_back(scale);
        }
        else if(nodelist[i].op == "PReLU"){
            NcnnParam slope;
            nscan = read(&magic, sizeof(float));
            slope.name = nodelist[i].name+"_s";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);
            slope.data_len = std::atoi(iter->second.c_str());
            slope.data = (float*)malloc(sizeof(float)*slope.data_len);
            read(slope.data, sizeof(float)*slope.data_len);
            slope.dims.push_back(slope.data_len);
            paramlist.push_back(slope);
        }
        else if(nodelist[i].op == "Scale"){
            NcnnParam scale;
            nscan = read(&magic, sizeof(float));
            scale.name = nodelist[i].name+"_s";
            std::map<int, std::string>::iterator iter;
            iter = nodelist[i].attrs.find(0);
            scale.data_len = std::atoi(iter->second.c_str());
            scale.data = (float*)malloc(sizeof(float)*scale.data_len);
            read(scale.data, sizeof(float)*scale.data_len);
            scale.dims.push_back(scale.data_len);
            paramlist.push_back(scale);

            iter = nodelist[i].attrs.find(1);
            int biasTerm = std::atoi(iter->second.c_str());
            if(biasTerm == 1){
                NcnnParam bias;
                bias.name = nodelist[i].name+"_b";
                bias.data_len = scale.data_len;
                bias.data = (float*)malloc(sizeof(float)*scale.data_len);
                read(bias.data, sizeof(float)*scale.data_len);
                bias.dims.push_back(scale.data_len);
                paramlist.push_back(bias);
            }   
        }
        else if(nodelist[i].op == "MemoryData"){
            NcnnParam const_data;
            //nscan = read(&magic, sizeof(float));
            std::map<int, std::string>::iterator iter;
            int data_len = 1;
            int size = (int)nodelist[i].attrs.size();
            std::vector<int> dims(size);
            for(iter = nodelist[i].attrs.begin(); iter != nodelist[i].attrs.end(); iter++)
            {
                std::pair<int, std::string> pair = *iter;
                data_len *= atoi(pair.second.c_str());
                dims[pair.first] = atoi(pair.second.c_str());
            }
            printf("%d.\n", data_len);
            const_data.name = nodelist[i].name;
            const_data.dim_size = (int) dims.size();
            const_data.dims = dims;
            const_data.data_len = data_len;
            const_data.data = (float*)malloc(sizeof(float)*data_len);
            read(const_data.data, sizeof(float)* data_len);
            paramlist.push_back(const_data);
        }
    }
    if(nscan < 0){
        LOG_ERROR() << "Cannot read the binary file: " << fname << "\n";
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
    /*for(int i = 0; i < paramlist.size(); i++)
    {
        printf("%s %f\n", paramlist[i].name.c_str(), ((float*)paramlist[i].data)[0]);
    }*/
    //printf("read done.\n");
    return true;
}

bool NcnnSerializer::LoadTextFile(const char* fname, std::vector<NcnnNode>& nodelist){
    fp = fopen(fname, "rb");
    if(!fp)
    {
        LOG_ERROR() << "Cannot open the param file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }
    // parse each key=value pair
    int id = 0;
    int res = 0;
    int magic = 0;
    res = fscanf(fp, "%d=", &magic);
    if(magic != 7767517){
        LOG_ERROR() << "param is too old, please regenerate \n";
    }
    int layer_count = 0;
    int blob_count = 0;
    res = fscanf(fp, "%d=", &layer_count);
    res = fscanf(fp, "%d=", &blob_count);
    //printf("layer_count: %d , blob_count: %d \n", layer_count, blob_count);
    
    for(int i = 0; i < layer_count; i++){
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
        for(int j = 0; j < bottom_count; j++){
            char bottom_name[256];
            res = fscanf(fp, "%255s=", bottom_name);
            node.inputs_name.push_back(bottom_name);
        }

        for(int j = 0; j < top_count; j++){
            char top_name[256];
            res = fscanf(fp, "%255s=", top_name);
            node.output_name.push_back(top_name);
        }

        if(res < 0){
            LOG_ERROR() <<"Read Param file data failed\n";
            return false;
        }
        while (fscanf(fp, "%d=", &id) == 1)
        {
            bool array_selection = id <= -23300;
             
            if(node.op == "Input" && array_selection == true){
                node.optimized = 1;
            }
            if (array_selection)
            {
                id = -id - 23300;
            }
            if (node.optimized == 1 && array_selection == true)
            {
                int len = 0;
                int nscan = fscanf(fp, "%d", &len);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array length failed\n");
                    return false;
                }
                
                params[id].f_data_array = (float*)malloc(sizeof(float)*len);
                params[id].i_data_array = (int*)malloc(sizeof(int)*len);
                //std::vector<std::string> opt_str;
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
                    if(str == ""){
                        str = vstr;
                    } else {
                        str = str + "," + vstr;
                    }
                    bool is_float = vstr_is_float(vstr);
                    if (is_float)
                    {
                        float* ptr = params[id].f_data_array;
                        nscan = sscanf(vstr, "%f", &ptr[j]);
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
                //printf("%s %d %s.\n", node.name.c_str(), id, str.c_str());  
                node.attrs.insert(std::pair<int, std::string>(id, str));
            }
            else
            {
                if(array_selection == true){
                    int len = 0;
                    int nscan = fscanf(fp, "%d", &len);
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict read array length failed\n");
                        return false;
                    }
                    std::string str = "";
                    params[id].f_data_array = (float*)malloc(sizeof(float)*len);
                    params[id].i_data_array = (int*)malloc(sizeof(int)*len);
                    for (int j = 0; j < len; j++)
                    {
                        char vstr[16];
                        nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                        if (nscan != 1)
                        {
                            fprintf(stderr, "ParamDict read array normal element failed\n");
                            return false;
                        }
                        if(str == ""){
                            str = vstr;
                        } else {
                            str = str + "," + vstr;
                        }
                        bool is_float = vstr_is_float(vstr);
                        if (is_float)
                        {
                            float* ptr = params[id].f_data_array;
                            nscan = sscanf(vstr, "%f", &ptr[j]);
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
                    //printf("%s %d %s.\n", node.name.c_str(), id, str.c_str());
                    node.attrs.insert(std::pair<int, std::string>(id, str));
                    free(params[id].f_data_array);
                    free(params[id].i_data_array);
                                      
                } else {
                    char vstr[16];

                    int nscan = fscanf(fp, "%15s", vstr);
                    if (nscan != 1)
                    {
                        fprintf(stderr, "ParamDict read value failed\n");
                        return false;
                    }
                    bool is_float = vstr_is_float(vstr);
                    //printf("string value: %s \n", vstr);
                    //printf("%s %d %s.\n", node.name.c_str(), id, vstr);
                    node.attrs.insert(std::pair<int, std::string>(id, vstr));
                    float f_data;
                    int i_data;
                    if (is_float){
                        nscan = sscanf(vstr, "%f", &f_data);
                    }else{
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
    
#if 0
    std::string nodeName = nodelist[(int)nodelist.size()-1].name;
    std::string nodeop = nodelist[(int)nodelist.size()-1].op;
    std::map<int, std::string>::iterator iter;  
    std::cout<<nodeName<<" "<<nodeop<<std::endl;
    //for(iter = nodelist[(int)nodelist.size()].attrs.begin(); iter !=  nodelist[(int)nodelist.size()].attrs.end(); iter++)  
    //   std::cout<<iter->first<<' '<<iter->second<<std::endl;

#endif
    return true;
}


bool NcnnSerializer::LoadConstTensor(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
                                      const std::vector<NcnnParam>& paramlist)
{
    int const_tensor_number = paramlist.size();
    std::set<std::string> node_name_set;


#ifdef DEBUG
    std::cout << "Load Const Tensor:" << std::endl;
#endif

    for(int i = 0; i < const_tensor_number; i++)
    {
        const NcnnParam& ncnn_tensor = paramlist.at(i);
#if 0
        std::cout << "    name: " << ncnn_tensor.name << std::endl;
#endif

        std::vector<int> dims = ncnn_tensor.dims;
        StaticTensor* tensor = CreateStaticConstTensor(graph, ncnn_tensor.name);
        SetTensorDim(tensor, dims);
        /*printf("%s ",ncnn_tensor.name.c_str());
        for(int i = 0 ; i < dims.size(); i++)
        {
            printf("%d ", dims[i]);
        }
        printf("\n");*/
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        SetTensorSize(tensor, ncnn_tensor.data_len);
        float* mem_buf = ( float* )std::malloc(ncnn_tensor.data_len*sizeof(float));
        float* raw_data = ( float* )ncnn_tensor.data;
        /* load data */
        for(int k = 0; k < ncnn_tensor.data_len; k++)
            mem_buf[k] = raw_data[k];

        SetConstTensorBuffer(tensor, mem_buf);
        SetConstTensorFileLocation(tensor, -1, 0);
        //printf("Load constTensor %s %d.\n", tensor->name.c_str(), tensor->mem_size);
        StaticOp* op = CreateStaticOp(graph, "Const");
        StaticNode* node = CreateStaticNode(graph, GetTensorName(tensor));
        SetNodeOp(node, op);
        AddNodeOutputTensor(node, tensor);
        
    }

    return true;
}       

static bool GetParam(const std::string name, const std::vector<NcnnParam>& paramlist, NcnnParam& param)
{
    for(unsigned int i = 0; i < paramlist.size(); i++)
    {
        if(paramlist.at(i).name == name)
        {
            param = paramlist.at(i);
            return true;
        }
    }
    return false;
}

void NcnnSerializer::CreateInputNode(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
                                      const std::vector<NcnnParam>& paramlist)
{
#ifdef DEBUG
    std::cout << "Create Input Node:" << std::endl;
#endif
    for(unsigned int i = 0; i < nodelist.size(); i++)
    {
        const NcnnNode& ncnn_node = nodelist.at(i);
        if(ncnn_node.op == "Input")
        {
            //printf("Create input tensor %s \n",  ncnn_node.name.c_str());
            std::string input_name = ncnn_node.name;

            StaticTensor* tensor = CreateStaticTensor(graph, input_name);

            SetTensorDataType(tensor, DataType::GetTypeID("float32"));

            NcnnParam param;
            if(GetParam(input_name, paramlist, param))
            {
                std::vector<int> dims = param.dims;
                SetTensorDim(tensor, dims);
            }

            StaticNode* node = CreateStaticNode(graph, input_name);
            StaticOp* op = CreateStaticOp(graph, "InputOp");

            SetNodeOp(node, op);

            AddNodeOutputTensor(node, tensor);

            /*add this node into graph input node list */
            AddGraphInputNode(graph, node);
#ifdef DEBUG
            std::cout << "    create an input node for " << ncnn_node.name << std::endl;
#endif
        }
    }
}

void NcnnSerializer::LoadNode(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node,
                              const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist)
{
    int input_number = ncnn_node.inputs_name.size();
#if 0
    std::cout << "name: " << ncnn_node.name << " input_number: " << input_number << std::endl;
#endif
    int size = (int)nodelist.size();
    for(int i = 0; i < input_number; i++)
    {
        std::string input_name = ncnn_node.inputs_name[i];
#if 0
        std::cout <<"  op name:  "<<ncnn_node.name <<"    input:  " << input_name << std::endl;
#endif
        StaticTensor* tensor = FindTensor(graph, input_name);
        //printf("%s input %s size is %d.\n", ncnn_node.name.c_str(), tensor->name.c_str(), tensor->mem_size);
        if(tensor == NULL){
            printf("Can not find tensor : %s \n", input_name.c_str());
        }
        AddNodeInputTensor(node, tensor);  
    }

    size = (int)paramlist.size();
    for(int i = 0; i < size; i++){
        std::string input_name = paramlist[i].name;
        std::string name = input_name.substr(0, input_name.length() - 2);
        if(name == ncnn_node.name){
            StaticTensor* tensor = FindTensor(graph, paramlist[i].name);

            AddNodeInputTensor(node, tensor);
        }
    }

    int out_size = (int)ncnn_node.output_name.size();
    for(int i = 0; i < out_size; i++ ){
        const std::string& output_name = ncnn_node.output_name[i];

        StaticTensor* tensor = CreateStaticTensor(graph, output_name);

        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        AddNodeOutputTensor(node, tensor);
    }
}

bool NcnnSerializer::LoadGraph(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
                                const std::vector<NcnnParam>& paramlist)
{
    SetGraphIdentity(graph, "ncnn", graph->model_name, "0");

    LoadConstTensor(graph, nodelist, paramlist);
    CreateInputNode(graph, nodelist, paramlist);
    std::vector<int> node_to_remove;
    unsigned int i;
    std::vector<std::string> no_supported_op;
    for(i = 0; i < nodelist.size(); i++)
    {
        NcnnNode ncnn_node = nodelist.at(i);
        
        if(ncnn_node.op == "Noop"&&ncnn_node.output_name.size() == 0)
            node_to_remove.push_back(i);
        if(!FindOpLoadMethod(ncnn_node.op))
        {
            auto it = find(no_supported_op.begin(),no_supported_op.end(),ncnn_node.op);
            if(it != no_supported_op.end())
                no_supported_op.push_back(ncnn_node.op);
        }
    }
    if(no_supported_op.size())
    {
        
        LOG_ERROR() << "These" <<no_supported_op.size() <<"ops are not supported \n";
        LOG_ERROR() << "{";
        for(int j = 0; j < (int)no_supported_op.size(); j++)
        {
            LOG_ERROR() << no_supported_op[j] <<",";
        }
        LOG_ERROR() << "}\n";
        return false;
    }
    for(i = 0; i < nodelist.size(); i++)
    {
        NcnnNode ncnn_node = nodelist.at(i);
        if(ncnn_node.op == "Input"||ncnn_node.op == "MemoryData")
            continue;

        if(ncnn_node.op == "Noop"&&ncnn_node.output_name.size() == 0)
            continue;

        StaticNode* node = nullptr;
        //printf("%s\n", ncnn_node.name.c_str());
        if(FindNode(graph, ncnn_node.name) != nullptr)
        {
            int counter = 0;  
            const std::string node_name = ncnn_node.name;
            std::string new_name = node_name + std::to_string(counter++); 
            while(FindNode(graph, new_name) != nullptr)
            {
                new_name = node_name + std::to_string(counter++);
                //printf("%s\n", new_name.c_str());
            }
            node = CreateStaticNode(graph, new_name);
        }
        else
        {
            node = CreateStaticNode(graph, ncnn_node.name);
        }
        LoadNode(graph, node, ncnn_node, nodelist, paramlist);

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(ncnn_node.op));

        if(!op_func(graph, node, ncnn_node))
            break;
    }
    if(i < nodelist.size()){
        return false;
    }
#if DEBUG
    std::cout << "Successfully load all nodes" << std::endl;
#endif
    for(int i = 0; i < graph->tensor_list.size(); i++)
    {
        StaticTensor* stttensor = graph->tensor_list[i].get();
        
        //printf("%s %d\n", stttensor->name.c_str(), stttensor->mem_size);

    }
    return true;
}

bool NcnnSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if(file_list.size() != GetFileNum())
        return false;

    std::vector<NcnnNode> nodelist;
    if(!LoadTextFile(file_list[0].c_str(), nodelist))
    {
        LOG_ERROR() << "Parse text file " << file_list[0].c_str() << " failed\n";
        return false;
    }
    
    std::vector<NcnnParam> paramlist;
    if(!LoadBinaryFile(file_list[1].c_str(), paramlist, nodelist))
    {
        LOG_ERROR() << "Parse binary file " << file_list[1].c_str() << " failed\n";
        return false;
    }
    
    SetGraphSource(graph, file_list[1]);
    SetGraphSourceFormat(graph, "ncnn");
    SetGraphConstTensorFile(graph, file_list[1]);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_NCNN);
    bool res = LoadGraph(graph, nodelist, paramlist);
    if(res == false){
        LOG_ERROR() << "Load Graph failed\n";
        return false;        
    }
    for(std::size_t ii = 0; ii < paramlist.size(); ++ii)
    {
        std::free(paramlist[ii].data);
    }
    return true;
}


float ParseNumber(const char* s,float d)
{
	bool bNegtiveBase,bNegtiveExp;
	int nPreZero = 0;
	const char* p;
	int sum_i = 0;
	float sum_f = 0.0;
	int sum_exp = 0;
	float sum = 0.0;
	bNegtiveBase = bNegtiveExp = false;
	if(!s)
		return false;
	if('-' == *s)
	{
		bNegtiveBase = true;
		s++;
	}
	for(;'0' == *s;nPreZero++,s++);
	for(;*s != '.' && *s != 'e' && *s != 'E' && *s != '\0';s++)
	{
		if(*s < '0' || *s > '9')
		{
			return false;
		}
		sum_i = sum_i*10 + *s - '0';
	}
	if(0 == sum_i && 0 == nPreZero)
		return false;
	if('.' == *s)
	{
		for(p = s;*p != 'e' && *p != 'E' && *p != '\0';p++);
		if(s==p-1)
			return false;
		s = p;
		p--;
		for(;*p != '.';p--)
		{
			if(*p < '0' || *p > '9')
				return false;
			sum_f = sum_f*0.1 + 0.1*(*p - '0');
		}
	}
	if('e' == *s || 'E' == *s)
	{
		s++;
		if('-' == *s)
		{
			bNegtiveExp = true;
			s++;
		}
		else if('+' == *s)
		{
			bNegtiveExp = false;
			s++;
		}
		nPreZero = 0;
		for(;*s != '\0';s++)
		{
			if(*s < '0' || *s > '9')
			{
				return false;
			}
			sum_exp = sum_exp*10 + *s - '0';
			nPreZero++;
		}
		if(0 == sum_exp && 0 == nPreZero)
			return false;
	}
	sum = sum_i + sum_f;
	if(bNegtiveExp)
	{
		while(sum_exp > 0)
		{
			sum /= 10;
			sum_exp--;
		}
	}
	else
	{
		while(sum_exp > 0)
		{
			sum *= 10;
			sum_exp--;
		}
	}
	if(bNegtiveBase)
		sum = -sum;
	d = sum;
    //printf("%f \n", d);
	return d;
}
std::vector<float>& split(const std::string& str, char delim, std::vector<float>& elems, bool skip_empty = true)
{
    std::istringstream iss(str);
    for(std::string item; getline(iss, item, delim);)
        if(skip_empty && item.empty())
            continue;
        else{ 
            float d = ParseNumber(item.c_str(), d);
            //printf("%f \n", d);
            elems.push_back(d);
        }
    return elems;
}

static void ParseAttr_n(const std::string str, std::vector<float>& result)
{
    split(str, ',', result);
}
static bool LoadNcnnSoftmax(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Softmax");

    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));
    const_iterator iter;

    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
        param.axis = std::atoi(iter->second.c_str());

    //param.axis = 1;
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end())
        param.axis = param.axis + std::atoi(iter->second.c_str());

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    AddGraphOutputNode(graph, node);

    return true;
}

static bool LoadNcnnConcat(StaticGraph* graph, StaticNode* node,  const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Concat");

    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));
    
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
        param.axis = std::atoi(iter->second.c_str()) + 1;
    
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnRelu(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{

    StaticOp* op = CreateStaticOp(graph, "ReLu");

    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));

    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
        param.negative_slope = 0;
    
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnSplit(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Split");
    SplitParam  param = any_cast<SplitParam>(OpManager::GetOpDefParam("Split"));
    param.is_caffe = true;
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}


static bool LoadNcnnConvolution(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Convolution");
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));

    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end()){
        param.output_channel = std::atoi(iter->second.c_str());
        if(ncnn_node.op == "ConvolutionDepthWise" ){
            param.group =  std::atoi(iter->second.c_str());
        }
    }

    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        param.kernel_h = std::atoi(iter->second.c_str());
        param.kernel_w = std::atoi(iter->second.c_str());
    } else {
        param.kernel_h = 0;
        param.kernel_w = 0;
    }
    
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end()){
        param.dilation_h = std::atoi(iter->second.c_str());
        param.dilation_w = std::atoi(iter->second.c_str());
    }else {
        param.dilation_h = 1;
        param.dilation_w = 1;
    }

    
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end()){
        param.stride_h = std::atoi(iter->second.c_str());
        param.stride_w = std::atoi(iter->second.c_str());
    }else {
        param.stride_h = 1;
        param.stride_w = 1;
    }


    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end()){
        param.pad_w0 = std::atoi(iter->second.c_str());
        param.pad_w1 = std::atoi(iter->second.c_str());
        param.pad_h0 = std::atoi(iter->second.c_str());
        param.pad_h1 = std::atoi(iter->second.c_str());
    }

    iter = ncnn_node.attrs.find(9);
    if(iter != ncnn_node.attrs.end()){
        param.activation = ActRELU;
    }
    //printf("%s %d %d %d %d %d %d.\n", node->name.c_str(), param.output_channel, param.kernel_h, param.stride_h, param.dilation_h, param.pad_h0, param.activation);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnPooling(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Pooling");
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end()){
        int type =  std::atoi(iter->second.c_str());
        if(type == 0){
            param.alg = kPoolMax;
        } else if(type == 1){
            param.alg = kPoolAvg;
        }
    }

    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        param.kernel_h = std::atoi(iter->second.c_str());
        param.kernel_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end()){
        param.stride_h = std::atoi(iter->second.c_str());
        param.stride_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end()){
        param.pad_h0 = std::atoi(iter->second.c_str());
        param.pad_h1 = std::atoi(iter->second.c_str());
        param.pad_w0 = std::atoi(iter->second.c_str());
        param.pad_w1 = std::atoi(iter->second.c_str());            
    }
    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end()){
        param.global = std::atoi(iter->second.c_str());
    }
    param.caffe_flavor = 1;


    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnDropout(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Dropout");
    
    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnBatchNorm(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");
    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    const_iterator iter;

    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end())
        param.eps = std::atoi(iter->second.c_str());
    param.caffe_flavor = 1;
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnScale(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Scale");
    ScaleParam param = any_cast<ScaleParam>(OpManager::GetOpDefParam("Scale"));

    const_iterator iter;

    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end())
        param.bias_term = std::atoi(iter->second.c_str());

    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnClip(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    ClipParam param = any_cast<ClipParam>(OpManager::GetOpDefParam("Clip"));

    const_iterator iter;

    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end())
        param.max = std::atoi(iter->second.c_str());

    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
        param.min = std::atoi(iter->second.c_str());

    StaticOp* op = CreateStaticOp(graph, "Clip");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnFC(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));

    const_iterator iter;

    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
        param.num_output = std::atoi(iter->second.c_str());

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnDetectionOutput(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{

    DetectionOutputParam param = any_cast<DetectionOutputParam>(OpManager::GetOpDefParam("DetectionOutput"));

    const_iterator iter;
    std::vector<float> v1, v2;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end()){
        param.num_classes = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v1);
        param.nms_threshold = v1.at(0);
    }else{
        param.nms_threshold = 0.05;
    }
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end()){
        param.nms_top_k = std::atoi(iter->second.c_str());
    }else{
        param.nms_top_k = 300;
    }
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end()){
        param.keep_top_k = std::atoi(iter->second.c_str());
    }else{
        param.keep_top_k = 100;
    }
    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v2);
        param.confidence_threshold = v2.at(0);
    }else {
        param.confidence_threshold = 0.5f;
    }
    StaticOp* op = CreateStaticOp(graph, "DetectionOutput");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnPriorBox(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    //printf("PriorBox \n");

    PriorBoxParam param = any_cast<PriorBoxParam>(OpManager::GetOpDefParam("PriorBox"));
    const_iterator iter;
    std::vector<float> v1, v2, v3, v4, vr1, vr2, vr3, vr4, ih, iw, sh, sw;
    
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v1);
        for(int i = 0; i < (int)v1.size(); i++){
            param.min_size.push_back(v1.at(i));
        }
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v2);
        for(int i = 0; i < (int)v2.size(); i++){
            param.max_size.push_back(v2.at(i));
        }
    }

    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v3);
        for(int i = 0; i < (int)v3.size(); i++){
            param.aspect_ratio.push_back(v3.at(i));
        }
    }

    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, vr1);
        param.variance.push_back(vr1.at(0));
    } else {
        param.variance.push_back(0.1);
    }
    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, vr2);
        param.variance.push_back(vr2.at(0));
    }else {
        param.variance.push_back(0.1);
    }
    iter = ncnn_node.attrs.find(5);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, vr3);
        param.variance.push_back(vr3.at(0));
    }else {
        param.variance.push_back(0.2);
    }
    iter = ncnn_node.attrs.find(6);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, vr4);
        param.variance.push_back(vr4.at(0));
    }else {
        param.variance.push_back(0.2);
    }
    iter = ncnn_node.attrs.find(7);
    if(iter != ncnn_node.attrs.end()){
        param.flip = std::atoi(iter->second.c_str());
    }else{
        param.flip = 1;
    }
    iter = ncnn_node.attrs.find(8);
    if(iter != ncnn_node.attrs.end()){
        param.clip = std::atoi(iter->second.c_str());
    }else{
        param.clip = 0;
    }
    iter = ncnn_node.attrs.find(9);
    if(iter != ncnn_node.attrs.end()){
        if( std::atoi(iter->second.c_str()) != -233)
            param.img_h = std::atoi(iter->second.c_str());
        else
            param.img_h = 0;
    }
    iter = ncnn_node.attrs.find(10);
    if(iter != ncnn_node.attrs.end()){
        if( std::atoi(iter->second.c_str()) != -233)
            param.img_w = std::atoi(iter->second.c_str());
        else
            param.img_w = 0;
    }
    iter = ncnn_node.attrs.find(11);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, sh);
        if(sh.at(0) != -233)
            param.step_h = sh.at(0);
        else 
            param.step_h = 0.f;
    }
    iter = ncnn_node.attrs.find(12);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, sw);
        if(sw.at(0) != -233)
            param.step_w = sw.at(0);
        else 
            param.step_w = 0.f;
    }

    iter = ncnn_node.attrs.find(13);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v4);
        param.offset = v4.at(0);
    }

    StaticOp* op = CreateStaticOp(graph, "PriorBox");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnFlatten(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));

    param.axis = 1;

    StaticOp* op = CreateStaticOp(graph, "Flatten");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnReshape(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));
    const_iterator iter;
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end()){
        if(-233 != std::atoi(iter->second.c_str())){
            param.re_shape.push_back(std::atoi(iter->second.c_str()));
        }
    } 
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end()){
        if(-233 != std::atoi(iter->second.c_str())){
            param.re_shape.push_back(std::atoi(iter->second.c_str()));
        }
    } else {
        param.re_shape.push_back(0);    
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        if(-233 != std::atoi(iter->second.c_str())){
            param.re_shape.push_back(std::atoi(iter->second.c_str()));
        }
    }
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end()){
        if(-233 != std::atoi(iter->second.c_str())){
            param.re_shape.push_back(std::atoi(iter->second.c_str()));
        }
    } 
    
    

    //param.is_mxnet = true;

    StaticOp* op = CreateStaticOp(graph, "Reshape");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnPermute(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    PermuteParam param = any_cast<PermuteParam>(OpManager::GetOpDefParam("Permute"));
    const_iterator iter;

    iter = ncnn_node.attrs.find(0);
    int orderType = 0;
    if(iter != ncnn_node.attrs.end())
        orderType = std::atoi(iter->second.c_str());   

    switch(orderType){

        case 3:
            param.order0 = 0;
            param.order1 = 2;
            param.order2 = 3;
            param.order3 = 1;
            break;
        default:
            break;
    }

    StaticOp* op = CreateStaticOp(graph, "Permute");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnEltwise(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    const_iterator iter;

    std::vector<float> coef;

    iter = ncnn_node.attrs.find(0);
    int opType = -1;
    if(iter != ncnn_node.attrs.end())
        opType = std::atoi(iter->second.c_str());   

    if(opType == 0){
        param.type = ELT_PROD;
    }else if(opType == 1){
        param.type = ELT_SUM;
    }else if(opType == 2){
        param.type = ELT_MAX;
    }else{
        param.type = ELT_SUM;
    }
    param.caffe_flavor = 0;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadNcnnInterp(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    InterpParam param = any_cast<InterpParam>(OpManager::GetOpDefParam("Interp"));
    
    std::vector<float> v1,v2;
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end()){
        param.resize_type = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v1);
        param.width_scale = v1.at(0);
    }
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v2);
        param.height_scale = v2.at(0);
    }
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end()){
        //printf("find 3.\n");
        ParseAttr_n(iter->second, v2);
        param.output_width = v2.at(0);
    }
    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end()){
        //printf("find 4.\n");
        ParseAttr_n(iter->second, v2);
        param.output_height = v2.at(0);
    }
    StaticOp* op = CreateStaticOp(graph, "Interp");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}
static bool LoadNcnnCrop(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    CropParam param = any_cast<CropParam>(OpManager::GetOpDefParam("Crop"));
    
    param.num_args = 2;
    StaticOp* op = CreateStaticOp(graph, "Crop");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadNcnnSlice(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam("Slice"));
    param.isncnn= true;
    param.iscaffe = false;
    param.ismxnet = false;
    param.isonnx = false;
    param.slice_point_.clear();
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    std::vector<float> v1;
    if(iter != ncnn_node.attrs.end()){
        ParseAttr_n(iter->second, v1);
        for(int i = 0; i < (int)v1.size(); i++){
            param.slice_point_.push_back((int)v1.at(i));
        }
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end()){
        param.axis = std::atoi(iter->second.c_str())+1;
    }
    StaticOp* op = CreateStaticOp(graph, "Slice");

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

static bool LoadNcnnNoop(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    
    StaticOp* op = CreateStaticOp(graph, "Noop");

    SetNodeOp(node, op);

    return true;
}

static bool LoadNcnnSigmoid(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    
    StaticOp* op = CreateStaticOp(graph, "Sigmoid");

    SetNodeOp(node, op);

    return true;
}

static bool LoadNcnnYolov3DetectionOutput(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    YOLOV3DetectionOutputParam param = any_cast<YOLOV3DetectionOutputParam>(OpManager::GetOpDefParam("YOLOV3DetectionOutput"));
    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
    {
        param.num_classes = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end())
    {
        param.num_box = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end())
    {
        param.confidence_threshold = std::atof(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end())
    {
        //printf("nms_threshold is %f\n", std::atof(iter->second.c_str()));
        param.nms_threshold = std::atof(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end())
    {
        std::vector<float> v1;
        ParseAttr_n(iter->second, v1);
        for(int i = 0; i < (int)v1.size(); i++){
            param.bias.push_back(v1.at(i));
        }
    }
    iter = ncnn_node.attrs.find(5);
    if(iter != ncnn_node.attrs.end())
    {
        std::vector<float> v1;
        ParseAttr_n(iter->second, v1);
        for(int i = 0; i < (int)v1.size(); i++){
            param.mask.push_back(v1.at(i));
        }
    }
    iter = ncnn_node.attrs.find(6);
    if(iter != ncnn_node.attrs.end())
    {
        std::vector<float> v1;
        ParseAttr_n(iter->second, v1);
        for(int i = 0; i < (int)v1.size(); i++){
            param.anchors_scale.push_back(v1.at(i));
        }
    }
    StaticOp* op = CreateStaticOp(graph, "YOLOV3DetectionOutput");

    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadNcnnDeconv(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    DeconvParam param = any_cast<DeconvParam>(OpManager::GetOpDefParam("Deconvolution"));
    const_iterator iter;
    std::vector<float> v1;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
    {
        param.num_output = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(1);
    if(iter != ncnn_node.attrs.end())
    {
        param.kernel_w = std::atoi(iter->second.c_str());
        param.kernel_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(11);
    if(iter != ncnn_node.attrs.end())
    {
        param.kernel_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(2);
    if(iter != ncnn_node.attrs.end())
    {
        param.dilation_w = std::atoi(iter->second.c_str());
        param.dilation_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(12);
    if(iter != ncnn_node.attrs.end())
    {
        param.dilation_h = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(3);
    if(iter != ncnn_node.attrs.end())
    {
        param.stride_h = std::atoi(iter->second.c_str());
        param.stride_w = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(13);
    if(iter != ncnn_node.attrs.end())
    {
        param.stride_w = std::atoi(iter->second.c_str());
    }   
    iter = ncnn_node.attrs.find(4);
    if(iter != ncnn_node.attrs.end())
    {
        param.pad_w0 = std::atoi(iter->second.c_str());
        param.pad_w1 = std::atoi(iter->second.c_str());
        param.pad_h0 = std::atoi(iter->second.c_str());
        param.pad_h1 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(15);
    if(iter != ncnn_node.attrs.end())
    {
        param.pad_w1 = std::atoi(iter->second.c_str());
    }   
    iter = ncnn_node.attrs.find(16);
    if(iter != ncnn_node.attrs.end())
    {
        param.pad_h0 = std::atoi(iter->second.c_str());
    }   
    iter = ncnn_node.attrs.find(17);
    if(iter != ncnn_node.attrs.end())
    {
        param.pad_h1 = std::atoi(iter->second.c_str());
    }
    iter = ncnn_node.attrs.find(7);
    if(iter != ncnn_node.attrs.end())
    {
        param.group = std::atoi(iter->second.c_str());
    }
    StaticOp* op = CreateStaticOp(graph, "Deconvolution");

    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;            
}

static bool LoadNcnnUnary(StaticGraph* graph, StaticNode* node, const NcnnNode& ncnn_node)
{
    StaticOp* op = CreateStaticOp(graph, "Unary");

    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));

    const_iterator iter;
    iter = ncnn_node.attrs.find(0);
    if(iter != ncnn_node.attrs.end())
        param.type = std::atoi(iter->second.c_str());
    
    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

bool NcnnSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("ncnn", serializer))
        return false;

    NcnnSerializer* p_ncnn = dynamic_cast<NcnnSerializer*>(serializer.get());
    p_ncnn->RegisterOpLoadMethod("Convolution", op_load_t(LoadNcnnConvolution));
    p_ncnn->RegisterOpLoadMethod("ConvolutionDepthWise", op_load_t(LoadNcnnConvolution));
    p_ncnn->RegisterOpLoadMethod("Pooling", op_load_t(LoadNcnnPooling));
    p_ncnn->RegisterOpLoadMethod("ReLU", op_load_t(LoadNcnnRelu));
    p_ncnn->RegisterOpLoadMethod("Split", op_load_t(LoadNcnnSplit));
    p_ncnn->RegisterOpLoadMethod("Concat", op_load_t(LoadNcnnConcat));  
    p_ncnn->RegisterOpLoadMethod("Softmax", op_load_t(LoadNcnnSoftmax)); 
    p_ncnn->RegisterOpLoadMethod("Dropout", op_load_t(LoadNcnnDropout));
    p_ncnn->RegisterOpLoadMethod("BatchNorm", op_load_t(LoadNcnnBatchNorm));
    p_ncnn->RegisterOpLoadMethod("Scale", op_load_t(LoadNcnnScale));
    p_ncnn->RegisterOpLoadMethod("Clip", op_load_t(LoadNcnnClip));
    p_ncnn->RegisterOpLoadMethod("InnerProduct", op_load_t(LoadNcnnFC));
    p_ncnn->RegisterOpLoadMethod("DetectionOutput", op_load_t(LoadNcnnDetectionOutput));
    p_ncnn->RegisterOpLoadMethod("PriorBox", op_load_t(LoadNcnnPriorBox));
    p_ncnn->RegisterOpLoadMethod("Flatten", op_load_t(LoadNcnnFlatten));
    p_ncnn->RegisterOpLoadMethod("Permute", op_load_t(LoadNcnnPermute));
    p_ncnn->RegisterOpLoadMethod("Reshape", op_load_t(LoadNcnnReshape));
    p_ncnn->RegisterOpLoadMethod("Eltwise", op_load_t(LoadNcnnEltwise));    
    p_ncnn->RegisterOpLoadMethod("Interp", op_load_t(LoadNcnnInterp));
    p_ncnn->RegisterOpLoadMethod("Crop", op_load_t(LoadNcnnCrop));  
    p_ncnn->RegisterOpLoadMethod("BinaryOp", op_load_t(LoadNcnnEltwise)); 
    p_ncnn->RegisterOpLoadMethod("Slice", op_load_t(LoadNcnnSlice));
    p_ncnn->RegisterOpLoadMethod("Noop", op_load_t(LoadNcnnNoop));
    p_ncnn->RegisterOpLoadMethod("Sigmoid", op_load_t(LoadNcnnSigmoid));
    p_ncnn->RegisterOpLoadMethod("UnaryOp", op_load_t(LoadNcnnUnary));
    p_ncnn->RegisterOpLoadMethod("Yolov3DetectionOutput", op_load_t(LoadNcnnYolov3DetectionOutput));
    p_ncnn->RegisterOpLoadMethod("DeconvolutionDepthWise", op_load_t(LoadNcnnDeconv));
    return true;
}

}