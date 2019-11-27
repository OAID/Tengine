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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <map>
#include <set>

#include "share_lib_parser.hpp"
#include "cpu_device.h"
#include "tengine_c_api.h"
#include "tengine_c_compat.h"
#include "tengine_c_helper.hpp"

#include "exec_context.hpp"
#include "graph_executor.hpp"
#include "tengine_errno.hpp"
#include "static_graph_interface.hpp"
#include "serializer.hpp"
#include "compiler_fp16.h"

#ifdef ENABLE_ONLINE_REPORT
#include "tenginereportmgr.hpp"
#endif

namespace TEngine {

extern int NodeSetParamGeneric(void* node, const char* param_name, const char* type_name, const void* param_val,
                               int size);
extern int NodeGetParamGeneric(void* node, const char* param_name, const char* type_name, void* param_val, int size);
extern int NodeAddParamGeneric(void* node, const char* param_name, const char* type_name, int size);
}    // namespace TEngine

using namespace TEngine;

static std::string gsHclVersion;

void set_cpu_list(const char* cpu_list_str)
{
    char* copy_str = strdup(cpu_list_str);

    std::vector<int> cpu_list;

    char* p = strtok(copy_str, ",");

    while(p)
    {
        int cpu_id = strtoul(p, NULL, 10);
        cpu_list.push_back(cpu_id);
        p = strtok(NULL, ",");
    }

    int* int_buf = cpu_list.data();

    set_working_cpu(int_buf, cpu_list.size());

    free(copy_str);
}

int remove_model(const char* model_name)
{
    int ret = 0;

    StaticGraphManager::Get();

    if(StaticGraphManager::Find(model_name))
        StaticGraphManager::Remove(model_name);
    else
    {
        ret = -1;
        set_tengine_errno(ENOENT);
    }

    StaticGraphManager::Put();

    return ret;
}

int destroy_runtime_graph(graph_t graph)
{
    GraphExecutor* executor = reinterpret_cast<GraphExecutor*>(graph);

    delete executor;

    return 0;
}

int dump_model(const char* model_name)
{
    StaticGraphPtr graph_ptr;

    if(StaticGraphManager::SafeGet(model_name, graph_ptr))
    {
        DumpStaticGraph(graph_ptr.get());
        return 0;
    }

    set_tengine_errno(ENOENT);

    return -1;
}

int node_get_attr_generic(void* node, const char* param_name, const char* type_name, void* param_val, int param_size)
{
    return NodeGetParamGeneric(node, param_name, type_name, param_val, param_size);
}

int node_set_attr_generic(void* node, const char* param_name, const char* type_name, const void* param_val,
                          int param_size)
{
    return NodeSetParamGeneric(node, param_name, type_name, param_val, param_size);
}

int node_add_attr(void* node, const char* param_name, const char* type_name, int param_size)
{
    return NodeAddParamGeneric(node, param_name, type_name, param_size);
}

static int real_vload_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                            int mem_size, va_list argp)
{
    SerializerPtr serializer;

    if(!SerializerManager::SafeGet(model_format, serializer))
    {
        /* try to load from plugin */
        std::string plugin_fname = std::string("lib") + model_format + "-serializer.so";
        std::string plugin_init_func = std::string(model_format) + "_plugin_init";

        if(load_tengine_plugin(model_format, plugin_fname.c_str(), plugin_init_func.c_str()) < 0)
        {
            LOG_ERROR() << "Get serializer failed, unknown model format: " << model_format << "\n";
            set_tengine_errno(ENOENT);
            return -1;
        }

        SerializerManager::SafeGet(model_format, serializer);
    }

    StaticGraph* static_graph = CreateStaticGraph(model_name);

    static_graph->exec_context = exec_context;

    int saved_file_number = serializer->GetFileNum();

    if(mem_size == 0)    // file mode
    {
        std::vector<std::string> file_list;
        file_list.push_back(( const char* )addr);

        for(int i = 1; i < saved_file_number; i++)
        {
            const char* file = va_arg(argp, const char*);
            file_list.emplace_back(file);
        }

        if(!serializer->LoadModel(file_list, static_graph) || !CheckGraphIntegraity(static_graph))
        {
            delete static_graph;
            return -1;
        }
    }
    else
    {
        std::vector<const void*> addr_list;
        std::vector<int> size_list;

        addr_list.push_back(addr);
        size_list.push_back(mem_size);

        for(int i = 1; i < saved_file_number; i++)
        {
            addr = va_arg(argp, const void*);
            mem_size = va_arg(argp, int);

            addr_list.push_back(addr);
            size_list.push_back(mem_size);
        }

        if(!serializer->LoadModel(addr_list, size_list, static_graph) || !CheckGraphIntegraity(static_graph))
        {
            delete static_graph;
            return -1;
        }
    }

    va_end(argp);

    if(!StaticGraphManager::Add(std::string(model_name), StaticGraphPtr(static_graph)))
    {
        XLOG_ERROR() << "replicated model name detected: " << model_name << " should not happen\n";
        set_tengine_errno(EBADSLT);
        return -1;
    }

    return 0;
}

static int vload_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                       int mem_size, va_list argp)
{
    StaticGraphManager::Get();

    if(StaticGraphManager::Find(model_name))
    {
        // set_tengine_errno(EEXIST);
        StaticGraphManager::Put();
        return 0;
    }

    int ret = real_vload_model(exec_context, model_name, model_format, addr, mem_size, argp);

    StaticGraphManager::Put();

    return ret;
}

int vload_file_model(context_t exec_context, const char* model_name, const char* model_format, const char* fname,
                     va_list argp)
{
    return vload_model(exec_context, model_name, model_format, fname, 0, argp);
}

int vload_mem_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                    int mem_size, va_list argp)
{
    return vload_model(exec_context, model_name, model_format, addr, mem_size, argp);
}

int get_model_format(graph_t graph)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();
    return g->GetModelFormat();

}

int save_graph_internal(graph_t graph, const char* model_format, const char* fname, va_list argp)
{
    /* Get the serializer according to model_format */
    SerializerPtr serializer;
    if(!SerializerManager::SafeGet(model_format, serializer))
    {
        /* try to load from plugin */
        std::string plugin_fname = std::string("lib") + model_format + "-serializer.so";
        std::string plugin_init_func = std::string(model_format) + "_plugin_init";

        if(load_tengine_plugin(model_format, plugin_fname.c_str(), plugin_init_func.c_str()) < 0)
        {
            LOG_ERROR() << "save graph failed, unknown model format: " << model_format << "\n";
            set_tengine_errno(ENOENT);
            return -1;
        }

        SerializerManager::SafeGet(model_format, serializer);
    }

    /* Create file list */
    std::vector<std::string> file_list;
    file_list.push_back(fname);

    for(unsigned int i = 1; i < serializer->GetFileNum(); i++)
    {
        const char* file = va_arg(argp, const char*);
        file_list.emplace_back(file);
    }
    va_end(argp);

    /* Get runtime graph pointer */
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();

    /* Save the graph to the files */
    if(!serializer->SaveModel(file_list, g))
        return -1;

    return 0;
}

static float get_absmax_val(float* data, int data_size)
{
    float max_val = 0.f;
    if(data != nullptr)
    {
        for(int i = 0; i < data_size; i++)
        {
            float abs_val = fabs(data[i]);
            if(abs_val > max_val)
                max_val = abs_val;
        }
    }
    return max_val;
}

static inline bool isSkipQuant(int nodeInedx, int node_no_quant_idxs[], int number)
{
    for(int i = 0; i < number; i++)
    {
        if(nodeInedx == node_no_quant_idxs[i])
            return true;
    }
    return false;
}
#if 1
void dump_graph_tensor_scale_internal(graph_t graph)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();

    for(unsigned int i = 0; i < g->seq_nodes.size(); i++)
    {
        Node* node = g->seq_nodes[i];
        Operator *op = node->GetOp();
        if(op->GetName() == "Const")
            continue;
        if(op->GetName() == "Convolution")
        {
            if(node->GetInputNum() > 2)
            {
                Tensor* bias_tensor = node->GetInputTensor(2);
                printf("bias datye type is %d\n",bias_tensor->GetDataType());
            }
        }
        for(unsigned int j = 0; j < node->GetOutputNum(); j++)
        {
            Tensor* output_tensor = node->GetOutputTensor(j);
            std::cout<<"tensor name is :" << output_tensor->GetName()<<"\n";
            // set the quant scale;
            auto p_quant = output_tensor->GetQuantParam();
            QuantParam& param = (*p_quant)[0];
            printf("scale is %f,zero is %d\n",param.scale,param.zero_point);
        }
    }

}
#endif

int post_train_graph_internal(graph_t graph, const char*file_name)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();
    std::map<std::string,float> layer_scale;
    bool parse_from_file = false;
    if(NULL != file_name)
    {
        std::ifstream scales(file_name);
        std::string line;
        while(std::getline(scales,line))
        {
            std::string layer_name;
            float scale_val = 0.f;
            size_t last = 0;
            size_t index = line.find_first_of(" ",last);
            layer_name = line.substr(last,index-last);
            last = index + 1;
            scale_val = atof((line.substr(last,line.size()- last)).c_str());
            layer_scale[layer_name] = scale_val;
        }
        parse_from_file = true;
        //only support one input
    }
    //set the input node quant param
    if(parse_from_file)
    {
        if(layer_scale.count("input") != 0)
        {
            Node* input_node = g->input_nodes[0];
            Tensor* input_tensor = input_node->GetOutputTensor(0);
            auto in_quant = input_tensor->GetQuantParam();
            if(in_quant->size()<=0)
            {
                in_quant->resize(1);
            }
            QuantParam& param = (*in_quant)[0];
            param.scale = layer_scale["input"];
            param.zero_point = 0;
        }
    }

    for(unsigned int i = 0; i < g->seq_nodes.size(); i++)
    {
        Node* node = g->seq_nodes[i];
        Operator *op = node->GetOp();
        if(op->GetName() == "Const")
            continue;
        if(op->GetName() == "Input" && parse_from_file)
            continue;
        std::string node_name = node->GetName();
        #if 1
        if(parse_from_file && (op->GetName() == "Flatten" || op->GetName() == "Dropout" || op->GetName() == "Reshape"||
           op->GetName() == "Squeeze"))
        {
            Tensor* input_tensor = node->GetInputTensor(0);
            auto in_quant = input_tensor->GetQuantParam();
            if(in_quant->size() != 1)
            {
                printf("input tensor quant param is null\n");
                return -1;
            }
            Tensor* out_tensor = node->GetOutputTensor(0);
            auto out_quant = out_tensor->GetQuantParam();
            float scale = (*in_quant)[0].scale;
            float zero = (*in_quant)[0].zero_point;

            if(out_quant->size()<=0)
            {
                out_quant->resize(1);
            }
            (*out_quant)[0].scale = scale;
            (*out_quant)[0].zero_point = zero;

            continue;
        }
        #endif
        #if 0
        if(op->GetName() == "Concat")
        {
            float scale = 0;
            float zero = 0;
            for(unsigned int j = 0; j < node->GetInputNum(); j++)
            {
                Tensor* input_tensor = node->GetInputTensor(j);
                auto in_quant = input_tensor->GetQuantParam();
                if(in_quant->size() != 1)
                {
                    printf("input tensor quant param is null\n");
                    return -1;
                }
                float tmp_scale = (*in_quant)[j].scale;
                float tmp_zero = (*in_quant)[j].zero_point;
                if(tmp_scale > scale)
                {
                   scale = tmp_scale;
                   zero = tmp_zero;
                }
            }

            Tensor* out_tensor = node->GetOutputTensor(0);
            auto out_quant = out_tensor->GetQuantParam();
            if(out_quant->size()<=0)
            {
                out_quant->resize(1);
            }
            (*out_quant)[0].scale = scale;
            (*out_quant)[0].zero_point = zero;

            continue;
        }
        #endif

        float output_scale = 0.f;
        if(parse_from_file)
        {
            for(auto it = layer_scale.begin();it != layer_scale.end(); ++it)
            {
                if(node_name.find(it->first) != node_name.npos)
                {
                    //std::cout<<"find the node:"<< node_name<<"layer_Name:" << it->first<<"\n";
                    output_scale = it->second;
                }
            }
        }
        for(unsigned int j = 0; j < node->GetOutputNum(); j++)
        {
            Tensor* output_tensor = node->GetOutputTensor(j);
            // set the quant scale;
            auto p_quant = output_tensor->GetQuantParam();
            if(p_quant->size()<=0)
            {
                p_quant->resize(1);
            }
            QuantParam& param = (*p_quant)[0];
            if(parse_from_file)
            {
                param.scale = output_scale;
                param.zero_point = 0;
            }
            else
            {
                param.scale = param.scale/127;
                param.zero_point = 0;
            }
        }
    }
    return 0;
}

#define GET_TENGINE_DT(a) (a + 1)
int quant_graph_internal(graph_t graph, int quant_mode, int node_no_quant_idxs[], int node_no_quant_number)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();

    for(unsigned int i = 0; i < g->seq_nodes.size(); i++)
    {
        if(isSkipQuant(i, node_no_quant_idxs, node_no_quant_number))
            continue;

        Node* node = g->seq_nodes[i];
        Operator* op = node->GetOp();
        if(op->GetName() == "Const")
            continue;

        /* set node output */
        for(unsigned int j = 0; j < node->GetOutputNum(); ++j)
        {
            Tensor* output = node->GetOutputTensor(j);
            output->SetDataType(GET_TENGINE_DT(quant_mode));
        }

        if(op->GetName() == "Convolution" || op->GetName() == "FullyConnected")
        {
            // quant weight
            // Tensor* input_tensor = node->GetInputTensor(0);
            Tensor* weight_tensor = node->GetInputTensor(1);
            if(weight_tensor->GetDataType() == TENGINE_DT_FP32)
            {
                int kernel_size = (weight_tensor->GetTotalSize()) / sizeof(float);
                float* kernel_org = ( float* )weight_tensor->GetMemAddr();

                // fp16 quant
                if(quant_mode == TENGINE_QUANT_FP16)
                {
                    __fp16* kernel_new = ( __fp16* )malloc(kernel_size * sizeof(__fp16));
                    for(int i = 0; i < kernel_size; i++)
                        kernel_new[i] = fp32_to_fp16(kernel_org[i]);

                    // set the memory
                    weight_tensor->FreeTensor();
                    weight_tensor->SetMemAddr(kernel_new);

                    // set the data type
                    weight_tensor->SetDataType(TENGINE_DT_FP16);
                }
                // int8 quant
                else if(quant_mode == TENGINE_QUANT_INT8)
                {
                    bool bperchannel = false;
                    int perchannel_size = 0;
                    int channel_num = weight_tensor->GetShape().GetN();
                    ;
                    if(op->GetName() == "Convolution")
                    {
                        int group = 0;
                        int k_h = 0;
                        int k_w = 0;

                        node_get_attr_generic(node, "group", typeid(int).name(), &group, sizeof(int));
                        // node_get_attr_generic(node,"input_channel",typeid(int).name(), &in_c,sizeof(int));
                        node_get_attr_generic(node, "kernel_h", typeid(int).name(), &k_h, sizeof(int));
                        node_get_attr_generic(node, "kernel_w", typeid(int).name(), &k_w, sizeof(int));

                        if(group == channel_num)
                        {
                            bperchannel = true;
                            perchannel_size = k_h * k_w;
                        }
                    }
                    int8_t* kernel_new = ( int8_t* )malloc(kernel_size);
                    if(bperchannel)
                    {
                        auto p_quant = weight_tensor->GetQuantParam();
                        p_quant->resize(channel_num);
                        for(int i = 0; i < channel_num; ++i)
                        {
                            float* kernel_tmp = kernel_org + i * perchannel_size;
                            int8_t* kernel_int8_tmp = kernel_new + i * perchannel_size;
                            float weight_max = get_absmax_val(kernel_tmp, perchannel_size);
                            float weight_scale = weight_max / 127;
                            int zero_point = 0;
                            QuantParam& param = (*p_quant)[i];
                            param.scale = weight_scale;
                            param.zero_point = zero_point;
                            for(int j = 0; j < perchannel_size; j++)
                            {
                                kernel_int8_tmp[j] = (int8_t)(round(kernel_tmp[j] / weight_scale) + zero_point);
                            }
                        }
                        // set the memory
                        weight_tensor->FreeTensor();
                        weight_tensor->SetMemAddr(kernel_new);
                        weight_tensor->SetDataType(TENGINE_DT_INT8);
                        bperchannel = false;
                    }
                    else
                    {
                        float weight_max = get_absmax_val(kernel_org, kernel_size);
                        float weight_scale = weight_max / 127;
                        int zero_point = 0;
                        for(int i = 0; i < kernel_size; i++)
                            kernel_new[i] = (int8_t)(round(kernel_org[i] / weight_scale) + zero_point);

                        // set the memory
                        weight_tensor->FreeTensor();
                        weight_tensor->SetMemAddr(kernel_new);

                        // set the data type
                        weight_tensor->SetDataType(TENGINE_DT_INT8);

                        // set the quant param
                        auto p_quant = weight_tensor->GetQuantParam();
                        p_quant->resize(1);
                        QuantParam& param = (*p_quant)[0];
                        param.scale = weight_scale;
                        param.zero_point = zero_point;
                    }
                }
            }

            // quant bias
            if(node->GetInputNum() > 2)
            {
                Tensor* bias_tensor = node->GetInputTensor(2);
                if(bias_tensor->GetDataType() == TENGINE_DT_FP32)
                {
                    int bias_size = (bias_tensor->GetTotalSize()) / sizeof(float);
                    float* bias_org = ( float* )bias_tensor->GetMemAddr();

                    if(quant_mode == TENGINE_QUANT_FP16)
                    {
                        __fp16* bias_new = ( __fp16* )malloc(bias_size * sizeof(__fp16));
                        for(int i = 0; i < bias_size; i++)
                            bias_new[i] = fp32_to_fp16(bias_org[i]);

                        // set the memory
                        bias_tensor->FreeTensor();
                        bias_tensor->SetMemAddr(bias_new);

                        // set the data type
                        bias_tensor->SetDataType(TENGINE_DT_FP16);
                    }
                    if(quant_mode == TENGINE_QUANT_INT8)
                    {
                        int32_t* bias_new = ( int32_t* )malloc(bias_size * sizeof(int32_t));
                        auto p_quant = weight_tensor->GetQuantParam();
                        Tensor* input_tensor = node->GetInputTensor(0);
                        auto in_quant = input_tensor->GetQuantParam();
                        QuantParam& in_param = (*in_quant)[0];
                        float input_scale = in_param.scale;
                        if(p_quant->size() > 1)
                        {
                            for(int i = 0; i < bias_size; i++)
                            {
                                QuantParam& param = (*p_quant)[i];
                                float weight_scale = param.scale;
                                bias_new[i] = (int32_t)(bias_org[i] / (input_scale * weight_scale));
                            }
                        }
                        else
                        {
                            QuantParam& param = (*p_quant)[0];
                            float weight_scale = param.scale;
                            for(int i = 0; i < bias_size; i++)
                            {
                                bias_new[i] = (int32_t)(bias_org[i] / (input_scale * weight_scale));
                            }
                        }
                        // set the memory
                        bias_tensor->FreeTensor();
                        bias_tensor->SetMemAddr(bias_new);
                        // set the data type
                        bias_tensor->SetDataType(TENGINE_DT_INT32);
                    }
                }
            }
        }
    }

    return 0;
}

graph_t create_graph_in_context(context_t exec_context, const char* graph_name, const char* model_name)
{
    GraphExecutor* executor = new GraphExecutor();

    if(!executor->CreateGraph(exec_context, graph_name, model_name))
    {
        delete executor;
        return nullptr;
    }

    return executor;
}

const char* get_model_name(graph_t graph)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);

    if(executor->GetModelName().empty())
        return nullptr;
    else
        return executor->GetModelName().c_str();
}

const char* get_tengine_hcl_version()
{
    return gsHclVersion.c_str();
}

namespace TEngine
{
    static int g_CurrentAuthedStatus = 1;
    std::function<void()> DUMP_HCL_VERSION_INFO_handle;

    void tengine_authed_status(int status)
    {
        printf("set status : %d\n",status);
        g_CurrentAuthedStatus = status;
    }
}

extern "C" int tengine_authed_test();
int is_tengine_auth()
{
    static bool bInit = false;
    if( !bInit )
    {
        tengine_authed_test();
        bInit = true;
    }

    return TEngine::g_CurrentAuthedStatus;
}

void about_tengine()
{
    printf("-------------------About Teninge----------------------\n");
    printf("Tengine Version : %s\n",get_tengine_version());
#ifdef CONFIG_KERNEL_FP32
printf("Support fp32 calc\n");
#endif

#ifdef CONFIG_KERNEL_FP16
printf("Support fp16 calc\n");
#endif

#ifdef CONFIG_KERNEL_INT8
printf("Support int8 calc\n");
#endif

#ifdef CONFIG_KERNEL_UINT8
printf("Support uint8 calc\n");
#endif

#ifdef ENABLE_ONLINE_REPORT
printf("Support online report\n");
#endif

    if( TEngine::DUMP_HCL_VERSION_INFO_handle )
    {
        TEngine::DUMP_HCL_VERSION_INFO_handle();
    }

    printf("-------------------------------------------------------");
}

extern void operator_plugin_init(void);
extern void serializer_plugin_init(void);
extern void executor_plugin_init(void);
extern void driver_plugin_init(void);

namespace TEngine {


typedef void (*REGISTER_AUTHED_HANDLE_CBK_T) ( int status );

int hclcpu_plugin_init(bool ignore_failure)
{
    if( ignore_failure )
    {
	    return 0;
    }
    static ShareLibParser so_handle;

    try
    {
        if(so_handle.Load("libhclcpu.so") < 0)
        {
            LOG_ERROR() << "cannot load libhclcpu.so\n";
            set_tengine_errno(ENOENT);
            return -1;
        }

        if(so_handle.ExecuteFunc<int()>("register_hclcpu_ops") < 0)
        {
            LOG_ERROR() << "register hcl cpu ops failed\n";
            set_tengine_errno(EFAULT);
            return -1;
        }
    }

    catch(const std::exception& e)
    {
        if(!ignore_failure)
        {
            LOG_ERROR() << e.what() << "\n";
            set_tengine_errno(EFAULT);
            return -1;
        }
    }

    try
    {
        so_handle.ExecuteFunc<void(REGISTER_AUTHED_HANDLE_CBK_T)>("set_tengine_authed_status_func",tengine_authed_status);
        DUMP_HCL_VERSION_INFO_handle = so_handle.GetFunction<bool()>("dump_hcl_version_info");
    }
    catch (const std::exception&)
    {

    }

    return 0;
}

int InitAllPlugin(void)
{
    operator_plugin_init();
    serializer_plugin_init();
    executor_plugin_init();
    driver_plugin_init();

    /* as for convert tool, it is all right to have no hclcpu */
    bool ignore_failure = false;

    const char* hcl_str = std::getenv("IGNORE_HCLCPU");

    if(hcl_str && hcl_str[0] == '1')
    {
        ignore_failure = true;
    }

    if(hclcpu_plugin_init(ignore_failure) < 0)
    {
        LOG_ERROR() << "no graph can be executed on CPU\n";
        return -1;
    }

    return 0;
}

struct tensor_entry
{
    Tensor* tensor;
    Node* node;
    Graph* graph;
    GraphExecutor* executor;
    bool consumed;
};

void replace_input_tensor(Tensor* new_tensor, Tensor* old_tensor)
{
    for(unsigned int i = 0; i < old_tensor->consumer.size(); i++)
    {
        NodePort* port = old_tensor->consumer[i];

        port->tensor = new_tensor;

        new_tensor->consumer.push_back(port);
    }

    old_tensor->consumer.clear();
}

static int check_graph_inputs(Graph* graph, std::map<std::string, tensor_entry>& out_map)
{
    int input_node_num = graph->input_nodes.size();
    std::vector<Node*> remove_node;
    std::vector<Tensor*> remove_tensor;

    for(int j = 0; j < input_node_num; j++)
    {
        Node* in_node = graph->input_nodes[j];

        if(in_node->GetInputNum() == 0)
        {
            Tensor* in_tensor = in_node->GetOutputTensor(0);

            auto ir = out_map.find(in_tensor->GetName());

            if(ir != out_map.end())
            {
                replace_input_tensor(ir->second.tensor, in_tensor);

                assert(in_tensor->consumer.size() == 0);

                remove_node.push_back(in_node);
                ir->second.consumed = true;
            }
        }
        else
        {
            for(unsigned int l = 0; l < in_node->GetInputNum(); l++)
            {
                Tensor* in_tensor = in_node->GetInputTensor(l);

                if(in_tensor->producer == nullptr)
                {
                    auto ir = out_map.find(in_tensor->GetName());

                    if(ir != out_map.end())
                    {
                        replace_input_tensor(ir->second.tensor, in_tensor);

                        assert(in_tensor->consumer.size() == 0);

                        remove_tensor.push_back(in_tensor);
                        ir->second.consumed = true;
                    }
                }
            }
        }
    }

    for(unsigned int i = 0; i < remove_node.size(); i++)
    {
        graph->RemoveNode(remove_node[i]);
    }

    for(unsigned int i = 0; i < remove_tensor.size(); i++)
    {
        graph->RemoveTensor(remove_tensor[i]);
    }

    return (remove_node.size() + remove_tensor.size());
}

GraphExecutor* do_merge_graph(std::vector<GraphExecutor*>& exec_list)
{
    std::map<std::string, tensor_entry> out_map;

    /* collect all output tensors */

    for(unsigned int i = 0; i < exec_list.size(); i++)
    {
        GraphExecutor* executor = exec_list[i];
        Graph* graph = executor->GetGraph();

        for(unsigned int n = 0; n < graph->output_nodes.size(); n++)
        {
            Node* node = graph->output_nodes[n];

            for(unsigned int m = 0; m < node->GetOutputNum(); m++)
            {
                Tensor* tensor = node->GetOutputTensor(m);

                if(tensor->consumer.size() == 0)
                {
                    tensor_entry entry;
                    entry.tensor = tensor;
                    entry.node = node;
                    entry.graph = graph;
                    entry.executor = executor;
                    entry.consumed = false;

                    if(out_map.count(tensor->GetName()))
                    {
                        LOG_ERROR() << "duplicated output tensor found: " << tensor->GetName() << "\n";
                        set_tengine_errno(EINVAL);
                        return nullptr;
                    }

                    out_map[tensor->GetName()] = entry;
                }
            }
        }
    }

    std::set<Graph*> keep_input_set;

    for(unsigned int i = 0; i < exec_list.size(); i++)
    {
        Graph* graph = exec_list[i]->GetGraph();

        int ret = check_graph_inputs(graph, out_map);

        if(ret == 0)
        {
            keep_input_set.insert(graph);
        }
    }

    GraphExecutor* first_executor = exec_list[0];

    std::string m_graph_name = first_executor->GetGraph()->GetName() + ".merged";
    ExecContext* m_context = ( ExecContext* )first_executor->GetExecAttr()->exec_context;

    GraphExecutor* m_executor = ( GraphExecutor* )create_graph_in_context(m_context, m_graph_name.c_str(), nullptr);
    Graph* m_graph = m_executor->GetGraph();

    for(unsigned int i = 0; i < exec_list.size(); i++)
    {
        Graph* graph = exec_list[i]->GetGraph();

        for(unsigned int j = 0; j < graph->seq_nodes.size(); j++)
        {
            Node* node = graph->seq_nodes[j];

            m_graph->AddNode(node, false);

            for(unsigned int l = 0; l < node->GetOutputNum(); l++)
            {
                Tensor* tensor = node->GetOutputTensor(l);

                m_graph->AddTensor(tensor, false);
            }
        }

        for(unsigned int j = 0; j < graph->input_nodes.size(); j++)
        {
            Node* node = graph->input_nodes[j];

            bool still_input_node = true;

            for(unsigned int l = 0; l < node->GetInputNum(); l++)
            {
                still_input_node = false;
                Tensor* tensor = node->GetInputTensor(l);

                if(tensor->producer == nullptr)
                {
                    still_input_node = true;
                    break;
                }
            }

            if(still_input_node)
                m_graph->input_nodes.push_back(node);
        }

        for(unsigned int j = 0; j < graph->output_nodes.size(); j++)
        {
            Node* node = graph->output_nodes[j];

            int output_tensor_number = 0;
            int output_tensor_in_map = 0;

            for(unsigned int l = 0; l < node->GetOutputNum(); l++)
            {
                Tensor* tensor = node->GetOutputTensor(l);

                if(tensor->consumer.size() == 0)
                {
                    output_tensor_number++;
                }

                if(out_map.count(tensor->GetName()))
                {
                    output_tensor_in_map++;
                }
            }

            if(output_tensor_number)
                m_graph->output_nodes.push_back(node);

            /* no output tensor has been consumed */
            if(output_tensor_number == output_tensor_in_map)
            {
                if(keep_input_set.count(graph))
                {
                    LOG_ERROR() << "graph: " << graph->GetName() << " is disconnected with others\n";
                    set_tengine_errno(EFAULT);
                    delete m_executor;
                    return nullptr;
                }
            }
        }
    }

    return m_executor;
}

}    // namespace TEngine
