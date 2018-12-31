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
 * Author: jingyou@openailab.com
 */
#include <random>
#include <fstream>
#include <sstream>
#include <cstring>

#include "tengine_c_api.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"

using namespace std;
using namespace tensorflow;

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------

TF_Status* TF_NewStatus() { return new TF_Status; }

void TF_DeleteStatus(TF_Status* s) { delete s; }

TF_Code TF_GetCode(const TF_Status* s)
{
    return static_cast<TF_Code>(s->status.code());
}

const char* TF_Message(const TF_Status* s)
{
    return s->status.error_message().c_str();
}

// -----------------------------------------------------------------------------

TF_Tensor* TF_NewTensor(TF_DataType dtype, const int64_t* dims, int num_dims,
                        void* data, size_t len,
                        void (*deallocator)(void* data, size_t len, void* arg),
                        void* deallocator_arg)
{
    TF_Tensor* tensor = new TF_Tensor;

    tensor->dtype = dtype;
    tensor->shape.resize(num_dims);

    for(int i = 0; i < num_dims; ++i)
    {
        tensor->shape[i] = dims[i];
    }

    tensor->data = data;
    tensor->count = len;

    return tensor;
}

void TF_DeleteTensor(TF_Tensor* t) { delete t; }

int64_t TF_Dim(const TF_Tensor* t, int dim_index)
{
    return static_cast<int64_t>(t->shape[dim_index]);
}

void* TF_TensorData(const TF_Tensor* t) { return t->data; }

// -----------------------------------------------------------------------------

TF_Operation* TF_GraphOperationByName(TF_Graph* graph, const char* oper_name)
{
    TF_Operation* oper = new TF_Operation;

    oper->node_name = oper_name;

    return oper;
}

// -----------------------------------------------------------------------------

TF_Graph* TF_NewGraph()
{
    TF_Graph* graph = new TF_Graph;
    graph->prerun_already = false;
    graph->graph_exe = nullptr;
    return graph;
}

void TF_DeleteGraph(TF_Graph* g)
{
    if(!g)
    {
        postrun_graph(g->graph_exe);
        destroy_graph(g->graph_exe);
    }

    delete g;
}

TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions()
{
    return new TF_ImportGraphDefOptions;
}

void TF_ImportGraphDefOptionsSetPrefix(TF_ImportGraphDefOptions* opts,
                                       const char* prefix)
{
    opts->prefix = prefix;
}

void TF_GraphImportGraphDef(TF_Graph* graph, const TF_Buffer* graph_def,
                            const TF_ImportGraphDefOptions* options,
                            TF_Status* status)
{
    init_tengine();

    // store the content of TF_Buffer in a temporary file
    string temp_modfile =  "/tmp/tf_modfile_";
    temp_modfile += options->prefix;

    ofstream fs(temp_modfile.c_str(), ios::binary | ios::out);
    if(!fs.good())
    {
        status->status = Status(TF_INTERNAL, "Can not open the temporary model file");
        return;
    }
    for(size_t i=0; i < graph_def->length; i++)
        fs.put(*((char *)graph_def->data + i));
    fs.close();

    // Create graph
    graph->graph_exe = create_graph(nullptr, "tensorflow", temp_modfile.c_str());
    if(graph->graph_exe == nullptr)
    {
        status->status = Status(TF_INVALID_ARGUMENT, "Create graph failed");
        return;
    }

}

TF_Graph::~TF_Graph()
{
    TF_DeleteGraph(this);
}

// -----------------------------------------------------------------------------

TF_SessionOptions* TF_NewSessionOptions() { return new TF_SessionOptions; }

// -----------------------------------------------------------------------------

TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opt,
                          TF_Status* status)
{
    TF_Session* sess = new TF_Session;
    sess->graph = graph;
    return sess;
}

void TF_CloseSession(TF_Session* s, TF_Status* status)
{
}

void TF_DeleteSession(TF_Session* s, TF_Status* status)
{
    delete s;
}

void TF_SessionRun(TF_Session* session, const TF_Buffer* run_options,
                   const TF_Output* inputs, TF_Tensor* const* input_values,
                   int ninputs, const TF_Output* outputs,
                   TF_Tensor** output_values, int noutputs,
                   const TF_Operation* const* target_opers, int ntargets,
                   TF_Buffer* run_metadata, TF_Status* status)
{
    graph_t graph = session->graph->graph_exe;

    // set input node name
    const char * input_node_name[ninputs];
    for(int i = 0; i < ninputs; ++i)
    {
        input_node_name[i] = inputs[i].oper->node_name;
    }

    if(set_graph_input_node(graph, input_node_name, ninputs) < 0)
    {
        status->status = Status(TF_INVALID_ARGUMENT, "Set input node failed");
        return;
    }

    // set output node name
    const char * output_node_name[noutputs];
    for(int i = 0; i < noutputs; ++i)
    {
        output_node_name[i] = outputs[i].oper->node_name;
    }

    if(set_graph_output_node(graph, output_node_name, noutputs) < 0)
    {
        status->status = Status(TF_INVALID_ARGUMENT, "Set output node failed");
        return;
    }

    // set input tensor
    for(int i = 0; i < ninputs; ++i)
    {
        // set input tensor shape
        tensor_t input_tensor = get_graph_input_tensor(graph,i,0);

        vector<int> shape = input_values[i]->shape;
        int dims[4];
        dims[0] = shape[0];
        dims[1] = shape[3];
        dims[2] = shape[1];
        dims[3] = shape[2];
        set_tensor_shape(input_tensor, dims, shape.size());

        // set input tensor buffer
        if(set_tensor_buffer(input_tensor, input_values[i]->data, input_values[i]->count) < 0)
        {
            status->status = Status(TF_INVALID_ARGUMENT, "Set input tensor buffer failed");
            return;
        }

        release_graph_tensor(input_tensor);
    }

    // prerun the graph
    if(!session->graph->prerun_already)
    {
        if(prerun_graph(graph) < 0)
        {
            status->status = Status(TF_INTERNAL, "Graph prerun failed");
            return;
        }
        else
            session->graph->prerun_already = true;
    }

    // set output tensor
    for(int i = 0; i < noutputs; ++i)
    {
        tensor_t output_tensor = get_graph_output_tensor(graph,i,0);

        int dims[4] = {0, 0, 0, 0};
        int num_dims = get_tensor_shape(output_tensor, dims, 4);
        int count = get_tensor_buffer_size(output_tensor);
        void *output_data = get_tensor_buffer(output_tensor);

        if(!output_values[i])
        {
            int64_t tf_dims[4];
            tf_dims[0] = dims[0];
            tf_dims[1] = dims[2];
            tf_dims[2] = dims[3];
            tf_dims[3] = dims[1];
            output_values[i] = TF_NewTensor(TF_FLOAT, tf_dims, num_dims, output_data, count, nullptr, nullptr);
        }
        else
            output_values[i]->data = output_data;

        release_graph_tensor(output_tensor);
    }

    // run the graph
    run_graph(graph, 1);
}

#ifdef __cplusplus
} /* end extern "C" */
#endif
