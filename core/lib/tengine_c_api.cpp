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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <iostream>
#include <stdarg.h>
#include <string>

#include "tengine_config.hpp"
#include "tengine_plugin.hpp"

#include "static_graph.hpp"
#include "resource_container.hpp"
#include "graph_executor.hpp"

#include "serializer.hpp"

#include "tengine_c_api.h"
#include "share_lib_parser.hpp"

using namespace TEngine;

struct tensor_handle
{
   GraphExecutor * executor;
   Tensor * tensor;
};



namespace TEngine {
        extern void tengine_init_executor(void);
}

#define TO_BE_IMPLEMENTED XLOG_WARN()<<"TODO: "<<__FUNCTION__<<" to be implemented\n"

static int vload_model(const char * model_name, const char * file_format, const char * fname, va_list ap);
static workspace_t get_default_workspace(void);
static user_context_t get_default_user_context();


/*** Level 0 API implementation */

graph_t create_graph(const char * graph_name, const char * format, const char * fname, ...)
{
	va_list argp;
	graph_t graph;
	int ret;

	if(init_tengine_library()<0)
		return nullptr;

	va_start(argp,fname);

        /*the model name is the same as graph name */

	ret=vload_model(graph_name,format,fname,argp);

	if(ret<0)
		return nullptr;

	/* use the default workspace to execute the graph */

	graph=create_runtime_graph(graph_name,graph_name,NULL);

	return graph; 
}


int check_graph_valid(graph_t graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	if(executor==nullptr)
		return 0;

	return 1;
}

const char * get_graph_name(graph_t graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetGraphName().c_str();
}

const char * get_model_name(graph_t graph)
{
        GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

        return executor->GetModelName().c_str();
}

int set_exec_device(graph_t graph, const char * device_name)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int run_inference(graph_t graph, void * input_data, int input_size)
{
	const char * tensor_name;
	const char * node_name;
	tensor_t tensor;
        int ret;

	/* should only one input node */
	if(get_input_node_number(graph)!=1)
		return -1;

        if(prerun_graph(graph)<0)
        {
            return -1;
        }

	node_name=get_input_node_name(graph,0);

	tensor_name=get_node_output_tensor(graph,node_name,0);

	tensor=get_graph_tensor(graph,tensor_name); 

	if(set_tensor_data(tensor,input_data,input_size)<0)
        {
            put_graph_tensor(tensor);
            return -1;
        }

        ret=run_graph(graph,1);


        put_graph_tensor(tensor);

        return ret;
}


int get_graph_output(graph_t graph, void * output_data, int output_size)
{
	const char * tensor_name;
	const char * node_name;
	tensor_t tensor;

	if(get_output_node_number(graph)!=1)
		return -1;

	node_name=get_output_node_name(graph,0);

	tensor_name=get_node_output_tensor(graph,node_name,0);

	tensor=get_graph_tensor(graph,tensor_name);

	int ret=get_tensor_data(tensor,output_data,output_size);

        put_graph_tensor(tensor);

        return ret;
}

void destroy_graph(graph_t graph)
{

        postrun_graph(graph);

	const char * model_name=get_model_name(graph);

	remove_model(model_name);

	destroy_runtime_graph(graph);
}

int get_output_size(graph_t graph)
{
	const char * tensor_name;
	const char * node_name;
	tensor_t tensor;

	if(get_output_node_number(graph)!=1)
		return -1;

	node_name=get_output_node_name(graph,0);

	tensor_name=get_node_output_tensor(graph,node_name,0);

	tensor=get_graph_tensor(graph,tensor_name);

	int ret=get_tensor_buffer_size(tensor);

        put_graph_tensor(tensor);

         return ret;
}

int set_input_shape(graph_t graph, int dims[], int dim_number)
{
	const char * tensor_name;
	const char * node_name;
	tensor_t tensor;

	/* should only one input node */
	if(get_input_node_number(graph)!=1)
		return -1;

	node_name=get_input_node_name(graph,0);

	tensor_name=get_node_output_tensor(graph,node_name,0);

	tensor=get_graph_tensor(graph,tensor_name); 

	int ret=set_tensor_shape(tensor,dims,dim_number);

        put_graph_tensor(tensor);

        return ret;
}

static user_context_t get_default_user_context(void)
{
     static user_context_t def_user_context=nullptr;

     if(def_user_context==nullptr)
     {
          def_user_context=create_user_context("default");
     }

     return def_user_context;
}

static workspace_t create_default_workspace(void)
{
    user_context_t user_context=get_default_user_context();

    return create_workspace("default",user_context);
}


static workspace_t get_default_workspace(void)
{
     static workspace_t default_workspace=nullptr;

     if(default_workspace==nullptr)
     {
          default_workspace=create_default_workspace();
     }

     return default_workspace;
}


/*** Level 1 API implementation */

int init_tengine_library(void)
{
    static int initialized=0;
    static std::mutex init_mutex;

    if(initialized)
        return 0;

    TEngineLock(init_mutex);
    
    if(initialized)
    {
        TEngineUnlock(init_mutex);
        return 0;
    }
      
    initialized=1;

    // Load the config file
    if(!TEngineConfig::Load("./etc/config"))
        return -1;
    TEnginePlugin::SetPluginManager();

    // TEngineConfig::DumpConfig();  // for debug
    // TEnginePlugin::DumpPlugin();  // for debug
    
    //create the default user context
    get_default_user_context();

    TEnginePlugin::LoadAll();

    //create the default context and workspace
    get_default_user_context();
    get_default_workspace();

    TEngineUnlock(init_mutex);
    return 0;
}

int request_tengine_version(const char * version)
{
	//TODO: the real version compatibility check
//	TO_BE_IMPLEMENTED;
	return 1;
}


static int vload_model(const char * model_name, const char * file_format, const char * fname, va_list argp)
{
	SerializerPtr serializer;

        if(!SerializerManager::SafeGet(file_format,serializer))
		return -1;

	int saved_file_number=serializer->GetFileNum();

	std::vector<std::string> file_list;

	file_list.push_back(fname);


	for(int i=1;i<saved_file_number;i++)
	{
		const char * file=va_arg(argp,const char *);
		file_list.emplace_back(file);
	}

	va_end(argp);

	StaticGraph * static_graph=CreateStaticGraph(model_name);

	if(!serializer->LoadModel(file_list,static_graph) ||
           !CheckGraphIntegraity(static_graph))
         {
             delete static_graph;
             return -1;
         }
         
	if(StaticGraphManager::SafeAdd(std::string(model_name),StaticGraphPtr(static_graph)))
	    return 0;   

        return -1;
}


int load_model(const char * model_name, const char * file_format, const char * fname, ...)
{
	va_list argp;
	va_start(argp,fname);

	return vload_model(model_name,file_format,fname,argp);
}

int save_model(const char * model_name, const char * file_format, const char * fname_prefix)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int remove_model(const char * model_name)
{
	
      if(!StaticGraphManager::Find(model_name))
            return -1;

        StaticGraphManager::Remove(model_name);

	return 0;
}

int dump_model(const char * model_name)
{
        StaticGraphPtr  graph_ptr;

        if(StaticGraphManager::SafeGet(model_name,graph_ptr))
        {
	    DumpStaticGraph(graph_ptr.get());
	    return 0;
        }

        return -1;
}



graph_t  create_runtime_graph(const char * graph_name, const char * model_name,workspace_t ws)
{
       if(ws==nullptr)
              ws=get_default_workspace();

	RuntimeWorkspace*  r_ws=static_cast<RuntimeWorkspace *>(ws);

	if(r_ws==nullptr)
	{
            return nullptr;
	}

	return r_ws->CreateGraphExecutor(graph_name,model_name);
}

int  destroy_runtime_graph(graph_t graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);
	RuntimeWorkspace * r_ws=executor->GetWorkspace();

	if(r_ws->DestroyGraphExecutor(executor))
		return 0;

	return -1;
}



int  set_graph_input_node(graph_t graph, const char * input_nodes[], int input_number)
{
	std::vector<std::string> inputs;

	for(int i=0;i<input_number;i++)
		inputs.push_back(input_nodes[i]);

	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	if(executor->SetGraphInputNode(inputs))
		return 0;

	return -1;

}

int  set_graph_output_node(graph_t graph, const char * output_nodes[], int output_number)
{
	std::vector<std::string> outputs;

	for(int i=0;i<output_number;i++)
		outputs.push_back(output_nodes[i]);

	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	if(executor->SetGraphOutputNode(outputs))
		return 0;

	return -1;
}


int get_input_node_number(graph_t graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetGraphInputNodeNum();
}

const char * get_input_node_name(graph_t graph, int idx)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetGraphInputNodeName(idx).c_str();
}

int get_node_input_number(graph_t graph, const char * node_name)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);
        return executor->GetNodeInputNum(node_name);
}

const char * get_node_input_tensor(graph_t graph, const char * node_name, int input_idx)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetNodeInputTensor(node_name,input_idx).c_str();
}


int get_output_node_number(graph_t graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetGraphOutputNodeNum();
}

const char * get_output_node_name(graph_t graph, int idx)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetGraphOutputNodeName(idx).c_str();
}

int get_node_output_number(graph_t graph, const char * node_name)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);
        return executor->GetNodeOutputNum(node_name);
}

const char * get_node_output_tensor(graph_t graph, const char * node_name, int output_idx)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->GetNodeOutputTensor(node_name,output_idx).c_str();
}


tensor_t  get_graph_tensor(graph_t graph, const char * tensor_name)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	Tensor * tensor=executor->FindTensor(tensor_name);

        if(tensor==nullptr)
            return nullptr;

        tensor_handle * h=new tensor_handle();

        h->executor=executor;
        h->tensor=tensor;

        return h;
}

int  check_tensor_valid(tensor_t tensor)
{
	tensor_handle * t=static_cast<tensor_handle *>(tensor);

	if(t==nullptr)
		return 0;

	return 1;
}

void  put_graph_tensor(tensor_t tensor)
{
        tensor_handle * h=static_cast<tensor_handle *>(tensor);
        delete h;
}


int  set_tensor_shape(tensor_t tensor, int dims[], int dim_number)
{

	std::vector<int> dim;

	for(int i=0;i<dim_number;i++)
		dim.push_back(dims[i]);

        tensor_handle * h=static_cast<tensor_handle*>(tensor);

        Tensor * real_tensor=h->tensor;

        TShape& shape=real_tensor->GetShape();

        shape.SetDim(dim);

	return 0;
}

int  get_tensor_shape(tensor_t tensor, int dims[], int dim_number)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);

        Tensor * real_tensor=h->tensor;

        TShape& shape=real_tensor->GetShape();

        std::vector<int>& dim=shape.GetDim();

        int dim_size=dim.size();

	if(dim_size>dim_number)
	     return -1;

        for(int i=0;i<dim_size;i++)
	     dims[i]=dim[i];

	return dim_size;
}

int  get_tensor_buffer_size(tensor_t tensor)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);

        Tensor * real_tensor=h->tensor;

        return real_tensor->GetTotalSize();
}

int  set_tensor_buffer_transfer(tensor_t tensor, void * buffer, int buffer_size,tensor_buf_cb_t cb, void * cb_arg)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);
	GraphExecutor * executor=h->executor;

        /* TODO: pass the callback to below layers*/
	if(executor->SetTensorBuffer(h->tensor,buffer,buffer_size))
		return 0;

	return -1;
}

int  set_tensor_buffer(tensor_t tensor, void * buffer, int buffer_size)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);
	GraphExecutor * executor=h->executor;
         
	if(executor->SetTensorBuffer(h->tensor,buffer,buffer_size))
		return 0;

        return -1;
}


int  set_tensor_data(tensor_t tensor, const void * input_data, int data_size)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);
	GraphExecutor * executor=h->executor;

	if(executor->SetTensorData(h->tensor,input_data,data_size))
		return 0;

	return -1;
}


int  get_tensor_data(tensor_t tensor, void * output_data, int data_size)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);
	GraphExecutor * executor=h->executor;

	if(executor->GetTensorData(h->tensor,output_data,data_size))
		return 0;

	return -1;
}

void * get_tensor_buffer(tensor_t tensor)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);

	GraphExecutor * executor=h->executor;

        return executor->GetTensorBuffer(h->tensor);

}

const char * get_tensor_name(tensor_t tensor)
{
        tensor_handle * h=static_cast<tensor_handle*>(tensor);
        Tensor *  real_tensor=h->tensor;

        return real_tensor->GetName().c_str();
}

int  prerun_graph(graph_t graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	if(!executor->InferShape())
           return -1;

	if(executor->Prerun())
		return 0;

	return -1;
}

int  run_graph(graph_t graph, int block)
{

	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

	return executor->Run(block);
}

int wait_graph(graph_t graph, int try_wait)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int  postrun_graph(graph_t  graph)
{
	GraphExecutor * executor=static_cast<GraphExecutor *>(graph);

        executor->Postrun();

	return 0;
}


int  get_graph_exec_status(graph_t graph)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int  set_graph_event_hook(graph_t graph, int event, graph_callback_t cb_func, void * cb_arg)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int get_engine_number(void)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

const char * get_engine_name(int idx)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int set_device_mode(const char * device_name, int mode)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int get_device_mode(const char * device_name)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int get_device_config(const char * device_name, const char * config_name, void * val, int size)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int set_device_config(const char * device_name, const char * config_name, void * val, int size)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int del_device_config(const char * device_name, const char * config_name)
{
	TO_BE_IMPLEMENTED;
	return 0;
}


user_context_t  create_user_context(const char * context_name)
{

        bool ret;

	UserContext * context=new UserContext(context_name);

        ret=UserContextManager::SafeAdd(context_name,context);

        if(ret)
	   return context;

        delete context;

        return nullptr;
}

int check_user_context_valid(user_context_t context)
{
	UserContext * user_context=static_cast<UserContext *>(context);

	if(user_context==nullptr)
		return 0;

	return 1;
}

user_context_t  get_user_context(const char * context_name)
{
        UserContext * user_context;

        if(UserContextManager::SafeGet(context_name,user_context))
            return user_context;

        return nullptr;
}

void destroy_user_context(user_context_t  context)
{
        UserContext * user_context=static_cast<UserContext *>(context);

        if(UserContextManager::SafeRemove(user_context->GetName()))
             delete user_context;
        
        XLOG_ERROR()<<"BUG: not managed user context: "<<user_context->GetName()<<"\n";
}

int set_user_context_config(user_context_t  context, const char * name, void * val, int size)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int get_user_context_config(user_context_t  context, const char * name, void * val, int size)
{
	TO_BE_IMPLEMENTED;
	return 0;
}
int del_user_context_config(user_context_t  context, const char * name)
{
	TO_BE_IMPLEMENTED;
	return 0;
}


workspace_t  create_workspace(const char * ws_name, user_context_t  context)
{
	UserContext *user_context=static_cast<UserContext *>(context);

	return user_context->CreateWorkspace(ws_name);
}

int check_workspace_valid(workspace_t ws)
{
	RuntimeWorkspace * r_ws=static_cast<RuntimeWorkspace *> (ws);

	if(r_ws==nullptr)
		return 0;
	return 1;
}

workspace_t  get_workspace(const char * ws_name,user_context_t context)
{
	UserContext *user_context=static_cast<UserContext *>(context);

	return user_context->FindWorkspace(ws_name);
}

void destroy_workspace(workspace_t  ws)
{
	RuntimeWorkspace * r_ws=static_cast<RuntimeWorkspace *> (ws);
	UserContext * user_context=r_ws->GetUserContext();

	user_context->DestroyWorkspace(r_ws);
}

int set_workspace_config(workspace_t  ws, const char * config_name, void * config_val)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int get_workspace_config(workspace_t  ws, const char * config_name, void * config_val)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int del_workspace_config(workspace_t  ws, const char * config_name)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int  set_graph_config(graph_t graph, const char * name, void * val, int size)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int  get_graph_config(graph_t graph, const char * name, void * val, int size)
{
	TO_BE_IMPLEMENTED;
	return 0;
}

int  del_graph_config(graph_t graph, const char * name)
{
	TO_BE_IMPLEMENTED;
	return 0;
}


void set_log_level(int level)
{
   SET_LOG_LEVEL((LogLevel)level);
}

