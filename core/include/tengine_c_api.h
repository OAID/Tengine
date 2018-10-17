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
#ifndef __TENGINE_C_API_H__ 
#define __TENGINE_C_API_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef void * user_context_t;
typedef void * workspace_t;
typedef void * graph_t;
typedef void * tensor_t;
typedef void * node_t;

typedef int (*graph_callback_t)(graph_t ,int, void * arg);
typedef int (*tensor_buf_cb_t)(void * buf, void * arg);

enum graph_exec_event {
   GRAPH_EXEC_START,
   GRAPH_EXEC_SUSPEND,
   GRAPH_EXEC_RESUME,
   GRAPH_EXEC_ABORT,
   GRAPH_EXEC_DONE
};

enum graph_exec_stat {
   GRAPH_STAT_CREATED,
   GRAPH_STAT_READY,
   GRAPH_STAT_RUN,
   GRAPH_STAT_DONE,
   GRAPH_STAT_ERROR
};

/* 
Level 0 API

Which is designed for simple graph usage, which has only one input tensor and one output tensor

*/

/*!
* @brief load a saved model and create a graph for execution 
*
* @param graph_name a name given to the loaded graph  
* @param format the saved model format, such as caffe/onnx/mxnet/tensorflow
* @param fname  variable list of files of the saved model
* @return the graph handle. 
* @note please call check_graph_valid to tell if the graph loaded correctly
*       need to call destroy_graph() to release the resource allocated
*/

graph_t create_graph(const char * graph_name, const char * format, const char * fname, ...);

/*!
* @brief check if the graph handle graph_t is valid
*
* @param graph_t the handle returned by create_graph 
* @return 1 when handle is valid, while -1 when handle is invalid
*/

int check_graph_valid(graph_t graph);

/*!
* @brief get the graph name of the graph handle
* @param  graph, the graph handle
* @return the name of the graph that the graph handle refers
*/

const char * get_graph_name(graph_t graph);

/*!
* @brief bind the device with the graph
* 
* @param graph the graph handle
* @param device_name the device name
* @return 0 when set success or -1 if failed
*/

int set_graph_device(graph_t graph, const char * device_name);

/*!
* @brief run the loaded graph with input data
* 
* @param graph the graph handle
* @param input_data the input data, which will be exactly the same shape as the input tensor
* @param input_data the byte size of the input data
* @return 0 success or -1 fail
*/

int run_inference(graph_t graph, void * input_data, int input_size);

/*!
* @brief get the graph output result
*
* @parami graph graph handle
* @param output_data the buffer to hold the output data
* @output_size size of the buffer
* @return 0 success -1 fail
* @note data will be copied to the output_data from graph output tensor
*/
int get_graph_output(graph_t graph, void * output_data, int output_size);

/*!
* @brief release the resource related with graph
*
* @graph graph handle
*/

void destroy_graph(graph_t graph);

/* auxiliary interfaces */

/*!
* @brief get the byte size of output result 
*
* @param graph the graph handle
* @return >0 the output size, <0 error
*/

int get_output_size(graph_t graph);

/*!
* @brief set the shape of the input tensor of the graph
* 
* @param graph the graph handle
* @param dims  the int array of tensor shape
* @param dim_number the size of the int array
* @return 0 sucess, or -1 fail
*/

int set_input_shape(graph_t graph, int dims[], int dim_number);


/*
   Level 1 API defines 

   runtime graph
   support mutilple inputs and outputs, as well as designate specific inputs and outputs

*/


/*!
* @brief initialize the library, should be called only once
* 
* @return 0 success, or -1 fail
*/

int init_tengine_library(void);

/*!
* @brief release the library, should be called only once
* 
*/

void release_tengine_library(void);

/*!
* @brief get the version of the tengine 
*
* @return const char * of version string
*/

const char * get_tengine_version(void);


/*!
* @brief check if the library supports the verson: major.minor.bugfix
*
* @param version a c string less than 64 bytes
* @return 1 support 0 not support
*/

int request_tengine_version(const char * version);


/*! 
* @brief load saved graph file into system, and represented by TEngine IR format
* 
* @param model_name the name assigned the loaded model
* @param model_format the model file format: caffe/onnx/tensorflow/mxnet/tengine
* @param fname file name of the saved model
* @return 0 success -1 fail
* @note  the saved model may have multiple files
*        please call remove_model() to release the loaded model
*/

int load_model(const char * model_name, const char * model_format, const char * fname, ...);

/*! 
* @brief load saved graph saved in memory into system, and represented by TEngine IR format
* 
* @param model_name the name assigned the loaded model
* @param model_format the model file format: caffe/onnx/tensorflow/mxnet/tengine
* @param mem_addr  the first mem block addr 
* @param mem_size  the first mem block size 
* @return 0 success -1 fail
* @note  the saved model may have multiple files
*        please call remove_model() to release the loaded model
*/

int load_mem_model(const char * model_name, const char * model_format, void * mem_addr,int mem_size, ...);

/*!
* @brief save the loaded model into file
*
* @param graph_t the runtime graph handle of the loaded model
* @param file_format the format to be stored
* @param fname the file name
* @return 0 success -1 fail
*/
int save_model(graph_t graph, const char * file_format, const char * fname, ...);

/*!
* @brief remove the loaded model 
*
* @param model_name the name of the loaded graph
* @return 0 success -1 fail
*/
int remove_model(const char * model_name);

/*!
* @brief dump the loaded model
* 
* @param model_name the name of loaded model
* @return 0 success -1 fail 
*/
int dump_model(const char * model_name);

/*!
* @brief create the graph for execution from a loaded model
* 
* @param graph_name the name the run-time graph
* @param model_name the name of loaded model
* @param ws  the workspace handle. It can be set to NULL, so that the default workspace will be used
*
* @return the handle of created graph. 
* @note   need to call check_graph_valid to verify the handle
*         call destory_runtime_graph() to release the resource
*/

graph_t   create_runtime_graph(const char * graph_name, const char * model_name,workspace_t ws);

/*!
* @brief check if a graph handle is valid or not 
*
* @param graph the graph handle
* @return 1 valid 0 invalid
*/
int  check_graph_valid(graph_t graph); 

/*!
* @brief destory the runtime graph and release resource allocated 
* 
* @param graph the graph handle
* @return 0 success -1 fail
*/

int   destroy_runtime_graph(graph_t  graph);

/*!
* @brief get the loaded model name when creating the graph
* 
* @param graph the graph handle
* @return the name of the loaded model
*/

const char * get_model_name(graph_t graph);

/*!
* @brief designate the input nodes of the graph
* 
* @param graph the graph handle
* @param input_nodes the node name list of input nodes
* @param input_number the number of input_nodes
*
* @note  if using the default input nodes of a graph, this call can be skipped
*/
int  set_graph_input_node(graph_t  graph, const char * input_nodes[], int input_number);


/*!
* @brief designate the output nodes of the graph
* 
* @param graph the graph handle
* @param output_nodes the node name list of output nodes
* @param output_number the number of output_nodes
*
* @note  if using the default output nodes of a graph, this call can be skipped
*/

int  set_graph_output_node(graph_t  graph, const char * output_nodes[], int output_number);

/*!
* @brief get the number of input node of the graph
* 
* @param graph the graph handle
* @return <0 error >0 the input node number
*/
int get_input_node_number(graph_t graph);

/*!
* @brief get the node name of #idx of input node 
*
* @param graph the graph handle
* @param idx the input node index
* @return the node name or NULL on error 
*/

const char * get_input_node_name(graph_t graph, int idx);

/*!
* @brief get the input tensor number of a  node
* 
* @param graph the graph handle
* @param node_name node name
* @return >=0 the number of input tensor, <0 error
*/


int get_node_input_number(graph_t graph, const char * node_name);


/*!
* @brief get the name of the input tensor of a node
*
* @param graph graph handle
* @param node_name the node name
* @param input_idx the index of the input tensor
* @return the tensor name or NULL on error
*/

const char * get_node_input_tensor(graph_t graph, const char * node_name, int input_idx);

/*!
* @brief get the number of output node of the graph 
* 
* @param graph the graph handle
* @return <0 error, >0 the output node number
*/

int get_output_node_number(graph_t graph);


/*!
* @brief get the node name of #idx of output node 
*
* @param graph the graph handle
* @param idx the output node index
* @return the node name or NULL on error 
*/

const char * get_output_node_name(graph_t graph, int idx);

/*!
* @brief get the output tensor number of a  node
* 
* @param graph the graph handle
* @param node_name node name
* @return >=1 the number of output tensor, <0 error
*/

int get_node_output_number(graph_t graph, const char * node_name);

/*!
* @brief get the name of the output tensor of a node
*
* @param graph graph handle
* @param node_name the node name
* @param output_idx the index of the output tensor
* @return the tensor name or NULL on error
*/

const char * get_node_output_tensor(graph_t graph, const char * node_name, int output_idx);

/*!
* @brief convert a tensor name into a tensor handle
*
* @param graph the graph handle
* @param tensor_name tensor name
* @return the tensor handle. 
* @note  Need to call check_tensor_valid to check if the handle is valid
*        Please call put_graph_tensor() to release the handle
*/
tensor_t  get_graph_tensor(graph_t graph, const char * tensor_name);

/*!
* @brief get a tensor handle of one of graph input notes
*
* @param graph the graph handle
* @param input_node_idx the input node index
* @param tensor_idx the tensor index
* @return the tensor handle
*/
tensor_t  get_graph_input_tensor(graph_t graph, int input_node_idx, int tensor_idx);

/*!
* @brief get a tensor handle of one of graph output notes
*
* @param graph the graph handle
* @param output_node_idx the output node index
* @param tensor_idx the tensor index
* @return the tensor handle
*/
tensor_t  get_graph_output_tensor(graph_t graph, int output_node_idx, int tensor_idx);

/*!
* @brief check if a tensor handle is valid or not
* 
* @param tensor the tensor handle
* @return 1 valid, or 0 invalid 
*/
int   check_tensor_valid(tensor_t tensor);

/*!
*  @brief release the tensor handle
* 
*  @param tensor the tensor handle
*/
void  put_graph_tensor(tensor_t tensor); 

/*!
* @brief set the shape of tensor 
* 
* @param tensor the tensor handle
* @param dims        an int array to represent shape
* @param dim_number  the array size
* @return  0 success or -1 fail
*/
int       set_tensor_shape(tensor_t tensor, int dims[], int dim_number);

/*!
* @brief get the shape of tensor 
* 
* @param tensor the tensor handle
* @param dims        an int array to get the returned shape
* @param dim_number  the array size
* @return  >=1 the valid dim number, or -1 fail
*/

int       get_tensor_shape(tensor_t tensor, int dims[], int dim_number);

/*!
* @brief get the byte size of a tensor should occupy
*
* @param tensor the tensor handle
* @return <0 error, >=0 tensor size
* @note   if return 0, it means the shape of the tensor is not set yet
*/

int  get_tensor_buffer_size(tensor_t tensor);

/*!
* @brief set the buffer of the tensor, transfer the owner ship
* 
* @param tensor the tensor handle
* @param buffer the buffer address
* @param buffer_size the buffer_size
* @param cb  the callback function used to release the buffer
* @param cb_arg args will be passed to the callback function
* @return 0 success -1 error
* @note  the owner of the buffer will be transferred to the tensor. 
*        The tensor will release it by calling the callback 
*        when tensor is destoryed
*/
int  set_tensor_buffer_transfer(tensor_t tensor, void * buffer, int buffer_size, tensor_buf_cb_t cb, void * cb_arg);

/*!
* @brief set the buffer of the tensor
* 
* @param tensor the tensor handle
* @param buffer the buffer address
* @param buffer_size the buffer_size
* @return 0 success or -1 error
* @note  the buffer is still managed by caller
*/
int  set_tensor_buffer(tensor_t target_tensor, void * buffer, int buffer_size);

/*!
* @brief copy the data to tensor buffer
*
* @param tensor the tensor handle
* @param input_data the input data 
* @param data_size  the input data size
* @return 0 success or -1 error
*/
int  set_tensor_data(tensor_t input_tensor, const void * input_data, int data_size);


/*!
* @brief copy tensor data to the output data buffer
* @param tensor the tensor handle
* @param output_data the output data buffer
* @param data_size  the output buffer size
* @return 0 success or -1 error
*/
int  get_tensor_data(tensor_t tensor, void * output_data, int data_size);


/*!
* @brief get the buffer of the tensor 
*
* @param tensor the tensor handle
* @return the buffer address. NULL maybe be returned if no buffer allocated yet
*/
void * get_tensor_buffer(tensor_t tensor);


/*!
* @brief  get the name of the tensor
*
* @param tensor the tensor handle
* @return a c string 
*/
const char * get_tensor_name(tensor_t tensor);

/*!
* @brief  get the handle of node
*
* @param  graph the graph
* @param  node_name the name of the node
* @return the node handle
*/
node_t get_graph_node(graph_t graph, const char * node_name);

/*!
* @brief  set the device to  execution a node
*
* @param  the node handle
* @param  dev_name the device name to run the node
*
* @return  0: bind ok
*         <0: error 
*/
int  set_node_device(node_t node, const char * dev_name);

/*!
* @brief  free the handle of node
*
* @param  the node handle
* @return none
*/
void put_graph_node(node_t node);



/*!
* @brief get the param value (int) of a node
*
* @param node, the target node
* @param param_name, the name of the param to be retrieval
* @param  param_val, pointer to the int val to be saved
* 
* @return 0, retrieval value successfully; 
*        <0, failed; probably the name does not exist or the type mismatch
*/

int get_node_param_int(node_t node, const char * param_name, int * param_val);

/*!
* @brief get the param value (float) of a node
*
* @param node, the target node
* @param param_name, the name of the param to be retrieval
* @param  param_val, pointer to the float val to be saved
* 
* @return 0, retrieval value successfully; 
*        <0, failed; probably the name does not exist or the type mismatch
*/

int get_node_param_float(node_t node, const char * param_name, float * param_val);

/*!
* @brief get the param value (int) of a node
*
* @param node, the target node
* @param param_name, the name of the param to be retrieval
* @param  param_val, pointer to the pointer val to be saved
* 
* @return 0, retrieval value successfully; 
*        <0, failed; the name does not exist 
*/

int get_node_param_pointer(node_t node, const char * param_name, void *  param_val);

/*!
* @brief get the param value of a node, the data type is indicated by type_info
*        this interface only works in c++, as type_info refers std::type_info
*
* @param node, the target node
* @param param_name, the name of the param to be retrieval
* @param type_info, pointer to the std::type_info of wanted type, NULL to skip type check
* @param param_val, pointer to the float val to be saved
* 
* @return 0, retrieval value successfully; 
*        <0, failed; probably the name does not exist or the type mismatch
*/

int get_node_param_generic(node_t node, const char * param_name, const void * type_info, void * param_val);

int set_node_param_int(node_t node, const char * param_name, const int * param_val);
int set_node_param_float(node_t node, const char * param_name, const float * param_val);
int set_node_param_pointer(node_t node, const char * param_name, const void* param_val);
int set_node_param_generic(node_t node, const char * param_name, const void * type_info, const void * param_val);

/*!
* @brief initialize resource for graph execution
*  
* @param graph graph handle
* @return 0 success or -1 fail
*/
int  prerun_graph(graph_t graph);

/*!
* @brief infer shape for graph
*  
* @param graph graph handle
* @return 0 success or -1 fail
*/
int  infer_shape(graph_t graph);

/*!
* @brief execute graph 
*
* @param graph the graph handle
* @param block blocking or nonlocking
* @return 0 success or -1 fail
* @note  if block is 0, need to call wait_graph to get result or set GRAPH_DONE event hook.
*/
int  run_graph(graph_t graph, int block);

/*!
* @brief wait graph execution done
*
* @param graph the graph handle
* @param try_wait if set, just check status and return 
* @return  1 graph is done 
*          0 try again
*/
int  wait_graph(graph_t graph, int try_wait);

/*!
* @brief release resource for graph execution
* @param graph graph handle
* @return 0 success or -1 fail
*/
int  postrun_graph(graph_t  graph);

/*!
* @brief get the status of graph execution
*
* @param graph the graph handle
* @return status
*
*/
int  get_graph_exec_status(graph_t graph);

/*!
* @brief set the event hook for graph execution
* 
* @param graph the graph handle
* @param event the event to be hooked
* @param cb_func callback funtion
* @param cb_arg argument will be passed to callback function
*/
int  set_graph_event_hook(graph_t graph, int event, graph_callback_t cb_func, void * cb_arg);




/* Level 3
   
   Resource allocation and configuration related 
*/

/* device management related */

/*!
* @brief get the device number 
*
* @return the number of device
*/
int get_device_number(void);

/*!
* @brief get the device name of specific index
*
* @param idx the device index
* @return the name of the device 
*/
const char * get_device_name(int idx);

/*!
* @brief set the device working mode
*
* @param device_name the device name 
* @param mode the device working mode
* @return 0 success -1 fail 
*/
int set_device_mode(const char * device_name, int mode);

/*!
* @brief get the device working mode
*
* @param device_name the device name 
* @return >=0 the mode  or -1 fail 
*/
int get_device_mode(const char * device_name);

/*!
* @brief get the config setting by config name. the config request may be passed to driver
*
* @param device_name the device name
* @param config_name the config item name
* @param val the buffer to hold the data
* @param size the buffer size
* @return 0 success -1 fail
*/

int get_device_config(const char * device_name, const char * config_name, void * val, int size);

/*!
* @brief set the config item of the device. the config item may be passed to driver 
*
* @param device_name the device name
* @param config_name  the config item name
* @param val     the buffer to hold the data to be set
* @param size    the buffer size
* @return 0 success -1 fail
*/
int set_device_config(const char * device_name, const char * config_name, void * val, int size);

/*!
* @brief delete the config item of the device
* @param device_name the device name
* @param config_name the config item name
* @return 0 success -1 fail  
*/
int del_device_config(const char * device_name, const char * config_name);


/*
*   user context related
*/

/*!
* @brief create one user context with name
*
* @param context_name the name of the created context 
* @return  user context handle. need to call check_user_context_valid() to verify the result
*/
user_context_t  create_user_context(const char * context_name); 

/*!
* @brief check if the user context handle is valid
*
* @param  context the context handle
* @return 1 valid handle or 0 invalid handle
*/

int check_user_context_valid(user_context_t context);

/*!
* @brief get the context handle by name
* 
* @param context_name the context name
* @return the context handle
*/
user_context_t  get_user_context(const char * context_name); 

/*!
* @brief destory and reclaim resource related with the context 
* @param context the context handle
*/
void destroy_user_context(user_context_t  context);

/*!
* @brief set config item  of user context
* 
* @param context the user context handle
* @param config_name the config item name
* @param val the buffer to hold the data to set
* @param size the buffer size
* @return 0 success or -1 fail
*/
int set_user_context_config(user_context_t  context, const char * config_name, void * val, int size);

/*!
* @brief get the config item of user context
* 
* @param context the user context handle
* @param config_name the config item name
* @param val the buffer to hold the data
* @param size the buffer size
* @return 0 succuess or -1 fail
*/
int get_user_context_config(user_context_t  context, const char * config_name, void * val, int size);

/*!
* @brief delete the config item of user context
*
* @param context the user context handle
* @param config_name the config item name
* @return 0 success or -1 fail
*/
int del_user_context_config(user_context_t  context, const char * config_name);

/*!
* @brief create a runtime workspace with name, inside a user context 
*  
* @param ws_name workspace name
* @param context user context handle
* @return workspace handle; need to call check_workspace_valid() to verify the result
*/
workspace_t  create_workspace(const char * ws_name, user_context_t  context);

/*!
* @brief check if workspace handle is valid
*
* @param ws workspace handle
* @return 1 valid or 0 invalid
*/
int check_workspace_valid(workspace_t ws);

/*!
* @brief get workspace handle by name 
* 
* @param ws_name workspace name
* @return workspace handle 
*/
workspace_t  get_workspace(const char * ws_name, user_context_t context); 

/*!
* @brief destroy workspace
* 
* @param ws workspace handle
*/
void destroy_workspace(workspace_t  ws);

/*!
* @brief  set the config item of workspace
* 
* @param  ws workspace handle
* @param  name config item name
* @param  val the buffer to hold the data
* @param  size the buffer size
* @return 0 success or -1 fail
*/
int set_workspace_config(workspace_t  ws, const char * name, void * val, int size);

/*!
* @brief  get the config item of workspace
* 
* @param  ws workspace handle
* @param  name config item name
* @param  val the buffer to hold the data
* @param  size the buffer size
* @return 0 success or -1 fail
*/
int get_workspace_config(workspace_t  ws, const char * name, void * val, int size);

/*!
* @brief delete the workspace config
*
* @param ws workspace handle
* @param name config item name
*
* @return 0 success or -1 fail
*/
int del_workspace_config(workspace_t  ws, const char * name);

/*!
* @brief interface to set some proprietary config items for graph.
*        It is probabaly the config will be passed to the DLA driver
* @param graph the graph handle
* @param name the config item name
* @param val the buffer to hold data
* @param size the buffer size
* @return 0 success or -1 fail 
*/

int       set_graph_config(graph_t graph, const char * name, void * val, int size);

/*!
* @brief interface to get some proprietary config items for graph.
*        It is probabaly the config will be passed to the DLA driver
* @param graph the graph handle
* @param name the config item name
* @param val the buffer to hold data
* @param size the buffer size
* @return 0 success or -1 fail 
*/
int       get_graph_config(graph_t graph, const char * name, void * val, int size);

/*!
* @brief delete the graph config
*
* @param graph graph handle
* @param name config item name
*/
int       del_graph_config(graph_t graph, const char * name);


/*
* misc API 
*/

/*!
* @brief set the logger level
*
* @param level log level
*/
void set_log_level(int level);


void dump_graph(graph_t graph);

/* Level 4
   
   Interface for developer and debug
*/

void set_config_file(const char * conf_file);

const char * get_config_file(void);

/*!
* @set the default device 
* 
* @param device the device name
* @return 0 valid or -1 invalid
*/

int set_default_device(const char * device);

/* for predefined device */
int load_device(const char * driver_name, const char * dev_name);


int unload_device(const char * driver_name, const char * dev_name);


/* plugin */

int load_tengine_plugin(const char * plugin_name, const char * fname, const char * init_func_name);
int unload_tengine_plugin(const char * plugin_name, const char * rel_func_name);
int get_tengine_plugin_number(void);
const char * get_tengine_plugin_name(int idx);



#ifdef __cplusplus
}
#endif

#endif
