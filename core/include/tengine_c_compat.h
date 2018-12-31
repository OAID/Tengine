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
#ifndef __TENGINE_C_COMPATIBLE_H__
#define __TENGINE_C_COMPATIBLE_H__

#include "tengine_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * @brief load saved graph file into system, and represented by TEngine IR format
 *
 * @param model_name the name assigned the loaded model
 * @param model_format the model file format: caffe/onnx/tensorflow/mxnet/tengine
 * @param fname file name of the saved model
 * @return 0 success -1 fail
 * @note  the saved model may have multiple files
 *        please call remove_model() to release the loaded model
 */

int load_model(const char* model_name, const char* model_format, const char* fname, ...);

/*!
 * @brief save the loaded model into file
 *
 * @param graph_t the runtime graph handle of the loaded model
 * @param file_format the format to be stored
 * @param fname the file name
 * @return 0 success -1 fail
 */
int save_model(graph_t graph, const char* file_format, const char* fname, ...);

/*!
 * @brief remove the loaded model
 *
 * @param model_name the name of the loaded graph
 * @return 0 success -1 fail
 */
int remove_model(const char* model_name);

/*!
 * @brief dump the loaded model
 *
 * @param model_name the name of loaded model
 * @return 0 success -1 fail
 */
int dump_model(const char* model_name);

/*!
 * @brief create the graph for execution from a loaded model
 *
 * @param graph_name the name the run-time graph
 * @param model_name the name of loaded model
 * @param context, the context to run the graph
 *
 * @return the handle of created graph.
 * @note   need to call check_graph_valid to verify the handle
 *         call destory_runtime_graph() to release the resource
 */

graph_t create_runtime_graph(const char* graph_name, const char* model_name, context_t context);

/*!
 * @brief check if a graph handle is valid or not
 *
 * @param graph the graph handle
 * @return 1 valid 0 invalid
 */
int check_graph_valid(graph_t graph);

/*!
 * @brief destory the runtime graph and release resource allocated
 *
 * @param graph the graph handle
 * @return 0 success -1 fail
 */

int destroy_runtime_graph(graph_t graph);

/*!
 * @brief check if a tensor handle is valid or not
 *
 * @param tensor the tensor handle
 * @return 1 valid, or 0 invalid
 */
int check_tensor_valid(tensor_t tensor);

/*!
 *  @brief release the tensor handle
 *
 *  @param tensor the tensor handle
 */
void put_graph_tensor(tensor_t tensor);

/*!
 * @brief  free the handle of node
 *
 * @param  the node handle
 * @return none
 */
void put_graph_node(node_t node);

/*!
 * @brief interface to set some proprietary config items for graph.
 *        It is probabaly the config will be passed to the DLA driver
 * @param graph the graph handle
 * @param name the config item name
 * @param val the buffer to hold data
 * @param size the buffer size
 * @return 0 success or -1 fail
 */

int set_graph_config(graph_t graph, const char* name, void* val, int size);

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

int get_node_param_int(node_t node, const char* param_name, int* param_val);

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

int get_node_param_float(node_t node, const char* param_name, float* param_val);

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

int get_node_param_pointer(node_t node, const char* param_name, void* param_val);

/*!
 * @brief get the param value of a node, the data type is indicated by type_info
 *        this interface only works in c++, as type_info refers std::type_info
 *
 * @param node, the target node
 * @param param_name, the name of the param to be retrieval
 * @param type_info, pointer to the std::type_info of wanted type, NULL to skip type check
 * @param param_val, pointer to the val to be saved
 * @param size, parameter size
 *
 * @return 0, retrieval value successfully;
 *        <0, failed; probably the name does not exist or the type mismatch
 */

int get_node_param_generic(node_t node, const char* param_name, const void* type_info, void* param_val, int size);

/*!
 * @brief infer shape for graph
 *
 * @param graph graph handle
 * @return 0 success or -1 fail
 */
int infer_shape(graph_t graph);
/*
 * @brief Get the layout of tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [out] layout: The layout of tensor.
 * @return >=1 the valid dim number, or -1 Fail.
 *
 */
int get_tensor_layout(tensor_t tensor, char* layout);

/*!
 * @brief Set the layout of tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] layout: The layout of tensor.
 * @return 0: Success; -1: Fail.
 *
 */
int set_tensor_layout(tensor_t tensor, const char* layout);

#ifdef __cplusplus
}
#endif

#endif
