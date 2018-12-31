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

#ifndef __TENGINE_C_API_H__
#define __TENGINE_C_API_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CONFIG_LEGACY_API
#define CONFIG_LEGACY_API
#endif

#define MAX_SHAPE_DIM_NUM 4

/* the data type of the tensor */
#define TENGINE_DT_FP32 0
#define TENGINE_DT_FP16 1
#define TENGINE_DT_INT8 2
#define TENGINE_DT_UINT8 3
#define TENGINE_DT_INT32 4
#define TENGINE_DT_INT16 5

/* layout type, not real layout */
#define TENGINE_LAYOUT_NCHW 0
#define TENGINE_LAYOUT_NHWC 1

/* tensor type: the content changed or not during inference */
#define TENSOR_TYPE_UNKNOWN 0
#define TENSOR_TYPE_VAR 1
#define TENSOR_TYPE_CONST 2
#define TENSOR_TYPE_INPUT 3
#define TENSOR_TYPE_DEP 4

/* node dump action definition */
#define NODE_DUMP_ACTION_DISABLE 0
#define NODE_DUMP_ACTION_ENABLE 1
#define NODE_DUMP_ACTION_START 2
#define NODE_DUMP_ACTION_STOP 3
#define NODE_DUMP_ACTION_GET 4

/* graph perf action definition */
#define GRAPH_PERF_STAT_DISABLE 0
#define GRAPH_PERF_STAT_ENABLE 1
#define GRAPH_PERF_STAT_STOP 2
#define GRAPH_PERF_STAT_START 3
#define GRAPH_PERF_STAT_RESET 4
#define GRAPH_PERF_STAT_GET 5

/* follow the std. UNIX log level definitioin */
enum log_level
{
    LOG_EMERG,
    LOG_ALERT,
    LOG_CRIT,
    LOG_ERR,
    LOG_WARNING,
    LOG_NOTICE,
    LOG_INFO,
    LOG_DEBUG
};

/* note: Android NN only define one event */
enum graph_exec_event
{
    GRAPH_EXEC_START,
    GRAPH_EXEC_SUSPEND,
    GRAPH_EXEC_RESUME,
    GRAPH_EXEC_ABORT,
    GRAPH_EXEC_DONE
};

/* todo: should add suspend? */
enum graph_exec_stat
{
    GRAPH_STAT_CREATED,
    GRAPH_STAT_READY,
    GRAPH_STAT_RUNNING,
    GRAPH_STAT_DONE,
    GRAPH_STAT_ERROR
};

enum device_policy
{
    DEFAULT_POLICY,
    LATENCY_POLICY,
    LOW_POWER_POLICY
};

typedef void* context_t;
typedef void* graph_t;
typedef void* tensor_t;
typedef void* node_t;

typedef int (*event_handler_t)(graph_t, int, void* arg);

typedef void (*log_print_t)(const char*);

/* performance profiling records */

struct perf_info
{
    const char* name; /* node name */
    const char* dev_name; /* device name */
    uint32_t count;
    uint32_t min;
    uint32_t max;
    uint64_t total_time; /* us or cycle, depends on devices */
    uint32_t base; /* 1ms second time number */
};

struct custom_kernel_tensor
{
    int dim[MAX_SHAPE_DIM_NUM]; /* the shape dim array */
    int dim_num; /* valid entry number */
    int element_num;
    int element_size; /* determined  by data_type */
    int data_type;
    int dev_type; /* indicate the tensor belongs to CPU/GPU ... */
    int layout_type; /*  NCHW type or NHWC type*/

    /* quant info */
    int quant_type; /* int8, int16 or int32 */
    float* scale;
    int* zero_point;
    int* quant_number;

    void* data; /* pointer to host memory (virtual address) */
    void* dev_mem; /* refers to device memory block */
    void* mapped_mem; /* the mapped dress for device memory block */
};

/* For user to add user defined kernel*/
struct custom_kernel_ops
{
    const char* kernel_name; /* name of the kernel */
    const char* op; /* name of the op to be implemented */
    int force; /* if not set, when bind() failed,
      try to use other kernel implementations*/
    void* kernel_param; /* used for kernel impl functions */
    int kernel_param_size;

    /*!
     * @brief generate output shape according to input shapes
     *        if not implemented, set it to NULL.
     *
     * @param [in]  ops: The point of custom defined kernel.
     * @param [in]  inputs[]:  pointer array to the shape of input tensors.
     *                         the shape has MAX_SHAPE_DIM_NUM elements,
     *                         and element with value 0  means the end of the shape.
     * @param [in]  input_num: The number of input tensors.
     * @param [out] outputs[]: pointer array to the shape of output tensors.
     *                         the memory has been allocated already
     * @param [in]  output_num: The number of output tensors
     * @param [in]  layout: the graph layout is NHWC or NCHW
     *
     * @return 0: success, -1: fail.
     */
    int (*infer_shape)(struct custom_kernel_ops* ops, const int* inputs[], int input_num, int* outputs[],
                       int output_num, int layout);

    /*!
     * @brief Get the inplace input tensor index for an output tensor.
     *
     * @param [in] ops: The point of custom defined kernel.
     * @param [in] output_idx: The index of custom defined kernel output.
     *
     * @return the inplace input tensor index for an output tensor.
     *         if the output tensor is not an inplace one, return -1.
     */
    int (*inplace_info)(struct custom_kernel_ops* ops, int output_idx);    // optional

    /*!
     * @brief Check if the kernel can work on the input and output shapes.
     *
     * @param [in] ops: The point of custom defined kernel.
     * @param [in] inputs[]: The custom kernel tensors for input.
     * @param [in] input_num: The number of the input tensors.
     * @param [in] outputs[]: The custom kernel output tensor for output
     * @param [in] output_num: The number of the output tensors.
     *
     * @return 0 if the input and output are supported
     *         otherwise, return -1.
     *
     * notes: If not implemented, set it NULL, which means always return 0.
     */
    int (*bind)(struct custom_kernel_ops* ops, const struct custom_kernel_tensor* inputs[], int input_num,
                const struct custom_kernel_tensor* outputs[], int output_num);

    /*!
     * @brief Prepare for run graph.
     *        dynamic_shape means it is not an abnormal case when input_num is zero.
     *
     * @param [in] ops: The point of custom defined kernel.
     * @param [in] inputs[]: The custom defined kernel input tensor.
     * @param [in] input_num: The number of the custom defined kernel input tensor.
     * @param [in] outputs[]: The custom defined kernel output tensor.
     * @param [in] output_num: The number of the custom defined kernel output tensor.
     * @param [in] dynamic_shape: It is not an abnormal case when input_num is zero.

     * @return 0: success, -1: fail.
     */
    int (*prerun)(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
                  struct custom_kernel_tensor* outputs[], int output_num, int dynamic_shape);

    /*!
     * @brief Reshape the graph.
     *
     * @param [in] ops: The point of custom defined kernel.
     * @param [in] inputs[]: The custom defined kernel input tensor.
     * @param [in] input_num: The number of the custom defined kernel input tensor.
     * @param [in] outputs[]: The custom defined kernel output tensor.
     * @param [in] output_num: The number of the custom defined kernel output tensor.

     * @return 0: success, -1: fail.
     *
     * notes: It will be called, when input shape changed.
     *        After prerun() has been called, need to reclaim and re-allocate run-time
     *        resource depends on input shape.
     */
    int (*reshape)(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
                   struct custom_kernel_tensor* outputs[], int output_num);

    /*!
     * @brief Run the graph.
     *
     * @param [in] ops: The point of custom defined kernel.
     * @param [in] inputs[]: The custom defined kernel input tensor.
     * @param [in] input_num: The number of the custom defined kernel input tensor.
     * @param [in] outputs[]: The custom defined kernel output tensor.
     * @param [in] output_num: The number of the custom defined kernel output tensor.

     * @return 0: success, -1: fail.
     */
    int (*run)(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
               struct custom_kernel_tensor* outputs[], int output_num);

    /*!
     * @brief Pause the graph and release the resources used when the graph is running.
     *
     * @param [in] ops: The point of custom defined kernel.
     * @param [in] inputs[]: The custom defined kernel input tensor.
     * @param [in] input_num: The number of the custom defined kernel input tensor.
     * @param [in] outputs[]: The custom defined kernel output tensor.
     * @param [in] output_num: The number of the custom defined kernel output tensor.

     * @return 0: success, -1: fail.
     */
    int (*postrun)(struct custom_kernel_ops* ops, struct custom_kernel_tensor* inputs[], int input_num,
                   struct custom_kernel_tensor* outputs[], int output_num);

    /*!
     * @brief Free the resource allocated this ops implementation.
     *
     * @param [in] ops: The point of custom defined kernel.
     *
     * @return None.
     */
    void (*release)(struct custom_kernel_ops* ops);
};

/************** Library intialization and version checking *******************/

/*!
 * @brief Initialize the tengine, only can be called once.
 *
 * @return 0: Success, -1: Fail.
 */
int init_tengine(void);

/*!
 * @brief Release the tengine, only can be called once.
 *
 * @return none.
 */
void release_tengine(void);

/*!
 * @brief Get the version of the tengine.
 *
 * @return const char * of version string.
 */
const char* get_tengine_version(void);

/*!
 * @brief Check the run-time library supports the verson.
 *        app developer should call get_tengine_version() to save the version used
 *        during developping.
 *
 *        this interface is designed for app built with dynamic tengine library.
 *        The app knows exactly that it can work on a tengine version, and it can
 *        check run-time tengine library supports that version.
 *
 * @param [in] version: A c string returned by get_tengine_version()
 * @return 1: support, 0: not support.
 */
int request_tengine_version(const char* version);

/*************************** graph operate set ********************************/

/*!
 * @brief Create the run-time graph for execution from a saved model.
 *        If model format is NULL, an empty graph handle will be returned.
 *
 * @param [in] context: The context the graph will run inside;
 *                   could be NULL and the graph is created in a private context.
 *
 * @param [in] model_format: The model format type,such as "caffe","tengine"
 * @param [in] file_name:  The name of model file.
 *
 * @return  The graph handler or NULL if failed.
 */

graph_t create_graph(context_t context, const char* model_format, const char* file_name, ...);

/*!
 * @brief save the graph into file using the model format
 *
 * @param [in] graph, the graph handle
 * @param [in] model_format, the name of the model format
 * @param [in] file_name, the file name of saved model
 *
 * @return  0 success or -1 fail
 */

int save_graph(graph_t graph, const char* model_format, const char* file_name, ...);

/*!
 * @brief Set the layout type of the graph
 *        the default layout of graph is NCHW
 * @param [in] graph, the graph handle
 * @param [in] layout_type, the layout type NCHW or NHWC
 *
 * @return 0 success, or -1 fail
 */

int set_graph_layout(graph_t graph, int layout_type);

/*!
 * @brief designate the input nodes of the graph
 *
 * @param [in] graph: the graph handle
 * @param [in] input_nodes: the node name list of input nodes
 * @param [in] input_number: the number of input_nodes
 *
 * @note  if using the default input nodes of a graph, this call can be skipped
 */
int set_graph_input_node(graph_t graph, const char* input_nodes[], int input_number);

/*!
 * @brief designate the output nodes of the graph
 *
 * @param [in] graph: the graph handle
 * @param [in] output_nodes: the node name list of output nodes
 * @param [in] output_number: the number of output_nodes
 *
 * @note  if using the default output nodes of a graph, this call can be skipped
 */

int set_graph_output_node(graph_t graph, const char* output_nodes[], int output_number);

/*!
 * @brief Merge several graph into one single graph
 *        all the graphs should be in the same context
 *
 * @param [in] graph_num: the number of graph array
 * @param [in] graph0: the first graph
 * @param [in] graph1: the second graph
 * @param [in] possible other graphs
 *
 * @return New graph or NULL in case of failure
 */

graph_t merge_graph(int graph_num, graph_t graph0, graph_t graph1, ...);

/*!
 * @brief Destory the runtime graph and release allocated resource.
 *
 * @param [in] graph: The graph handle.
 * @return 0: Success, -1: Fail.
 */
int destroy_graph(graph_t graph);

/*!
 * @brief Get the number of input node of the graph.
 *
 *  @param [in] graph The graph handle.
 *  @return <0 Fail, >0 the input node number.
 */
int get_graph_input_node_number(graph_t graph);

/*!
 * @brief Get the node handle of #idx of input node of the graph.
 *
 * @param [in] graph The graph handle.
 * @param [in] idx The input node index,starting from zero.
 *
 * @return The node name or NULL on error.
 */
node_t get_graph_input_node(graph_t graph, int idx);

/*!
 * @brief Get the number of output node of the graph.
 *
 *  @param [in] graph The graph handle.
 *
 *  @return <0 error, >0 the input node number.
 */
int get_graph_output_node_number(graph_t graph);

/*!
 * @brief Get the node handle #idx of a graph output node.
 *
 * @param [in] graph The graph handle.
 * @param [in] idx The input node index, starting from zero.
 *
 * @return The node name or NULL on error.
 */
node_t get_graph_output_node(graph_t graph, int idx);

/*!
 * @brief Get a tensor handle of a graph output node.
 *
 * @param [in] graph: The graph handle.
 * @param [in] output_node_idx: The output node index.
 * @param [in] tensor_idx: The output tensor index of the output node.
 *
 * @return The tensor handle or NULL on error.
 *
 */
tensor_t get_graph_output_tensor(graph_t graph, int output_node_idx, int tensor_idx);

/*!
 * @brief Get tensor handle of one graph input tensor.
 *
 * @param [in] graph: The graph handle.
 * @param [in] input_node_idx: The input node index, starting from zero.
 * @param [in] tensor_idx: The output tensor index of the input node, starting from zero.
 *
 * @return The tensor handle or NULL on error.
 */
tensor_t get_graph_input_tensor(graph_t graph, int input_node_idx, int tensor_idx);

/******************* node operate set ****************************/
/*!
 * @brief Create a node for the graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] node_name: The name of the node.
 * @param [in] op_name: The name of the operate.
 *
 * @return The node handle or NULL on error.
 */
node_t create_graph_node(graph_t graph, const char* node_name, const char* op_name);

/*!
 * @brief  Get the node handle of the graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] node_name: The name of the node.
 *
 * @return The node handle or NULL on error.
 */
node_t get_graph_node(graph_t graph, const char* node_name);

/*!
 * @brief Get the node name.
 *
 * @param [in] node: The node handle.
 *
 * @return The node name, NULL on error.
 */
const char* get_node_name(node_t node);

/*!
 * @brief Get the node op.
 *
 * @param [in] node: The node handle.
 *
 * @return The op name, NULL on error.
 */
const char* get_node_op(node_t node);

/*!
 * @brief  Release the node handle.
 *
 * @param  [in] node: The node handle.
 *
 * @return None.
 */
void release_graph_node(node_t node);

/*!
 * @brief Get the input tensor handle of a node.
 *
 * @param [in] graph: The graph handle.
 * @param [in] node_name: The node name.
 * @param [in] input_idx: The index of the input tensor.
 * @return The tensor name or NULL on error.
 *
 */
tensor_t get_node_input_tensor(node_t node, int input_idx);

/*!
 * @brief Get the output tensor handle of a node.
 *
 * @param [in] node: The node handle.
 * @param [in] output_idx: The index of the output tensor.
 *
 * @return The tensor handle or NULL on error.
 *
 */
tensor_t get_node_output_tensor(node_t node, int output_idx);

/*!
 * @brief Set a node's the #idx input tensor.
 *
 * @param [in] graph: The graph handle.
 * @param [in] input_idx: The index of the input tensor.
 * @param [in] tesnor: The tensor handle.
 *
 * @return 0 on success or -1 on error.
 *
 */
int set_node_input_tensor(node_t node, int input_idx, tensor_t tensor);

/*!
 * @brief Set a node's the #idx output tensor.
 *
 * @param [in] graph: The graph handle.
 * @param [in] output_idx: The index of the output tensor.
 * @param [in] tensor: The tensor handle.
 * @param [in] tensor_type: The tensor type: VAR/CONST/INPUT/DEP
 *
 *  @return 0 on success or -1 on error.
 *
 */
int set_node_output_tensor(node_t node, int output_idx, tensor_t tensor, int tensor_type);

/*!
 * @brief Get the output tensor number of a node.
 *
 * @param [in] graph: The graph handle.
 * @param [in] node: The node hanle.
 *
 * @return >=1 the number of output tensor,
 *         -1 on error.
 *
 */

int get_node_output_number(node_t node);

/*!
 * @brief Get the input tensor number of a node.
 *
 * @param [in] graph: The graph handle.
 * @param [in] node: The node hanle.
 *
 * @return >=1 the number of output tensor,
 *         -1 on error.
 *
 */
int get_node_input_number(node_t node);

/*!
 * @brief Add an attribute to a node.
 *
 * @param [in] node: The target node handle.
 * @param [in] attr_name: The name of the attribute to be added.
 * @param [in] type_info: The pointer to the std::type_info of expected type
 *                       can be set to NULL to skip type match checking.
 * @param [in] size: The size of the attribute
 *
 * @return 0: Successfully,
 *         -1: Failed.
 */
int add_node_attr(node_t node, const char* attr_name, const void* type_info, int size);

/*!
 * @brief Get the attribute value (int) of a node
 *        backended by get_node_attr_generic().
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [out] attr_val: Pointer to the int val to be saved.
 *
 * @return 0: Retrieval value Successfully,
 *         -1: Failed, the name does not exist or the type mismatch.
 *
 */
int get_node_attr_int(node_t node, const char* attr_name, int* attr_val);

/*!
 * @brief Get the attribute value (float) of a node
 *        backended by get_node_attr_generic().
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [out] attr_val: Pointer to the float val to be saved.
 *
 * @return 0: Retrieval value Successfully,
 *         -1: Failed, the name does not exist or the type mismatch.
 */
int get_node_attr_float(node_t node, const char* attr_name, float* attr_val);

/*!
 * @brief Get the attribute value (pointer) of a node
 *        backended by get_node_attr_generic().
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [out] attr_val: Pointer to the pointer val to be saved.
 *
 * @return  0: Retrieval value Successfully,
 *         -1: Failed, the name does not exist or the type mismatch.
 */
int get_node_attr_pointer(node_t node, const char* attr_name, void* attr_val);

/*!
 * @brief Get the attribute value of a node, the data type is indicated by type_info.
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [in] type_info: The pointer to the std::type_info of expected type
 *                   can be set to NULL to skip type match checking.
 * @param [out] buf: The pointer to the buffer to save val.
 * @param [in] size: The buffer size.
 *
 * @return  0: Retrieval value Successfully,
 *         -1: Failed; The name does not exist or the type mismatch.
 *
 */
int get_node_attr_generic(node_t node, const char* attr_name, const void* type_info, void* buf, int size);

/*!
 * @brief Set the attribute value (int) of a node
 *        backended by set_node_attr_generic().
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [in] attr_val: The pointer to the int val to be set.
 *
 * @return  0: Retrieval value Successfully;
 *         -1: Failed, The name does not exist or the type mismatch.
 *
 */
int set_node_attr_int(node_t node, const char* attr_name, const int* attr_val);

/*!
 * @brief Set the attribute value (float) of a node
 *        backended by set_node_attr_generic().
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [in] attr_val: The pointer to the float val to be set.
 *
 * @return  0: Retrieval value Successfully.
 *         -1: Failed, The name does not exist or the type mismatch.
 *
 */
int set_node_attr_float(node_t node, const char* attr_name, const float* attr_val);

/*!
 * @brief Set the attribute value (pointer) of a node
 *        backended by set_node_attr_generic().
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [in] attr_val: The pointer to the pointer val to be set.
 *
 * @return  0: Retrieval value Successfully
 *         -1: Failed, The name does not exist or the type mismatch.
 *
 */
int set_node_attr_pointer(node_t node, const char* attr_name, const void* attr_val);

/*!
 * @brief Set the attribute value of a node, the data type is indicated by type_info.
 *
 * @param [in] node: The target node.
 * @param [in] attr_name: The name of the attribute to be retrieval.
 * @param [in] type_info: The pointer to the std::type_info of wanted type,
 *                   can be set to NULL to skip type match checking.
 * @param [in] buf: The pointer to the buffer to hold val.
 * @param [in] size: The buffer size.
 *
 * @return  0: Retrieval value Successfully.
 *         -1: Failed, The name does not exist or the type mismatch.
 *
 */
int set_node_attr_generic(node_t node, const char* attr_name, const void* type_info, const void* buf, int size);

/*!
 * @brief Set customer kernel of a node, on a specific device,
 *        the operate in kernel_ops must be the same as node's operate.
 *
 * @param [in] node: The node handle.
 * @param [in] device: The kernel works for which device. NULL means for default device.
 * @param [in] kernel_ops: The custom implemented kernel operates.
 *
 * @return 0: Success, -1: Fail.
 */
int set_custom_kernel(node_t node, const char* dev_name, struct custom_kernel_ops* kernel_ops);

/*!
 * @brief Remove customer kernel of a node, on a specific device.
 *
 * @param [in] node: The node handle.
 * @param [in] device: The kernel works for which device. NULL means for default device.
 *
 * @return 0: Success, -1: Fail.
 */
int remove_custom_kernel(node_t node, const char* dev_name);

/********************* Tensor operate set ***********************************/

/*!
 * @brief create a tensor handle by tensor name.
 *
 * @param [in] graph: The graph handle
 * @param [in] tensor_name: Tensor name.
 * @param [in] data_type:  the data type.
 *
 * @return The tensor handle or NULL on error.
 *
 */
tensor_t create_graph_tensor(graph_t graph, const char* tensor_name, int data_type);

/*!
 * @brief Get a tensor handle by tensor name.
 *
 * @param [in] graph: The graph handle.
 * @param [in] tensor_name: Tensor name.
 *
 * @return The tensor handle or NULL on error.
 *
 */
tensor_t get_graph_tensor(graph_t graph, const char* tensor_name);

/*!
 * @brief  Get the name of the tensor handle.
 *
 * @param [in] tensor: the tensor handle.
 *
 * @return A c string.

 */
const char* get_tensor_name(tensor_t tensor);

/*!
 * @brief Release the tensor handle.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return None.
 */
void release_graph_tensor(tensor_t tensor);

/*!
 * @brief Get the shape of tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [out] dims: An int array to get the returned shape.
 * @param [in] dim_number: The array size.
 * @return >=1 the valid dim number, or -1 Fail.
 *
 */
int get_tensor_shape(tensor_t tensor, int dims[], int dim_number);

/*!
 * @brief Set the shape of tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] dims: An int array to represent shape.
 * @param [in] dim_number: The array size.
 * @return 0: Success; -1: Fail.
 *
 */
int set_tensor_shape(tensor_t tensor, const int dims[], int dim_number);

/*!
 * @brief Get the byte size of a tensor should occupy.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return <0: Error; >=0: Tensor size.
 * @note   If return 0, it means the shape of the tensor is not set yet.
 */
int get_tensor_buffer_size(tensor_t tensor);

/*!
 * @brief Get the buffer of the tensor.
 *    A tensor may deny to expose its internal buffer, so that get_tensor_buffer()
 *    will fail but get_tensor_buffer_size()/set_tensor_data() succeed.
 *
 * @param [in] tensor: The tensor handle.
 * @return The buffer address. if no buffer allocated return NULL.
 */
void* get_tensor_buffer(tensor_t tensor);

/*!
 * @brief Set the buffer of the tensor.
 *    A tensor may deny to change its internal buffer setting.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] buffer: The buffer address.
 * @param [in] buffer_size: The buffer_size.
 *
 * @return 0: Success; -1: Fail.
 * @note  The buffer is still managed by caller.
 */
int set_tensor_buffer(tensor_t tensor, void* buffer, int buffer_size);

/*!
 * @brief Copy tensor data to the output data buffer.
 * @param [in] tensor: The tensor handle.
 * @param [out] output_data: The output data buffer.
 *
 * @param [in] data_size: the output buffer size.
 * @return 0: Success; or -1: Fail.
 *
 */
int get_tensor_data(tensor_t tensor, void* output_data, int data_size);

/*!
 * @brief Copy the data to tensor buffer.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] input_data: The input data.
 * @param [in] data_size: The input data size.
 *
 * @return 0: Success; -1: Fail.
 *
 */
int set_tensor_data(tensor_t tensor, const void* input_data, int data_size);

/*!
 * @brief Get the data type of the tensor.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return The tensor type, see TENGINE_DT_FP32 etc, -1 on error.
 */
int get_tensor_data_type(tensor_t tensor);

/*!
 * @brief Set the data type of the tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] data_type: The data type. see TENGINE_DT_FP32 etc.
 *
 * @return 0 on sucess, -1 on error.
 */
int set_tensor_data_type(tensor_t tensor, int data_type);

/*!
 * @brief Set tensor quant parameters
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] scale:  The scale array address.
 * @param [in] zero point: The zero point address.
 * @param [in] number:  The element number of array.
 *
 * @return 0 on sucess, -1 on error.
 */
int set_tensor_quant_param(tensor_t tensor, const float* scale, const int* zero_point, int number);

/*!
 * @brief Get tensor quant parameters.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] scale:  The scale array address.
 * @param [in] zero point: The zero point address.
 * @param [in] number:  The element number of array.
 *
 * @return 0 on sucess, -1 on error.
 */

int get_tensor_quant_param(tensor_t tensor, float* scale, int* zero_point, int number);

/************************** Graph run related interface *********************/

/*!
 * @brief The interface to set some proprietary attribute items for graph.
 *        The backend device to run the graph may use the attribute item.
 *
 * @param [in] graph: The graph handle.
 * @param [in] attr_name: The attribute name.
 * @param [in] buf: The buffer to hold data.
 * @param [in] size: The buffer size.
 *
 * @return 0: Success, -1: Fail.
 */
int set_graph_attr(graph_t graph, const char* attr_name, const void* buf, int size);

/*!
 * @brief The interface to get some proprietary config items for graph.
 *        It is probabaly the config will be passed to the DLA driver.
 * @param [in] graph: the graph handle.
 * @param [in] name: The attribute name.
 * @param [in] buf: The buffer to hold data.
 * @param [in] size: The buffer size.
 *
 * @return 0: Success, -1: Fail.
 *
 */
int get_graph_attr(graph_t graph, const char* attr_name, void* buf, int size);

/*!
 * @brief Set the gradient descent method.
 *
 * @param [in] graph: The graph handle.
 * @param [in] gd_methd: The gradient descent.
 * @param [in] parameters: For each gd_method.
 *
 * @return 0: Success, -1: Fail.
 *
 */
int set_graph_gd_method(graph_t graph, int gd_method, ...);

/*!
 * @brief Initialize resource for graph execution.
 *
 * @param [in] graph: The graph handle.
 *
 * @return 0: Success, -1: Fail.
 *
 */
int prerun_graph(graph_t graph);

/*!
 * @brief Execute graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] block: Blocking or nonlocking.
 * @return 0: Success, -1: Fail.
 * @note  If block is 0, need to call wait_graph to get result or set GRAPH_DONE event hook.
 *
 */
int run_graph(graph_t graph, int block);

/*!
 * @brief Wait graph execution done.
 *
 * @param [in] graph: The graph handle.
 * @param [in] try_wait: If set, just check status and return.
 * @return  1: Graph is done.
 *          0: Try again.
 *
 */
int wait_graph(graph_t graph, int try_wait);

/*!
 * @brief Release the resource for graph execution.
 * @param [in] graph: graph handle.
 *
 * @return 0: Success, -1: Fail.
 */
int postrun_graph(graph_t graph);

/*!
 * @brief Get the status of graph execution.
 *
 * @param [in] graph: The graph handle.
 *
 * @return status
 */
int get_graph_exec_status(graph_t graph);

/*!
 * @brief Set the event hook for graph execution.
 *
 * @param [in] graph: The graph handle.
 * @param [in] event: The event to be hooked.
 * @param [in] cb_func: The callback funtion.
 * @param [in] cb_arg: The argument will be passed to callback function.
 * @return 0: Success, -1: Fail.
 *
 */
int set_graph_event_hook(graph_t graph, int event, event_handler_t cb_func, void* cb_arg);

/***************** Device related *****************************/

/*!
 * @set The default device.
 *
 * @param [in] device: The device name.
 * @return 0: valid, -1: invalid.
 *
 */
int set_default_device(const char* device);

/*!
 * @brief Set the device to execution a graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] dev_name: The device name to run the node.
 *
 * @return  =0: Bind success.
 *          <0: error.
 *
 */
int set_graph_device(graph_t graph, const char* dev_name);

/*!
 * @brief Set the device to execution a node.
 *
 * @param [in] node: The node handle.
 * @param [in] dev_name: The device name to run the node.
 *
 * @return =0: Bind ok.
 *         <0: Fail
 */
int set_node_device(node_t node, const char* dev_name);

/*!
 * @brief get the device the node runs on
 *
 * @param [in] node: the node handle
 *
 * @return the device name or NULL if no device assigned yet
 */

const char* get_node_device(node_t node);

/*!
 * @brief Enable dump function pre-defined on device on a node,
 *        the dump buffer will be returned by get_node_tensor_dump()
 *
 * @param [in] node: The node handle.
 * @param [in] action: 1 enable, 0 disable.
 *
 * @return 0 success, or -1 on error.
 */

int do_node_dump(node_t node, int action);

/*!
 * @brief Get the dump buffer pointer generated by target device
 *        exact meaning of the dump buffer is decided by device.
 *        If the return number equals buf_size, it is possible that some buffers
 *        do not be retrieved yet.
 *
 * @param [in]  node: The node handle
 * @param [out] buf: the pointer array to the dump buffers
 * @param [in]  buf_size: the pointer array size
 *
 * @return The number of retrieved buffer, or -1 on fail
 *
 */

int get_node_dump_buffer(node_t node, void** buf, int buf_size);

/*!
 * @brief Start or stop the perf stats
 *
 * @param [in] graph: the graph handle
 * @param [in] action: 0 stop, 1 start, 2 reset counter
 *
 * @return 0 success, -1 fail
 */

int do_graph_perf_stat(graph_t graph, int action);

/*!
 * @brief get graph performance stats records
 *        If the returned number equals buf_size, there may be som records do not be
 *        retrieved yet
 *
 * @param [in] graph: the graph handle
 * @param [out] buf: the pointer array to struct perf_info  buffer
 * @param [in] buf_size: the number of record pointer can be stored in buf
 *
 * @return the number of record retrieved or -1 on fail
 */

int get_graph_perf_stat(graph_t graph, struct perf_info** buf, int buf_size);

/*!
 * @brief Get the device number in the system.
 *
 * @return The number of device.
 */
int get_device_number(void);

/*!
 * @brief Get the device name by specific index.
 *
 * @param [in] idx: The device index.
 *
 * @return the name of the device.
 */
const char* get_device_name(int idx);

/*!
 * @brief Get the default name of device.
 *
 * @return The name of the default device.
 */

const char* get_default_device(void);

/*!
 * @brief Create device, for predefined device but driver does not auto probed device.
 *
 * @param [in] driver_name: The driver name.
 * @param [in] dev_name: The device name.
 * @return =0: Success.
 *         <0: Fail.
 */
int create_device(const char* driver_name, const char* dev_name);

/*!
 * @brief Destroy device, for predefined device but driver does not auto probed device.
 *
 * @param [in] driver_name: The driver name.
 * @param [in] dev_name: The device name.
 *
 * @return =0: Success.
 *         <0: Fail.
 */
int destroy_device(const char* driver_name, const char* dev_name);

/*!
 * @brief Set the device working policy.
 *
 * @param [in] device_name: The device name.
 * @param [in] policy: The device working mode.
 *
 * @return 0: Success, -1: Fail.
 */
int set_device_policy(const char* device_name, device_policy policy);

/*!
 * @brief Get the device working mode.
 *
 * @param [in] device_name: The device name.
 * @return >=0: The mode, -1: Fail.
 */
int get_device_policy(const char* device_name);

/*!
 * @brief Get the config setting by config name. the config request may be passed to driver.
 *
 * @param [in] device_name: The device name.
 * @param [in] attr_name: The attribute name.
 * @param [out] val: The buffer to hold the data.
 * @param [in] size: The buffer size.
 * @return 0: Success, -1: Fail.
 */
int get_device_attr(const char* device_name, const char* attr_name, void* val, int size);

/*!
 * @brief Set the config item of the device. The config item may be passed to driver.
 *
 * @param [in] device_name: The device name.
 * @param [in] attr_name: The config item name.
 * @param [in] val: The buffer to hold the data to be set.
 * @param [in] size: The buffer size.
 * @return 0: Success, -1: Fail.
 */
int set_device_attr(const char* device_name, const char* attr_name, void* val, int size);

/******************** execution context *****************************/

/*!
 * @brief Create one execution context with name.
 *
 * @param [in] context_name: The name of the created context.
 * @param [in] empty_context: No device is assigned with this context
 *                            otherwise, all proved devices will be added into the context.
 *
 * @return Execution context handle.
 *         If create Failed, return NULL.
 */
context_t create_context(const char* context_name, int empty_context);

/*!
 * @brief Destory and reclaim the resource related with the context.
 *
 * @param [in] context: The context handle.
 */
void destroy_context(context_t context);

/*!
 * @brief Get the device number assigned to a context.
 *
 * @param [in] context: The context handle.
 *
 * @return The number of devices inside the context.
 */

int get_context_device_number(context_t context);

/*!
 * @brief Get the name of the idx device in a context.
 *
 * @param [in] context: The context handle.
 * @param [in] idx: The device idx.
 *
 * @return  The name of device or NULL.
 */

const char* get_context_device_name(context_t context, int idx);

/*!
 *  @brief Add a device into one context.
 *
 *  @param [in] context: The context handle.
 *  @param [in] dev_name: The device name.
 *
 *  @return 0: Success, -1: Fail.
 */
int add_context_device(context_t context, const char* dev_name);

/*!
 *  @brief Remove a device from one context.
 *
 *  @param [in] context: The context handle.
 *  @param [in] dev_name: The device name.
 *
 *  @return 0: Success, -1: Fail.
 */
int remove_context_device(context_t context, const char* dev_name);

/*!
 * @brief Set attribute item of a context.
 *
 * @param [in] context: The context handle.
 * @param [in] attr_name: The attribute item name.
 * @param [in] val: The buffer to hold the data to set.
 * @param [in] size: The buffer size.
 * @return 0: Success, -1: Fail.
 */
int set_context_attr(context_t context, const char* attr_name, const void* val, int val_size);

/*!
 * @brief Get the attribute item of a context.
 *
 * @param [in] context: The context handle.
 * @param [in] attr_name: The attribute item name.
 * @param [out] val: The buffer to hold the data.
 * @param [in] size: The buffer size.
 * @return 0: Succuess, -1: Fail.
 */
int get_context_attr(context_t context, const char* attr_name, void* val, int val_size);

/*
 * Misc API
 */

/*!
 * @brief return the error number
 *        list of the symbolic error name follows glibc definitions
 *
 * @return the last error set in library
 *
 * @note It is MT-safe
 */

int get_tengine_errno(void);

/*!
 * @brief Set the logger level.
 *
 * @param [in] level: The log level.
 */
void set_log_level(enum log_level level);

/*!
 * @brief set the print function of log.
 *
 * @param [in] func: The print function.
 *
 * @return None.
 *
 * @note  default log output is stdout
 */

void set_log_output(log_print_t func);

/*!
 * @brief Dump the run-time graph.
 *        If the graph is dumpped after prerun(), it will dump the optimized graph instead of the origin one.
 *
 * @param [in] graph: The graph handle.
 */
void dump_graph(graph_t graph);

/**************************** Plug-in operate set *******************/
/*!
 * @brief Load one plugin from disk, and execute the init function.
 *
 * @param [in] plugin_name: Plugin name.
 * @param [in] fname: Plugin file name.
 * @param [in] init_func_name: The name of the init function.
 *
 * @return 0: Plugin loaded and inited Success,
 *      -1: Fail
 */
int load_tengine_plugin(const char* plugin_name, const char* fname, const char* init_func_name);

/*!
 * @brief Unload one plugin and call the release function.
 *
 * @param [in] plugin_name: The name of plugin.
 * @param [in] rel_func_name: The release function name.
 *
 * @return  0: Success;
 *      -1: Fail.
 */
int unload_tengine_plugin(const char* plugin_name, const char* rel_func_name);

/*!
 * @brief Get the number of loaded plugin.
 *
 * @return The plugin number.
 */
int get_tengine_plugin_number(void);

/*!
 * @brief Get the name of #idx plugin.
 *
 * @param [in] idx: The index of loaded plugin.
 *
 * @return The name of plugin.
 */
const char* get_tengine_plugin_name(int idx);

#ifdef __cplusplus
}
#endif

#ifdef CONFIG_LEGACY_API
#include "tengine_c_compat.h"
#endif

#endif
