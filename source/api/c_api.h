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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#if defined __GNUC__
#define DLLEXPORT __attribute((visibility("default")))
#elif defined(_MSC_VER)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#if defined __GNUC__
#define DEPRECATED_BEFORE
#define DEPRECATED_AFTER __attribute__((deprecated))
#elif defined(_MSC_VER)
#pragma deprecated()
#define DEPRECATED_BEFORE __declspec(deprecated)
#define DEPRECATED_AFTER
#else
#define DEPRECATED_BEFORE
#define DEPRECATED_AFTER
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SHAPE_DIM_NUM 8

/* the data type of the tensor */
#define TENGINE_DT_FP32  0
#define TENGINE_DT_FP16  1
#define TENGINE_DT_INT8  2
#define TENGINE_DT_UINT8 3
#define TENGINE_DT_INT32 4
#define TENGINE_DT_INT16 5

/* layout type, not real layout */
#define TENGINE_LAYOUT_NCHW 0
#define TENGINE_LAYOUT_NHWC 1

/* tensor type: the content changed or not during inference */
#define TENSOR_TYPE_UNKNOWN 0
#define TENSOR_TYPE_VAR     1
#define TENSOR_TYPE_CONST   2
#define TENSOR_TYPE_INPUT   3
#define TENSOR_TYPE_DEP     4

/* cluster type: big-LITTLE and DynamIQ defined */
#define TENGINE_CLUSTER_ALL    0
#define TENGINE_CLUSTER_BIG    1
#define TENGINE_CLUSTER_MEDIUM 2
#define TENGINE_CLUSTER_LITTLE 3

#define TENGINE_MODE_FP32        0
#define TENGINE_MODE_FP16        1
#define TENGINE_MODE_HYBRID_INT8 2
#define TENGINE_MODE_UINT8       3
#define TENGINE_MODE_INT8        4

/* node dump action definition */
#define NODE_DUMP_ACTION_DISABLE 0
#define NODE_DUMP_ACTION_ENABLE  1
#define NODE_DUMP_ACTION_START   2
#define NODE_DUMP_ACTION_STOP    3
#define NODE_DUMP_ACTION_GET     4

/* graph perf action definition */
#define GRAPH_PERF_STAT_DISABLE 0
#define GRAPH_PERF_STAT_ENABLE  1
#define GRAPH_PERF_STAT_STOP    2
#define GRAPH_PERF_STAT_START   3
#define GRAPH_PERF_STAT_RESET   4
#define GRAPH_PERF_STAT_GET     5

/* follow the std. UNIX log level definition */
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

/* TODO: should add suspend? */
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

/* graph exec options */
typedef struct options
{
    int num_thread;
    int cluster;
    int precision;
    uint64_t affinity;
} options_t;

struct custom_kernel_tensor
{
    int dim[MAX_SHAPE_DIM_NUM]; /* the shape dim array */
    int dim_num;                /* valid entry number */
    int element_num;
    int element_size; /* determined  by data_type */
    int data_type;
    int dev_type;    /* indicate the tensor belongs to CPU/GPU ... */
    int layout_type; /*  NCHW type or NHWC type*/

    /* quant info */
    int quant_type; /* int8, int16 or int32 */
    float* scale;
    int* zero_point;
    int* quant_number;

    void* data;       /* pointer to host memory (virtual address) */
    void* dev_mem;    /* refers to device memory block */
    void* mapped_mem; /* the mapped address for device memory block */
};

/* For user to add user defined kernel*/
struct custom_kernel_ops
{
    const char* kernel_name; /* name of the kernel */
    const char* op;          /* name of the op to be implemented */
    int force;               /* if not set, when bind() failed,
      try to use other kernel implementations*/
    void* kernel_param;      /* used for kernel impl functions */
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
    int (*inplace_info)(struct custom_kernel_ops* ops, int output_idx); // optional

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
DLLEXPORT int init_tengine(void);

/*!
 * @brief Release the tengine, only can be called once.
 *
 * @return none.
 */
DLLEXPORT void release_tengine(void);

/*!
 * @brief Get the version of the tengine.
 *
 * @return const char * of version string.
 */
DLLEXPORT const char* get_tengine_version(void);

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
DLLEXPORT int request_tengine_version(const char* version);

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

DLLEXPORT graph_t create_graph(context_t context, const char* model_format, const char* file_name, ...);

/*!
 * @brief Set the layout type of the graph
 *        the default layout of graph is NCHW
 * @param [in] graph, the graph handle
 * @param [in] layout_type, the layout type NCHW or NHWC
 *
 * @return 0 success, or -1 fail
 */

DLLEXPORT int set_graph_layout(graph_t graph, int layout_type);

/*!
 * @brief designate the input nodes of the graph
 *
 * @param [in] graph: the graph handle
 * @param [in] input_nodes: the node name list of input nodes
 * @param [in] input_number: the number of input_nodes
 *
 * @note  if using the default input nodes of a graph, this call can be skipped
 */
DLLEXPORT int set_graph_input_node(graph_t graph, const char* input_nodes[], int input_number);

/*!
 * @brief designate the output nodes of the graph
 *
 * @param [in] graph: the graph handle
 * @param [in] output_nodes: the node name list of output nodes
 * @param [in] output_number: the number of output_nodes
 *
 * @note  if using the default output nodes of a graph, this call can be skipped
 */

DLLEXPORT int set_graph_output_node(graph_t graph, const char* output_nodes[], int output_number);

/*!
 * @brief Destroy the runtime graph and release allocated resource.
 *
 * @param [in] graph: The graph handle.
 * @return 0: Success, -1: Fail.
 */
DLLEXPORT int destroy_graph(graph_t graph);

/*!
 * @brief Get the number of input node of the graph.
 *
 *  @param [in] graph The graph handle.
 *  @return <0 Fail, >0 the input node number.
 */
DLLEXPORT int get_graph_input_node_number(graph_t graph);

/*!
 * @brief Get the node handle of #idx of input node of the graph.
 *
 * @param [in] graph The graph handle.
 * @param [in] idx The input node index,starting from zero.
 *
 * @return The node name or NULL on error.
 */
DLLEXPORT node_t get_graph_input_node(graph_t graph, int idx);

/*!
 * @brief Get the number of output node of the graph.
 *
 *  @param [in] graph The graph handle.
 *
 *  @return <0 error, >0 the input node number.
 */
DLLEXPORT int get_graph_output_node_number(graph_t graph);

/*!
 * @brief Get the node handle #idx of a graph output node.
 *
 * @param [in] graph The graph handle.
 * @param [in] idx The input node index, starting from zero.
 *
 * @return The node name or NULL on error.
 */
DLLEXPORT node_t get_graph_output_node(graph_t graph, int idx);

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
DLLEXPORT tensor_t get_graph_output_tensor(graph_t graph, int output_node_idx, int tensor_idx);

/*!
 * @brief Get tensor handle of one graph input tensor.
 *
 * @param [in] graph: The graph handle.
 * @param [in] input_node_idx: The input node index, starting from zero.
 * @param [in] tensor_idx: The output tensor index of the input node, starting from zero.
 *
 * @return The tensor handle or NULL on error.
 */
DLLEXPORT tensor_t get_graph_input_tensor(graph_t graph, int input_node_idx, int tensor_idx);

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
DLLEXPORT node_t create_graph_node(graph_t graph, const char* node_name, const char* op_name);

/*!
 * @brief  Get the node handle of the graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] node_name: The name of the node.
 *
 * @return The node handle or NULL on error.
 */
DLLEXPORT node_t get_graph_node(graph_t graph, const char* node_name);

/*!
 * @brief  Get the node handle of the graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] idx: The index of the node.
 *
 * @return The node handle or NULL on error.
 */
DLLEXPORT node_t get_graph_node_by_idx(graph_t graph, int idx);

/*!
 * @brief  Get the node handle of the graph.
 *
 * @param [in] graph: The graph handle.
 *
 * @return The quantities of all nodes.
 */
DLLEXPORT int get_graph_node_num(graph_t graph);

/*!
 * @brief Get the node name.
 *
 * @param [in] node: The node handle.
 *
 * @return The node name, NULL on error.
 */
DLLEXPORT const char* get_node_name(node_t node);

/*!
 * @brief Get the node op.
 *
 * @param [in] node: The node handle.
 *
 * @return The op name, NULL on error.
 */
DLLEXPORT const char* get_node_op(node_t node);

/*!
 * @brief  Release the node handle.
 *
 * @param  [in] node: The node handle.
 *
 * @return None.
 */
DLLEXPORT void release_graph_node(node_t node);

/*!
 * @brief Get the input tensor handle of a node.
 *
 * @param [in] node: The node handle.
 * @param [in] input_idx: The index of the input tensor.
 * @return The tensor handle or NULL on error.
 *
 */
DLLEXPORT tensor_t get_node_input_tensor(node_t node, int input_idx);

/*!
 * @brief Get the output tensor handle of a node.
 *
 * @param [in] node: The node handle.
 * @param [in] output_idx: The index of the output tensor.
 *
 * @return The tensor handle or NULL on error.
 *
 */
DLLEXPORT tensor_t get_node_output_tensor(node_t node, int output_idx);

/*!
 * @brief Set a node's the #idx input tensor.
 *
 * @param [in] node: The node handle.
 * @param [in] input_idx: The index of the input tensor.
 * @param [in] tesnor: The tensor handle.
 *
 * @return 0 on success or -1 on error.
 *
 */
DLLEXPORT int set_node_input_tensor(node_t node, int input_idx, tensor_t tensor);

/*!
 * @brief Set a node's the #idx output tensor.
 *
 * @param [in] node: The node handle.
 * @param [in] output_idx: The index of the output tensor.
 * @param [in] tensor: The tensor handle.
 * @param [in] tensor_type: The tensor type: VAR/CONST/INPUT/DEP
 *
 *  @return 0 on success or -1 on error.
 *
 */
DLLEXPORT int set_node_output_tensor(node_t node, int output_idx, tensor_t tensor, int tensor_type);

/*!
 * @brief Get the output tensor number of a node.
 *
 * @param [in] node: The node hanle.
 *
 * @return >=1 the number of output tensor,
 *         -1 on error.
 *
 */
DLLEXPORT int get_node_output_number(node_t node);

/*!
 * @brief Get the input tensor number of a node.
 *

 * @param [in] node: The node hanle.
 *
 * @return >=1 the number of output tensor,
 *         -1 on error.
 *
 */
DLLEXPORT int get_node_input_number(node_t node);

/*!
 * @brief Add an attribute to a node.
 *
 * @param [in] node: The target node handle.
 * @param [in] attr_name: The name of the attribute to be added.
 * @param [in] type_name: The name of the std::type_info of the type,
 *                       can be set to NULL to skip type match checking.
 * @param [in] size: The size of the attribute
 *
 * @return 0: Successfully,
 *         -1: Failed.
 */
DLLEXPORT int add_node_attr(node_t node, const char* attr_name, const char* type_name, int size);

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
DLLEXPORT DEPRECATED_BEFORE int get_node_attr_int(node_t node, const char* attr_name, int* attr_val) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int get_node_attr_float(node_t node, const char* attr_name, float* attr_val) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int get_node_attr_pointer(node_t node, const char* attr_name, void* attr_val) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int get_node_attr_generic(node_t node, const char* attr_name, const char* type_name, void* buf, int size) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_node_attr_int(node_t node, const char* attr_name, const int* attr_val) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_node_attr_float(node_t node, const char* attr_name, const float* attr_val) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_node_attr_pointer(node_t node, const char* attr_name, const void* attr_val) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_node_attr_generic(node_t node, const char* attr_name, const char* type_name, const void* buf, int size) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_custom_kernel(node_t node, const char* dev_name, struct custom_kernel_ops* kernel_ops) DEPRECATED_AFTER;

/*!
 * @brief Remove customer kernel of a node, on a specific device.
 *
 * @param [in] node: The node handle.
 * @param [in] device: The kernel works for which device. NULL means for default device.
 *
 * @return 0: Success, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int remove_custom_kernel(node_t node, const char* dev_name) DEPRECATED_AFTER;

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
DLLEXPORT tensor_t create_graph_tensor(graph_t graph, const char* tensor_name, int data_type);

/*!
 * @brief Get a tensor handle by tensor name.
 *
 * @param [in] graph: The graph handle.
 * @param [in] tensor_name: Tensor name.
 *
 * @return The tensor handle or NULL on error.
 *
 */
DLLEXPORT tensor_t get_graph_tensor(graph_t graph, const char* tensor_name);

/*!
 * @brief  Get the name of the tensor handle.
 *
 * @param [in] tensor: the tensor handle.
 *
 * @return A c string.

 */
DLLEXPORT const char* get_tensor_name(tensor_t tensor);

/*!
 * @brief Release the tensor handle.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return None.
 */
DLLEXPORT void release_graph_tensor(tensor_t tensor);

/*!
 * @brief Get the shape of tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [out] dims: An int array to get the returned shape.
 * @param [in] dim_number: The array size.
 * @return >=1 the valid dim number, or -1 Fail.
 *
 */
DLLEXPORT int get_tensor_shape(tensor_t tensor, int dims[], int dim_number);

/*!
 * @brief Set the shape of tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] dims: An int array to represent shape.
 * @param [in] dim_number: The array size.
 * @return 0: Success; -1: Fail.
 *
 */
DLLEXPORT int set_tensor_shape(tensor_t tensor, const int dims[], int dim_number);

/*!
 * @brief Get the byte size of a tensor should occupy.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return <0: Error; >=0: Tensor size.
 * @note   If return 0, it means the shape of the tensor is not set yet.
 */
DLLEXPORT int get_tensor_buffer_size(tensor_t tensor);

/*!
 * @brief Get the buffer of the tensor.
 *    A tensor may deny to expose its internal buffer, so that get_tensor_buffer()
 *    will fail but get_tensor_buffer_size()/set_tensor_data() succeed.
 *
 * @param [in] tensor: The tensor handle.
 * @return The buffer address. if no buffer allocated return NULL.
 */
DLLEXPORT void* get_tensor_buffer(tensor_t tensor);

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
DLLEXPORT int set_tensor_buffer(tensor_t tensor, void* buffer, int buffer_size);

/*!
 * @brief Copy tensor data to the output data buffer.
 * @param [in] tensor: The tensor handle.
 * @param [out] output_data: The output data buffer.
 *
 * @param [in] data_size: the output buffer size.
 * @return 0: Success; or -1: Fail.
 *
 */
DLLEXPORT int get_tensor_data(tensor_t tensor, void* output_data, int data_size);

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
DLLEXPORT int set_tensor_data(tensor_t tensor, const void* input_data, int data_size);

/*!
 * @brief Get the data type of the tensor.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return The tensor type, see TENGINE_DT_FP32 etc, -1 on error.
 */
DLLEXPORT int get_tensor_data_type(tensor_t tensor);

/*!
 * @brief Set the data type of the tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] data_type: The data type. see TENGINE_DT_FP32 etc.
 *
 * @return 0 on sucess, -1 on error.
 */
DLLEXPORT int set_tensor_data_type(tensor_t tensor, int data_type);

/*!
 * @brief Set the data layout of the tensor.
 *
 * @param [in] tensor: The tensor handle.
 *
 * @return The tensor type, 0 : nchw, 1 : nhwc.
 */
DLLEXPORT int get_tensor_layout(tensor_t tensor);

/*!
 * @brief Set the data layout of the tensor.
 *
 * @param [in] tensor: The tensor handle.
 * @param [in] layout: The data layout, 0 : nchw, 1 : nhwc.
 *
 * @return 0 on sucess, -1 on error.
 */
DLLEXPORT int set_tensor_layout(tensor_t tensor, int layout);

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
DLLEXPORT int set_tensor_quant_param(tensor_t tensor, const float* scale, const int* zero_point, int number);

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
DLLEXPORT int get_tensor_quant_param(tensor_t tensor, float* scale, int* zero_point, int number);

/************************** Graph run related interface *********************/

/*!
 * @brief The interface to get possible cpu mask bits, when specified cluster of cpu,
 *        function will return the mask bits of the cluster.
 *
 * @param [in] cluster: The be queried cluster.
 *
 * @return affinity mask.
 */
DLLEXPORT size_t get_cluster_affinity_mask(int cluster);

/*!
 * @brief The interface to set cluster and threads count will used.
 *
 * @param [in] graph: The graph handle.
 * @param [in] cluster: The wanted cluster of all cpu clusters.
 * @param [in] threads: The threads count of graph will be used to run.
 *
 * @return 0: Success, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int set_graph_thread(graph_t graph, int cluster, int threads) DEPRECATED_AFTER;

/*!
 * @brief The interface to directly set used cpu mask.
 *
 * @param [in] graph: The graph handle.
 * @param [in] cpu_mask: The mask bits of graph will be used to run.
 *
 * @return 0: Success, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int set_graph_thread_mask(graph_t graph, size_t cpu_mask) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_graph_attr(graph_t graph, const char* attr_name, const void* buf, int size) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int get_graph_attr(graph_t graph, const char* attr_name, void* buf, int size) DEPRECATED_AFTER;

/*!
 * @brief Initialize resource for graph execution, and set cluster and threads count will used.
 *
 * @param [in] graph: The graph handle.
 * @param [in] opt: The runtime options.
 *
 * @return 0: Success, -1: Fail.
 *
 */
DLLEXPORT int prerun_graph_multithread(graph_t graph, struct options opt);

/*!
 * @brief Initialize resource for graph execution.
 *
 * @param [in] graph: The graph handle.
 *
 * @return 0: Success, -1: Fail.
 *
 */
DLLEXPORT int prerun_graph(graph_t graph);

/*!
 * @brief Execute graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] block: Blocking or nonlocking.
 * @return 0: Success, -1: Fail.
 * @note  If block is 0, need to call wait_graph to get result or set GRAPH_DONE event hook.
 *
 */
DLLEXPORT int run_graph(graph_t graph, int block);

/*!
 * @brief Wait graph execution done.
 *
 * @param [in] graph: The graph handle.
 * @param [in] try_wait: If set, just check status and return.
 * @return  1: Graph is done.
 *          0: Try again.
 *
 */
DLLEXPORT int wait_graph(graph_t graph, int try_wait);

/*!
 * @brief Release the resource for graph execution.
 * @param [in] graph: graph handle.
 *
 * @return 0: Success, -1: Fail.
 */
DLLEXPORT int postrun_graph(graph_t graph);

/*!
 * @brief Get the status of graph execution.
 *
 * @param [in] graph: The graph handle.
 *
 * @return status
 */
DLLEXPORT DEPRECATED_BEFORE int get_graph_exec_status(graph_t graph) DEPRECATED_AFTER;

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
DLLEXPORT DEPRECATED_BEFORE int set_graph_event_hook(graph_t graph, int event, event_handler_t cb_func, void* cb_arg) DEPRECATED_AFTER;

/***************** Device related *****************************/

/*!
 * @set The default device.
 *
 * @param [in] device: The device name.
 * @return 0: valid, -1: invalid.
 *
 */
DLLEXPORT DEPRECATED_BEFORE int set_default_device(const char* device) DEPRECATED_AFTER;

/*!
 * @brief Set the device to execute a graph.
 *
 * @param [in] graph: The graph handle.
 * @param [in] dev_name: The device name to run the node.
 *
 * @return  =0: Bind success.
 *          <0: error.
 *
 */
DLLEXPORT int set_graph_device(graph_t graph, const char* dev_name);

/*!
 * @brief get the device the node runs on
 *
 * @param [in] node: the node handle
 *
 * @return the device name or NULL if no device assigned yet
 */
DLLEXPORT DEPRECATED_BEFORE const char* get_node_device(node_t node) DEPRECATED_AFTER;

/*!
 * @brief Get the default name of device.
 *
 * @return The name of the default device.
 */
DLLEXPORT const char* get_default_device(void);

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
DLLEXPORT context_t create_context(const char* context_name, int empty_context);

/*!
 * @brief Destory and reclaim the resource related with the context.
 *
 * @param [in] context: The context handle.
 */
DLLEXPORT void destroy_context(context_t context);

/*!
 * @brief Get the device number assigned to a context.
 *
 * @param [in] context: The context handle.
 *
 * @return The number of devices inside the context.
 */
DLLEXPORT DEPRECATED_BEFORE int get_context_device_number(context_t context) DEPRECATED_AFTER;

/*!
 *  @brief Add a device into one context.
 *
 *  @param [in] context: The context handle.
 *  @param [in] dev_name: The device name.
 *  @param [in ...: optional params.
 *      @remarks void*  option: address of device option
 *      @remarks size_t size: size of struct option
 *
 *  @return 0: Success, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int add_context_device(context_t context, const char* dev_name) DEPRECATED_AFTER;

/*!
 *  @brief Set device for one context.
 *
 *  @param [in] context: The context handle.
 *  @param [in] dev_name: The device name.
 *  @param [in ...: optional params.
 *      @remarks void*  option: address of device option
 *      @remarks size_t size: size of struct option
 *
 *  @return 0: Success, -1: Fail.
 */
DLLEXPORT int set_context_device(context_t context, const char* dev_name, const void* dev_option, size_t dev_opt_size);

/*!
 *  @brief Remove a device from one context.
 *
 *  @param [in] context: The context handle.
 *  @param [in] dev_name: The device name.
 *
 *  @return 0: Success, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int remove_context_device(context_t context, const char* dev_name) DEPRECATED_AFTER;

/*!
 * @brief Set attribute item of a context.
 *
 * @param [in] context: The context handle.
 * @param [in] attr_name: The attribute item name.
 * @param [in] val: The buffer to hold the data to set.
 * @param [in] size: The buffer size.
 * @return 0: Success, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int set_context_attr(context_t context, const char* attr_name, const void* val, int val_size) DEPRECATED_AFTER;

/*!
 * @brief Get the attribute item of a context.
 *
 * @param [in] context: The context handle.
 * @param [in] attr_name: The attribute item name.
 * @param [out] val: The buffer to hold the data.
 * @param [in] size: The buffer size.
 * @return 0: Succuess, -1: Fail.
 */
DLLEXPORT DEPRECATED_BEFORE int get_context_attr(context_t context, const char* attr_name, void* val, int val_size) DEPRECATED_AFTER;

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

DLLEXPORT DEPRECATED_BEFORE int get_tengine_errno(void) DEPRECATED_AFTER;

/*!
 * @brief return and clear the error number
 *        list of the symbolic error name follows glibc definitions
 *
 * @return the last error set in library
 *
 * @note It is MT-safe
 */

DLLEXPORT DEPRECATED_BEFORE int clr_tengine_errno(void) DEPRECATED_AFTER;

/*!
 * @brief Set the logger level.
 *
 * @param [in] level: The log level.
 */
DLLEXPORT void set_log_level(enum log_level level);

/*!
 * @brief set the print function of log.
 *
 * @param [in] func: The print function.
 *
 * @return None.
 *
 * @note  default log output is stdout
 */
DLLEXPORT void set_log_output(log_print_t func);

/*!
 * @brief Dump the run-time graph.
 *        If the graph is dumpped after prerun(), it will dump the optimized graph instead of the origin one.
 *
 * @param [in] graph: The graph handle.
 */
DLLEXPORT void dump_graph(graph_t graph);

/**************************** Plug-in operate set *******************/
/*!
 * @brief Load one plugin from disk, and execute the init function.
 *
 * @param [in] plugin_name: Plugin name.
 * @param [in] file_name: Plugin file name.
 * @param [in] init_func_name: The name of the init function.
 *
 * @return 0: Plugin loaded and inited Success,
 *      -1: Fail
 */
DLLEXPORT int load_tengine_plugin(const char* plugin_name, const char* file_name, const char* init_func_name);

/*!
 * @brief Unload one plugin and call the release function.
 *
 * @param [in] plugin_name: The name of plugin.
 * @param [in] rel_func_name: The release function name.
 *
 * @return  0: Success;
 *      -1: Fail.
 */
DLLEXPORT int unload_tengine_plugin(const char* plugin_name, const char* rel_func_name);

/*!
 * @brief Get the number of loaded plugin.
 *
 * @return The plugin number.
 */
DLLEXPORT int get_tengine_plugin_number(void);

/*!
 * @brief Get the name of #idx plugin.
 *
 * @param [in] idx: The index of loaded plugin.
 *
 * @return The name of plugin.
 */
DLLEXPORT const char* get_tengine_plugin_name(int idx);

DLLEXPORT const char* get_tengine_hcl_version(void);

#ifdef __cplusplus
}
#endif
