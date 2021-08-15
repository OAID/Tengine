# C API

## Initial

It implements the functions of Tengine framework basic resource initialization, release function, and version number query. 

Example:

```c++
/* inital tengine */
if (init_tengine() != 0)
{
    fprintf(stderr, "Initial tengine failed.\n");
    return -1;
}
fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

/* some codes */

/* release tengine */
release_tengine();
```

### `int init_tengine(void)`

Brief：
- `Initialize the tengine, only can be called once.`

Return：
- `0: Success, -1: Fail.`

### `void release_tengine(void)`

Brief：
- `Release the tengine, only can be called once.`

### `const char* get_tengine_version(void)`

Brief：
- `Get the version of the tengine.`

Return：
- `const char * of version string.`

## Graph

It implements the functions of Tengine calculation graph creation, release, and parameter acquisition.

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP32;
opt.affinity = affinity;

/* create graph, load tengine model xxx.tmfile */
graph_t graph = create_graph(NULL, "tengine", model_file);

/* set the shape, data buffer of input_tensor of the graph */
tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

/* prerun graph, set work options(num_thread, cluster, precision) */
prerun_graph_multithread(graph, opt);

/* run graph */
run_graph(graph, 1);

/* get the result of classification */
tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

/* release tengine */
postrun_graph(graph);
destroy_graph(graph);
```

### `graph_t create_graph(context_t context, const char* model_format, const char* file_name, ...)`

Brief：
- `Create the run-time graph for execution from a saved model. If model format is NULL, an empty graph handle will be returned.`

Params：
- `context: The context the graph will run inside could be NULL and the graph is created in a private context`
- `model_format: The model format type,such as "caffe","tengine"`
- `file_name:  The name of model file.`

Return：
- `0: Success, -1: Fail.`

### `int prerun_graph_multithread(graph_t graph, struct options opt)`

Brief：
- `Initialize resource for graph execution, and set cluster and threads count will used.`

Params：
- `graph: The graph handle.`
- `opt: The graph exec options`

Return：
- `0: Success, -1: Fail.`

### `int run_graph(graph_t graph, int block)`

Brief：
- `Execute graph.`

Params：
- `graph: The graph handle.`
- `block: Blocking or nonlocking.`

Return：
- `0: Success, -1: Fail.`

### `int postrun_graph(graph_t graph)`

Brief：
- `Release the resource for graph execution.`

Params：
- `graph: graph handle.`

Return：
- `0: Success, -1: Fail.`

### `int destroy_graph(graph_t graph)`

Brief：
- `Destory the runtime graph and release allocated resource.`

Params：
- `graph: The graph handle.`

Return：
- `0: Success, -1: Fail.`

### `int set_graph_layout(graph_t graph, int layout_type)`

Brief：
- `Set the layout type of the graph the default layout of graph is NCHW.`

Params：
- `graph, the graph handle`
- `layout_type, the layout type NCHW or NHWC`

Return：
- `0: Success, -1: Fail.`

### `int set_graph_input_node(graph_t graph, const char* input_nodes[], int input_number)`

Brief：
- `designate the input nodes of the graph.`

Params：
- `graph: the graph handle`
- `input_nodes: the node name list of input nodes`
- `input_number: the number of input_nodes`

Return：
- `0: Success, -1: Fail.`

### `int set_graph_output_node(graph_t graph, const char* output_nodes[], int output_number)`

Brief：
- `designate the output nodes of the graph.`

Params：
- `graph: the graph handle`
- `output_nodes: the node name list of output nodes`
- `output_number: the number of output_nodes`

Return：
- `0: Success, -1: Fail.`

### `int get_graph_input_node_number(graph_t graph)`

Brief：
- `Get the number of input node of the graph.`

Params：
- `graph: The graph handle.`

Return：
- `the input node number.`

### `node_t get_graph_input_node(graph_t graph, int idx)`

Brief：
- `Get the node handle of #idx of input node of the graph.`

Params：
- `graph: The graph handle.`
- `idx: The input node index,starting from zero.`

Return：
- `The node name or NULL on error.`

### `int get_graph_output_node_number(graph_t graph)`

Brief：
- `Get the number of output node of the graph.`

Params：
- `graph: The graph handle.`

Return：
- `The input node number.`

### `node_t get_graph_output_node(graph_t graph, int idx)`

Brief：
- `Get the node handle #idx of a graph output node.`

Params：
- `graph: The graph handle.`
- `idx: The input node index, starting from zero.`

Return：
- `The node name or NULL on error.`

### `tensor_t get_graph_output_tensor(graph_t graph, int output_node_idx, int tensor_idx)`

Brief：
- `Get a tensor handle of a graph output node.`

Params：
- `graph: The graph handle.`
- `output_node_idx: The output node index.`
- `tensor_idx: The output tensor index of the output node.`

Return：
- `The tensor handle or NULL on error.`

### `tensor_t get_graph_input_tensor(graph_t graph, int input_node_idx, int tensor_idx)`

Brief：
- `Get a tensor handle of a graph output node.`

Params：
- `graph: The graph handle.`
- `input_node_idx: The input node index, starting from zero.`
- `tensor_idx: The output tensor index of the input node, starting from zero.`

Return：
- `The tensor handle or NULL on error.`

## Node

Operations related to Node

### `node_t create_graph_node(graph_t graph, const char* node_name, const char* op_name)`

Brief：
- `Create a node for the graph.`

Params：
- `graph: The graph handle.`
- `node_name: The name of the node.`
- `op_name: The name of the operate.`

Return：
- `The node handle or NULL on error.`

### `node_t get_graph_node(graph_t graph, const char* node_name)`

Brief：
- `Get the node handle of the graph.`

Params：
- `graph: The graph handle.`
- `node_name: The name of the node.`

Return：
- `The node handle or NULL on error.`

## Tensor

Operations related to tensor data.

```c++
/* set the shape, data buffer of input_tensor of the graph */
int img_size = img_h * img_w * 3;
int dims[] = {1, 3, img_h, img_w};    // nchw
float* input_data = ( float* )malloc(img_size * sizeof(float));

tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
set_tensor_shape(input_tensor, dims, 4);
set_tensor_buffer(input_tensor, input_data, img_size * 4);
 
/* get the result of classification */
tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
float* output_data = ( float* )get_tensor_buffer(output_tensor);
int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
```

### `tensor_t create_graph_tensor(graph_t graph, const char* tensor_name, int data_type)`

Brief：
- `create a tensor handle by tensor name.`

Params：
- `graph: The graph handle`
- `tensor_name: Tensor name.`
- `data_type: the data type.`

Return：
- `The tensor handle or NULL on error.`

### `tensor_t get_graph_tensor(graph_t graph, const char* tensor_name)`

Brief：
- `Get a tensor handle by tensor name.`

Params：
- `graph: The graph handle`
- `tensor_name: Tensor name.`

Return：
- `The tensor handle or NULL on error.`

### `const char* get_tensor_name(tensor_t tensor)`

Brief：
- `Get the name of the tensor handle.`

Params：
- `tensor: the tensor handle.`

Return：
- `const char * of version string.`

### `int get_tensor_shape(tensor_t tensor, int dims[], int dim_number)`

Brief：
- `Get the shape of tensor.`

Params：
- `tensor: The tensor handle.`
- `dims: An int array to get the returned shape.`
- `dim_number: The array size.`

Return：
- `>=1 the valid dim number, or -1 Fail.`

### `int set_tensor_shape(tensor_t tensor, const int dims[], int dim_number)`

Brief：
- `Set the shape of tensor.`

Params：
- `tensor: The tensor handle.`
- `dims: An int array to get the returned shape.`
- `dim_number: The array size.`

Return：
- `0: Success; -1: Fail.`

## Device

## Exection context

Set and execute the related operations of the session module, which is mainly used to display and set the hardware backend of various heterogeneous computing.

```c++
/* create VeriSilicon TIM-VX backend */
context_t timvx_context = create_context("timvx", 1);
int rtt = add_context_device(timvx_context, "TIMVX");

/* create graph, load tengine model xxx.tmfile */
graph_t graph = create_graph(timvx_context, "tengine", model_file);
```

### `context_t create_context(const char* context_name, int empty_context)`

Brief：
- `Create one execution context with name.`

Params：
- `context_name: The name of the created context.`
- `empty_context: No device is assigned with this context otherwise, all proved devices will be added into the context.`

Return：
- `Execution context handle. If create Failed, return NULL.`

### `int add_context_device(context_t context, const char* dev_name)`

Brief：
- `Add a device into one context.`

Params：
- `context: The context handle.`
- `dev_name: The device name.`

Return：
- `0: Success, -1: Fail.`

### `void destroy_context(context_t context)`

Brief：
- `Destory and reclaim the resource related with the context.`

Params：
- `context: The context handle.`

## Misc

Other auxiliary API.

```
/* set the level of log with INFO */
set_log_level(LOG_INFO);

/* dump the graph to console */
dump_graph(graph);
```

### `void set_log_level(enum log_level level)`

Brief：
- `Set the logger level.`

Params：
- `level: The log level.`

### `void dump_graph(graph_t graph)`

Brief：
- `Dump the run-time graph. If the graph is dumpped after prerun(), it will dump the optimized graph instead of the origin one.`

Params：
- `graph: The graph handle.`

## Plugin

## Macro definition

## Structure

## Custom operator

