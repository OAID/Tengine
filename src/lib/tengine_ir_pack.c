
#include "tengine_ir.h"

#define ADDR_ALIGN(a) ((a + 3) & (~0x3))

struct packed_tensor
{
    int32_t size;
    uint16_t idx;
    int16_t producer; /* node idx: -1 means no producer */
    int16_t consumer[MAX_CONSUMER_NUM]; /* idx of node */

    uint8_t consumer_num;
    uint8_t tensor_type; /* const, input, var, dep */
    uint8_t data_type; /* int8,uint8,fp32,fp16,int32 */
    uint8_t dim_num;
    uint8_t elem_size;
    uint8_t layout;

    int32_t elem_num;
    int32_t data_size;
    int32_t dims[MAX_SHAPE_DIM_NUM];
    char data[0];
};

struct packed_op
{
    int32_t size;
    uint16_t op_type;
    uint8_t op_version;
    uint8_t same_shape;
    uint16_t param_size;
    char param_mem[0];
};

struct packed_node
{
    int32_t size;
    int32_t input_offset;
    int32_t output_offset;
    uint16_t idx;
    uint8_t dynamic_shape;
    uint8_t input_num;
    uint8_t output_num;
    uint8_t node_type;

    struct packed_op op;

    /* input > 2 */
    /* output > 2 */
};

struct packed_ir_graph
{
    int32_t size;
    int32_t node_offset;
    int32_t tensor_offset;
    uint16_t tensor_num;
    uint16_t node_num;
    uint16_t input_num;
    uint16_t output_num;

    int8_t graph_layout;
    int8_t model_layout;
    int8_t model_format;

    uint16_t node_io[0]; /* idx of inputs + outputs */

    /* node */
    /* tensor */
};

static struct packed_tensor* get_first_packed_tensor(struct packed_ir_graph* p_graph)
{
    char* addr = (char*)p_graph;

    return ( struct packed_tensor* )(addr + p_graph->tensor_offset);
}

static struct packed_tensor* get_next_packed_tensor(struct packed_tensor* p_tensor)
{
	char* addr = (char*)p_tensor;
    return ( struct packed_tensor* )(addr + p_tensor->size);
}

static struct packed_node* get_first_packed_node(struct packed_ir_graph* p_graph)
{
	char* addr = (char*)p_graph;

    return ( struct packed_node* )(addr + p_graph->node_offset);
}

static struct packed_node* get_next_packed_node(struct packed_node* p_node)
{
	char* addr = (char*)p_node;

    return ( struct packed_node* )(addr + p_node->size);
}

static struct packed_op* pack_ir_op(struct ir_op* op)
{
    int packed_size = sizeof(struct packed_op) + op->param_size;

    packed_size = ADDR_ALIGN(packed_size);

    struct packed_op* packed = ( struct packed_op* )sys_malloc(packed_size);

    packed->size = packed_size;
    packed->op_type = op->op_type;
    packed->op_version = op->op_version;
    packed->same_shape = op->same_shape;
    packed->param_size = op->param_size;

    if (packed->param_size)
    {
        memcpy(packed->param_mem, op->param_mem, packed->param_size);
    }

    return packed;
}

static struct packed_node* pack_ir_node(struct ir_node* node)
{
    struct packed_op* p_op = pack_ir_op(&node->op);

    int node_size = p_op->size + sizeof(struct packed_node) - sizeof(struct packed_op);

    if (node->input_num > 2)
        node_size += sizeof(int16_t) * node->input_num;

    if (node->output_num > 2)
        node_size += sizeof(int16_t) * node->output_num;

    node_size = ADDR_ALIGN(node_size);

    struct packed_node* packed = ( struct packed_node* )sys_malloc(node_size);

    packed->size = node_size;
    packed->idx = node->idx;
    packed->dynamic_shape = node->dynamic_shape;
    packed->input_num = node->input_num;
    packed->output_num = node->output_num;
    packed->node_type = node->node_type;
    memcpy(&packed->op, p_op, p_op->size);

    int offset = p_op->size + sizeof(struct packed_node) - sizeof(struct packed_op);

    if (node->input_num == 1)
        packed->input_offset = node->input_tensors[0];
    else
    {
        packed->input_offset = offset;

        int16_t* ptr = ( int16_t* )(( char* )packed + offset);

        for (int i = 0; i < node->input_num; i++)
            ptr[i] = node->input_tensors[i];

        offset += sizeof(int16_t) * node->input_num;
    }

    if (node->output_num == 1)
        packed->output_offset = node->output_tensors[0];
    else
    {
        packed->output_offset = offset;
        int16_t* ptr = ( int16_t* )(( char* )packed + offset);

        for (int i = 0; i < node->output_num; i++)
            ptr[i] = node->output_tensors[i];
    }

    sys_free(p_op);

    return packed;
}

static struct packed_tensor* pack_ir_tensor(struct ir_tensor* tensor)
{
    int data_size;

    if (tensor->data)
        data_size = tensor->elem_size * tensor->elem_num;
    else
        data_size = 0;

    int packed_size = data_size + sizeof(struct packed_tensor);

    packed_size = ADDR_ALIGN(packed_size);

    struct packed_tensor* packed = ( struct packed_tensor* )sys_malloc(packed_size);

    packed->size = packed_size;
    packed->idx = tensor->idx;
    packed->tensor_type = tensor->tensor_type;
    packed->data_type = tensor->data_type;
    packed->dim_num = tensor->dim_num;
    packed->layout = tensor->layout;
    packed->data_size = data_size;
    packed->elem_num = tensor->elem_num;
    packed->elem_size = tensor->elem_size;
    packed->consumer_num = tensor->consumer_num;
    packed->producer = tensor->producer;

    memcpy(packed->consumer, tensor->consumer, MAX_CONSUMER_NUM);

    if (data_size)
        memcpy(packed->data, tensor->data, data_size);

    for (int i = 0; i < tensor->dim_num; i++)
        packed->dims[i] = tensor->dims[i];

    return packed;
}

int pack_ir_graph(struct ir_graph* ir_graph, void** mem, int* mem_size)
{
    int packed_size = sizeof(struct packed_ir_graph);
    struct packed_ir_graph* packed = ( struct packed_ir_graph* )sys_malloc(packed_size);

    packed->size = packed_size;
    packed->node_offset = 0;
    packed->tensor_offset = 0;
    packed->tensor_num = ir_graph->tensor_num;
    packed->node_num = ir_graph->node_num;
    packed->input_num = ir_graph->input_num;
    packed->output_num = ir_graph->output_num;

    packed->graph_layout = ir_graph->graph_layout;
    packed->model_layout = ir_graph->model_layout;
    packed->model_format = ir_graph->model_format;

    /* packing input/out nodes */

    packed_size += sizeof(uint16_t) * (ir_graph->input_num + ir_graph->output_num);
    packed_size = ADDR_ALIGN(packed_size);

    packed = ( struct packed_ir_graph* )sys_realloc(packed, packed_size);

    for (int i = 0; i < ir_graph->input_num; i++)
        packed->node_io[i] = ir_graph->input_nodes[i];

    for (int i = 0; i < ir_graph->output_num; i++)
        packed->node_io[i + ir_graph->input_num] = ir_graph->output_nodes[i];

    packed->node_offset = packed_size;

    /* packing nodes */
    for (int i = 0; i < packed->node_num; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, i);
        struct packed_node* packed_node = pack_ir_node(ir_node);

        packed = sys_realloc(packed, packed_size + packed_node->size);

        memcpy(( char* )packed + packed_size, packed_node, packed_node->size);

        packed_size += packed_node->size;

        sys_free(packed_node);
    }

    packed->tensor_offset = packed_size;

    /*packing tensors*/
    for (int i = 0; i < packed->tensor_num; i++)
    {
        struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, i);
        struct packed_tensor* packed_tensor = pack_ir_tensor(ir_tensor);

        packed = sys_realloc(packed, packed_size + packed_tensor->size);

        memcpy(( char* )packed + packed_size, packed_tensor, packed_tensor->size);

        packed_size += packed_tensor->size;
        sys_free(packed_tensor);
    }

    packed->size = packed_size;

    *mem = packed;
    *mem_size = packed_size;

    return 0;
}

struct ir_graph* unpack_ir_graph(const void* mem, int mem_size)
{
    struct packed_ir_graph* packed = ( struct packed_ir_graph* )mem;

    if (packed->size != mem_size)
        return NULL;

    struct ir_graph* ir_graph = create_ir_graph(NULL);

    ir_graph->input_num = packed->input_num;
    ir_graph->output_num = packed->output_num;
    ir_graph->graph_layout = packed->graph_layout;
    ir_graph->model_layout = packed->model_layout;
    ir_graph->model_format = packed->model_format;

    ir_graph->input_nodes = ( int16_t* )sys_malloc(sizeof(int16_t) * ir_graph->input_num);

    for (int i = 0; i < ir_graph->input_num; i++)
        ir_graph->input_nodes[i] = packed->node_io[i];

    ir_graph->output_nodes = ( int16_t* )sys_malloc(sizeof(int16_t) * ir_graph->output_num);

    for (int i = 0; i < ir_graph->output_num; i++)
        ir_graph->output_nodes[i] = packed->node_io[i + ir_graph->input_num];

    struct packed_node* packed_node = NULL;

    for (int i = 0; i < packed->node_num; i++)
    {
        if (packed_node)
            packed_node = get_next_packed_node(packed_node);
        else
            packed_node = get_first_packed_node(packed);

        struct ir_node* ir_node = create_ir_node(ir_graph, NULL, packed_node->op.op_type, packed_node->op.op_version);

        if (packed_node->op.param_size)
        {
            /* todo: double check if packed_node->op.param_size == ir_node->op.param_size */
            memcpy(ir_node->op.param_mem, packed_node->op.param_mem, packed_node->op.param_size);
        }

        ir_node->dynamic_shape = packed_node->dynamic_shape;
        ir_node->input_num = packed_node->input_num;
        ir_node->output_num = packed_node->output_num;
        ir_node->node_type = packed_node->node_type;

        ir_node->input_tensors = ( int16_t* )sys_malloc(sizeof(int16_t) * packed_node->input_num);
        ir_node->output_tensors = ( int16_t* )sys_malloc(sizeof(int16_t) * packed_node->output_num);

        if (packed_node->input_num == 1)
            ir_node->input_tensors[0] = packed_node->input_offset;
        else
        {
            int16_t* ptr = ( int16_t* )(( char* )packed_node + packed_node->input_offset);
            for (int j = 0; j < packed_node->input_num; j++)
                ir_node->input_tensors[j] = ptr[j];
        }

        if (packed_node->output_num == 1)
            ir_node->output_tensors[0] = packed_node->output_offset;
        else
        {
            int16_t* ptr = ( int16_t* )(( char* )packed_node + packed_node->output_offset);
            for (int j = 0; j < packed_node->output_num; j++)
                ir_node->output_tensors[j] = ptr[j];
        }
    }

    struct packed_tensor* packed_tensor = NULL;

    for (int i = 0; i < packed->tensor_num; i++)
    {
        if (packed_tensor)
            packed_tensor = get_next_packed_tensor(packed_tensor);
        else
            packed_tensor = get_first_packed_tensor(packed);

        struct ir_tensor* ir_tensor = create_ir_tensor(ir_graph, NULL, packed_tensor->data_type);

        ir_tensor->producer = packed_tensor->producer;
        ir_tensor->consumer_num = packed_tensor->consumer_num;

        for (int j = 0; j < packed_tensor->consumer_num; j++)
            ir_tensor->consumer[j] = packed_tensor->consumer[j];

        ir_tensor->tensor_type = packed_tensor->tensor_type;
        ir_tensor->data_type = packed_tensor->data_type;
        ir_tensor->elem_size = packed_tensor->elem_size;
        ir_tensor->elem_num = packed_tensor->elem_num;
        ir_tensor->dim_num = packed_tensor->dim_num;

        for (int j = 0; j < ir_tensor->dim_num; j++)
            ir_tensor->dims[j] = packed_tensor->dims[j];

        ir_tensor->free_host_mem = 1;
        ir_tensor->internal_allocated = 0;

        if (packed_tensor->data_size)
        {
            ir_tensor->data = sys_malloc(packed_tensor->data_size);
            memcpy(ir_tensor->data, packed_tensor->data, packed_tensor->data_size);
        }
    }

    return ir_graph;
}
