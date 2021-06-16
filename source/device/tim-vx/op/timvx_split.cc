#include "timvx_executor.hpp"
extern "C"
{
#include "utility/vector.h"
#include "operator/op.h"
#include "split_param.h"
}
bool VXEngine::AddSplitNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    std::vector<std::shared_ptr<tim::vx::Tensor> > split_out_tensor(ir_node->output_num);
    for (int i = 0; i < ir_node->output_num; i++)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);
        split_out_tensor[i] = this->vx_tensor_map[output_tensor->index];
    }
    struct split_param* param = (struct split_param*)ir_node->op.param_mem;
    uint32_t axis = input_tensor->dim_num - 1 - param->axis;
    std::vector<uint32_t> slices;
    for (int i = 0; i < ir_node->output_num; i++)
    {
        uint32_t split_slice = ((uint32_t*)get_vector_data(param->split_sizes_, i))[0];
        slices.push_back(  split_slice  );
    }
    auto split = graph->CreateOperation<tim::vx::ops::Split>(axis, slices);
    vx_node_map[ir_node->index] = split;
    (*split)
            .BindInput({ this->vx_tensor_map[input_tensor->index] })
            .BindOutputs( split_out_tensor );
    return true;
}
