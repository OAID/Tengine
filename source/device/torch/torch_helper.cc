#include "torch_helper.hpp"

torch::nn::Conv2dOptions
create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,
                    int64_t dilation = 1, bool bias = false)
{
    torch::nn::Conv2dOptions conv_options =
        torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size)
            .stride(stride)
            .padding(padding)
            .bias(bias)
            .groups(groups)
            .dilation(dilation);

    return conv_options;
}


Net::Net(struct subgraph* subgraph_map, dict_irt2vxt tensor_map)
{
    subgraph = subgraph_map;
    ir_graph = subgraph->graph;
    torch_tensor_map.insert(tensor_map.begin(), tensor_map.end());

    struct graph* ir_graph = subgraph->graph;

    /* Node Register */
    for (uint16_t i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;
        std::string node_name(ir_node->name);

        switch (op_type)
        {
        case OP_CONST:
        case OP_INPUT:
            continue;
        case OP_CONV:
        {
            this->AddConvolutionNode(ir_node);
            break;
        }
//        case OP_POOL:
//            this->AddPoolingNode(ir_node);
//            break;
        default:
            fprintf(stderr, "Tengine TORCH Prerun: Cannot support OP(%d).\n", ir_node->index);
            break;
        }
    }
}


std::vector<std::shared_ptr<torch::Tensor> > Net::forward(std::vector<std::shared_ptr<torch::Tensor> > torch_input)
{
    /* 多输入 */
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        *torch_tensor_map[subgraph->input_tensor_list[0]] = *torch_input[i];
    }

    for (uint16_t i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;
        std::string node_name(ir_node->name);

        switch (op_type)
        {
        case OP_CONST:
        case OP_INPUT:
            continue;
        case OP_CONV:
        {
            struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;

            if (param->activation < 0)
            {
                *torch_tensor_map[ir_node->output_tensors[0]] = std::any_cast<torch::nn::Conv2d>(torch_node_map[node_id])(*torch_tensor_map[ir_node->input_tensors[0]]);
            }
            else if (param->activation == 0)
            {
                *torch_tensor_map[ir_node->output_tensors[0]] = torch::relu(std::any_cast<torch::nn::Conv2d>(torch_node_map[node_id])(*torch_tensor_map[ir_node->input_tensors[0]]));
            }
            break;
        }
            //            case OP_POOL:
            //                this->AddPoolingNode(ir_node);
            //                break;
        default:
            fprintf(stderr, "Tengine TORCH Run: Cannot support OP(%d).\n", ir_node->index);
            break;
        }
    }

    std::vector<std::shared_ptr<torch::Tensor> > torch_output;
    /* 多输出 */
    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        torch_output.push_back(torch_tensor_map[subgraph->output_tensor_list[i]]);
    }
    return torch_output;
}








