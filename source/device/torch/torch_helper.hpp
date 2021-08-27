
#pragma once

extern "C" {
#include "api/c_api.h"
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"

#include "convolution_param.h"

}

#include <torch/torch.h>
#include <any>


typedef std::map<uint32_t, std::shared_ptr<torch::Tensor> > dict_irt2vxt;
typedef std::map<uint32_t, std::any > dict_irt2vxo;


torch::nn::Conv2dOptions
create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                    int64_t stride, int64_t padding, int64_t groups,
                    int64_t dilation, bool bias);

struct Net : torch::nn::Module
{
    struct subgraph* subgraph;
    struct graph* ir_graph;
    dict_irt2vxt torch_tensor_map;
    dict_irt2vxo torch_node_map;

    Net(struct subgraph* subgraph_map, dict_irt2vxt tensor_map)
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
                struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;

                struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
                struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
                struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
                struct tensor* bias_tensor;

                bool bias = false;
                if (ir_node->input_num > 2)
                {
                    bias = true;
                    bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
                }

                torch::nn::Conv2d layer
                    = torch::nn::Conv2d{create_conv_options(
                        /*in_planes = */ input_tensor->dims[1], /*out_planes = */ output_tensor->dims[1],
                        /*kerner_size = */ param->kernel_h, /*stride = */ param->stride_h, /*padding = */ param->pad_h0,
                        /*groups = */ param->group, /*dilation = */ param->dilation_h, /*bias = */ bias
                    )};
                register_module(std::to_string(ir_node->index), layer);
                torch_node_map[ir_node->index] = layer;

                {
                    torch::Tensor t = torch::rand({weight_tensor->dims[0], weight_tensor->dims[1], weight_tensor->dims[2], weight_tensor->dims[3]});
                    void* date_mem = t.data_ptr();
                    memcpy(date_mem, weight_tensor->data, weight_tensor->elem_num*weight_tensor->elem_size);
                    layer->weight = register_parameter(std::to_string(ir_node->index) + "_weight", t);
                }

                if (bias)
                {
                    torch::Tensor t = torch::rand({output_tensor->dims[1]});
                    void* date_mem = t.data_ptr();
                    memcpy(date_mem, bias_tensor->data, bias_tensor->elem_num*bias_tensor->elem_size);
                    layer->bias = register_parameter(std::to_string(ir_node->index) + "_bias", t);
                }

                break;
            }
//            case OP_POOL:
//                this->AddPoolingNode(ir_node);
//                break;
            default:
                fprintf(stderr, "Tengine TORCH Prerun: Cannot support OP(%d).\n", ir_node->index);
                break;
            }
        }
    }

    std::vector<std::shared_ptr<torch::Tensor> > forward(std::vector<std::shared_ptr<torch::Tensor> > torch_input)
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
                        *torch_tensor_map[ir_node->output_tensors[0]] = torch::relu( std::any_cast<torch::nn::Conv2d>(torch_node_map[node_id])(*torch_tensor_map[ir_node->input_tensors[0]]) );
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
        return  torch_output;
    }
};

