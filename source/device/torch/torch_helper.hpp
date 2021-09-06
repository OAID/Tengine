
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
typedef std::map<uint32_t, std::any> dict_irt2vxo;

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

    Net(struct subgraph* subgraph_map, dict_irt2vxt tensor_map);

    std::vector<std::shared_ptr<torch::Tensor> > forward(std::vector<std::shared_ptr<torch::Tensor> > torch_input);

    bool AddConvolutionNode(struct node* ir_node);
};
