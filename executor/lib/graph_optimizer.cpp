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
#include <cmath>
#include <cstring>

#include "node.hpp"
#include "graph.hpp"
#include "graph_optimizer.hpp"
#include "operator/fused_operator.hpp"
#include "operator/batch_norm.hpp"
#include "operator/convolution.hpp"
#include "operator/relu.hpp"
#include "operator/scale.hpp"
#include "operator/eltwise.hpp"
#include "tensor_mem.hpp"

namespace TEngine {

static bool GraphFuseBNScale(Graph* graph, GraphOptimizer* opt);
static bool GraphFuseConvBN(Graph* graph, GraphOptimizer* opt);
static bool GraphFuseConvReLu(Graph* graph, GraphOptimizer* opt);
static bool GraphFuseConvReLu6(Graph* graph, GraphOptimizer* opt);
static bool GraphFuseRelu6(Graph* graph, GraphOptimizer* opt);
static void AddConstNodeToSubGraph(Subgraph* graph, Tensor* tensor, Node* fused_node, int fused_port_index);

static bool Weight_Bn(Subgraph* graph, Node* ConvNode, float* mean, float* var, float* gamma, float* beta, float eps,
                      float rescale_factor, Tensor* bias_tensor)
{
    Tensor* kernel_tensor = ConvNode->GetInputTensor(1);
    Convolution* conv_op = dynamic_cast<Convolution*>(ConvNode->GetOp());
    ConvParam* param = conv_op->GetParam();
    const TShape& kernel_shape = kernel_tensor->GetShape();

    int group = param->group;
    // int input_chan = kernel_shape.Shape(1);
    int input_chan = kernel_shape.GetC();
    int output_chan = kernel_shape.Shape(0) / group;
    int kernel_x = param->kernel_w;
    int kernel_y = param->kernel_h;
    int kernel_size = input_chan * kernel_x * kernel_y;
    float* kernel_org = ( float* )get_tensor_mem(kernel_tensor);
    int channel_num = kernel_shape.Shape(0);
    float* kernel_new = ( float* )(malloc(kernel_size * channel_num * sizeof(float) + 128));

    memcpy(kernel_new, kernel_org, sizeof(float) * kernel_size * channel_num);

    kernel_tensor->SetMemAddr(kernel_new);
    kernel_tensor->SetAttr("free_mem", 1);

    float* scale_mean = ( float* )malloc(channel_num * sizeof(float));
    float* scale_var_inv = ( float* )malloc(channel_num * sizeof(float));

    float rescale_factor_tmp = rescale_factor;

    // fuse the bias;
    float* bias = NULL;
    std::string bias_name;
    if(bias_tensor)
    {
        bias = ( float* )get_tensor_mem(bias_tensor);
        bias_name = bias_tensor->GetName() + ".bn";
    }
    else
    {
        bias_name = ConvNode->GetName() + ".bias.bn";
    }

    /*
     *  create the bias node,,, ugly code..
     *
     */

    {
        Tensor* new_bias_tensor = new Tensor(bias_name);
        std::vector<int> dims{channel_num};

        TShape bias_shape;
        bias_shape.SetDim(dims);

        new_bias_tensor->Reshape(bias_shape);
        new_bias_tensor->SetType(kConstTensor);

        void* bias_new = ( void* )malloc(channel_num * sizeof(float) + 128);

        new_bias_tensor->SetMemAddr(bias_new);

        AddConstNodeToSubGraph(graph, new_bias_tensor, ConvNode, 2);

        delete new_bias_tensor;

        // set the free flag
        new_bias_tensor = ConvNode->GetInputTensor(2);
        new_bias_tensor->SetAttr("free_mem", 1);
    }

    rescale_factor_tmp = rescale_factor_tmp ? 1 / rescale_factor_tmp : 0;

    if(NULL == bias)
    {
        for(int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / sqrt(var[c] * rescale_factor_tmp + eps);
            scale_mean[c] = -mean[c] * rescale_factor_tmp * scale_var_inv[c];
        }
    }
    else
    {
        for(int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = 1.f / sqrt(var[c] * rescale_factor_tmp + eps);
            scale_mean[c] = (bias[c] - mean[c] * rescale_factor_tmp) * scale_var_inv[c];
        }
    }

    if(NULL != gamma)
    {
        for(int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] = gamma[c] * scale_var_inv[c];
            scale_mean[c] = gamma[c] * scale_mean[c];
        }
    }
    if(NULL != beta)
    {
        for(int c = 0; c < channel_num; c++)
        {
            scale_mean[c] = scale_mean[c] + beta[c];
        }
    }
    if(kernel_shape.GetDataLayout() == TENGINE_LAYOUT_NCHW)
    {
        for(int g = 0; g < group; g++)
        {
            float* kernel = kernel_new + g * output_chan * kernel_size;
            for(int o_c = 0; o_c < output_chan; o_c++)
            {
                float w_scale = scale_var_inv[g * output_chan + o_c];
                for(int i = 0; i < kernel_size; i++)
                {
                    kernel[o_c * kernel_size + i] = kernel[o_c * kernel_size + i] * w_scale;
                }
            }
        }
    }
    else
    {
        for(int o_c = 0; o_c < output_chan; o_c++)
        {
            for(int k_h = 0; k_h < kernel_y; ++k_h)
            {
                for(int k_w = 0; k_w < kernel_x; ++k_w)
                {
                    for(int g = 0; g < group; ++g)
                    {
                        float w_scale = scale_var_inv[g * output_chan + o_c];
                        for(int i_c = 0; i_c < input_chan; ++i_c)
                        {
                            int weight_idx = o_c * group * kernel_size + k_h * kernel_x * input_chan * group +
                                             k_w * input_chan * group + g * input_chan + i_c;
                            kernel_new[weight_idx] = kernel_new[weight_idx] * w_scale;
                        }
                    }
                }
            }
        }
    }
    float* bias_tmp = ( float* )get_tensor_mem(ConvNode->GetInputTensor(2));

    for(int i = 0; i < channel_num; i++)
    {
        bias_tmp[i] = scale_mean[i];
    }

    free(scale_var_inv);
    free(scale_mean);

    return true;
}

static void AddConstNodeToSubGraph(Subgraph* graph, Tensor* tensor, Node* fused_node, int fused_port_index)
{
    Tensor* new_tensor = new Tensor(*tensor);

    std::string new_tensor_name = new_tensor->GetName();

    //   if(new_tensor_name.rfind(".fused")==std::string::npos)
    new_tensor_name += ".fused";

    new_tensor->SetName(new_tensor_name);

    Node* new_node = new Node(new_tensor->GetName());
    Operator* op = new_node->GetOp();

    op = OpManager::CreateOp("Const");
    new_node->SetOp(op);
    new_node->AddOutputTensor(new_tensor);
    new_tensor->producer = new_node->GetOutputPort(0);
    fused_node->AddInputTensor(new_tensor);
    new_tensor->consumer.clear();
    new_tensor->consumer.push_back(fused_node->GetInputPort(fused_port_index));
    graph->seq_nodes.push_back(new_node);
    graph->SetNodeOwner(new_node);
    graph->SetTensorOwner(new_tensor);

    if(tensor->ExistAttr("free_mem"))
        tensor->RemoveAttr("free_mem");

    // return new_tensor;
}

static bool GraphFuseBNScale(Graph* graph, GraphOptimizer* opt)
{
    int node_number = graph->seq_nodes.size();
    std::vector<Subgraph*> orig_sub;

    /*get all bn_scale chain*/
    for(int i = 0; i < node_number; i++)
    {
        Node* Scale_node = graph->seq_nodes[i];
        Operator* op = Scale_node->GetOp();

        if(op->GetName() != "Scale")
            continue;

        /*Check if it is a bn-->Scale*/
        Tensor* input_tensor;
        Node* Bn_node;

        input_tensor = Scale_node->GetInputTensor(0);
        Bn_node = input_tensor->producer->owner;
        op = Bn_node->GetOp();

        if(op->GetName() != "BatchNormalization")
            continue;

        /*Create a subgrah represent the chain*/
        Subgraph* sub = new Subgraph("BnScale_chain");

        sub->seq_nodes.push_back(Bn_node);
        sub->seq_nodes.push_back(Scale_node);

        sub->input_nodes.push_back(Bn_node);
        sub->output_nodes.push_back(Scale_node);

        /* add const node into seq nodes */
        for(unsigned int i = 1; i < Bn_node->GetInputNum(); i++)
        {
            Tensor* tensor = Bn_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }

        for(unsigned int i = 1; i < Scale_node->GetInputNum(); i++)
        {
            Tensor* tensor = Scale_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }
        orig_sub.push_back(sub);
    }

    /*replace the nodes of the grah*/
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph fused("fused");
        Subgraph* orig = orig_sub[i];

        Node* orig_output = orig->output_nodes[0];
        Node* orig_input = orig->input_nodes[0];

        std::string node_name = orig_input->GetName() + "-" + orig_output->GetName();

        /*create new Node node*/
        Node* fused_node = new Node(node_name);
        Operator* new_bn_op = OpManager::CreateOp("BatchNormalization");

        fused_node->SetDynamicShape(orig_input->IsDynamicShape());
        fused_node->MergeAttr(orig_output);
        fused_node->MergeAttr(orig_input);
        fused_node->SetOp(new_bn_op);

        // 1. Add the input tensor and ouput tensor to the fused node

        Tensor* output_tensor = orig_output->GetOutputTensor(0);
        fused_node->AddOutputTensor(output_tensor);
        Tensor* input_tensor = orig_input->GetInputTensor(0);
        fused_node->AddInputTensor(input_tensor);

        fused.seq_nodes.push_back(fused_node);
        fused.input_nodes.push_back(fused_node);
        fused.output_nodes.push_back(fused_node);
        fused.SetNodeOwner(fused_node);

        // 2. Create the new const nodes

        Node* orig_bn = orig->seq_nodes[0];
        Node* orig_scale = orig->seq_nodes[1];

        Tensor* orig_gamma = orig_scale->GetInputTensor(1);
        Tensor* orig_beta = orig_scale->GetInputTensor(2);
        Tensor* orig_mean = orig_bn->GetInputTensor(3);
        Tensor* orig_var = orig_bn->GetInputTensor(4);

        /*create the const node and add to the sub graph*/
        AddConstNodeToSubGraph(&fused, orig_gamma, fused_node, 1);
        AddConstNodeToSubGraph(&fused, orig_beta, fused_node, 2);
        AddConstNodeToSubGraph(&fused, orig_mean, fused_node, 3);
        AddConstNodeToSubGraph(&fused, orig_var, fused_node, 4);

        // 3. Set new Batch Norm
        BatchNorm* bn_op = dynamic_cast<BatchNorm*>(orig_bn->GetOp());
        BatchNormParam* param_org = bn_op->GetParam();
        BatchNormParam* param_new = (( BatchNorm* )new_bn_op)->GetParam();
        param_new->caffe_flavor = 0;
        param_new->eps = param_org->eps;
        param_new->rescale_factor = param_org->rescale_factor;

        graph->Replace(orig, &fused);
    }

    /* release orig_sub */
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph* orig = orig_sub[i];
        delete orig;
    }

    return true;
}

static bool GraphFuseRelu6(Graph* graph, GraphOptimizer* opt)
{
    int node_number = graph->seq_nodes.size();
    std::vector<Subgraph*> orig_sub;

    /*get all relu_minimum chain*/
    for(int i = 0; i < node_number; i++)
    {
        Node* min_node = graph->seq_nodes[i];
        Operator* op = min_node->GetOp();

        if(op->GetName() != "Eltwise")
            continue;

        Eltwise* eltwise_op = dynamic_cast<Eltwise*>(op);
        EltwiseParam* param = eltwise_op->GetParam();
        if(param->type != ELT_MIN_SCALAR)
            continue;    // todo:  verify 6

        /*Check if it is a  relu + minimum*/
        Tensor* input_tensor;
        Node* relu_node;

        input_tensor = min_node->GetInputTensor(0);
        relu_node = input_tensor->producer->owner;
        op = relu_node->GetOp();

        if(op->GetName() != "ReLu")
            continue;

        /*Create a subgrah represent the chain*/
        Subgraph* sub = new Subgraph("relu6_chain");

        sub->seq_nodes.push_back(relu_node);
        sub->seq_nodes.push_back(min_node);

        sub->input_nodes.push_back(relu_node);
        sub->output_nodes.push_back(min_node);

        /* add const node into seq nodes */
        for(unsigned int i = 1; i < relu_node->GetInputNum(); i++)
        {
            Tensor* tensor = relu_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }

        for(unsigned int i = 1; i < min_node->GetInputNum(); i++)
        {
            Tensor* tensor = min_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }
        orig_sub.push_back(sub);
    }

    /*replace the nodes of the grah*/
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph fused("fused");
        Subgraph* orig = orig_sub[i];

        Node* orig_output = orig->output_nodes[0];
        Node* orig_input = orig->input_nodes[0];

        std::string node_name = orig_input->GetName() + orig_output->GetName();

        /*create new Node node*/
        Node* fused_node = new Node(node_name);
        Operator* new_bn_op = OpManager::CreateOp("ReLu6");

        fused_node->SetDynamicShape(orig_input->IsDynamicShape());
        fused_node->MergeAttr(orig_output);
        fused_node->MergeAttr(orig_input);
        fused_node->SetOp(new_bn_op);

        // 1. Add the input tensor and ouput tensor to the fused node

        Tensor* output_tensor = orig_output->GetOutputTensor(0);
        fused_node->AddOutputTensor(output_tensor);
        Tensor* input_tensor = orig_input->GetInputTensor(0);
        fused_node->AddInputTensor(input_tensor);

        fused.seq_nodes.push_back(fused_node);
        fused.input_nodes.push_back(fused_node);
        fused.output_nodes.push_back(fused_node);
        fused.SetNodeOwner(fused_node);

        // 2. replace
        graph->Replace(orig, &fused);
    }

    /* release orig_sub */
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph* orig = orig_sub[i];
        delete orig;
    }
    return true;
}

static bool GraphFuseConvBN(Graph* graph, GraphOptimizer* opt)
{
    int node_number = graph->seq_nodes.size();
    std::vector<Subgraph*> orig_sub;

    /*get all bn_scale chain*/
    for(int i = 0; i < node_number; i++)
    {
        Node* Bn_node = graph->seq_nodes[i];
        Operator* op = Bn_node->GetOp();

        if(op->GetName() != "BatchNormalization")
            continue;

        /*Check if it is a Conv-->Bn*/
        Tensor* input_tensor;
        Node* Conv_node;

        input_tensor = Bn_node->GetInputTensor(0);
        Conv_node = input_tensor->producer->owner;
        op = Conv_node->GetOp();

        if(op->GetName() != "Convolution")
            continue;

        /*Create a subgrah represent the chain*/
        Subgraph* sub = new Subgraph("ConvBn_chain");

        sub->seq_nodes.push_back(Conv_node);
        sub->seq_nodes.push_back(Bn_node);

        sub->input_nodes.push_back(Conv_node);
        sub->output_nodes.push_back(Bn_node);

        /* add const node into seq nodes */
        for(unsigned int i = 1; i < Conv_node->GetInputNum(); i++)
        {
            Tensor* tensor = Conv_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }

        for(unsigned int i = 1; i < Bn_node->GetInputNum(); i++)
        {
            Tensor* tensor = Bn_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }
        orig_sub.push_back(sub);
    }

    /*replace the nodes of the graph*/
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph fused("fused");
        Subgraph* orig = orig_sub[i];

        Node* orig_output = orig->output_nodes[0];
        Node* orig_input = orig->input_nodes[0];

        std::string node_name = orig_input->GetName() + "-" + orig_output->GetName();

        /*create new Node node*/
        Node* fused_node = new Node(node_name);
        Operator* new_conv_op = OpManager::CreateOp("Convolution");

        fused_node->SetDynamicShape(orig_input->IsDynamicShape());
        fused_node->MergeAttr(orig_output);
        fused_node->MergeAttr(orig_input);
        fused_node->SetOp(new_conv_op);

        /*copy conv param*/
        fused_node->SetAttr("Fused.Batch", true);

        Convolution* fused_op = dynamic_cast<Convolution*>(new_conv_op);
        ConvParam* fused_param = fused_op->GetParam();
        Convolution* orig_op = dynamic_cast<Convolution*>(orig_input->GetOp());
        ConvParam* orig_param = orig_op->GetParam();
        *fused_param = *orig_param;

        Tensor* output_tensor = orig_output->GetOutputTensor(0);
        fused_node->AddOutputTensor(output_tensor);

        Tensor* input_tensor = orig_input->GetInputTensor(0);
        fused_node->AddInputTensor(input_tensor);

        fused.seq_nodes.push_back(fused_node);
        fused.input_nodes.push_back(fused_node);
        fused.output_nodes.push_back(fused_node);
        fused.SetNodeOwner(fused_node);

        /* create new const node for convolution */
        Tensor* weight = orig_input->GetInputTensor(1);
        AddConstNodeToSubGraph(&fused, weight, fused_node, 1);

        BatchNorm* bn_op = dynamic_cast<BatchNorm*>(orig_output->GetOp());
        BatchNormParam* param_org = bn_op->GetParam();

        Tensor* orig_mean = orig_output->GetInputTensor(3);
        Tensor* orig_var = orig_output->GetInputTensor(4);

        /* cal the scale mean and var */

        float* mean = ( float* )get_tensor_mem(orig_mean);
        float* var = ( float* )get_tensor_mem(orig_var);
        float* gamma = NULL;
        float* beta = NULL;

        if(!param_org->caffe_flavor)
        {
            Tensor* orig_gamma = orig_output->GetInputTensor(1);
            Tensor* orig_beta = orig_output->GetInputTensor(2);
            gamma = ( float* )get_tensor_mem(orig_gamma);
            beta = ( float* )get_tensor_mem(orig_beta);
        }

        Tensor* bias_tensor;

        if(orig_input->GetInputNum() > 2)
            bias_tensor = orig_input->GetInputTensor(2);
        else
            bias_tensor = nullptr;

        Weight_Bn(&fused, fused_node, mean, var, gamma, beta, param_org->eps, param_org->rescale_factor, bias_tensor);

        graph->Replace(orig, &fused);
    }

    /* release orig_sub */
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph* orig = orig_sub[i];
        delete orig;
    }

    return true;
}

bool GraphOptimizerManager::RunOpt(const std::string& name, Graph* graph)
{
    if(!Find(name))
        return false;

    GraphOptimizer* opt = Get(name);

    return opt->optimizer(graph, opt);
}

void GraphOptimizerManager::Init(void)
{
    // register a few predefined optimizer

    GraphOptimizer* opt = new GraphOptimizer();
    opt->name = "BNScale";
    opt->optimizer = graph_opt_t(GraphFuseBNScale);
    Add(opt->name, opt);

    opt = new GraphOptimizer();
    opt->name = "ConvBN";
    opt->optimizer = graph_opt_t(GraphFuseConvBN);
    Add(opt->name, opt);

    opt = new GraphOptimizer();
    opt->name = "ConvReLu";
    opt->optimizer = graph_opt_t(GraphFuseConvReLu);
    Add(opt->name, opt);

    opt = new GraphOptimizer();
    opt->name = "ConvReLu6";
    opt->optimizer = graph_opt_t(GraphFuseConvReLu6);
    Add(opt->name, opt);

    opt = new GraphOptimizer();
    opt->name = "Relu6";
    opt->optimizer = graph_opt_t(GraphFuseRelu6);
    Add(opt->name, opt);
}

static bool NodeInGraph(Node* node, Graph* graph)
{
    int number = graph->seq_nodes.size();

    for(int i = 0; i < number; i++)
    {
        if(node == graph->seq_nodes[i])
            return true;
    }

    return false;
}

/* the graph optimizer: conv_relu */
static bool GraphFuseConvReLuCommon(Graph* graph, GraphOptimizer* opt, bool relu6)
{
    int node_number = graph->seq_nodes.size();
    std::vector<Subgraph*> orig_sub;

    for(int i = 0; i < node_number; i++)
    {
        Node* node = graph->seq_nodes[i];
        Operator* op = node->GetOp();

        if(relu6)
        {
            if(op->GetName() != "ReLu6")
                continue;
        }
        else
        {
            if(op->GetName() != "ReLu")
                continue;
            if(op->GetName() == "ReLu" && dynamic_cast<ReLu*>(op)->GetParam()->negative_slope != 0.f)
            {
                continue;
            }
        }
        Tensor* input_tensor = node->GetInputTensor(0);

        Node* conv_node = input_tensor->producer->owner;

        op = conv_node->GetOp();

        if(op->GetName() != "Convolution")
            continue;

        // check if node in seq_nodes

        if(!NodeInGraph(conv_node, graph))
            continue;
        // if parents has muti_consumer: not fuse
        Tensor* conv_otensor = conv_node->GetOutputTensor(0);
        if(conv_otensor->consumer.size() > 1)
            continue;
        Subgraph* sub = new Subgraph("conv_relu");

        sub->seq_nodes.push_back(conv_node);
        sub->seq_nodes.push_back(node);

        sub->input_nodes.push_back(conv_node);
        sub->output_nodes.push_back(node);

        /* add const node into seq nodes,
        so that they will be removed from origin graph too */

        for(unsigned int i = 1; i < conv_node->GetInputNum(); i++)
        {
            Tensor* tensor = conv_node->GetInputTensor(i);
            sub->seq_nodes.push_back(tensor->producer->owner);
        }

        orig_sub.push_back(sub);
    }

    /* construct new node */
    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph fused("fused");
        Subgraph* orig = orig_sub[i];

        Node* orig_output = orig->output_nodes[0];
        Node* orig_input = orig->input_nodes[0];

        std::string node_name = orig_input->GetName() + "-" + orig_output->GetName();

        Node* fused_node = new Node(node_name);
        Operator* op = OpManager::CreateOp("Convolution");

        fused_node->SetDynamicShape(orig_input->IsDynamicShape());

        fused_node->SetOp(op);
        fused_node->MergeAttr(orig_input);
        fused_node->MergeAttr(orig_output);

        Convolution* fused_op = dynamic_cast<Convolution*>(op);
        ConvParam* fused_param = fused_op->GetParam();

        Convolution* orig_op = dynamic_cast<Convolution*>(orig_input->GetOp());
        ConvParam* orig_param = orig_op->GetParam();

        *fused_param = *orig_param;

        if(relu6)
            fused_param->activation = ActRELU6;
        else
            fused_param->activation = ActRELU;

        Tensor* output_tensor = orig_output->GetOutputTensor(0);
        fused_node->AddOutputTensor(output_tensor);

        Tensor* input_tensor = orig_input->GetInputTensor(0);
        fused_node->AddInputTensor(input_tensor);

        fused.seq_nodes.push_back(fused_node);
        fused.input_nodes.push_back(fused_node);
        fused.output_nodes.push_back(fused_node);
        fused.SetNodeOwner(fused_node);

        /* create new const node for convolution */
        Tensor* weight = orig_input->GetInputTensor(1);

        AddConstNodeToSubGraph(&fused, weight, fused_node, 1);

        bool has_bias = orig_input->GetInputNum() > 2 ? true : false;

        if(has_bias)
        {
            Tensor* orig_bias = orig_input->GetInputTensor(2);
            AddConstNodeToSubGraph(&fused, orig_bias, fused_node, 2);
        }

        graph->Replace(orig, &fused);
    }

    for(unsigned int i = 0; i < orig_sub.size(); i++)
    {
        Subgraph* orig = orig_sub[i];

        delete orig;
    }

    return true;
}
static bool GraphFuseConvReLu(Graph* graph, GraphOptimizer* opt)
{
    return GraphFuseConvReLuCommon(graph, opt, false);
}
static bool GraphFuseConvReLu6(Graph* graph, GraphOptimizer* opt)
{
    return GraphFuseConvReLuCommon(graph, opt, true);
}

}    // namespace TEngine
