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

#include "node.hpp"
#include "graph.hpp"
#include "graph_optimizer.hpp"
#include "operator/fused_operator.hpp"
#include "operator/batch_norm.hpp"
#include "operator/convolution.hpp"
#include "operator/relu.hpp"
#include "operator/scale.hpp"


namespace TEngine {

static bool GraphFuseBNScaleReLu(Graph * graph,GraphOptimizer * opt);
static bool GraphFuseConvReLu(Graph * graph,GraphOptimizer * opt);

bool GraphOptimizerManager::RunOpt(const std::string& name,Graph * graph)
{
     if(!Find(name))
        return false;

    GraphOptimizer * opt=Get(name);

    return opt->optimizer(graph,opt);
}


void GraphOptimizerManager::Init(void)
{
   //register a few predefined optimizer

   GraphOptimizer * opt= new GraphOptimizer();

   opt->name="BNScaleReLu";
   opt->optimizer=graph_opt_t(GraphFuseBNScaleReLu);

   Add(opt->name,opt);


   opt=new GraphOptimizer();
   opt->name="ConvReLu";
   opt->optimizer=graph_opt_t(GraphFuseConvReLu);

   Add(opt->name,opt);
}

/* the graph optimizer: conv_relu */
static bool GraphFuseConvReLu(Graph * graph,GraphOptimizer * opt)
{
    int node_number=graph->seq_nodes.size();

    std::vector<Subgraph *> orig_sub;

    for(int i=0;i<node_number;i++)
    {
        Node * node=graph->seq_nodes[i];
        Operator * op=node->GetOp();

        if(op->GetName()!="ReLu")
            continue;

         Tensor * input_tensor=node->GetInputTensor(0);

         Node * conv_node=input_tensor->producer->owner;

         op=conv_node->GetOp();

         if(op->GetName()!="Convolution")
             continue;

         Subgraph * sub= new Subgraph("conv_relu");

         sub->seq_nodes.push_back(conv_node);
         sub->seq_nodes.push_back(node);

         sub->input_nodes.push_back(conv_node);
         sub->output_nodes.push_back(node);

         /* add const node into seq nodes, 
            so that they will be removed from origin graph too */

         for(unsigned int i=1;i<conv_node->GetInputNum();i++)
         {
             Tensor * tensor=conv_node->GetInputTensor(i);
             sub->seq_nodes.push_back(tensor->producer->owner);
         }

        orig_sub.push_back(sub);
    }


    /* construct new node */
    for(unsigned int i=0;i<orig_sub.size();i++)
    {
         Subgraph fused("fused");
         Subgraph * orig=orig_sub[i];

         Node * orig_output=orig->output_nodes[0];
         Node * orig_input=orig->input_nodes[0];

         std::string node_name=orig_input->GetName()+std::string(".fused");

         Node * fused_node=new Node(node_name);
         Operator * op=OpManager::CreateOp("Convolution");

         fused_node->SetOp(op);

         Convolution * fused_op=dynamic_cast<Convolution*>(op);
         ConvParam * fused_param=fused_op->GetParam();

         Convolution * orig_op=dynamic_cast<Convolution *>(orig_input->GetOp());
         ConvParam * orig_param=orig_op->GetParam();

         fused_node->SetAttr("Fused.ReLu",true);

         *fused_param=*orig_param;


        Tensor * output_tensor=orig_output->GetOutputTensor(0);
        fused_node->AddOutputTensor(output_tensor);

        Tensor * input_tensor=orig_input->GetInputTensor(0);
        fused_node->AddInputTensor(input_tensor);

        fused.seq_nodes.push_back(fused_node);
        fused.input_nodes.push_back(fused_node);
        fused.output_nodes.push_back(fused_node);
        fused.SetNodeOwner(fused_node);

        /* create new const node for convolution */
        Tensor * weight=orig_input->GetInputTensor(1);

        Tensor * new_weight=new Tensor(*weight);

        std::string new_tensor_name;

        new_tensor_name=new_weight->GetName()+".fused";
        new_weight->SetName(new_tensor_name);

        Node * new_node;

        new_node=new Node(new_weight->GetName());

        op=OpManager::CreateOp("Const");
        new_node->SetOp(op);

        new_node->AddOutputTensor(new_weight);
        new_weight->producer=new_node->GetOutputPort(0);

        fused_node->AddInputTensor(new_weight);
        new_weight->consumer.clear();
        new_weight->consumer.push_back(fused_node->GetInputPort(1));

        fused.seq_nodes.push_back(new_node);
        fused.SetNodeOwner(new_node);
        fused.SetTensorOwner(new_weight);
 
        bool has_bias=orig_input->GetInputNum()>2?true:false;

        if(has_bias)
        {
             Tensor * orig_bias=orig_input->GetInputTensor(2);
             Tensor * new_bias=new Tensor(*orig_bias);

             new_tensor_name=new_bias->GetName()+".fused";
             new_bias->SetName(new_tensor_name);   

             new_node=new Node(new_bias->GetName());
             op=OpManager::CreateOp("Const");

             new_node->SetOp(op);

             new_node->AddOutputTensor(new_bias);
             new_bias->producer=new_node->GetOutputPort(0);

             fused_node->AddInputTensor(new_bias);
             new_bias->consumer.clear();
             new_bias->consumer.push_back(fused_node->GetInputPort(2));

             fused.seq_nodes.push_back(new_node);
             fused.SetNodeOwner(new_node);
             fused.SetTensorOwner(new_bias);
        }
    

        graph->Replace(orig,&fused);
    }

    for(unsigned int i=0;i<orig_sub.size();i++)
    {
        Subgraph * orig=orig_sub[i];
        
        delete orig; 
    }
   
    return true;
}

/* the graph  optimizer: fuse_bn_scale_relu */

static bool GraphFuseBNScaleReLu(Graph * graph,GraphOptimizer * opt)
{
    int node_number=graph->seq_nodes.size();

    std::vector<Subgraph *> orig_sub;

    
    for(int i=0;i<node_number;i++)
    {
        Node * node=graph->seq_nodes[i];
        Operator * op=node->GetOp();

        if(op->GetName()!="ReLu")
            continue;

         /* check if it is a bn-->scale-->relue seq */

         Tensor * input_tensor;
         Node *   scale_node;
         
         input_tensor=node->GetInputTensor(0);

         scale_node=input_tensor->producer->owner;

         op=scale_node->GetOp();

         if(op->GetName()!="Scale")
               continue;

         input_tensor=scale_node->GetInputTensor(0);

         Node * bn_node=input_tensor->producer->owner;

         op=bn_node->GetOp();

         if(op->GetName()!="BatchNormalization")
               continue;

         /* create a subgraph to represent the chain */

         Subgraph * sub= new Subgraph("relu_chain");


         sub->seq_nodes.push_back(bn_node);
         sub->seq_nodes.push_back(scale_node);
         sub->seq_nodes.push_back(node);
         sub->input_nodes.push_back(bn_node);
         sub->output_nodes.push_back(node);

         /* add const node into seq nodes */

         for(unsigned int i=1;i<bn_node->GetInputNum();i++)
         {
             Tensor * tensor=bn_node->GetInputTensor(i);
             sub->seq_nodes.push_back(tensor->producer->owner);
         }

         for(unsigned int i=1;i<scale_node->GetInputNum();i++)
         {
             Tensor * tensor=scale_node->GetInputTensor(i);
             sub->seq_nodes.push_back(tensor->producer->owner);
         }

         orig_sub.push_back(sub);

   }

   for(unsigned int i=0;i<orig_sub.size();i++)
   {
         Subgraph fused("fused");
         Subgraph * orig=orig_sub[i];

         Node * orig_output=orig->output_nodes[0];
         Node * orig_input=orig->input_nodes[0];

         std::string node_name=orig_output->GetName()+std::string(".fused");

         Node * fused_node=new Node(node_name);
         Operator * op=OpManager::CreateOp(FusedBNScaleReLu::class_name);
         fused_node->SetOp(op);

         Tensor * output_tensor=orig_output->GetOutputTensor(0);
         fused_node->AddOutputTensor(output_tensor);

         Tensor * input_tensor=orig_input->GetInputTensor(0);
         fused_node->AddInputTensor(input_tensor);


         fused.seq_nodes.push_back(fused_node);
         fused.input_nodes.push_back(fused_node);
         fused.output_nodes.push_back(fused_node);
         fused.SetNodeOwner(fused_node);


         Node * orig_bn=orig->seq_nodes[0];
         Node * orig_scale=orig->seq_nodes[1];
         

         /* create new tensors for gamma,beta,mean,var */


         Tensor * orig_gamma=orig_scale->GetInputTensor(1);
         Tensor * orig_beta=orig_scale->GetInputTensor(2);
         Tensor * orig_mean=orig_bn->GetInputTensor(3);
         Tensor * orig_var=orig_bn->GetInputTensor(4);

         Tensor * new_gamma=new Tensor(*orig_gamma);
         Tensor * new_beta=new Tensor(*orig_beta);
         Tensor * new_mean=new Tensor(*orig_mean);
         Tensor * new_var=new Tensor(*orig_var);

         std::string new_tensor_name;

         new_tensor_name=new_gamma->GetName()+".fused";
         new_gamma->SetName(new_tensor_name);

         new_tensor_name=new_beta->GetName()+".fused";
         new_beta->SetName(new_tensor_name);

         new_tensor_name=new_mean->GetName()+".fused";
         new_mean->SetName(new_tensor_name);

         new_tensor_name=new_var->GetName()+".fused";
         new_var->SetName(new_tensor_name);


         /* create new node */

         Node * new_node=new Node(new_gamma->GetName());

         op=OpManager::CreateOp("Const");

         new_node->SetOp(op);
         new_node->AddOutputTensor(new_gamma);
         new_gamma->producer=new_node->GetOutputPort(0);

         fused_node->AddInputTensor(new_gamma);
         new_gamma->consumer.clear();
         new_gamma->consumer.push_back(fused_node->GetInputPort(1));

         fused.seq_nodes.push_back(new_node);
         fused.SetNodeOwner(new_node);
         fused.SetTensorOwner(new_gamma);


         new_node=new Node(new_beta->GetName());

         op=OpManager::CreateOp("Const");

         new_node->SetOp(op);
         new_node->AddOutputTensor(new_beta);
         new_beta->producer=new_node->GetOutputPort(0);

         fused_node->AddInputTensor(new_beta);
         new_beta->consumer.clear();
         new_beta->consumer.push_back(fused_node->GetInputPort(2));

         fused.seq_nodes.push_back(new_node);
         fused.SetNodeOwner(new_node);
         fused.SetTensorOwner(new_beta);

         
         new_node=new Node(new_mean->GetName());

         op=OpManager::CreateOp("Const");

         new_node->SetOp(op);
         new_node->AddOutputTensor(new_mean);
         new_mean->producer=new_node->GetOutputPort(0);

         fused_node->AddInputTensor(new_mean);
         new_mean->consumer.clear();
         new_mean->consumer.push_back(fused_node->GetInputPort(3));

         fused.seq_nodes.push_back(new_node);
         fused.SetNodeOwner(new_node);
         fused.SetTensorOwner(new_mean);


         new_node=new Node(new_var->GetName());

         op=OpManager::CreateOp("Const");

         new_node->SetOp(op);
         new_node->AddOutputTensor(new_var);
         new_var->producer=new_node->GetOutputPort(0);

         fused_node->AddInputTensor(new_var);
         new_var->consumer.clear();
         new_var->consumer.push_back(fused_node->GetInputPort(4));

         fused.seq_nodes.push_back(new_node);
         fused.SetNodeOwner(new_node);
         fused.SetTensorOwner(new_var);


         /* do parameter conversion for mean and var*/

         BatchNorm * bn_op=dynamic_cast<BatchNorm *>(orig_bn->GetOp());
         BatchNormParam * param=bn_op->GetParam();

         const TShape& shape=new_mean->GetShape();

         int channel_number=shape.GetSize();

         float * scale_var_inv=(float *)std::malloc(channel_number*sizeof(float));
         float * scale_mean=(float *)std::malloc(channel_number*sizeof(float));

         float * var=(float *)orig_var->GetMemAddr();
         float * mean=(float *)orig_mean->GetMemAddr();


         float eps=param->eps;
         float rescale_factor=param->rescale_factor?1/param->rescale_factor:0;

         for(int c=0;c<channel_number;c++)
         {
              scale_var_inv[c]=1.f/std::sqrt(var[c]*rescale_factor + eps);
              scale_mean[c]=-mean[c]*rescale_factor*scale_var_inv[c];
         }
         
         new_mean->SetAttr("free_mem",1);
         new_var->SetAttr("free_mem",1);
         new_mean->SetMemAddr(scale_mean);
         new_var->SetMemAddr(scale_var_inv);

         graph->Replace(orig,&fused);
    }


    /* release orig_sub */

    for(unsigned int i=0;i<orig_sub.size();i++)
    {
        Subgraph * orig=orig_sub[i];
        
        delete orig; 
    }
   
    return true;
}


} //namespace TEngine
