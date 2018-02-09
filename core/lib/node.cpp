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
#include <iostream>
#include <vector>
#include <string>

#include "static_graph.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "logger.hpp"



namespace TEngine {

void Node::DumpNode(void)
{
    LOG_INFO()<<"\nNode: "<<name_<<" OP: "<<GetOp()->GetName()<<" idx: "<<GetNodeIndex()<<std::endl;
    LOG_INFO()<<"\tInput: "<<inputs_.size()<<" Output: "<<outputs_.size()<<std::endl;

    LOG_INFO()<<"\tInput Tensors:"<<std::endl;
    for(unsigned int i=0;i<inputs_.size();i++)
    {
         Tensor * p=inputs_[i]->tensor;
         LOG_INFO()<<"\t\t"<<inputs_[i]->port_index<<": ";

         Node * parent;
         parent=p->producer->owner;  

         LOG_INFO()<<" from node: "<<parent->GetName()<<" \ttensor: ";

         p->DumpTensor(LOG_INFO());

         LOG_INFO()<<"\n";
        
    }
   
    LOG_INFO()<<"\tOutput Tensors:"<<std::endl;
    for(unsigned int i=0;i<outputs_.size();i++)
    {
         Tensor * p=outputs_[i]->tensor;
         LOG_INFO()<<"\t\t"<<outputs_[i]->port_index<<": ";
         p->DumpTensor(LOG_INFO());
         LOG_INFO()<<" connects to: "<< p->consumer.size()<<" nodes \n";

         for(unsigned k=0;k<p->consumer.size();k++)
         {
             Node * child;
             child=p->consumer[k]->owner;  
             LOG_INFO()<<"\t\tC"<<k<<": " <<child->GetName()<<"\n";
         }


         LOG_INFO()<<"\n";
    }

   
}

int Node::GetParentNum(void)
{
   return GetInputNum();
}

Node * Node::GetParentNode(int idx)
{
    NodePort * port=inputs_[idx].get();
    Tensor * tensor=port->tensor;

    return tensor->producer->owner;
}


/* in-correct implement, should not be used */
int Node::GetChildNum(void)
{
   return GetOutputNum();
}


/* in-correct implement, should not be used */
Node * Node::GetChildNode(int idx)
{
    NodePort * port=outputs_[idx].get();
    Tensor * tensor=port->tensor;

    return tensor->consumer[0]->owner;
}

float Node::GetFops(void)
{
    std::vector<TShape> inputs;

    for(unsigned int i=0;i<GetInputNum();i++)
    {
         Tensor * tensor=GetInputTensor(i);
         inputs.push_back(tensor->GetShape());
    }

    std::vector<TShape> outputs;

    for(unsigned int i=0;i<GetOutputNum();i++)
    {
         Tensor * tensor=GetOutputTensor(i);
         outputs.push_back(tensor->GetShape());
    }

   return op_->GetFops(inputs,outputs);
    
}



} //namespace TEngine
