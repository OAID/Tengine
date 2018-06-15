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
#ifndef __NODE_HPP__
#define __NODE_HPP__

#include <vector>
#include <string>

#include "base_object.hpp"
#include "operator.hpp"
#include "tensor.hpp"

namespace TEngine {

class Node;
class Tensor;

struct NodePort {
    Node * owner;
    int port_index;
    Tensor * tensor;
};

using NodePortPtr=std::shared_ptr<NodePort>;

class Node : public BaseObject {

public:

    ~Node() { }

    Node(const std::string& name)  { name_=name; op_=nullptr;dynamic_shape_=false;}

    Operator * GetOp(void) const {return op_.get();}

    unsigned int GetInputNum(void)  const { return inputs_.size(); }
    unsigned int GetOutputNum(void) const { return outputs_.size(); }

    int GetNodeIndex(void) const { return index_;}
    const std::string& GetName(void) const { return name_;}

    Tensor * GetInputTensor(int idx) 
    {
         NodePort * ptr=GetNodePort(inputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr->tensor;
    }

    const Tensor * GetInputTensor(int idx) const 
    { 
        const NodePort * ptr=GetNodePort(inputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr->tensor; 
    }

    Tensor * GetOutputTensor(int idx) 
    {
        NodePort * ptr=GetNodePort(outputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr->tensor; 
    }

    const Tensor * GetOutputTensor(int idx) const 
    {
        const NodePort * ptr=GetNodePort(outputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr->tensor; 
    }

    NodePort * GetInputPort (int idx)
    {
        NodePort * ptr=GetNodePort(inputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr;
    }
    const NodePort * GetInputPort (int idx) const
    {
        const NodePort * ptr=GetNodePort(inputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr;
    }

    Tensor * GetInputTensorSeq(int idx)
    {
        return GetInputPortSeq(idx)->tensor;
    }

    NodePort * GetInputPortSeq (int idx)
    {
        return inputs_[idx].get();
    }

    Tensor * GetOutputTensorSeq(int idx)
    {
        return GetOutputPortSeq(idx)->tensor;
    }

    NodePort * GetOutputPortSeq (int idx)
    {
        return outputs_[idx].get();
    }

    const NodePort * GetOutputPort (int idx) const
    {
        const NodePort * ptr=GetNodePort(outputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr;
    }

    NodePort * GetOutputPort (int idx) 
    {
        NodePort * ptr=GetNodePort(outputs_,idx);

        if(ptr==nullptr)
            return nullptr;

        return ptr;
    }

    NodePort * GetNodePort(std::vector<NodePortPtr>& port_list, int idx)
    {
         auto ir=port_list.begin();

         while(ir!=port_list.end())
         {
             if((*ir)->port_index==idx)
                  return (*ir).get();

              ir++;
         }

         return nullptr;
    }

    const NodePort * GetNodePort(const std::vector<NodePortPtr>& port_list, int idx) const
    {
         auto ir=port_list.begin();

         while(ir!=port_list.end())
         {
             if((*ir)->port_index==idx)
                  return (*ir).get();
              ir++;
         }

         return nullptr;
    }

    void AddInputTensor(Tensor * tensor) 
    {
       int idx=inputs_.size();

       SetInputPort(idx,tensor);
    }

    void AddOutputTensor(Tensor * tensor) 
    { 
       int idx=outputs_.size();

       SetOutputPort(idx,tensor);
    }

    void SetInputPort(unsigned int idx, Tensor * tensor) 
    { 

        SetNodePort(inputs_,idx,tensor);
    }

    void SetOutputPort(int idx, Tensor * tensor)
    {
        SetNodePort(outputs_,idx,tensor);
    }

    void SetNodePort(std::vector<NodePortPtr>& port_list, int idx, Tensor * tensor)
    {
        NodePort * port=new NodePort();

        port->owner=this;
        port->port_index=idx;
        port->tensor=tensor;
        auto ir=port_list.begin();

        while(ir!=port_list.end())
        {
            if((*ir)->port_index==idx)
             {
               (*ir).reset(port);
               return;
             }
              ir++;
        }


        port_list.push_back(NodePortPtr(port));
        
    }

    bool RemoveOutputPort(int idx)
    {
        return RemoveNodePort(outputs_,idx);
    }

    bool RemoveInputPort(int idx)
    {
        return RemoveNodePort(inputs_,idx);
    }

    bool RemoveNodePort(std::vector<NodePortPtr>& port_list, int idx)
    {
        auto ir=port_list.begin();
        
        while(ir!=port_list.end())
        {
            if((*ir)->port_index==idx)
             {
               port_list.erase(ir);
               return true;
             }
              ir++;
        }

        return false;
    }

	void MergeAttr(Node * orig);

    std::vector<NodePortPtr>& GetAllInputs(void) { return inputs_;}
    std::vector<NodePortPtr>& GetAllOutputs(void) { return outputs_;}
     

    void SetOp(Operator * op) {return op_.reset(op);}

    void SetNodeIndex(int idx)   {index_=idx;}
    void SetName(const std::string& n) {name_=n;}

    void DumpNode(void);

    float GetFops(void);

    bool IsDynamicShape(void) { return dynamic_shape_;}
    bool SetDynamicShape(bool val) { dynamic_shape_=val; return true; }

    /* Deprecated, should not use in new code */
    int GetParentNum(void);
    Node * GetParentNode(int idx);

    int GetChildNum(void);
    Node * GetChildNode(int idx);  


    inline void* GetRepID(){
        return rep_id;
    }

    inline void SetRepID(void* id){
        rep_id = id;
    }


protected:
     OperatorPtr op_;
     std::vector<NodePortPtr> inputs_;
     std::vector<NodePortPtr> outputs_;

     std::string name_;
     int  index_; //index in seq node list of graph
     bool dynamic_shape_;

     void* rep_id;
};



} //namespace TEngine

#endif
