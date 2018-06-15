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
#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <iostream>
#include <string>

#include "base_object.hpp"
#include "tensor_shape.hpp"
#include "logger.hpp"
#include "node.hpp"

namespace TEngine {

class Node;
struct NodePort;
struct StaticConstTensor;

class Tensor : public BaseObject {

public:
	Tensor (const std::string& name)
        {
             name_=name;
             data_type_="float32";
             static_tensor_=nullptr;
        }

	virtual ~Tensor()
        {

            if(type_==kConstTensor 
               && ExistAttr("free_mem")
               && ExistAttr("mem_addr"))
            {
               void * mem=any_cast<void *>(GetAttr("mem_addr"));
               std::free(mem);
            }
        }

	Tensor(const Tensor& other)=default;
	Tensor& operator=(const Tensor& rhs)=default;

        const std::string& GetName(void) const { return name_;}
        void SetName(const std::string& n) { name_=n;}

	const TShape&   GetShape(void)  const   { return shape_;}
	TShape&   GetShape(void)    { return shape_;}

	void Reshape(const TShape& shape);


	const std::string& GetDatatype(void) const { return data_type_;}
        void SetDataType(const std::string& dtype_name){ data_type_=dtype_name;}

	unsigned int GetTotalSize() const;
        void DumpTensor(std::ostream& os) const;


        TensorType GetType(void) const { return type_;}
        void SetType(TensorType t) { type_=t;}

        void AddConsumer(NodePort * in_port)
        {
             consumer.push_back(in_port);
        }

        bool RemoveConsumer(NodePort * in_port)
        {
           auto ir=consumer.begin();

           while(ir!=consumer.end())
           {
               if((*ir)==in_port)
               {
                  consumer.erase(ir);
                  return true;
               }
               ir++;
           }

           return false;
        }

        NodePort * producer;
        std::vector<NodePort *> consumer;

        Node * GetConsumerNode(int idx);

        /* note: as tensor.hpp is defined in representation level,
                so that the memory allocated is only valid for const tensor
                to hold the trained parameters
           please use get_tensor_mem()/set_tensor_mem() to get/set tensor memory
               in operator run functioins

        */

        void * GetMemAddr(void) const
        {
           if(!ExistAttr("mem_addr"))
               return nullptr;
           
            return any_cast<void *>(GetAttr("mem_addr"));
        }

        void SetMemAddr(void * addr)
        {
            (*this)["mem_addr"]=addr;
        }

        void FreeMem(void);
        void BindStaticTensor(StaticConstTensor *);

protected:
        TensorType   type_;
        std::string  name_;
	std::string  data_type_;
	TShape       shape_;
        StaticConstTensor * static_tensor_;

       

};


} //namesapce TEngine



#endif


