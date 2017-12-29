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
#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__

#include <functional>

#include "attribute.hpp"
#include "base_object.hpp"
#include "operator_manager.hpp"
#include "generic_factory.hpp"
#include "tensor_shape.hpp"
#include "static_graph.hpp"

namespace TEngine {


class Node;
class StaticOp;

class Operator: public BaseObject {

public:

    using io_str_t=std::pair<std::string,std::string >;
    using infer_shape_t=std::function<bool(const std::vector<TShape>&,std::vector<TShape>&)>;


    virtual Operator * Clone(void) =0;
 
    virtual void SetSchema(void) {};
    virtual void ParseParam(void) {};
    virtual bool ParamFromStaticOp(StaticOp * s_op) {return true;}

    virtual any GetDefParam(void)  
    {
        return any();
    }

    virtual bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>&oshape)
    { 
        oshape=ishape; 
        return true;
    }

    virtual float GetFops(const std::vector<TShape>& ishape, const std::vector<TShape>&oshape)
    {
        return 0.0f;
    }


    void SetName(const std::string& new_name) { name_=new_name;}
    std::string& GetName(void) { return name_;}
    
    Operator& Input(std::vector<std::string> & args)
    {
       return ParseInputOutput(std::move(args),inputs_);
    }

    Operator& Input(std::vector<std::string> && args)
    {
       return ParseInputOutput(std::move(args),inputs_);
    }
 
    Operator& Input(std::initializer_list<std::string> args)
    {
        std::vector<std::string> input_str(args);
	return ParseInputOutput(std::move(input_str),inputs_);
     }

    Operator& Output(std::vector<std::string> && args)
    {
       return ParseInputOutput(std::move(args),outputs_);
    }

    Operator& Output(std::vector<std::string> & args)
    {
       return ParseInputOutput(std::move(args),outputs_);
    }

	Operator& Output(std::initializer_list<std::string> args)
	{
		std::vector<std::string> output_str(args);
		return ParseInputOutput(std::move(output_str),outputs_);
	}

	Operator& SetLayout(const std::string& layout_str)
	{
		layout_=layout_str;

	    return *this;
     }

     Operator& SetDoc(std::string&& doc_str)
	 {
	    doc_=doc_str;
	    return *this;
     }

     Operator& SetAttr(const std::string& name, const any& val)
     {
          BaseObject::SetAttr(name,val);
          return *this;
     }

     const std::string& GetDoc(void) const { return doc_;}

     int          GetInputNum(void) const { return inputs_.size();}
     const std::string& GetInputName(int idx) const {return inputs_[idx].first;}
     const std::string& GetInputDtype(int idx) const { return inputs_[idx].second;}
     int          GetOutputNum(void) const { return outputs_.size();}
     const std::string& GetOutputName(int idx) const {return outputs_[idx].first;}

     const std::string& GetOutputDtype(int idx) const  { return outputs_[idx].second;}
     const std::string& GetLayout(void) const { return layout_;}



     Operator()=default;
     Operator(const Operator& )=default;

     virtual ~Operator() {};

protected:

	std::string name_;
	std::vector<io_str_t> inputs_;
   	std::vector<io_str_t> outputs_;
   	std::string    layout_;
	std::string    doc_;

private:

   Operator& ParseInputOutput(std::vector<std::string>&& str_vector,
            std::vector<io_str_t>& parsed)
    {

        for(unsigned int i=0;i<str_vector.size();i++)
        {
            std::string& str=str_vector[i];
            std::string::size_type n=str.find(':');

            if(n == std::string::npos)
            {
                /* not found, set the default DataType */
                parsed.emplace_back(str,"float32");
            }
            else
            {
                parsed.emplace_back(str.substr(0,n),str.substr(n+1));
            }
        }

        return *this;

    }

};


template <typename T> 
class OperatorNoParam: public Operator {

public:

   OperatorNoParam(void) =default;
   OperatorNoParam(const OperatorNoParam&) =default;
  virtual ~OperatorNoParam() {}


   Operator * Clone(void) override
   {
      T * new_obj= new T(*dynamic_cast<T *>(this));
      return new_obj;
   }

};


template <typename T, typename P>
class OperatorWithParam: public Operator {

public:

   OperatorWithParam() =default;
   OperatorWithParam(const OperatorWithParam&) =default;
   virtual ~OperatorWithParam() {}


   Operator * Clone(void) override
   {
      T * new_obj= new T(*dynamic_cast<T *>(this));
      return new_obj;
   }


  P* GetParam(void) 
  { 
         return &param_;
  }


  void ParseDefParam(void) 
  {
     ParseParam(param_,this);

  }

  virtual void ParseParam(P& param, Operator * op)
  {
       P::Parse(param,op);
  }


  bool ParamFromStaticOp(StaticOp * s_op) 
  {
     param_=any_cast<P>(s_op->param);
     
     return true;
  }

  any GetDefParam(void)  
  {
     P param;

     ParseParam(param,this);
        
     return param;
  }

protected:

  P param_;

};




using OperatorPtr=std::shared_ptr<Operator>;

using OpFactory=SpecificFactory<Operator>;

template <typename T>
void RegisterOp(const std::string& name)
{
    using op_clone_t=std::function<Operator *(Operator *)>;

    T * p=new T();
    OpManager::SafeAdd(name,p);
    p->SetSchema();

    OpFactory::GetFactory()->RegisterCreator(name,op_clone_t([](Operator * b){
                       return new T(*dynamic_cast<T*>(b));
             }));

}

#define OP_CLONE_METHOD(type) \
         Operator * Clone(void) \
         {\
             type * new_obj=new type(*this); \
             return new_obj;\
         }

#define OP_GET_DEF_PARAM(param_type) \
        any GetDefParam(void) \
        {\
             param_type param;\
             param_type::Parse(param,*this);\
             return param;\
        }

#define CREATE_OPERATOR(name) OpManager::CreateOp(name)

} //namespace TEngine

#endif
