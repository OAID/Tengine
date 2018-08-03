
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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */


#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "tf_serializer.hpp"

#include "type_name.hpp"
#include "operator_manager.hpp"
#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/scale_param.hpp"
#include "operator/lrn_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/eltwise.hpp"
#include "operator/relu_param.hpp"
#include "operator/resize_param.hpp"

namespace TEngine {

using op_load_t=std::function<bool(TFNode * tf_node,TFGraph& tf_graph, StaticGraph * graph)>;

namespace tf_serializer {
   static void CreateInputNode(TFNode * tf_node, StaticGraph * graph);
   static bool LoadConstTensor(TFNode * tf_node, StaticGraph * graph);

}

void TFSerializer::DumpTFGraph(TFGraph& tf_graph)
{
   int node_number=tf_graph.seq_nodes.size();

   LOG_INFO()<<"total node number: "<<node_number<<"\n";

   for(int i=0;i<node_number;i++)
   {
      TFNode * node=tf_graph.seq_nodes[i];

      LOG_INFO()<<i<<"\t"<<node->name<<" OP: "<<node->op<<" IN: "<< node->inputs.size()
                <<" OUT: "<<node->outputs.size()<<" PB_DEFS: "<<node->pb_defs.size()<<"\n";

      for(unsigned int j=0;j<node->inputs.size();j++)
      {
          TFNode * input=node->inputs[j];
          LOG_INFO()<<"\tI"<<j<<": "<<input->name<<"  "<<input->op<<"\n";
      }

      for(unsigned int j=0;j<node->outputs.size();j++)
      {
          TFNode * output=node->outputs[j];
          LOG_INFO()<<"\tO"<<j<<": "<<output->name<<"  "<<output->op<<"\n";
      }

   }
}

bool TFSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph * graph)
{
    tensorflow::GraphDef tf_net;

    if(//!LoadTextFile(file_list[0].c_str(), tf_net) &&
       !LoadBinaryFile(file_list[0].c_str(), tf_net))
         return false;

    return LoadGraph(tf_net,graph); 
}

bool TFSerializer::LoadTextFile(const char * fname, tensorflow::GraphDef& tf_net)
{
    std::ifstream is(fname, std::ios::in);

    if(!is.is_open())
    {
        LOG_ERROR()<<"cannot open file: "<<fname<<"\n";
        return false;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    bool ret= google::protobuf::TextFormat::Parse(&input_stream,&tf_net);

    is.close();


    return ret;

}

bool TFSerializer::LoadBinaryFile(const char * fname, tensorflow::GraphDef& tf_net)
{
   std::ifstream is(fname, std::ios::in|std::ios::binary);

   if(!is.is_open())
   {
       LOG_ERROR()<<"cannot open file: "<<fname<<"\n";
       return false;
   }

   google::protobuf::io::IstreamInputStream input_stream(&is);
   google::protobuf::io::CodedInputStream coded_input(&input_stream);

   coded_input.SetTotalBytesLimit(512<<20, 64<<20);

   bool ret=tf_net.ParseFromCodedStream(&coded_input);

   is.close();

   if(!ret)
       LOG_ERROR()<<"parse file: "<<fname<<" failed\n";

   return ret;
}

bool TFSerializer::LoadGraph(tensorflow::GraphDef& tf_net,StaticGraph * graph)
{
     TFGraph tf_graph;

     //step 1: construct whole graph

     if(!ConstructGraph(tf_net,tf_graph))
        return false;


     //step 2: scanning and fusing nodes

     if(!OptimizeGraph(tf_graph))
        return false;

    //step 3: create static graph
     if(!GenerateStaticGraph(tf_graph,graph))
         return false;

    return true;
}

bool TFSerializer::ConstructGraph(tensorflow::GraphDef& tf_net, TFGraph& tf_graph)
{
    int node_number=tf_net.node_size();
    std::unordered_map<std::string, TFNode *> node_map;

    /* first scan, setup all nodes */

    for(int i=0;i<node_number;i++)
    {
         const tensorflow::NodeDef& node_param=tf_net.node(i);

         TFNode * tf_node=new TFNode();

         tf_node->idx=i;
         tf_node->name=node_param.name();
         tf_node->op=node_param.op();
         tf_node->pb_defs.push_back(&tf_net.node(i));

         tf_graph.seq_nodes.push_back(tf_node);

         node_map[tf_node->name]=tf_node;
    }

    /* the second scan, setup connections */
    for(int i=0;i< node_number;i++)
    {
         const tensorflow::NodeDef& node_param=tf_net.node(i);
         const std::string& name=node_param.name();

         TFNode * cur_node=node_map[name];

         for(int j=0;j<node_param.input_size();j++)
         {
             const std::string& input_name=node_param.input(j);

             TFNode * input_node=node_map[input_name];

             if(input_node==nullptr)
             {
                XLOG_ERROR()<<"cannot find input: "<<input_name<<" for node: "<<name<<"\n";
                return false;
             }

             cur_node->inputs.push_back(input_node);
             input_node->outputs.push_back(cur_node);
         }
    }

    return true;
}

void TFSerializer::DisconnectNode(TFNode * cur_node)
{
    TFNode * input_node;

    for(unsigned int i=0;i<cur_node->inputs.size();i++)
    {
         input_node=cur_node->inputs[i];

         auto ir=input_node->outputs.begin();

         while(ir!=input_node->outputs.end())
         {
            if(*ir!=cur_node)
                ir++;
             else
                break;
         }

         if(ir== input_node->outputs.end())
         {
             XLOG_ERROR()<<"ERROR on node connection!!\n";
         }

         input_node->outputs.erase(ir);
    }

    cur_node->inputs.clear();

    TFNode * output_node;

    for(unsigned int i=0;i<cur_node->outputs.size();i++)
    {
         output_node=cur_node->outputs[i];

         auto ir=output_node->inputs.begin();

         while(ir!=output_node->inputs.end())
         {
            if(*ir!=cur_node)
                ir++;
             else
                break;
         }

         if(ir== output_node->inputs.end())
         {
             XLOG_ERROR()<<"ERROR on node connection!!\n";
         }

         output_node->inputs.erase(ir);
    }

    cur_node->outputs.clear();
    
}

bool TFSerializer::MergeParentNode(TFNode * base_node, TFNode * parent_node)
{
      /* remove the input for parent node */

      auto input_ir=base_node->inputs.begin();

      while (input_ir!=base_node->inputs.end())
      {
	    if(*input_ir==parent_node)
		  break;

	    input_ir++;
      }

      base_node->inputs.erase(input_ir);

      /* connect parent's input node and base node */

      base_node->inputs.insert(base_node->inputs.end(),parent_node->inputs.begin(),parent_node->inputs.end());

     for(auto node: parent_node->inputs)
     {
	  for(unsigned int i=0;i<node->outputs.size();i++)
	  {
	       if(node->outputs[i]==parent_node)
		{
			node->outputs[i]=base_node;
			break;
		}
	  }
     }

     /* bridge parent's output*/

     auto output_ir=parent_node->outputs.begin();

     while(output_ir!= parent_node->outputs.end())
     {
	   TFNode * node=*output_ir;

	   if(node!=base_node)
	   {
		  base_node->outputs.push_back(node);

		  for(unsigned int i=0;i<node->inputs.size();i++)
	          {
			  if(node->inputs[i]==parent_node)
			  {
				  node->inputs[i]=base_node;
				  break;
			  }
			  
		  } 
	   }

	   output_ir++;
     }

     /* handle TF definitions */

     base_node->pb_defs.insert(base_node->pb_defs.end(),parent_node->pb_defs.begin(),parent_node->pb_defs.end());

     parent_node->inputs.clear();
     parent_node->outputs.clear();

     return true;
}

bool TFSerializer::CheckComposedBNAdd(TFNode * cur_node)
{ 
     if(cur_node->op != "Add") 
	     return false;

     TFNode * input0=cur_node->inputs[0];
     TFNode * input1=cur_node->inputs[1];

     if(input0->op!= "Mul" || input1->op != "Sub")
	     return false;

     /* further check: batchnorm/add_1 int name */
     if(cur_node->name.find("batchnorm/add_1")!=std::string::npos)
	     return true;

     return false;
}

void TFSerializer::BNRecursiveInputMerge(TFNode * node)
{
     bool mul_1_node=false;

     if(node->name.find("/batchnorm/mul")!=std::string::npos)
     {
         if(node->name.find("/batchnorm/mul_1")!=std::string::npos)
         {
	     mul_1_node=true;
	  }else if(node->name.find("/batchnorm/mul_2")==std::string::npos)
	  {
	      
	    //disconnect the connection between mul and mul2
	    auto ir=node->outputs.begin();

	    if((*ir)->name.find("/batchnorm/mul2")==std::string::npos)
		    ir++;

	    TFNode * mul2_node=*ir;

	    node->outputs.erase(ir);

	    ir=mul2_node->inputs.begin();

	    if((*ir)->name.find("/batchnorm/mul")==std::string::npos)
		    ir++;

	    mul2_node->inputs.erase(ir);
	  }
     }

     int orig_input_size=node->inputs.size();
     std::vector<TFNode *> input_cpy=node->inputs;

     for(int i=0; i<orig_input_size;i++)
     {
	
	if(mul_1_node && i==0) 
		continue;

        TFNode * input_node=input_cpy[i];

	if(input_node->op=="Const")
		continue;

	BNRecursiveInputMerge(input_node);
	MergeParentNode(node,input_node);
     }
}


void TFSerializer::FuseComposedBN(TFNode * cur_node)
{
     BNRecursiveInputMerge(cur_node);
     cur_node->op="ComposedBN";

     /* set new name */
     auto pos=cur_node->name.find("batchnorm/add_1");
     cur_node->name.replace(pos,strlen("batchnorm/add_1"),"bn.fused");

     /* skip to create static node for add/y */

     for(unsigned int i=0;i < cur_node->inputs.size();i++)
     {
	  TFNode * node=cur_node->inputs[i];

	  if(node->name.find("/batchnorm/add/y")!=std::string::npos)
		  node->no_static_node=true;
     }


}

bool TFSerializer::MergeChildNode(TFNode * base_node, TFNode * child_node)
{
     auto output_ir=base_node->outputs.begin();

     while(output_ir!=base_node->outputs.end())
     {
         if(*output_ir==child_node)
           break;
          output_ir++;
     }

     base_node->outputs.erase(output_ir);
     base_node->outputs.insert(base_node->outputs.end(),child_node->outputs.begin(),child_node->outputs.end());
 
     for(auto node: child_node->outputs)
     {
          for(unsigned int i=0;i<node->inputs.size();i++)
          {
               if(node->inputs[i]==child_node)
               {
                   node->inputs[i]=base_node;
                   break;
               }
          }
     }

     
     auto ir=child_node->inputs.begin();

     while(ir!=child_node->inputs.end())
     {
           TFNode * node=*ir;

           if(node!=base_node)
           {
               base_node->inputs.push_back(node);
            
               for(unsigned int i=0;i<node->outputs.size();i++)
               {
                   if(node->outputs[i]==child_node)
                    {
                         node->outputs[i]=base_node;
                         break;
                    }   
               }
           }

            ir++;
     }

     base_node->pb_defs.insert(base_node->pb_defs.end(),child_node->pb_defs.begin(),child_node->pb_defs.end());

     child_node->inputs.clear();
     child_node->outputs.clear();
     
     return true;
}

void TFSerializer::CleanupResizeNearestNeighbor(TFGraph& tf_graph)
{
	auto ir=tf_graph.seq_nodes.begin();

	while(ir!=tf_graph.seq_nodes.end())
        {
            TFNode * cur_node=*ir;

	    if(cur_node->op=="ResizeNearestNeighbor")
	    {
		    TFNode * data_node=cur_node->inputs[0];
		    TFNode * data_shape_node=nullptr;

		    for(unsigned int i=0;i<data_node->outputs.size();i++)
		    {
			   data_shape_node=data_node->outputs[i];

			  if(data_shape_node->op=="Shape")
				  break;
		    }

		    DisconnectNode(data_shape_node);

		    TFNode * mul_node=cur_node->inputs[1];
		    TFNode * stride_slice=mul_node->inputs[0];

		    DisconnectNode(stride_slice);
		    DisconnectNode(mul_node);

	    }

	    ir++;

	}
}

void TFSerializer::MergeReluMinimum(TFGraph & tf_graph)
{
	for(auto ir=tf_graph.seq_nodes.begin(); ir!=tf_graph.seq_nodes.end();ir++)
	{
            TFNode * cur_node=*ir;

	    if(cur_node->inputs.size()==0)
		    continue;

	    TFNode * input0=cur_node->inputs[0];

	    if(cur_node->op=="Minimum" && 
		input0->op=="Relu")
	     {

		  TFNode * const_node=cur_node->inputs[1];

		  DisconnectNode(const_node);

		  MergeChildNode(input0,cur_node);

		  input0->op="Relu6";
		//  input0->op="Relu";
	     }
	}
}

bool TFSerializer::OptimizeGraph(TFGraph& tf_graph)
{
    /* first clean up the predictions module of TF */
    auto ir=tf_graph.seq_nodes.begin();

    while(ir!=tf_graph.seq_nodes.end())
    {
        TFNode * cur_node=*ir;

        if(cur_node->op=="Reshape")
        {
           /* Reshape should have two inputs */

           TFNode * input_node0=cur_node->inputs[0];
           TFNode * input_node1=cur_node->inputs[1];

           if(input_node0->op=="Softmax" ||
              input_node1->op=="Softmax")
            {
                  DisconnectNode(cur_node);
                  ir=tf_graph.seq_nodes.erase(ir);
                   delete cur_node;
                   continue;
                    
            }
           
            TFNode * output_node=cur_node->outputs[0];

            if(output_node->op=="Softmax")
            {
               TFNode * input_node0=cur_node->inputs[0];
               TFNode * input_node1=cur_node->inputs[1];
               TFNode * input_node;

               if(input_node0->op=="Const")
               {
                   DisconnectNode(input_node0);
                   input_node=input_node1;
               }
               else
               {
                   DisconnectNode(input_node1);
                   input_node=input_node0;   
               }

               MergeChildNode(input_node,cur_node);

               ir=tf_graph.seq_nodes.erase(ir);
               delete cur_node;
               continue;
            }
        }

        ir++;
    }

    /* remove the squeeze node and identity */

    ir=tf_graph.seq_nodes.begin();

    while(ir!=tf_graph.seq_nodes.end())
    {
        TFNode * cur_node=*ir;

        if(cur_node->op=="Squeeze")
        {
           TFNode * softmax_node=nullptr;
           TFNode * shape_node=nullptr;

           for(unsigned int j=0;j<cur_node->outputs.size();j++)
           {
              if(cur_node->outputs[j]->op=="Softmax")
                    softmax_node=cur_node->outputs[j];
              else if(cur_node->outputs[j]->op=="Shape")
                      shape_node=cur_node->outputs[j];
           }

           if(softmax_node)
           {
               if(shape_node)
                   DisconnectNode(shape_node);

               TFNode * input_node=cur_node->inputs[0];
               MergeChildNode(input_node,cur_node);
               ir=tf_graph.seq_nodes.erase(ir);
               delete cur_node;
               continue;
           }
        }


        if (cur_node->op == "Identity")
        {
            TFNode *input_node = cur_node->inputs[0];
            MergeChildNode(input_node, cur_node);

            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;
            continue;
        }

        if (cur_node->op == "ConcatV2")
        {
            TFNode *axis_node = nullptr;

            for (unsigned int i = 0; i < cur_node->inputs.size(); i++)
            {
                TFNode *check_node = cur_node->inputs[i];

                if (check_node->op == "Const")
                {
                    axis_node = check_node;
                    break;
                }
            }

            if (axis_node)
            {
                cur_node->pb_defs.push_back(axis_node->pb_defs[0]);

                //remove it from graph
                DisconnectNode(axis_node);

                auto axis_ir = tf_graph.seq_nodes.begin();

                while (*axis_ir != axis_node)
                    axis_ir++;

                ir = tf_graph.seq_nodes.erase(axis_ir);
                continue;
            }
        }

        ir++;
    }

    /* merge biasadd and conv */
    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode *cur_node = *ir;

        if (cur_node->op == "Conv2D" || cur_node->op == "DepthwiseConv2dNative")
        {
            TFNode *output_node = cur_node->outputs[0];

            if (output_node->op == "BiasAdd")
            {
                MergeChildNode(cur_node, output_node);
            }
        }

        ir++;
    }


    /* merge composed BatchNormal */

    ir=tf_graph.seq_nodes.begin();

    while(ir!=tf_graph.seq_nodes.end())
    {
        TFNode * cur_node=*ir;

	if(CheckComposedBNAdd(cur_node))
	    FuseComposedBN(cur_node);
	ir++;
    }

    /* cleanup ResizeNearestNeighbor */
    CleanupResizeNearestNeighbor(tf_graph);

    /* merge Minimum and Relu */

    MergeReluMinimum(tf_graph);


    /* remove no input and output nodes */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode *cur_node = *ir;

        if (cur_node->inputs.size() == 0 &&
            cur_node->outputs.size() == 0)
        {
            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;
        }
        else
            ir++;
    }

    /* remove no input but not placeholder/const nodes */
    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode *cur_node = *ir;

        if (cur_node->inputs.size() == 0 &&
            cur_node->op != "Const" &&
            cur_node->op != "Placeholder")
        {
            DisconnectNode(cur_node);
            tf_graph.seq_nodes.erase(ir);
            delete cur_node;

            ir = tf_graph.seq_nodes.begin(); //restart
        }
        else
            ir++;
    }

    //DumpTFGraph(tf_graph);
    return true;
}

bool TFSerializer::GenerateStaticGraph(TFGraph& tf_graph, StaticGraph * graph)
{
    int node_number=tf_graph.seq_nodes.size();
    int i;

    bool debug_graph=false;
    const char * debug_env=std::getenv("DEBUG_G");
    if((debug_env) && (debug_env[0]=='1'))
    {
        debug_graph=true;
    }
    for(i=0;i< node_number;i++)
    {
        TFNode * tf_node=tf_graph.seq_nodes[i];

        if(debug_graph)
        {
            std::cout<<i<<"\t"<<tf_node->op<<"\t"<<tf_node->name<<"\n";
        }

       if(tf_node->no_static_node)
		continue;

       if(tf_node->op=="Const")
       {
           tf_serializer::LoadConstTensor(tf_node,graph);
           continue;
       }
 
       if(tf_node->op=="Placeholder")
       {
           tf_serializer::CreateInputNode(tf_node,graph);
           continue;
       }

       StaticNode *  node=CreateStaticNode(graph,tf_node->name);

       /* create tensor */
       StaticTensor * tensor=CreateStaticTensor(graph,tf_node->name);

       SetTensorDataLayout(tensor,"NCHW");
       SetTensorDataType(tensor,"float32");

       AddNodeOutputTensor(node,tensor);

       tf_node->static_node=node;
       tf_node->static_tensor=tensor;

       if(!FindOpLoadMethod(tf_node->op))
       {
             LOG_ERROR()<<"cannot find load function for operator: "<<tf_node->op<<"\n";
             break;
       }

       op_load_t op_func=any_cast<op_load_t>(GetOpLoadMethod(tf_node->op));

       if(!op_func(tf_node,tf_graph,graph))
       {
            LOG_ERROR()<<"error on load node: "<<tf_node->name
                       <<" op: "<<tf_node->op<<"\n";
            break;
       }
    }

    if(i<node_number)
        return false;

    return true;
}

namespace tf_serializer {

/*************************************************/

/* 
AvgPool
Conv2D
DepthwiseConv2dNative
FusedBatchNorm
Relu6
Softmax
*/


static bool GetAttrValue(const tensorflow::NodeDef * node, const char* key, tensorflow::AttrValue& value)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node->attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

static void CreateInputNode(TFNode * tf_node, StaticGraph * graph)
{
    StaticNode * node=CreateStaticNode(graph,tf_node->name);

    StaticTensor * tensor=CreateStaticTensor(graph,tf_node->name);

    SetTensorDataLayout(tensor,"NCHW");
    SetTensorDataType(tensor,"float32");

    //if has shape, set it
    tensorflow::AttrValue shape;

    if(GetAttrValue(tf_node->pb_defs[0],"shape",shape))
    {
        std::vector<int> dim;

        dim.resize(4);

        dim[0]=shape.shape().dim(0).size();
        dim[1]=shape.shape().dim(3).size();
        dim[2]=shape.shape().dim(1).size();
        dim[3]=shape.shape().dim(2).size();
       
        SetTensorDim(tensor,dim);
    }
    
    AddNodeOutputTensor(node,tensor);

    StaticOp * op=CreateStaticOp(graph,"InputOp");
    SetNodeOp(node,op);

    AddGraphInputNode(graph,node);

    tf_node->static_node=node;
    tf_node->static_tensor=tensor;
    
}

static void GetScalarTensor(const tensorflow::TensorProto& tf_tensor, std::vector<int>& dim, void * mem_ptr, std::string& layout)
{
    dim.push_back(1);
    float * dst=(float *)mem_ptr;

    dst[0]=tf_tensor.float_val(0);

    layout="W";
}

static void Get1DimTensor(const tensorflow::TensorProto& tf_tensor, std::vector<int>& dim, void * mem_ptr, std::string& layout)
{
      const tensorflow::TensorShapeProto& shape = tf_tensor.tensor_shape();

      dim.push_back(shape.dim(0).size());

      const float * src=reinterpret_cast<const float*>(tf_tensor.tensor_content().c_str());
      float * dst=(float *)mem_ptr;

      std::memcpy(dst,src,dim[0]*sizeof(float));
 
      layout="W";

}

static void Get2DimTensor(const tensorflow::TensorProto& tf_tensor, std::vector<int>& dim, void * mem_ptr,std::string& layout)
{
      dim.resize(2);
      const tensorflow::TensorShapeProto& shape = tf_tensor.tensor_shape();

      dim[0]=shape.dim(0).size();
      dim[1]=shape.dim(1).size();

      const float * src=reinterpret_cast<const float*>(tf_tensor.tensor_content().c_str());
      float * dst=(float *)mem_ptr;

      std::memcpy(dst,src,dim[0]*dim[1]*sizeof(float));

      layout="HW";
}

static void Get4DimTensor(const tensorflow::TensorProto& tf_tensor, std::vector<int>& dim, void * mem_ptr, std::string& layout)
{
      dim.resize(4);
      const tensorflow::TensorShapeProto& shape = tf_tensor.tensor_shape();

      dim[0]=shape.dim(3).size();
      dim[1]=shape.dim(2).size();
      dim[2]=shape.dim(0).size();
      dim[3]=shape.dim(1).size();

      const float * src=reinterpret_cast<const float*>(tf_tensor.tensor_content().c_str());
      float * dst=(float *)mem_ptr;

      for(int n=0;n<dim[0];n++)
         for(int c=0;c<dim[1];c++)
            for(int h=0;h<dim[2];h++)
              for(int w=0;w<dim[3];w++)
        {
              *dst++=src[h*dim[3]*dim[1]*dim[0]+w*dim[1]*dim[0]+c*dim[0]+n];
        }

     layout="NCHW";
 
}

static void GetTensorContentAndDim(const tensorflow::TensorProto& tf_tensor, std::vector<int>& dim, void * mem_ptr, std::string& layout )
{
    const tensorflow::TensorShapeProto& shape = tf_tensor.tensor_shape();

    int dim_size=shape.dim_size();

    switch(dim_size)
    {
      case 0:
           GetScalarTensor(tf_tensor,dim,mem_ptr,layout);
           break;
      case 1:
           Get1DimTensor(tf_tensor,dim,mem_ptr,layout);
           break;
      case 2:
           Get2DimTensor(tf_tensor,dim,mem_ptr,layout);
           break;
      case 4:
           Get4DimTensor(tf_tensor,dim,mem_ptr,layout);
           break;
      default:
           break;
    }
        
}

static void * LoadConstParam(TFNode * tf_node)
{
    tensorflow::AttrValue value;

    const tensorflow::NodeDef * node_def=tf_node->pb_defs[0];

    if(GetAttrValue(node_def,"value",value))
    {
        const tensorflow::TensorProto& tf_tensor=value.tensor();


        int mem_size=tf_tensor.tensor_content().size();

        if(mem_size==0)
            mem_size=sizeof(float);

        void * mem_ptr=std::malloc(mem_size);

        std::vector<int> dims;
        std::string layout;

        GetTensorContentAndDim(tf_tensor,dims,mem_ptr,layout);
     
        return mem_ptr;	
    }

    return nullptr;
}


static bool LoadConstTensor(TFNode * tf_node, StaticGraph * graph)
{
    StaticNode * node=CreateStaticNode(graph,tf_node->name);
    StaticTensor * tensor=CreateStaticConstTensor(graph,tf_node->name);

    SetTensorDataType(tensor,"float32");

    tensorflow::AttrValue value;

    const tensorflow::NodeDef * node_def=tf_node->pb_defs[0];

    if(GetAttrValue(node_def,"value",value))
    {
        const tensorflow::TensorProto& tf_tensor=value.tensor();


        int mem_size=tf_tensor.tensor_content().size();

        if(mem_size==0)
            mem_size=sizeof(float);

        void * mem_ptr=std::malloc(mem_size);

        std::vector<int> dims;
        std::string layout;

        GetTensorContentAndDim(tf_tensor,dims,mem_ptr,layout);
        
        SetTensorDim(tensor,dims);
        SetTensorSize(tensor,mem_size);
        SetTensorDataLayout(tensor,layout);
        SetConstTensorBuffer(tensor,mem_ptr);
    }

    SetConstTensorFileLocation(tensor,-1,0);

    AddNodeOutputTensor(node,tensor);

    StaticOp * const_op=CreateStaticOp(graph,"Const");
    SetNodeOp(node,const_op);

    tf_node->static_node=node;
    tf_node->static_tensor=tensor;

    return true;
}

static bool LoadConv2D(TFNode * tf_node,TFGraph& tf_graph, StaticGraph * graph)
{
    /* handle inputs first */
    TFNode * input0=tf_node->inputs[0]; /* input */
    TFNode * input1=tf_node->inputs[1]; /* weight */

    StaticNode * node=tf_node->static_node;

    AddNodeInputTensor(node,input0->static_tensor);
    AddNodeInputTensor(node,input1->static_tensor);

    if(tf_node->inputs.size()>2)
    {
        TFNode * input2=tf_node->inputs[2];
        AddNodeInputTensor(node,input2->static_tensor);
    }

    /* conv param */

    const tensorflow::NodeDef * node_def=tf_node->pb_defs[0];
    
    ConvParam  param=any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));

    tensorflow::AttrValue value;

    if(GetAttrValue(node_def,"dilations",value))
    {
         param.dilation_h=value.list().i(1);
         param.dilation_h=value.list().i(2);
    }
    
   if(GetAttrValue(node_def,"padding",value))
   {
        if(value.s()=="VALID")
        {
           param.pad_h=0;
           param.pad_w=0;
        }
        else if(value.s()=="SAME")
        {
           param.pad_h=-1;
           param.pad_w=-1;
        }
   }

   if(GetAttrValue(node_def,"strides",value))
   {
       param.stride_h=value.list().i(1);
       param.stride_w=value.list().i(2);
   }

   int in_channel=1,out_channel=1,kernel_h=0,kernel_w=0;
   int group=1;
   //Tensorflow has to get those information from weights

   const tensorflow::NodeDef * weight_def=input1->pb_defs[0];

   if(GetAttrValue(weight_def,"value",value))
   {
       const tensorflow::TensorShapeProto& shape = value.tensor().tensor_shape();

       kernel_h=shape.dim(0).size();
       kernel_w=shape.dim(1).size();
       in_channel=shape.dim(2).size();
       out_channel=shape.dim(3).size();
   }

   if(tf_node->op=="DepthwiseConv2dNative")
   {
        group=in_channel;
        out_channel=in_channel*out_channel;

        StaticTensor * weight_tensor=input1->static_tensor;

        //reset tensor's shape
        std::vector<int> dims;

        dims.push_back(out_channel);
        dims.push_back(1);
        dims.push_back(kernel_h);
        dims.push_back(kernel_w);

        SetTensorDim(weight_tensor,dims);
   }
  
   param.kernel_h=kernel_h;
   param.kernel_w=kernel_w;
   param.output_channel=out_channel;
   param.group=group;

   StaticOp * op=CreateStaticOp(graph,"Convolution");

   SetOperatorParam(op,param);

   SetNodeOp(node,op);

   return true;
}

static bool LoadPool(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
   TFNode * input=tf_node->inputs[0];
   StaticNode * node=tf_node->static_node;

   AddNodeInputTensor(node,input->static_tensor);

   PoolParam  param=any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

   const tensorflow::NodeDef * node_def=tf_node->pb_defs[0];
   tensorflow::AttrValue  value;

   if(GetAttrValue(node_def,"ksize",value))
   {
     param.kernel_h=value.list().i(1);
     param.kernel_w=value.list().i(2);
   }

   if(GetAttrValue(node_def,"strides",value))
   {
     param.stride_h=value.list().i(1);
     param.stride_w=value.list().i(2);
   }

   if(GetAttrValue(node_def,"padding",value))
   {
      if(value.s()=="VALID")
      {
        param.pad_h=0;
        param.pad_w=0;
      }
      else if(value.s()=="SAME")
      {
        param.pad_h=-1;
        param.pad_w=-1;
      }
   }


   if(tf_node->op=="AvgPool")
   {
       param.alg=kPoolAvg;
   }
   else if(tf_node->op=="MaxPool")
   {
       param.alg=kPoolMax;
   }

   //convert to onnx format
   param.kernel_shape.resize(2);
   param.kernel_shape[0]=param.kernel_h;
   param.kernel_shape[1]=param.kernel_w;
   
   param.pads.resize(4);
   param.pads[0]=param.pad_h;
   param.pads[1]=param.pad_w;
   param.pads[2]=param.pad_h;
   param.pads[3]=param.pad_w;    

   param.strides.resize(2);
   param.strides[0]=param.stride_h;
   param.strides[1]=param.stride_w;

   StaticOp * op=CreateStaticOp(graph,"Pooling");
   SetOperatorParam(op,param);
   SetNodeOp(node,op);

   return true;
}

static bool LoadBatchNorm(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
    TFNode * input0=tf_node->inputs[0];
    TFNode * gamma=tf_node->inputs[1];
    TFNode * beta=tf_node->inputs[2];
    TFNode * mean=tf_node->inputs[3];
    TFNode * var=tf_node->inputs[4];

    StaticNode * node=tf_node->static_node;

    AddNodeInputTensor(node,input0->static_tensor);
    AddNodeInputTensor(node,gamma->static_tensor);   
    AddNodeInputTensor(node,beta->static_tensor);   
    AddNodeInputTensor(node,mean->static_tensor);   
    AddNodeInputTensor(node,var->static_tensor);   

    BatchNormParam param=any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    const tensorflow::NodeDef * node_def=tf_node->pb_defs[0];
    tensorflow::AttrValue  value;

    if(GetAttrValue(node_def,"epsilon",value))
    {
       param.eps=value.f();
    }
    
    StaticOp * op=CreateStaticOp(graph,"BatchNormalization");
    SetOperatorParam(op,param);
    SetNodeOp(node,op);

    return true;

}

static bool LoadSoftmax(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
    TFNode * input=tf_node->inputs[0];
    StaticNode * node=tf_node->static_node;
    AddNodeInputTensor(node,input->static_tensor);

    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));
    /* it seems tensorflow justs support last dimension */

    StaticOp * op=CreateStaticOp(graph,"Softmax");
    SetOperatorParam(op,param);
    SetNodeOp(node,op);

    AddGraphOutputNode(graph,node);

    return true;
}

static bool LoadRelu(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
    TFNode * input=tf_node->inputs[0];
    StaticNode * node=tf_node->static_node;

    AddNodeInputTensor(node,input->static_tensor);

    ReLuParam  param=any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    param.negative_slope=0.f;

    StaticOp * op=CreateStaticOp(graph,"ReLu");
    SetOperatorParam(op,param);
    SetNodeOp(node,op);

    return true;
}

static bool LoadResize(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
    TFNode * input=tf_node->inputs[0];
    StaticNode * node=tf_node->static_node;

    AddNodeInputTensor(node,input->static_tensor);

    ResizeParam  param=any_cast<ResizeParam>(OpManager::GetOpDefParam("Resize"));
    param.scale_h  = 2;
    param.scale_w  = 2;
    param.type = 0;
    StaticOp * op=CreateStaticOp(graph,"Resize");
    SetOperatorParam(op,param);
    SetNodeOp(node,op);

    return true;
}



static bool LoadRelu6(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
    TFNode * input=tf_node->inputs[0];
    StaticNode * node=tf_node->static_node;

    AddNodeInputTensor(node,input->static_tensor);

    StaticOp * op=CreateStaticOp(graph,"ReLu6");
    SetNodeOp(node,op);

    return true;
}

static int nhwc_axis_swap[]={0,2,3,1};

static bool LoadConcat(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{
    TFNode * input;
    StaticNode * node=tf_node->static_node;

    for(unsigned int i=0;i<tf_node->inputs.size();i++)
    {
        input=tf_node->inputs[i];
        AddNodeInputTensor(node,input->static_tensor);
    }


    
    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));

    const tensorflow::NodeDef * node_def=tf_node->pb_defs[1];
    tensorflow::AttrValue  value;

    if(GetAttrValue(node_def,"value",value))
    {
        const tensorflow::TensorProto& tf_tensor=value.tensor();

        int axis=tf_tensor.int_val(0);

        //TF is NHWC, TEngine is NCHW
        param.axis=nhwc_axis_swap[axis];
    }

    StaticOp * op=CreateStaticOp(graph,"Concat");
    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    return true;
}

static EltType MapEltwise(TFNode * tf_node, const std::string& elt_op)
{
     if(elt_op=="Add")
         return ELT_SUM;
     else if(elt_op=="Mul")
         return ELT_PROD;
     else if(elt_op=="Sub")
         return ELT_SUB;
     else if(elt_op=="Rsqrt")
         return ELT_RSQRT;
     else if(elt_op =="Minimum")
         return ELT_MIN_SCALAR;
     else
         return ELT_LAST;
}

static bool LoadEltwise(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{

    //sanity check
    if(tf_node->op=="Add" || tf_node->op=="Mul" || tf_node->op=="Sub" || tf_node->op=="Minimum")
    {
        if(tf_node->inputs.size()!=2)
            return false; 

    }
    else if(tf_node->op=="Rsqrt")
    {
        if(tf_node->inputs.size()!=1)
            return false; 
    }
    else 
    {
        XLOG_ERROR()<<"Unsupported op: "<<tf_node->op<<"\n";
        return false; 
    }
    
    StaticNode * node=tf_node->static_node;

    for(unsigned int i=0;i<tf_node->inputs.size();i++)
    {
        AddNodeInputTensor(node,tf_node->inputs[i]->static_tensor);
    }

    StaticOp * op=CreateStaticOp(graph,"Eltwise");

    EltwiseParam  param=any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

    param.type=MapEltwise(tf_node,tf_node->op);

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    return true;
}

static bool LoadComposedBN(TFNode * tf_node, TFGraph& tf_graph, StaticGraph * graph)
{

    TFNode * input0=tf_node->inputs[0];
    TFNode * gamma=tf_node->inputs[1];
    TFNode * var=tf_node->inputs[2];
    TFNode * add_y=tf_node->inputs[3];
    TFNode * beta=tf_node->inputs[4];
    TFNode * mean=tf_node->inputs[5];

    StaticNode * node=tf_node->static_node;

    AddNodeInputTensor(node,input0->static_tensor);
    AddNodeInputTensor(node,gamma->static_tensor);   
    AddNodeInputTensor(node,beta->static_tensor);   
    AddNodeInputTensor(node,mean->static_tensor);   
    AddNodeInputTensor(node,var->static_tensor);   

    BatchNormParam param=any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    /* add_y is epison in deed */

    float * eps_ptr=(float *)LoadConstParam(add_y);

    param.eps=eps_ptr[0];

    free(eps_ptr);

   // printf("eps=%.20f\n",param.eps);
		    
    StaticOp * op=CreateStaticOp(graph,"BatchNormalization");
    SetOperatorParam(op,param);
    SetNodeOp(node,op);

    return true;
}

} //namespace tf_serializer

using namespace tf_serializer;

bool TFSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;
    
    if(!SerializerManager::SafeGet("tensorflow",serializer))
          return false; 

    TFSerializer * p_tf=dynamic_cast<TFSerializer *>(serializer.get());

    p_tf->RegisterOpLoadMethod("AvgPool",op_load_t(LoadPool));
    p_tf->RegisterOpLoadMethod("MaxPool",op_load_t(LoadPool));
    p_tf->RegisterOpLoadMethod("Conv2D",op_load_t(LoadConv2D));
    p_tf->RegisterOpLoadMethod("DepthwiseConv2dNative",op_load_t(LoadConv2D));
    p_tf->RegisterOpLoadMethod("FusedBatchNorm",op_load_t(LoadBatchNorm));
    p_tf->RegisterOpLoadMethod("Relu6",op_load_t(LoadRelu6));
    p_tf->RegisterOpLoadMethod("Relu",op_load_t(LoadRelu));
    p_tf->RegisterOpLoadMethod("Softmax",op_load_t(LoadSoftmax));
    p_tf->RegisterOpLoadMethod("ConcatV2",op_load_t(LoadConcat));
    p_tf->RegisterOpLoadMethod("Add",op_load_t(LoadEltwise));
    p_tf->RegisterOpLoadMethod("Sub",op_load_t(LoadEltwise));
    p_tf->RegisterOpLoadMethod("Mul",op_load_t(LoadEltwise));
    p_tf->RegisterOpLoadMethod("Minimum",op_load_t(LoadEltwise));
    p_tf->RegisterOpLoadMethod("Rsqrt",op_load_t(LoadEltwise));
    p_tf->RegisterOpLoadMethod("ResizeNearestNeighbor",op_load_t(LoadResize));
    p_tf->RegisterOpLoadMethod("ComposedBN",op_load_t(LoadComposedBN));

    return true;
}

void test_tfserializer(void)
{
    std::vector<std::string> file_list;   

    const char * model_fname="/home/haitao/workshop/Tengine_models/mobilenet/tensorflow/frozen_mobilenet_v1_224.prototxt";
    //const char * model_fname="/home/haitao/workshop/Tengine_models/mobilenet/tensorflow/frozen_mobilenet_v1_224.pb";
    //const char * model_fname="/home/haitao/github/tensorflow/tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb";

    file_list.push_back(model_fname);

    /* test */

    SerializerPtr p_tf;

    SerializerManager::SafeGet("tensorflow",p_tf);
    StaticGraph * graph=CreateStaticGraph("test");

   if(!p_tf->LoadModel(file_list,graph))
   {
       LOG_ERROR()<<"Load model failed\n";
       return;
   }

   LOG_INFO()<<"Load model successfully\n";

   DumpStaticGraph(graph);

  if( CheckGraphIntegraity(graph))
      LOG_INFO()<<"check passed\n";

}




} //namespace TEngine

