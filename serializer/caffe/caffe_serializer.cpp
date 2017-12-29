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
#include <functional>
#include <unordered_map>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>


#include "type_name.hpp"
#include "caffe_serializer.hpp"
#include "operator_manager.hpp"
#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/scale_param.hpp"
#include "operator/lrn_param.hpp"
#include "operator/softmax_param.hpp"


namespace TEngine {

using op_load_t=std::function<bool(StaticGraph *, StaticNode * ,const caffe::LayerParameter&)>;
using blob_load_t=std::function<bool(StaticGraph *, StaticNode * , const caffe::LayerParameter&)>;

std::unordered_map<std::string,blob_load_t> blob_load_map;

bool CaffeSingle::LoadBinaryFile(const char * fname, caffe::NetParameter& caffe_net)
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

   bool ret=caffe_net.ParseFromCodedStream(&coded_input);

   is.close();

   if(!ret)
       LOG_ERROR()<<"parse file: "<<fname<<" failed\n";

   return ret;
}

bool CaffeSingle::LoadTextFile(const char * fname, caffe::NetParameter& caffe_net)
{
      std::ifstream is(fname, std::ios::in);

      if(!is.is_open())
      {
          LOG_ERROR()<<"cannot open file: "<<fname<<"\n";
          return false;
      }

     google::protobuf::io::IstreamInputStream input_stream(&is);
     bool ret= google::protobuf::TextFormat::Parse(&input_stream,&caffe_net);

     is.close();

     if(!ret)
          LOG_ERROR()<<"parse file: "<<fname<<" failed\n";

     return ret;
}

bool CaffeSingle::LoadModel(const std::vector<std::string>& file_list, StaticGraph * graph)
{
    caffe::NetParameter caffe_net;

    if(file_list.size()!=GetFileNum())
          return false;

    if(!LoadBinaryFile(file_list[0].c_str(), caffe_net))
         return false;


    SetGraphSource(graph,file_list[0]);
    SetGraphSourceFormat(graph,"caffe");
    SetGraphConstTensorFile(graph,file_list[0]);

    return LoadGraph(caffe_net,graph); 

}

bool CaffeSingle::LoadNode(StaticGraph * graph, StaticNode * node,const caffe::LayerParameter& layer_param, name_map_t& tensor_name_map)
{
     for(int i=0;i<layer_param.bottom_size();i++)
     {
         const std::string& orig_name=layer_param.bottom(i);

         std::string& tensor_name=tensor_name_map[orig_name];

         StaticTensor * tensor=FindTensor(graph,tensor_name);

         AddNodeInputTensor(node,tensor);
     }

     for(int i=0;i<layer_param.top_size();i++)
     {

         const std::string& orig_name=layer_param.top(i);
         std::string tensor_name;

        if(tensor_name_map.count(orig_name))
             tensor_name=GetNodeName(node)+"/"+std::to_string(i);
        else
             tensor_name=orig_name;

         StaticTensor * tensor=CreateStaticTensor(graph,tensor_name);

         SetTensorDataLayout(tensor,"NCHW");
         SetTensorDataType(tensor,"float32");

         AddNodeOutputTensor(node,tensor);

         //record the name mapping

         tensor_name_map[orig_name]=tensor_name;
     }

     return true;
}


bool CaffeSingle::LoadGraph(caffe::NetParameter& caffe_net, StaticGraph * graph)
{
     SetGraphIdentity(graph, "caffe",caffe_net.name(),"0");

     name_map_t tensor_name_map;

     int layer_num=caffe_net.layer_size();
     int i;

     for(i=0;i<layer_num;i++)
     {
          const caffe::LayerParameter& layer_param=caffe_net.layer(i);
          const std::string& caffe_op_name=layer_param.type();

          if(!FindOpLoadMethod(caffe_op_name))
          {
               LOG_ERROR()<<"cannot find load function for operator: "<<caffe_op_name<<"\n";
               break;
          }

          StaticNode *  node=CreateStaticNode(graph,layer_param.name());

          if(!LoadNode(graph,node,layer_param,tensor_name_map))
              break;

          op_load_t op_func=any_cast<op_load_t>(GetOpLoadMethod(caffe_op_name));

           if(!op_func(graph,node,layer_param))
               break;
     }

     if(i<layer_num)
           return false;

     return true;    
  
}

bool CaffeBuddy::LoadModel(const std::vector<std::string>& file_list, StaticGraph * graph)
{
    if(file_list.size()!=GetFileNum())
         return false;

     caffe::NetParameter test_net;

     if(!LoadTextFile(file_list[0].c_str(),test_net))
     {
        std::cout<<"FAILED\n";
        return false;
     }


     caffe::NetParameter train_net;

     if(!LoadBinaryFile(file_list[1].c_str(), train_net))
            return false;

     SetGraphSource(graph,file_list[1]);
     SetGraphSourceFormat(graph,"caffe");
     SetGraphConstTensorFile(graph,file_list[1]);

     return LoadGraph(test_net,train_net,graph);
 
}

bool CaffeBuddy::LoadGraph(caffe::NetParameter& test_net, caffe::NetParameter& train_net,StaticGraph * graph)
{
     name_map_t tensor_name_map;

     SetGraphIdentity(graph, "caffe",test_net.name(),"0");

     /* create the layer name map of the train_net */
     std::unordered_map<std::string, const caffe::LayerParameter *> train_name_map;

     int layer_number;

     layer_number=train_net.layer_size();

     int i;

     for(i=0;i<layer_number;i++)
     {
        const caffe::LayerParameter& layer_param=train_net.layer(i);

        train_name_map[layer_param.name()]=&layer_param;
     }

     layer_number=test_net.layer_size();
     int n;

     for(n=0;n<layer_number;n++)
     {
        const caffe::LayerParameter& layer_param=test_net.layer(n);
        const std::string& caffe_op_name=layer_param.type();

        if(!FindOpLoadMethod(caffe_op_name))
        {
               LOG_ERROR()<<"cannot find load function for operator: "<<caffe_op_name<<"\n";
               break;
        }

        StaticNode *  node=CreateStaticNode(graph,layer_param.name());

        if(!LoadNode(graph,node,layer_param,tensor_name_map))
              break;

        op_load_t op_func=any_cast<op_load_t>(GetOpLoadMethod(caffe_op_name));

        if(!op_func(graph,node,layer_param))
               break;

         /*Load pre-trained parameters*/
        if(train_name_map.count(layer_param.name()))
        {
            const caffe::LayerParameter * p_train;

            p_train=train_name_map[layer_param.name()];

            if(p_train->blobs_size())
            {
                 blob_load_t func=blob_load_map[caffe_op_name];

                 if(!func(graph,node,*p_train))
                     break;
            }

        }

     }


     if(n<layer_number)
          return false;

     return true;
}




static void LoadCaffeBlob(StaticGraph * graph, StaticNode * node, const std::vector<std::string>& name_list, 
                          const std::vector<std::string>& layout_list,
                          const caffe::LayerParameter& layer_param)

{
    unsigned int blob_num=layer_param.blobs_size();

    

    for(unsigned int i=0;i<blob_num && i< name_list.size();i++)
    {
       std::string new_tensor_name=GetNodeName(node)+"/"+name_list[i];

       StaticTensor * tensor=CreateStaticConstTensor(graph,new_tensor_name);


       /* load tensor data*/

       const caffe::BlobProto& blob=layer_param.blobs(i);

       std::vector<int> dims;

       if(blob.has_shape())
       {

           for(int i=0;i<blob.shape().dim_size();i++)
           {
                dims.push_back(blob.shape().dim(i));
           }
       }
       else
       {
          std::vector<int> temp;
          temp.push_back(blob.num());
          temp.push_back(blob.channels());
          temp.push_back(blob.height());
          temp.push_back(blob.width());

          int start=0;

          while(temp[start]==1)  start++;

          for(unsigned int i=start;i<temp.size();i++)
              dims.push_back(temp[i]);

       }

       SetTensorDim(tensor,dims);
       SetTensorDataType(tensor,"float32");
       SetTensorDataLayout(tensor,layout_list[i]);

       int mem_size=blob.data_size()*4;

       SetTensorSize(tensor,mem_size);

       float * ptr=(float *)std::malloc(mem_size);

       for(int i=0;i<blob.data_size();i++)
              ptr[i]=blob.data(i);

       SetConstTensorBuffer(tensor,ptr);
       SetConstTensorFileLocation(tensor,-1,0);

       StaticNode * new_node=CreateStaticNode(graph,new_tensor_name);

       StaticOp * const_op=CreateStaticOp(graph,"Const");

       SetNodeOp(new_node,const_op);

       AddNodeOutputTensor(new_node,tensor);

       AddNodeInputTensor(node,tensor);

   }


}

static void CreatePresetNode(StaticGraph * graph, StaticNode * node, const char * name, const char * layout,
               std::vector<int>& dims, float val)
{
     std::string new_tensor_name=GetNodeName(node)+"/"+name;
     StaticTensor * tensor=CreateStaticConstTensor(graph,new_tensor_name);

     SetTensorDim(tensor,dims);
     SetTensorDataType(tensor,"float32");
     SetTensorDataLayout(tensor,layout);
   
     int elem_size=1;

     for(unsigned int i=0;i < dims.size();i++)
     {
         elem_size*=dims[i];
     }

     SetTensorSize(tensor,elem_size*sizeof(float));

     float* ptr=(float *)std::malloc(elem_size*sizeof(float));

     for(int i=0;i<elem_size;i++)
         ptr[i]=val;

     SetConstTensorBuffer(tensor,ptr);
     SetConstTensorFileLocation(tensor,-1,0);

     StaticNode * new_node=CreateStaticNode(graph,new_tensor_name);

     StaticOp * const_op=CreateStaticOp(graph,"Const");

     SetNodeOp(new_node,const_op);

     AddNodeOutputTensor(new_node,tensor);

     AddNodeInputTensor(node,tensor);
}

static bool LoadBatchNormBlob(StaticGraph * graph, StaticNode* node, const caffe::LayerParameter& layer_param)
{

    const caffe::BlobProto& rescale_blob=layer_param.blobs(2);

    StaticOp * op=GetNodeOp(node);

    BatchNormParam param=any_cast<BatchNormParam>(GetOperatorParam(op));

    param.rescale_factor=rescale_blob.data(0);

    SetOperatorParam(op,param);

    /* for compatible reason, create the two tensors: gamma (1.0) and beta (0.0) */

    /* get the dim, i.e., channel size */

    const caffe::BlobProto& mean_blob=layer_param.blobs(0);

    std::vector<int> dims;
    dims.push_back(mean_blob.shape().dim(0));

    CreatePresetNode(graph,node,"gamma","W",dims,1.0f);
    CreatePresetNode(graph,node,"beta","W",dims,0.0f);

    std::vector<std::string> name_list={"means","vars"};
    std::vector<std::string> layout_list={"W","W"};

    LoadCaffeBlob(graph, node,name_list,layout_list,layer_param);

    return true;
}

static bool LoadFullyConnectedBlob(StaticGraph * graph, StaticNode* node, const caffe::LayerParameter& layer_param)
{
    std::vector<std::string> name_list={"weight","bias"};
    std::vector<std::string> layout_list={"HW","W"};

    LoadCaffeBlob(graph, node,name_list,layout_list,layer_param);
    
    return true;
}

static bool LoadScaleBlob(StaticGraph * graph, StaticNode* node, const caffe::LayerParameter& layer_param)
{
    std::vector<std::string> name_list={"gamma","beta"};
    std::vector<std::string> layout_list={"CHW","W"};

    LoadCaffeBlob(graph, node,name_list,layout_list,layer_param);
    
    return true;
}


static bool LoadConvolutionBlob(StaticGraph * graph, StaticNode* node, const caffe::LayerParameter& layer_param)
{
     std::vector<std::string> name_list={"weight","bias"};
     std::vector<std::string> layout_list={"NCHW","W"};

     LoadCaffeBlob(graph, node,name_list,layout_list,layer_param);

     return true;
}

static bool  LoadCaffeInputOp(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    StaticOp * op=CreateStaticOp(graph,"InputOp");

    SetNodeOp(node,op);

    const caffe::InputParameter& input_param=layer_param.input_param();

    if(input_param.shape_size())
    {
       std::vector<int> dim;
       const caffe::BlobShape& blob_shape=input_param.shape(0);

       for(int i=0;i<blob_shape.dim_size();i++)
       {
          dim.push_back(blob_shape.dim(i));
       }

       StaticTensor * tensor=GetNodeOutputTensor(graph,node,0);

       SetTensorDim(tensor,dim);
    }
	
    AddGraphInputNode(graph,node);

    return true;     
}

static bool  LoadCaffeSoftMax(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    const caffe::SoftmaxParameter& softmax_param=layer_param.softmax_param();

    SoftmaxParam param=any_cast<SoftmaxParam>(OpManager::GetOpDefParam("SoftMax"));

    if(softmax_param.has_axis())
        param.axis=softmax_param.axis();
    else
        param.axis=1;

    StaticOp * op=CreateStaticOp(graph,"SoftMax");

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    AddGraphOutputNode(graph,node);

    return true;
}

static bool  LoadCaffeReLu(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    StaticOp * op=CreateStaticOp(graph,"ReLu");

    SetNodeOp(node,op);

    return true;
}

static bool  LoadCaffeSplit(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    

    StaticOp * op=CreateStaticOp(graph,"Split");

    SetNodeOp(node,op);

    return true;
}

static bool  LoadCaffeConcat(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    ConcatParam  param=any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));

    const caffe::ConcatParameter& concat_param=layer_param.concat_param();

    if(concat_param.has_concat_dim())
        param.axis=static_cast<int>(concat_param.concat_dim());
    else
        param.axis=concat_param.axis();

    StaticOp * op=CreateStaticOp(graph,"Concat");

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    return true;
}

static bool  LoadCaffeDropout(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    StaticOp * op=CreateStaticOp(graph,"Dropout");

    SetNodeOp(node,op);

    return true;
}

static bool  LoadCaffeAccuracy(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    StaticOp * op=CreateStaticOp(graph,"Accuracy");

    SetNodeOp(node,op);

    AddGraphOutputNode(graph,node);

    return true;
}

static bool LoadCaffeConvolution(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
	 
     const caffe::ConvolutionParameter& conv_param=layer_param.convolution_param();

     ConvParam  param=any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));


     param.kernel_h=conv_param.kernel_size(0);
     param.kernel_w=conv_param.kernel_size(0);

     if(conv_param.stride_size())
     {
        param.stride_h=conv_param.stride(0);
        param.stride_w=conv_param.stride(0);
     }


     if(conv_param.pad_size())
     {
        param.pad_h=conv_param.pad(0);
        param.pad_w=conv_param.pad(0);
     }

     param.output_channel=conv_param.num_output();

     if(conv_param.has_group())
         param.group=conv_param.group();


     StaticOp * op=CreateStaticOp(graph,"Convolution");

     SetOperatorParam(op,param);

     SetNodeOp(node,op);

	 
     /* create new Node and tensor for pre-trained weights */

     std::vector<std::string> name_list={"weight","bias"};
     std::vector<std::string> layout_list={"NCHW","W"};

     LoadCaffeBlob(graph, node,name_list,layout_list,layer_param);

     return true;
}

static PoolArg ConvertCaffePool(caffe::PoolingParameter_PoolMethod method)
{
      if(method == caffe::PoolingParameter_PoolMethod_AVE)
           return kPoolAvg; 
      else  if (method == caffe::PoolingParameter_PoolMethod_STOCHASTIC)
          return kPoolRand;
 
      /* for others, return MAX */
  
       return kPoolMax;
}

static bool LoadCaffePooling(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
   
    const caffe::PoolingParameter& pool_param=layer_param.pooling_param();

    PoolParam  param=any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

    param.alg=ConvertCaffePool(pool_param.pool());
    param.kernel_h=pool_param.kernel_size();
    param.kernel_w=pool_param.kernel_size();
    param.global=pool_param.global_pooling();

    param.kernel_shape.resize(2);
    param.kernel_shape[0]=param.kernel_h;
    param.kernel_shape[1]=param.kernel_w;

    if(pool_param.has_pad())
    { 
        param.pad_h=pool_param.pad();
        param.pad_w=pool_param.pad();
        param.pads.resize(4);
        param.pads[0]=param.pad_h;
        param.pads[1]=param.pad_w;
        param.pads[2]=param.pad_h;
        param.pads[3]=param.pad_w;
    }

    if(pool_param.has_stride())
    {
        param.stride_h=pool_param.stride();
        param.stride_w=pool_param.stride();

        param.strides.resize(2);
        param.strides[0]=param.stride_h;
        param.strides[1]=param.stride_w;
    }

    param.caffe_flavor=1;

    StaticOp * op=CreateStaticOp(graph,"Pooling");

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    return true;

}


static bool LoadCaffeInnerProduct(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
	
    const caffe::InnerProductParameter & ip_param=layer_param.inner_product_param();
	

    FCParam  param=any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));
    param.num_output=ip_param.num_output();

    /* Load weight and bias blob */
    std::vector<std::string> name_list={"weight","bias"};
    std::vector<std::string> layout_list={"HW","W"};

    LoadCaffeBlob(graph, node,name_list,layout_list,layer_param);


    StaticOp * op=CreateStaticOp(graph,"FullyConnected");

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    return true;

}

static bool LoadCaffeBatchNorm(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    BatchNormParam param=any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    const caffe::BatchNormParameter& bn_param=layer_param.batch_norm_param();

    param.eps=bn_param.eps();
    param.caffe_flavor=1;


    StaticOp * op=CreateStaticOp(graph,"BatchNormalization");

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    if(layer_param.blobs_size())
    {
         LoadBatchNormBlob(graph,node,layer_param);
    }

    return true;

}


static bool LoadCaffeScale(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    ScaleParam param=any_cast<ScaleParam>(OpManager::GetOpDefParam("Scale"));

    const caffe::ScaleParameter& scale_param=layer_param.scale_param();

    if(scale_param.has_axis())
         param.axis=scale_param.axis();
   
    if(scale_param.has_num_axes())
         param.num_axes=scale_param.num_axes();

    if(scale_param.has_bias_term())
         param.bias_term=scale_param.bias_term();


    StaticOp * op=CreateStaticOp(graph,"Scale");

    SetOperatorParam(op,param);

    SetNodeOp(node,op);

    if(layer_param.blobs_size())
    {
         LoadScaleBlob(graph,node,layer_param);
    } 

    return true;
}

static bool LoadCaffeLRN(StaticGraph * graph, StaticNode * node, const caffe::LayerParameter& layer_param)
{
    LRNParam param=any_cast<LRNParam>(OpManager::GetOpDefParam("LRN"));
    const ::caffe::LRNParameter& caffe_param=layer_param.lrn_param();

    if(caffe_param.norm_region() == caffe::LRNParameter_NormRegion_WITHIN_CHANNEL)
        param.norm_region=LRN_WITHIN_CHANNEL;
    else
        param.norm_region=LRN_ACROSS_CHANNELS;

    param.k=caffe_param.k();
    param.alpha=caffe_param.alpha();
    param.beta=caffe_param.beta();
    param.local_size=caffe_param.local_size(); 
   
    StaticOp * op=CreateStaticOp(graph,"LRN");
    SetOperatorParam(op,param);
    SetNodeOp(node,op);

    return true;
}

bool CaffeSerializerRegisterOpLoader(void)
{

    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("caffe_single",serializer))
          return false;

    CaffeSingle * p_caffe=dynamic_cast<CaffeSingle *>(serializer.get());

    p_caffe->RegisterOpLoadMethod("Data",op_load_t(LoadCaffeInputOp));
    p_caffe->RegisterOpLoadMethod("Input",op_load_t(LoadCaffeInputOp));
    p_caffe->RegisterOpLoadMethod("Convolution",op_load_t(LoadCaffeConvolution));
    p_caffe->RegisterOpLoadMethod("Pooling",op_load_t(LoadCaffePooling));
    p_caffe->RegisterOpLoadMethod("Softmax",op_load_t(LoadCaffeSoftMax));
    p_caffe->RegisterOpLoadMethod("SoftmaxWithLoss",op_load_t(LoadCaffeSoftMax));
    p_caffe->RegisterOpLoadMethod("ReLU",op_load_t(LoadCaffeReLu));
    p_caffe->RegisterOpLoadMethod("InnerProduct",op_load_t(LoadCaffeInnerProduct));
    p_caffe->RegisterOpLoadMethod("Split",op_load_t(LoadCaffeSplit));
    p_caffe->RegisterOpLoadMethod("Concat",op_load_t(LoadCaffeConcat));
    p_caffe->RegisterOpLoadMethod("Dropout",op_load_t(LoadCaffeDropout));
    p_caffe->RegisterOpLoadMethod("Accuracy",op_load_t(LoadCaffeAccuracy));
    p_caffe->RegisterOpLoadMethod("BatchNorm",op_load_t(LoadCaffeBatchNorm));
    p_caffe->RegisterOpLoadMethod("Scale",op_load_t(LoadCaffeScale));
    p_caffe->RegisterOpLoadMethod("LRN",op_load_t(LoadCaffeLRN));


    if(!SerializerManager::SafeGet("caffe",serializer))
          return false;

    CaffeBuddy * p_buddy=dynamic_cast<CaffeBuddy *>(serializer.get());

    p_buddy->RegisterOpLoadMethod("Data",op_load_t(LoadCaffeInputOp));
    p_buddy->RegisterOpLoadMethod("Input",op_load_t(LoadCaffeInputOp));
    p_buddy->RegisterOpLoadMethod("Convolution",op_load_t(LoadCaffeConvolution));
    p_buddy->RegisterOpLoadMethod("Pooling",op_load_t(LoadCaffePooling));
    p_buddy->RegisterOpLoadMethod("Softmax",op_load_t(LoadCaffeSoftMax));
    p_buddy->RegisterOpLoadMethod("SoftmaxWithLoss",op_load_t(LoadCaffeSoftMax));
    p_buddy->RegisterOpLoadMethod("ReLU",op_load_t(LoadCaffeReLu));
    p_buddy->RegisterOpLoadMethod("InnerProduct",op_load_t(LoadCaffeInnerProduct));
    p_buddy->RegisterOpLoadMethod("Split",op_load_t(LoadCaffeSplit));
    p_buddy->RegisterOpLoadMethod("Concat",op_load_t(LoadCaffeConcat));
    p_buddy->RegisterOpLoadMethod("Dropout",op_load_t(LoadCaffeDropout));
    p_buddy->RegisterOpLoadMethod("Accuracy",op_load_t(LoadCaffeAccuracy));
    p_buddy->RegisterOpLoadMethod("BatchNorm",op_load_t(LoadCaffeBatchNorm));
    p_buddy->RegisterOpLoadMethod("Scale",op_load_t(LoadCaffeScale));
    p_buddy->RegisterOpLoadMethod("LRN",op_load_t(LoadCaffeLRN));

    blob_load_map["Convolution"]=LoadConvolutionBlob;
    blob_load_map["InnerProduct"]=LoadFullyConnectedBlob;
    blob_load_map["BatchNorm"]=LoadBatchNormBlob;
    blob_load_map["Scale"]=LoadScaleBlob;

    return true;
}

} //namespace TEngine
