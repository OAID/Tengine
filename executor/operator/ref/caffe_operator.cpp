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
#include <cmath>

#include <caffe/caffe.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/scale_layer.hpp>
#include <caffe/layers/batch_norm_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include <caffe/layers/prelu_layer.hpp>
#include <caffe/layers/deconv_layer.hpp>

#include "graph.hpp"
#include "operator/convolution.hpp"
#include "operator/input_op.hpp"
#include "operator/pooling.hpp"
#include "operator/softmax.hpp"
#include "operator/fully_connected.hpp"
#include "operator/relu.hpp"
#include "operator/split.hpp"
#include "operator/concat.hpp"
#include "operator/accuracy.hpp"
#include "operator/dropout.hpp"
#include "operator/batch_norm.hpp"
#include "operator/scale.hpp"
#include "operator/lrn.hpp"
#include "operator/prelu.hpp"
#include "operator/eltwise.hpp"
#include "operator/deconvolution.hpp"

#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "prof_utils.hpp"


/* this is an experimental file to use caffe's function to implement operator calculation */

using namespace caffe;

namespace TEngine {

static bool CheckShape(const Tensor * p_tensor, const Blob<float> * blob)
{
     const TShape& shape=p_tensor->GetShape();

     const std::vector<int>& dim0=shape.GetDim();
     const std::vector<int>& dim1=blob->shape();

     return (dim0==dim1);
}


static bool CopyBlobToTensor(Tensor * p_tensor, const Blob<float> * blob)
{
     //check shape 

     if(!CheckShape(p_tensor,blob))
     {
          std::cerr<<"copy data to blob failed due to shape mistach: tensor"<<p_tensor->GetName()<<"\n";
          return false;
     }

     const float * blob_data=blob->cpu_data();
     int     blob_size=blob->count();
     int    mem_size=4*blob_size;

     void * addr=get_tensor_mem(p_tensor);

     if(addr==nullptr)
     {
        addr=std::malloc(mem_size);
        set_tensor_mem(p_tensor,addr,mem_size,std::free); 
     }

     std::memcpy(addr,blob_data,mem_size);

     return true;
}

static void ReshapeBlob(Blob<float>&blob, const Tensor * p_tensor)
{
     const TShape& shape=p_tensor->GetShape();

     const std::vector<int>& dim0=shape.GetDim();

     blob.Reshape(dim0);
}

 
static bool CopyBlobFromTensor(const Tensor * p_tensor, Blob<float> * blob)
{
     if(!CheckShape(p_tensor,blob))
          return false;
    
    
     float * blob_data=blob->mutable_cpu_data();
     int     blob_size=blob->count();
     int     mem_size=blob_size*sizeof(float);

     void *  addr=get_tensor_mem(p_tensor);

     std::memcpy(blob_data,addr,mem_size);

     return true;
}


template <typename Dtype>
struct WrapConv: public ConvolutionLayer<Dtype> {

  WrapConv(const LayerParameter& param) : ConvolutionLayer<Dtype> (param) {}

  void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        ConvolutionLayer<Dtype>::LayerSetUp(bottom,top);
        ConvolutionLayer<Dtype>::Reshape(bottom,top);
  }

  void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        ConvolutionLayer<Dtype>::Forward_cpu(bottom,top);
  }


};


bool  caffe_run_convolution(Node *node, int rep, unsigned long * time)
{
     LayerParameter layer_param;
     ConvolutionParameter* caffe_param =layer_param.mutable_convolution_param();

     layer_param.set_name(node->GetName()+".te");

     Convolution * conv_op=dynamic_cast<Convolution *>(node->GetOp());

     ConvParam * p_param=conv_op->GetParam();
     caffe_param->add_kernel_size(p_param->kernel_h);
     caffe_param->add_kernel_size(p_param->kernel_w);
     caffe_param->add_stride(p_param->stride_h);
     caffe_param->add_stride(p_param->stride_w);
     caffe_param->add_dilation(p_param->dilation_h);
     caffe_param->add_dilation(p_param->dilation_w);

     if(p_param->pad_h<0)
        caffe_param->add_pad(p_param->pads[2]);
     else
        caffe_param->add_pad(p_param->pad_h);

     if(p_param->pad_w<0)
        caffe_param->add_pad(p_param->pads[3]);
     else
        caffe_param->add_pad(p_param->pad_w);

     caffe_param->set_num_output(p_param->output_channel);
     caffe_param->set_bias_term((node->GetInputNum()>2));
     caffe_param->set_group(p_param->group);
     
     
     WrapConv<float> caffe_layer(layer_param);

     Blob<float> input,output;

     std::vector<Blob<float>*> bottom,top;

     bottom.push_back(&input);
     top.push_back(&output);


 
     /* input */    
     const Tensor * tensor=node->GetInputTensor(0);
     ReshapeBlob(input, tensor);
     CopyBlobFromTensor(tensor,&input);

     caffe_layer.Init(bottom,top);


     /* weight */

     std::vector<boost::shared_ptr<Blob<float> > > blob_vector=caffe_layer.blobs();

     boost::shared_ptr<Blob<float> > blob_ptr=blob_vector[0];

     tensor=node->GetInputTensor(1);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     /* bias */
     if( blob_vector.size()>1)
     {
         blob_ptr=blob_vector[1];

         tensor=node->GetInputTensor(2);
         ReshapeBlob(*blob_ptr, tensor);
         CopyBlobFromTensor(tensor,blob_ptr.get());
     }

      if(rep)
      {
          unsigned long start=get_cur_time();

          for(int i=0;i<rep;i++)
             caffe_layer.Forward(bottom,top);

          unsigned long end=get_cur_time();

          (*time)=end-start;
      }
      else
          caffe_layer.Forward(bottom,top);
          

     Tensor * output_tensor=node->GetOutputTensor(0);

     CopyBlobToTensor(output_tensor,&output);

     /* if we needs to do relu fusion */

    if(node->ExistAttr("Fused.ReLu"))
    {
       float * data=(float *)get_tensor_mem(output_tensor);
       const TShape & shape=output_tensor->GetShape();
       int number=shape.GetSize();

       for(int i=0;i<number;i++)
       {
            if(data[i]<0)
                data[i]=0;
       }
       
    }

     return true;
}

bool  caffe_run_convolution(Node *node)
{
   return caffe_run_convolution(node,0,nullptr);
}
static PoolingParameter_PoolMethod MapPool(PoolArg arg)
{
 
    if(arg==kPoolAvg)
         return PoolingParameter_PoolMethod_AVE;

    if(arg==kPoolRand)
         return PoolingParameter_PoolMethod_STOCHASTIC;   

    return PoolingParameter_PoolMethod_MAX;
}

template <typename Dtype>
struct WrapPoolingLayer: public PoolingLayer<Dtype> {

 WrapPoolingLayer(const LayerParameter& param): PoolingLayer<Dtype>(param) {};

 void Init(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
      { 
           PoolingLayer<Dtype>::LayerSetUp(bottom,top);
           PoolingLayer<Dtype>::Reshape(bottom,top);
      }


 void Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { PoolingLayer<Dtype>::Forward_cpu(bottom,top);}

};

bool  caffe_run_pooling(Node *node)
{
    LayerParameter layer_param;
    PoolingParameter* caffe_param = layer_param.mutable_pooling_param();

    Pooling * pooling_op=dynamic_cast<Pooling*>(node->GetOp());

    PoolParam * p_param=pooling_op->GetParam();
    // ONNX param
    caffe_param->set_kernel_h(p_param->kernel_shape[0]);
    caffe_param->set_kernel_w(p_param->kernel_shape[1]);
    //caffe_param->set_kernel_size(p_param->kernel_shape[0]);
    caffe_param->set_pool(MapPool(p_param->alg));
    caffe_param->set_pad_h(p_param->pads[0]); 
    caffe_param->set_pad_w(p_param->pads[1]); 
    caffe_param->set_stride_h(p_param->strides[0]); 
    caffe_param->set_stride_w(p_param->strides[1]); 
    // origin param 
    // caffe_param->set_kernel_size(p_param->kernel_h);
    // caffe_param->set_pool(MapPool(p_param->alg));
    // caffe_param->set_pad(p_param->pad_h); 
    // caffe_param->set_stride(p_param->stride_h); 


   
    Blob<float> input, output;

    Tensor * tensor=node->GetInputTensor(0);

    ReshapeBlob(input,tensor);
    CopyBlobFromTensor(tensor,&input);

    tensor=node->GetOutputTensor(0);
    ReshapeBlob(output,tensor);

    std::vector<Blob<float>*> bottom,top;
    bottom.push_back(&input);
    top.push_back(&output);

    WrapPoolingLayer<float>  caffe_layer(layer_param);

    caffe_layer.Init(bottom,top);
    caffe_layer.Forward(bottom,top);

 
    CopyBlobToTensor(tensor,&output);  

    return true;
    
}

template <typename Dtype>
struct WrapReLULayer: public ReLULayer<Dtype> {

 WrapReLULayer(const LayerParameter& param): ReLULayer<Dtype>(param) {}

 void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
   {    ReLULayer<Dtype>::Forward_cpu(bottom,top); }

};

bool caffe_run_relu(Node * node)
{
    LayerParameter layer_param;
    Blob<float> input, output;

    Tensor * tensor=node->GetInputTensor(0);

    ReshapeBlob(input,tensor);
    CopyBlobFromTensor(tensor,&input);
    
    tensor=node->GetOutputTensor(0);
    ReshapeBlob(output,tensor);

    std::vector<Blob<float>*> bottom,top;

    bottom.push_back(&input);
    top.push_back(&output);

    WrapReLULayer<float>  caffe_layer(layer_param);

    caffe_layer.Forward(bottom,top);

    CopyBlobToTensor(tensor,&output);  

    return true;
}
// add prelu

template <typename Dtype>
struct WrapPReLULayer: public PReLULayer<Dtype> {

 WrapPReLULayer(const LayerParameter& param): PReLULayer<Dtype>(param) {}
 
 void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        PReLULayer<Dtype>::LayerSetUp(bottom,top);
        PReLULayer<Dtype>::Reshape(bottom,top);
  }
 void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
   {    PReLULayer<Dtype>::Forward_cpu(bottom,top); }

};
// add deconv
template <typename Dtype>
struct WrapDeconvLayer: public DeconvolutionLayer<Dtype> 
{

 WrapDeconvLayer(const LayerParameter& param): DeconvolutionLayer<Dtype>(param) {}
 
 void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        DeconvolutionLayer<Dtype>::LayerSetUp(bottom,top);
        DeconvolutionLayer<Dtype>::Reshape(bottom,top);
  }
 void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
   {    DeconvolutionLayer<Dtype>::Forward_cpu(bottom,top); }

};
bool caffe_run_deconv(Node * node,int rep, unsigned long * time)
{
    LayerParameter layer_param;
     ConvolutionParameter* caffe_param =layer_param.mutable_convolution_param();

     layer_param.set_name(node->GetName()+".te");

     Deconvolution * deconv_op=dynamic_cast<Deconvolution *>(node->GetOp());

     DeconvParam * p_param=deconv_op->GetParam();
     caffe_param->add_kernel_size(p_param->kernel_size);
     caffe_param->add_kernel_size(p_param->kernel_size);
     caffe_param->add_stride(p_param->stride);
     caffe_param->add_stride(p_param->stride);
     caffe_param->add_pad(p_param->pad);
     caffe_param->set_num_output(p_param->num_output);
     caffe_param->set_bias_term((node->GetInputNum()>2));
     caffe_param->add_dilation(p_param->dilation);
     caffe_param->add_dilation(p_param->dilation);
  
    WrapDeconvLayer<float> caffe_layer(layer_param);
    //input
    Blob<float> input, output;
    std::vector<Blob<float>*> bottom,top;

    Tensor * tensor=node->GetInputTensor(0);
    ReshapeBlob(input,tensor);
    CopyBlobFromTensor(tensor,&input);

    bottom.push_back(&input);
    top.push_back(&output);
      caffe_layer.Init(bottom,top);
     /* weight */

     std::vector<boost::shared_ptr<Blob<float> > > blob_vector=caffe_layer.blobs();
     boost::shared_ptr<Blob<float> > blob_ptr=blob_vector[0];
     tensor=node->GetInputTensor(1);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     /* bias */
     if( blob_vector.size()>1)
     {
         blob_ptr=blob_vector[1];

         tensor=node->GetInputTensor(2);
         ReshapeBlob(*blob_ptr, tensor);
         CopyBlobFromTensor(tensor,blob_ptr.get());
     }

      if(rep)
      {
          unsigned long start=get_cur_time();

          for(int i=0;i<rep;i++)
             caffe_layer.Forward(bottom,top);

          unsigned long end=get_cur_time();

          (*time)=end-start;
      }
      else
          caffe_layer.Forward(bottom,top);
          

     Tensor * output_tensor=node->GetOutputTensor(0);
     ReshapeBlob(output, output_tensor);
     CopyBlobToTensor(output_tensor,&output); 
     return true;
}
bool  caffe_run_deconv(Node *node)
{
   return caffe_run_deconv(node,0,nullptr);
}
bool caffe_run_prelu(Node * node)
{
 
    LayerParameter layer_param;
  
    WrapPReLULayer<float> caffe_layer(layer_param);

    Blob<float> input, output;
     std::vector<Blob<float>*> bottom,top;
     bottom.push_back(&input);
     top.push_back(&output);

    //input
    Tensor * tensor=node->GetInputTensor(0);
    ReshapeBlob(input,tensor);
    CopyBlobFromTensor(tensor,&input);
    
    caffe_layer.Init(bottom,top);
  
    std::vector<boost::shared_ptr<Blob<float> > > blob_vector=caffe_layer.blobs();
     /* weight */
    boost::shared_ptr<Blob<float> > blob_ptr=blob_vector[0];
    tensor=node->GetInputTensor(1);
    ReshapeBlob(*blob_ptr, tensor);
    CopyBlobFromTensor(tensor,blob_ptr.get());
    // forward
    caffe_layer.Forward(bottom,top);
   //output
    tensor=node->GetOutputTensor(0);
    ReshapeBlob(output,tensor);
    CopyBlobToTensor(tensor,&output);  

    return true;
}

// end prelu

template <typename Dtype>
struct WrapSoftmaxLayer: public SoftmaxLayer<Dtype> {

 WrapSoftmaxLayer(const LayerParameter& param): SoftmaxLayer<Dtype>(param) {}

 void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
 {
      SoftmaxLayer<Dtype>::LayerSetUp(bottom,top);
      SoftmaxLayer<Dtype>::Reshape(bottom,top);
 }

 void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
   {    SoftmaxLayer<Dtype>::Forward_cpu(bottom,top); }

};

bool caffe_run_softmax(Node * node)
{
    LayerParameter layer_param;
    Blob<float> input, output;

    Tensor * tensor=node->GetInputTensor(0);
 
    ReshapeBlob(input,tensor);
    CopyBlobFromTensor(tensor,&input);

    tensor=node->GetOutputTensor(0);
    ReshapeBlob(output,tensor);

    std::vector<Blob<float>*> bottom,top;

    bottom.push_back(&input);
    top.push_back(&output);

    WrapSoftmaxLayer<float>  caffe_layer(layer_param);

    caffe_layer.Init(bottom,top);
    caffe_layer.Forward(bottom,top);

    CopyBlobToTensor(tensor,&output);

    return true;
}

template <typename Dtype>
struct WrapIP: public InnerProductLayer<Dtype>{

   WrapIP(const LayerParameter& param): InnerProductLayer<Dtype>(param) {};

   void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
    {
          InnerProductLayer<Dtype>::LayerSetUp(bottom,top);
          InnerProductLayer<Dtype>::Reshape(bottom,top);
    }

   void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
         InnerProductLayer<Dtype>::Forward_cpu(bottom,top);
    }

};

bool caffe_run_fully_connected(Node * node, int rep, unsigned long * time)
{
    LayerParameter layer_param;
    InnerProductParameter * caffe_param=layer_param.mutable_inner_product_param();
    FullyConnected * fc_op=dynamic_cast<FullyConnected *>(node->GetOp());

    FCParam * p_param=fc_op->GetParam();

    caffe_param->set_num_output(p_param->num_output);

    caffe_param->set_bias_term((node->GetInputNum()>2));

    Blob<float> input,output;
         
    std::vector<Blob<float>*> bottom,top;

    bottom.push_back(&input);
    top.push_back(&output);

    /* prepare input */

     Tensor * tensor=node->GetInputTensor(0);
     ReshapeBlob(input, tensor);
     CopyBlobFromTensor(tensor,&input);


    WrapIP<float> caffe_layer(layer_param);

     caffe_layer.Init(bottom,top);


     /* weight */

     std::vector<boost::shared_ptr<Blob<float> > > blob_vector=caffe_layer.blobs();

     boost::shared_ptr<Blob<float> > blob_ptr=blob_vector[0];

     tensor=node->GetInputTensor(1);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     /* bias */

     if( blob_vector.size()>1)
     {
         blob_ptr=blob_vector[1];

         tensor=node->GetInputTensor(2);
         ReshapeBlob(*blob_ptr, tensor);
         CopyBlobFromTensor(tensor,blob_ptr.get());
     }

     if(rep)
     {
        unsigned long start=get_cur_time();
        for(int i=0;i<rep;i++)
           caffe_layer.Forward(bottom,top);
        unsigned long end=get_cur_time();

        (*time)=end-start;
     }
     else
     {
         caffe_layer.Forward(bottom,top);
     }

    tensor=node->GetOutputTensor(0);

    /* Caffe use 2D blobs in InnerProductLayer, and tensor is 4D,
       so ReshapeBlob is required before CopyBlobToTensor */
    ReshapeBlob(output, tensor);

    CopyBlobToTensor(tensor,&output);

    return true;
        

}

bool caffe_run_fully_connected(Node * node)
{
    return caffe_run_fully_connected(node,0,nullptr);
}

bool caffe_run_split(Node * node)
{
     Tensor * tensor=node->GetInputTensor(0);
     int mem_size=tensor->GetTotalSize();
     void * src_addr=get_tensor_mem(tensor);

     for(unsigned int i=0;i<node->GetOutputNum();i++)
     {
          Tensor *out_tensor=node->GetOutputTensor(i);
          void *  dst_addr=get_tensor_mem(out_tensor);

          std::memcpy(dst_addr,src_addr,mem_size);
     }

     return true;
}

bool caffe_run_concat(Node * node)
{
     Tensor * out_tensor=node->GetOutputTensor(0);
     char * out_addr=(char *)get_tensor_mem(out_tensor);

     for(unsigned int i=0;i<node->GetInputNum();i++)
     {
          Tensor *in_tensor=node->GetInputTensor(i);

          void * in_addr=get_tensor_mem(in_tensor);

          int in_size=in_tensor->GetTotalSize();

          std::memcpy(out_addr,in_addr,in_size);

          out_addr+=in_size;

     }

     return true;
}


bool caffe_run_dropout(Node * node)
{
     Tensor * in_tensor=node->GetInputTensor(0);
     Tensor * out_tensor=node->GetOutputTensor(0);

     void * in_addr=get_tensor_mem(in_tensor);
     void * out_addr=get_tensor_mem(out_tensor);

     int mem_size=in_tensor->GetTotalSize();

     std::memcpy(out_addr,in_addr,mem_size);

     return true;
       
}

template <typename Dtype>
struct WrapBatchNorm: public BatchNormLayer<Dtype> {

  WrapBatchNorm(const LayerParameter& param) : BatchNormLayer<Dtype> (param) {}

  void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        BatchNormLayer<Dtype>::LayerSetUp(bottom,top);
        BatchNormLayer<Dtype>::Reshape(bottom,top);
  }

  void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        BatchNormLayer<Dtype>::Forward_cpu(bottom,top);
  }


};

bool caffe_run_batch_norm(Node * node)
{
     LayerParameter layer_param;
     BatchNormParameter * caffe_param=layer_param.mutable_batch_norm_param();

     layer_param.set_name(node->GetName()+".te");

     BatchNorm * bn_op=dynamic_cast<BatchNorm *>(node->GetOp());

     BatchNormParam * p_param=bn_op->GetParam();

     caffe_param->set_eps(p_param->eps);
     caffe_param->set_use_global_stats(true);

     WrapBatchNorm<float> caffe_layer(layer_param);

     Blob<float> input,output;

     std::vector<Blob<float>*> bottom,top;

     bottom.push_back(&input);
     top.push_back(&output);

     /*input */

     const Tensor * tensor=node->GetInputTensor(0);
     ReshapeBlob(input, tensor);
     CopyBlobFromTensor(tensor,&input);

     caffe_layer.Init(bottom,top);

     std::vector<boost::shared_ptr<Blob<float> > > blob_vector=caffe_layer.blobs();

     /*means*/
     boost::shared_ptr<Blob<float> > blob_ptr=blob_vector[0];
     tensor=node->GetInputTensor(3);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     /*vars*/
     blob_ptr=blob_vector[1];
     tensor=node->GetInputTensor(4);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     /*rescale_factor*/
     blob_ptr=blob_vector[2];

     float * blob_data=blob_ptr->mutable_cpu_data();

     blob_data[0]=p_param->rescale_factor;
     
     caffe_layer.Forward(bottom,top);

     Tensor * output_tensor=node->GetOutputTensor(0);

     CopyBlobToTensor(output_tensor,&output);

     return true;
}

template <typename Dtype>
struct WrapScale: public ScaleLayer<Dtype> {

  WrapScale(const LayerParameter& param) : ScaleLayer<Dtype> (param) {}

  void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        ScaleLayer<Dtype>::LayerSetUp(bottom,top);
        ScaleLayer<Dtype>::Reshape(bottom,top);
  }

  void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        ScaleLayer<Dtype>::Forward_cpu(bottom,top);
  }
};


bool caffe_run_scale(Node * node)
{
     LayerParameter layer_param;
     ScaleParameter * caffe_param=layer_param.mutable_scale_param();

     layer_param.set_name(node->GetName()+".te");

     Scale * scale_op=dynamic_cast<Scale *>(node->GetOp());

     ScaleParam * p_param=scale_op->GetParam();

     caffe_param->set_axis(p_param->axis);
     caffe_param->set_num_axes(p_param->num_axes);
     caffe_param->set_bias_term(p_param->bias_term);

     WrapScale<float> caffe_layer(layer_param);

     Blob<float> input,output;

     std::vector<Blob<float>*> bottom,top;

     bottom.push_back(&input);
     top.push_back(&output);

     /* input */
     const Tensor * tensor=node->GetInputTensor(0);
     ReshapeBlob(input, tensor);
     CopyBlobFromTensor(tensor,&input);

     caffe_layer.Init(bottom,top);


     std::vector<boost::shared_ptr<Blob<float> > > blob_vector=caffe_layer.blobs();

     /* gamma */
     boost::shared_ptr<Blob<float> > blob_ptr=blob_vector[0];
     tensor=node->GetInputTensor(1);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     /* beta */
     blob_ptr=blob_vector[1];
     tensor=node->GetInputTensor(2);
     ReshapeBlob(*blob_ptr, tensor);
     CopyBlobFromTensor(tensor,blob_ptr.get());

     caffe_layer.Forward(bottom,top);

     Tensor * output_tensor=node->GetOutputTensor(0);

     CopyBlobToTensor(output_tensor,&output);

     return true;
}

template <typename Dtype>
struct WrapLRN: public LRNLayer<Dtype> {

  WrapLRN(const LayerParameter& param) : LRNLayer<Dtype> (param) {}

  void Init(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        LRNLayer<Dtype>::LayerSetUp(bottom,top);
        LRNLayer<Dtype>::Reshape(bottom,top);
  }

  void Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
        LRNLayer<Dtype>::Forward_cpu(bottom,top);
  }
};

bool caffe_run_lrn(Node * node)
{
     LayerParameter layer_param;
     caffe::LRNParameter * caffe_param=layer_param.mutable_lrn_param();

     LRN * lrn_op=dynamic_cast<LRN *>(node->GetOp());

     LRNParam * param=lrn_op->GetParam();

     layer_param.set_name(node->GetName()+".te");

     caffe_param->set_local_size(param->local_size);
     caffe_param->set_alpha(param->alpha);
     caffe_param->set_beta(param->beta);
     caffe_param->set_k(param->k);

     if(param->norm_region ==  LRN_ACROSS_CHANNELS)
         caffe_param->set_norm_region(caffe::LRNParameter_NormRegion_ACROSS_CHANNELS);
     else
         caffe_param->set_norm_region(caffe::LRNParameter_NormRegion_WITHIN_CHANNEL);

     
     WrapLRN<float> caffe_layer(layer_param);

     Blob<float> input,output;

     std::vector<Blob<float>*> bottom,top;

     bottom.push_back(&input);
     top.push_back(&output);

     /* input */
     const Tensor * tensor=node->GetInputTensor(0);
     ReshapeBlob(input, tensor);
     CopyBlobFromTensor(tensor,&input);

     caffe_layer.Init(bottom,top);
   
     caffe_layer.Forward(bottom,top);

     Tensor * output_tensor=node->GetOutputTensor(0);

     CopyBlobToTensor(output_tensor,&output);

     return true;
}

bool caffe_run_eltwise(Node * node)
{
    //input
    Tensor * input_tensor0=node->GetInputTensor(0);
    const TShape& ishape=input_tensor0->GetShape();
    int input_count4=ishape.GetSize();
    void * input0=get_tensor_mem(input_tensor0);

    Tensor * input_tensor1=nullptr;
    void* input1=nullptr;

    if(node->GetInputNum()>1)
    {
       input_tensor1=node->GetInputTensor(1);
       input1=get_tensor_mem(input_tensor1);
    }

    // this version only support for input_num=2
   // int input_number=node->GetInputNum();

    // output
    Tensor * output_tensor=node->GetOutputTensor(0);
    void * output=get_tensor_mem(output_tensor);
    float* out_ptr=(float*)output;
    float* in0=(float*)input0;
    float* in1=(float*)input1;
    Eltwise * eltwise_op=dynamic_cast<Eltwise *>(node->GetOp());
    EltwiseParam*  param=eltwise_op->GetParam();

    switch (param->type)
    {
    case ELT_SUM_SCALAR:
        for (int i = 0; i < input_count4; ++i)
        {
            *out_ptr++ = (*in0++)+in1[0];
        }
        break;
    case ELT_SUM:
        for (int i = 0; i < input_count4; ++i)
        {
            *out_ptr++ = (*in0++)+(*in1++);
        }
        break;
    case ELT_SUB:
        for (int i = 0; i < input_count4; ++i)
        {
            *out_ptr++ = (*in0++)-(*in1++);
        }
        break;
    case ELT_MAX:
        for (int i = 0; i < input_count4; ++i)
        {
            *out_ptr++ =std::max (in0[i],in1[i]);
        }
        break;
    case ELT_PROD:
        for (int i = 0; i < input_count4; ++i)
        {
            *out_ptr++ =in0[i]*in1[i];
        }
        break;
    case ELT_RSQRT:
        for (int i = 0; i < input_count4; ++i)
        {
            *out_ptr++ =1/std::sqrt(in0[i]);
        }
        break;
    default:
        return false;
    }
    return true;

}

struct CaffeNodeOps: public NodeOps {

using node_exec_t =bool(*)(Node *);
	  
bool Run(Node * node) override 
{
      return ops_func(node);	  
}

   node_exec_t ops_func;
};

void RegisterNodeRun(const std::string& name, bool(*func)(Node *))

{
    CaffeNodeOps * caffe_ops=new CaffeNodeOps();

      caffe_ops->ops_func=func;

      NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME,
               name,caffe_ops);
} 


void RegisterCaffeExecutors(void)
{
    RegisterNodeRun("Convolution",caffe_run_convolution);
    RegisterNodeRun("Pooling",caffe_run_pooling);
    RegisterNodeRun("ReLu",caffe_run_relu);
    RegisterNodeRun("Softmax",caffe_run_softmax);
    RegisterNodeRun("FullyConnected",caffe_run_fully_connected);
    RegisterNodeRun("Split",caffe_run_split);
    RegisterNodeRun("Concat",caffe_run_concat);
    RegisterNodeRun("Dropout",caffe_run_dropout);
    RegisterNodeRun(BatchNormName,caffe_run_batch_norm);
    RegisterNodeRun("Scale",caffe_run_scale);
    RegisterNodeRun("LRN",caffe_run_lrn);
    RegisterNodeRun("PReLU",caffe_run_prelu);
    RegisterNodeRun("Eltwise",caffe_run_eltwise);
    RegisterNodeRun("Deconvolution",caffe_run_deconv);
}

} //namespace TEngine

