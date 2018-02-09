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
#include <unistd.h>


#include "share_lib_parser.hpp"
#include "tensor_mem.hpp"
#include "graph_executor.hpp"
#include "graph.hpp"
#include "operator/convolution.hpp"
#include "prof_utils.hpp"
#include "debug_utils.hpp"
#include "node_ops.hpp"
#include "test_soc_info.hpp"
#include "test_device.hpp"

using namespace TEngine;


void init_tensor_data(float * addr, int number, int fixed_val)
{
    for(int i=0;i<number;i++)
    {
       if(fixed_val>=0)
           addr[i]=fixed_val;
       else
           addr[i]=i%23;
    }


}


float op_fops;
float im2col_fops;
int input_c=1024;
int input_h=7;
int input_w=7;
int input_n=1;
int kernel_h=1;
int kernel_w=1;
int stride_h=1;
int pad_h=0;
int output_channel=32;
int group=1;

Node *  create_convolution_node(void)
{
    Operator * op=OpManager::CreateOp("Convolution");
    Convolution * conv_op=dynamic_cast<Convolution *>(op);

    ConvParam*  param=conv_op->GetParam();

    param->kernel_h=kernel_h;
    param->kernel_w=kernel_w;
    param->stride_h=stride_h;
    param->stride_w=stride_h;
    param->pad_h=pad_h;
    param->pad_w=pad_h;
    param->output_channel=output_channel;
    param->group=group;
    param->dilation_h=1;
    param->dilation_w=1;

    /* calculate shapes */


    int output_h=(input_h-param->kernel_h+2*param->pad_h)/param->stride_h+1;
    int output_w=(input_w-param->kernel_w+2*param->pad_w)/param->stride_w+1;

    std::cout<<"Convolution Settings:\n";
    std::cout<<"kernel: "<<param->kernel_h<<" stride: "<<param->stride_h<<" pad: "<<param->pad_h;
    std::cout<<" group: "<<param->group<<" output channel: "<<param->output_channel<<"\n";
    std::cout<<"input  n: "<<input_n<<" c: "<<input_c<<" h: "<<input_h<<" w: "<<input_w<<"\n";
    std::cout<<"output n: "<<input_n<<" c: "<<param->output_channel<<" h: "<<output_h<<" w: "<<output_w<<"\n";


    std::vector<int> input_dims={input_n,input_c,input_h,input_w};
    std::vector<int> weight_dims={param->output_channel,input_c/param->group,param->kernel_h,param->kernel_w};
    std::vector<int> bias_dims={param->output_channel};
    std::vector<int> output_dims={input_n,param->output_channel,output_h,output_w};

    int im2col_h=input_c/param->group*param->kernel_h*param->kernel_w;
    int im2col_w=output_h*output_w;

    std::printf("im2col matrix: wegith (%d,%d) input_col(%d,%d)\n", param->output_channel,im2col_h,
                                     im2col_h,im2col_w);
    
    im2col_fops=1.0*input_n*param->output_channel*im2col_h*im2col_w*2;

    op_fops=1.0*input_n*output_h*output_w*param->output_channel*(param->kernel_h*param->kernel_w* input_c*2);
    op_fops=op_fops/param->group;

    Node * node=new Node("test_convolution");

    node->SetOp(conv_op);

    //prepare tensor: input/weight/bias/output
    Tensor * tensor;
    int mem_size;
    void * addr;

    tensor=new Tensor("input");
    
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
         
    TShape * shape=&tensor->GetShape();

    shape->SetDataLayout("NCHW");
    shape->SetDim(input_dims);

    node->SetInputPort(0,tensor);

    mem_size=tensor->GetTotalSize();
    addr=std::malloc(mem_size);
    set_tensor_mem(tensor,addr,mem_size,std::free);

    init_tensor_data((float *)addr,mem_size/sizeof(float),-1);

    tensor=new Tensor("weight");
    
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
         
    shape=&tensor->GetShape();

    shape->SetDataLayout("NCHW");
    shape->SetDim(weight_dims);

    node->SetInputPort(1,tensor);

    mem_size=tensor->GetTotalSize();
    addr=std::malloc(mem_size);
    set_tensor_mem(tensor,addr,mem_size,std::free);

    init_tensor_data((float *)addr,mem_size/sizeof(float),-1);

#if 0
    tensor=new Tensor("bias");
    
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
         
    shape=&tensor->GetShape();

    shape->SetDataLayout("W");
    shape->SetDim(bias_dims);

    node->SetInputPort(2,tensor);

    mem_size=tensor->GetTotalSize();
    addr=std::malloc(mem_size);
    set_tensor_mem(tensor,addr,mem_size,std::free);
    init_tensor_data((float *)addr,mem_size/sizeof(float),0);
#endif
   
    tensor=new Tensor("output");
    
    tensor->SetDataType("float32");
    tensor->SetType(kVarTensor);
         
    shape=&tensor->GetShape();

    shape->SetDataLayout("NCHW");
    shape->SetDim(output_dims);

    node->SetOutputPort(0,tensor);

    mem_size=tensor->GetTotalSize();
    addr=std::malloc(mem_size);
    set_tensor_mem(tensor,addr,mem_size,std::free);

    return node;
}

namespace TEngine {

extern bool  caffe_run_convolution(Node *node, int rep, unsigned long * time);

}

void test_convolution(int rep)
{
    Node * node= create_convolution_node();

    unsigned long time=0;

    caffe_run_convolution(node,rep,&time);


    std::cout<<"***** reference performance ********\n";
    std::cout<<"rep="<<rep<<" used time: "<<time<<" us \n";
    std::printf("total fops: %.2f M, per op fops: %.2f K\n",
             op_fops*rep/1000000, op_fops/1000);
    std::printf("performance rate: %.2f Mops\n", op_fops*rep/time);
    std::printf("converted to im2col performance rate: %.2f Mops\n", im2col_fops*rep/time);

    delete node;
}

void test_new_operator(int rep)
{
    Node * node0=create_convolution_node();
    Node * node1=create_convolution_node();

      
    caffe_run_convolution(node0,0,nullptr);

    SocInfo * soc_info=TestGetSocInfo();

    TestDevice * device=new TestDevice();


    device->soc_info=*soc_info;

     std::vector<int> cpu_list={4,5};

     device->soc_info.SetWorkingCPU(cpu_list,4);

     device->LaunchAider();

     device->BindMaster();

    NodeOps * conv_ops=NodeOpsRegistryManager::FindNodeOps(&device->soc_info,node1);

    std::cout<<"HERE: "<<__FILE__<<":"<<__LINE__<<"\n";

    auto f=std::bind(&TestDevice::TaskDispatch,device,std::placeholders::_1,
                 std::placeholders::_2);

    conv_ops->SetHelper(std::malloc,std::free,f);

    std::cout<<"HERE: "<<__FILE__<<":"<<__LINE__<<"\n";
    if(!conv_ops->Prerun(node1))
    {
       std::cout<<"Prerun failed\n";
    }

    std::cout<<"HERE: "<<__FILE__<<":"<<__LINE__<<"\n";
    if(!conv_ops->Run(node1))
    {
       std::cout<<"Run failed\n";
    }

    std::cout<<"HERE: "<<__FILE__<<":"<<__LINE__<<"\n";
    /* compare data */

    Tensor * tensor=node0->GetOutputTensor(0);
    float *  data0=(float *)get_tensor_mem(tensor);

    tensor=node1->GetOutputTensor(0);

    float *  data1=(float *)get_tensor_mem(tensor);

    std::vector<int> output_dim=tensor->GetShape().GetDim();

    std::vector<int> mismatch;

    if(!CompareFloatTensor(data0,data1,output_dim,mismatch))
    {
        std::cout<<"MISMATCH: ";
        for(unsigned int i=0;i<mismatch.size();i++)
           std::cout<<" "<<mismatch[i];
        std::cout<<"\n";

        DumpFloat("/tmp/data0",data0,tensor->GetShape().GetSize());
        DumpFloat("/tmp/data1",data1,tensor->GetShape().GetSize());

        return ;
    }


    std::cout<<"Performance Benchmark ....\n";

    /* performance benchmark ... */

    unsigned long start=get_cur_time();

    for(int i=0;i<rep;i++)
         conv_ops->Run(node1);

    unsigned long end=get_cur_time();

    std::printf("****** OUR IMPL ***************\n");
    std::cout<<"rep="<<rep<<" used time: "<<end-start<<" us \n";
    std::printf("total fops: %.2f M, per op fops: %.2f K\n",
             op_fops*rep/1000000, op_fops/1000);
    std::printf("performance rate: %.2f Mops\n", op_fops*rep/(end-start));
    std::printf("converted to im2col performance rate: %.2f Mops\n", im2col_fops*rep/(end-start));
   

    if(!conv_ops->Postrun(node1))
    {
       std::cout<<"Postrun failed\n";
    }

    conv_ops->Release();

    
}


void sys_init(void)
{
   ShareLibParser p0("./build/operator/liboperator.so");
   p0.ExcecuteFunc<int()>("tengine_plugin_init");

   ShareLibParser p1("./build/serializer/libserializer.so");
   p1.ExcecuteFunc<int()>("tengine_plugin_init");

   ShareLibParser p2("./build/executor/libexecutor.so");
   p2.ExcecuteFunc<int()>("tengine_plugin_init");
}

int main(int argc, char * argv[])
{
    int rep=1;
    int res;

    while((res=getopt(argc,argv,"r:i:h:w:o:k:p:s:g:"))!=-1)
    {
       switch(res)
       {
       case 'r':
          rep=strtoul(optarg,NULL,10);
          break;
       case 'i':
          input_c=strtoul(optarg,NULL,10);
          break;
       case 'h':
          input_h=strtoul(optarg,NULL,10);
          break;
       case 'w':
          input_w=strtoul(optarg,NULL,10);
          break;
       case 'o':
          output_channel=strtoul(optarg,NULL,10);
          break;
       case 'k':
          kernel_h=strtoul(optarg,NULL,10);
          break;
       case 'p':
          pad_h=strtoul(optarg,NULL,10);
          break;
       case 's':
          stride_h=strtoul(optarg,NULL,10);
          break;
       case 'g':
          group=strtoul(optarg,NULL,10);
          break;
       
       default:
          break;
       }
    }

    sys_init();
 
    test_new_operator(rep);

    test_convolution(rep);

    std::cout<<"ALL TESTS DONE\n";

    return 0;

}
