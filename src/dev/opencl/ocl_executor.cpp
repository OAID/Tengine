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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#include "ocl_executor.hpp"
#include "ocl_helper.hpp"

extern "C"
{
#include "tengine_op.h"
#include "convolution_param.h"
}


bool OCLEngine::init()
{
    this->queue_list.clear();
    bin_num = -1;

    /**Step 1: Getting platforms and choose an available one(first).*/
    getPlatform(platform);

    /**Step 2:Query the platform and choose the first GPU device if has one.*/
    devices=getCl_device_id(platform);

    /**Step 3: Create context.*/
    context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);

    /**Step 4: Creating command queue associate with the context.*/
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);


}

bool OCLEngine::build_kernel(const char *filename, const char *kernel_name)
{
    /**Step 5: Create program object */
    std::string sourceStr ;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();

    size_t sourceSize[] = {strlen(source)};
    program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

    /**Step 6: Build program. */
    status=clBuildProgram(program, 1, devices,NULL,NULL,NULL);
    if (status != CL_SUCCESS)
    {
        fprintf(stderr,"Log: clBuildProgram status %d\n",status);
        size_t len;
        char buffer[8 * 1024];

        fprintf(stderr,"Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr,"%s\n", buffer);
        return false;
    }
//    else
//        fprintf(stderr,"Log: clBuildProgram SUCCESS\n");

    unsigned char *programBinary;
    FILE *pf;
    bin_num += 1;
    char binaryFileName[200] = "";
    strcat(binaryFileName, std::to_string(bin_num).c_str());
    strcat(binaryFileName, ".bin");

    // ???? build ? program ???
    size_t programBinarySize;
    status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(cl_device_id), &programBinarySize, NULL); // ?? build ? program ???
    if (status != CL_SUCCESS)
    {
        fprintf(stderr,"Log: clGetProgramInfo0 status %d\n",status);
        return false;
    }
    programBinary = (unsigned char *)malloc(sizeof(unsigned char)*programBinarySize);
    status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &programBinary, NULL);      // ????
    if (status != CL_SUCCESS)
    {
        fprintf(stderr,"Log: clGetProgramInfo1 status %d\n",status);
        return false;
    }

    // ??clCreateProgramWithBinary ??? program
    program = clCreateProgramWithBinary(context, 1, devices, &programBinarySize, (const unsigned char **)&programBinary, NULL, NULL);
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        fprintf(stderr,"Log: clBuildProgram status %d\n",status);
        return false;
    }

    /**Step 7: Create kernel object */
    kernel = clCreateKernel(program, kernel_name, NULL);

    return true;
}

int OCLEngine::BuildKernel(struct subgraph* subgraph)
{
    struct ir_graph* ir_graph = subgraph->graph;

    /**Step 8: Create kernel object */
   for (int i = 0; i < subgraph->node_num; i++)
   {
       uint16_t node_id = subgraph->node_list[i];
       struct ir_node* ir_node = get_ir_graph_node(ir_graph, node_id);
       auto op_type = ir_node->op.op_type;

        switch (op_type)
        {
            case OP_CLIP:
                this->AddClipNode(ir_node);
                break;
            case OP_CONCAT:
                this->AddConcatNode(ir_node);
                break;
            case OP_CONST:
            case OP_INPUT:
                continue;
            case OP_CONV:
                this->AddConvolutionNode(ir_node);
                break;
            case OP_DROPOUT:
                this->AddDropoutNode(ir_node);
                break;
            case OP_ELTWISE:
                this->AddEltwiseNode(ir_node);
                break;
            case OP_FC:
                this->AddFullyConnectionNode(ir_node);
                break;
            case OP_FLATTEN:
                this->AddFlattenNode(ir_node);
                break;
//            case OP_PERMUTE:
//                this->AddPermuteNode(ir_node);
//                break;
            case OP_POOL:
                this->AddPoolingNode(ir_node);
                break;
            case OP_RELU:
                this->AddReluNode(ir_node);
                break;
            case OP_RESHAPE:
                this->AddReshapeNode(ir_node);
                break;
            case OP_SLICE:
                this->AddSliceNode(ir_node);
                break;
//            case OP_SOFTMAX:
//                this->AddSoftmaxNode(ir_node);
            default:
                TLOG_INFO("Tengine OpenCL GPU: Cannot support OP(%d).\n", ir_node->idx);
                break;
        }
   }

//   /** Test Hello World */
//    this->AddHelloWorldNode();
    return 0;
}

bool OCLEngine::OCLTensorMap(struct ir_graph* ir_graph, int ir_tensor_idx, cl_mem_flags flag)
{
    auto iter = this->ocl_tensor_map.find(ir_tensor_idx);
    struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);

    cl_mem clBuffer;
    if ( TENSOR_TYPE_CONST == ir_tensor->tensor_type || TENSOR_TYPE_DEP == ir_tensor->tensor_type )
    {
//        fprintf(stderr,"Upload weight/bias %s\n",ir_tensor->name);
        clBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE , ir_tensor->elem_num * ir_tensor->elem_size, NULL, NULL);
    }
    else
        clBuffer = clCreateBuffer(context, flag , ir_tensor->elem_num * ir_tensor->elem_size, NULL, NULL);

    this->ocl_tensor_map[ir_tensor_idx] = clBuffer;
}

int OCLEngine::BuildTensor(struct subgraph* subgraph)
{
    struct ir_graph* ir_graph = subgraph->graph;

    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        this->OCLTensorMap(ir_graph, ir_tensor_idx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR);
    }
    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (int j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            this->OCLTensorMap(ir_graph, ir_tensor_idx, CL_MEM_READ_WRITE);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            this->OCLTensorMap(ir_graph, ir_tensor_idx, CL_MEM_READ_WRITE);
        }
    }

//    /** Test Hello World */
////    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, (NUM) * sizeof(float),(void *) input, NULL);
//    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , NUM * sizeof(float), NULL, NULL);
//    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , NUM * sizeof(float), NULL, NULL);
//
//    this->ocl_tensor_map[0] = inputBuffer;
//    this->ocl_tensor_map[1] = outputBuffer;
}

int OCLEngine::OCLEnginePreRun(struct subgraph* subgraph)
{
//    get_device_message();
    // dump_sub_graph(subgraph);

    /* init opencl setting */
    this->init();

    /* build opencl tensor */
    this->BuildTensor(subgraph);

    /* build opencl kernel */
    this->BuildKernel(subgraph);

    return 0;
};

int OCLEngine::OCLEngineRun(struct subgraph* subgraph)
{
    struct ir_graph* ir_graph = subgraph->graph;

    /* upload data */
    fprintf(stderr,"Upload date\n");
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        cl_event enentPoint;
        CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueWriteBuffer(commandQueue, this->ocl_tensor_map[ir_tensor_idx], CL_TRUE, 0, input_tensor->elem_num * input_tensor->elem_size, input_tensor->data, 0, NULL, &enentPoint));
        clWaitForEvents(1,&enentPoint); ///wait
        clReleaseEvent(enentPoint);
    }

    /* run */
    /**Step 10: Running the kernel.*/
    if (queue_list.empty())
    {
        fprintf(stderr,"queue_list is empty\n");
        return 0;
    }
    int term = 0;
    fprintf(stderr,"queue_list size %d\n",queue_list.size());
    cl_event enentPoint;
    for (int i = 0; i<queue_list.size(); ++i)
    {
//        fprintf(stderr,"term %d\n",term++);
        auto it = queue_list[i];
        fprintf(stderr,"queue_work_size %s %d %d\n",it.name.c_str(), it.queue_global_work_size[0], it.queue_local_work_size[0]);
//        double start0 = get_current_time();
        cl_event enentPoint;
        it.enentPoint = enentPoint;
        CHECK_ENQUEUE_KERNEL_STATUS(clEnqueueNDRangeKernel(commandQueue, it.queue_kernel, it.dims, NULL, it.queue_global_work_size, it.queue_local_work_size, 0, NULL, &enentPoint) );
//        if (0 == (i+1) % 10)
//            clFinish(commandQueue);
//        clWaitForEvents(1,&enentPoint); ///wait
//        clFinish(commandQueue);
//        double end0 = get_current_time();
//        double cur0 = end0 - start0;
//        fprintf(stderr,"time %s %lf\n",it.name.c_str(), cur0);

//        size_t len;
//        cl_ulong queued;
//        cl_ulong submit;
//        cl_ulong start;
//        cl_ulong end;
//        clGetEventProfilingInfo(enentPoint, CL_PROFILING_COMMAND_QUEUED, sizeof(queued)*2, &queued, &len);
//        clGetEventProfilingInfo(enentPoint, CL_PROFILING_COMMAND_SUBMIT, sizeof(submit)*2, &submit, &len);
//        clGetEventProfilingInfo(enentPoint, CL_PROFILING_COMMAND_START, sizeof(start)*2, &start, &len);
//        clGetEventProfilingInfo(enentPoint, CL_PROFILING_COMMAND_END, sizeof(end)*2, &end, &len);
//        auto time = (end - start) / 1000000.0;
//        fprintf(stderr,"time %lf\n",time);
//        fprintf(stderr,"time queued %ld\n", queued);
//        fprintf(stderr,"time submit %ld\n", submit);
//        fprintf(stderr,"time start  %ld\n", start);
//        fprintf(stderr,"time end    %ld\n\n", end);

//        clReleaseEvent(enentPoint);
    }
    clFinish(commandQueue);
//    clReleaseEvent(enentPoint);

    /* download data */
    /**Step 11: Read the cout put back to host memory.*/
    fprintf(stderr,"Download date\n");
    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        if (output_tensor->data == NULL)
        {
            TLOG_INFO("Log:download data malloc\n");
            float* fp32_data = (float*)malloc(output_tensor->elem_size * output_tensor->elem_num);
            output_tensor->data = fp32_data;
        }
        cl_event enentPoint;
        CHECK_ENQUEUE_BUFFER_STATUS(clEnqueueReadBuffer(commandQueue, this->ocl_tensor_map[ir_tensor_idx], CL_TRUE, 0, output_tensor->elem_num * output_tensor->elem_size, output_tensor->data, 0, NULL, &enentPoint) );
        clWaitForEvents(1,&enentPoint); ///wait
        clReleaseEvent(enentPoint);

        float* data_out = (float*)output_tensor->data;
        std::cout<<"data out "<<output_tensor->name<<" "<<data_out[0]<<std::endl;
    }


#ifdef DEBUG_DATA
    for (auto iter = this->gpu_addr_map.begin(); iter != this->gpu_addr_map.end(); iter++)
    {
        struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, iter->first);
        cudaMemcpy(ir_tensor->data, iter->second, ir_tensor->elem_num * ir_tensor->elem_size, cudaMemcpyDeviceToHost);
    }
#endif

    return 0;
}

void OCLEngine::OCLEnginePostRun()
{
    /**Step 12: Clean the resources.*/
    status = clReleaseKernel(kernel);//*Release kernel.
    status = clReleaseProgram(program);    //Release the program object.

    for (auto iter = this->ocl_tensor_map.begin(); iter != this->ocl_tensor_map.end(); iter++)
    {
        clReleaseMemObject(iter->second);
    }
    status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
    status = clReleaseContext(context);//Release context.

    if (devices != NULL)
    {
        free(devices);
        devices = NULL;
    }
};