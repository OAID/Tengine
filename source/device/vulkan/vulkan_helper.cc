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

#include "vulkan_helper.hpp"

// bool CHECK_ENQUEUE_BUFFER_STATUS(cl_int status)
// {
//     if (status != CL_SUCCESS)
//     {
//         TLOG_INFO("Log: clEnqueue****Buffer status %d\n",status);
//         if (status == CL_INVALID_COMMAND_QUEUE  )
//             TLOG_INFO("Log: CL_INVALID_COMMAND_QUEUE   \n");
//         else if (status == CL_INVALID_CONTEXT   )
//             TLOG_INFO("Log: CL_INVALID_CONTEXT    \n");
//         else if (status == CL_INVALID_MEM_OBJECT  )
//             TLOG_INFO("Log: CL_INVALID_MEM_OBJECT   \n");
//         else if (status == CL_INVALID_VALUE  )
//             TLOG_INFO("Log: CL_INVALID_VALUE   \n");
//         else if (status == CL_INVALID_EVENT_WAIT_LIST  )
//             TLOG_INFO("Log: CL_INVALID_EVENT_WAIT_LIST   \n");
//         else if (status == CL_MISALIGNED_SUB_BUFFER_OFFSET   )
//             TLOG_INFO("Log: CL_MISALIGNED_SUB_BUFFER_OFFSET    \n");
//         else if (status == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST  )
//             TLOG_INFO("Log: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST   \n");
//         else if (status == CL_MEM_OBJECT_ALLOCATION_FAILURE   )
//             TLOG_INFO("Log: CL_MEM_OBJECT_ALLOCATION_FAILURE     \n");
//         else if (status == CL_INVALID_OPERATION    )
//             TLOG_INFO("Log: CL_INVALID_OPERATION    \n");
//         else if (status == CL_OUT_OF_RESOURCES    )
//             TLOG_INFO("Log: CL_OUT_OF_RESOURCES      \n");
//         else if (status == CL_OUT_OF_HOST_MEMORY     )
//             TLOG_INFO("Log: CL_OUT_OF_HOST_MEMORY     \n");
//         return false;
//     }
// //    else
// //        TLOG_INFO("Log: clEnqueue****Buffer SUCCESS\n");
//     return true;
// }

// bool CHECK_ENQUEUE_KERNEL_STATUS(cl_int status)
// {
//     if (status != CL_SUCCESS)
//     {
//         TLOG_INFO("Log: clEnqueueNDRangeKernel status %d\n",status);
//         if (status == CL_INVALID_PROGRAM_EXECUTABLE   )
//             TLOG_INFO("Log: CL_INVALID_PROGRAM_EXECUTABLE    \n");
//         else if (status == CL_INVALID_COMMAND_QUEUE    )
//             TLOG_INFO("Log: CL_INVALID_COMMAND_QUEUE     \n");
//         else if (status == CL_INVALID_KERNEL   )
//             TLOG_INFO("Log: CL_INVALID_KERNEL    \n");
//         else if (status == CL_INVALID_CONTEXT   )
//             TLOG_INFO("Log: CL_INVALID_CONTEXT    \n");
//         else if (status == CL_INVALID_KERNEL_ARGS   )
//             TLOG_INFO("Log: CL_INVALID_KERNEL_ARGS    \n");
//         else if (status == CL_INVALID_WORK_DIMENSION    )
//             TLOG_INFO("Log: CL_INVALID_WORK_DIMENSION     \n");
//         else if (status == CL_INVALID_GLOBAL_WORK_SIZE   )
//             TLOG_INFO("Log: CL_INVALID_GLOBAL_WORK_SIZE    \n");
//         else if (status == CL_INVALID_GLOBAL_OFFSET    )
//             TLOG_INFO("Log: CL_INVALID_GLOBAL_OFFSET      \n");
//         else if (status == CL_INVALID_WORK_GROUP_SIZE     )
//             TLOG_INFO("Log: CL_INVALID_WORK_GROUP_SIZE     \n");
//         else if (status == CL_INVALID_WORK_ITEM_SIZE     )
//             TLOG_INFO("Log: CL_INVALID_WORK_ITEM_SIZE       \n");
//         else if (status == CL_MISALIGNED_SUB_BUFFER_OFFSET      )
//             TLOG_INFO("Log: CL_MISALIGNED_SUB_BUFFER_OFFSET      \n");
//         else if (status == CL_INVALID_IMAGE_SIZE    )
//             TLOG_INFO("Log: CL_INVALID_IMAGE_SIZE     \n");
//         else if (status == CL_OUT_OF_RESOURCES     )
//             TLOG_INFO("Log: CL_OUT_OF_RESOURCES       \n");
//         else if (status == CL_MEM_OBJECT_ALLOCATION_FAILURE     )
//             TLOG_INFO("Log: CL_MEM_OBJECT_ALLOCATION_FAILURE     \n");
//         else if (status == CL_INVALID_EVENT_WAIT_LIST      )
//             TLOG_INFO("Log: CL_INVALID_EVENT_WAIT_LIST        \n");
//         else if (status == CL_OUT_OF_RESOURCES       )
//             TLOG_INFO("Log: CL_OUT_OF_RESOURCES       \n");
//         else if (status == CL_OUT_OF_HOST_MEMORY        )
//             TLOG_INFO("Log: CL_OUT_OF_HOST_MEMORY        \n");
//         return false;
//     }
// //    else
// //        TLOG_INFO("Log: clEnqueueNDRangeKernel SUCCESS\n");
//     return true;
// }

// bool CHECK_SET_KERNEL_STATUS(cl_int status)
// {
//     if (status != CL_SUCCESS)
//     {
//         TLOG_INFO("Log: clSetKernelArg status %d\n",status);
//         if (status == CL_INVALID_KERNEL )
//             TLOG_INFO("Log: CL_INVALID_KERNEL  \n");
//         else if (status == CL_INVALID_ARG_INDEX  )
//             TLOG_INFO("Log: CL_INVALID_ARG_INDEX   \n");
//         else if (status == CL_INVALID_ARG_VALUE )
//             TLOG_INFO("Log: CL_INVALID_ARG_VALUE  \n");
//         else if (status == CL_INVALID_MEM_OBJECT )
//             TLOG_INFO("Log: CL_INVALID_MEM_OBJECT  \n");
//         else if (status == CL_INVALID_SAMPLER )
//             TLOG_INFO("Log: CL_INVALID_SAMPLER  \n");
//         else if (status == CL_INVALID_ARG_SIZE  )
//             TLOG_INFO("Log: CL_INVALID_ARG_SIZE   \n");
//         else if (status == CL_INVALID_ARG_VALUE )
//             TLOG_INFO("Log: CL_INVALID_ARG_VALUE  \n");
//         else if (status == CL_OUT_OF_RESOURCES   )
//             TLOG_INFO("Log: CL_OUT_OF_RESOURCES    \n");
//         else if (status == CL_OUT_OF_HOST_MEMORY  )
//             TLOG_INFO("Log: CL_OUT_OF_HOST_MEMORY   \n");
//         return false;
//     }
// //    else
// //    {
// //        TLOG_INFO("Log: clSetKernelArg SUCCESS   \n");
// //    }
//     return true;
// }

/** convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout<<"Error: failed to open file\n"<<filename<<std::endl;
    return -1;
}

/**Getting platforms and choose an available one.*/
// int getPlatform(cl_platform_id &platform)
// {
    // platform = NULL;//the chosen platform

    // cl_uint numPlatforms;//the NO. of platforms
    // cl_int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // if (status != CL_SUCCESS)
    // {
    //     std::cout<<"Error: Getting platforms!"<<std::endl;
    //     return -1;
    // }

    // /**For clarity, choose the first available platform. */
    // if(numPlatforms > 0)
    // {
    //     cl_platform_id* platforms =
    //         (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
    //     status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    //     platform = platforms[0];
    //     free(platforms);
    // }
    // else
    //     return -1;

    // return 0;
// }

/**Step 2:Query the platform and choose the first GPU device if has one.*/
// cl_device_id *getCl_device_id(cl_platform_id &platform)
// {
//     cl_uint numDevices = 0;
//     cl_device_id *devices=NULL;
//     cl_int    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
//     if (numDevices > 0) //GPU available.
//     {
//         devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
//         status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
//     }
//     return devices;
// }

void get_device_message()
{
    // /* Host/device data structures */
    // cl_platform_id *platforms;
    // cl_device_id *devices;
    // cl_uint num_platforms;
    // cl_uint num_devices, addr_data;
    // cl_int i, err;

    // /* Extension data */
    // char name_data[48000], ext_data[409600];

    // err = clGetPlatformIDs(5, NULL, &num_platforms);
    // if(err < 0) {
    //     perror("Couldn't find any platforms.");
    //     exit(1);
    // }

    // /* 选取所有的platforms*/
    // platforms = (cl_platform_id*)
    //     malloc(sizeof(cl_platform_id) * num_platforms);
    // err = clGetPlatformIDs(num_platforms, platforms, NULL);
    // if(err < 0) {
    //     perror("Couldn't find any platforms");
    //     exit(1);
    // }

    // //循环查看所有platforms的devices信息，一般intel和AMD的都可以有两个devices：CPU和显卡
    // //如果是nvidia的就一般只有一个显卡device了。
    // printf("\nnum_platforms %d\n", num_platforms);
    // for (int j = 0; j < (int)num_platforms; j++)
    // {
    //     printf("\nplatform %d\n", j+1);
    //     /* 步骤和platforms的一样 */
    //     err = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 1, NULL, &num_devices);
    //     if(err < 0) {
    //         perror("Couldn't find any devices!!!");
    //         exit(1);
    //     }

    //     /* Access connected devices */
    //     devices = (cl_device_id*)
    //         malloc(sizeof(cl_device_id) * num_devices);
    //     clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL,
    //                    num_devices, devices, NULL);

    //     /*循环显示platform的所有device（CPU和显卡）信息。*/
    //     for(i=0; i<(int)num_devices; i++) {

    //         err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
    //                               sizeof(name_data), name_data, NULL);
    //         if(err < 0) {
    //             perror("Couldn't read extension data");
    //             exit(1);
    //         }
    //         clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS,
    //                         sizeof(ext_data), &addr_data, NULL);

    //         clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS,
    //                         sizeof(ext_data), ext_data, NULL);

    //         printf("NAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s\n\n",
    //                name_data, addr_data, ext_data);
    //     }
    // }

    // free(platforms);
    // free(devices);
    // printf("\n");
}

void dump_sub_graph(struct subgraph* sub_graph)
{
    // TLOG_INFO("Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->index, sub_graph->device->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
    // TLOG_INFO("\tSub nodes: [ ");

    // for (int j = 0; j < sub_graph->node_num - 1; j++)
    // {
    //     int node_id = sub_graph->node_list[j];
    //     TLOG_INFO("%d, ", node_id);
    // }
    // TLOG_INFO("%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);

    // TLOG_INFO("\tSub input tensors: [ ");
    // for (int j = 0; j < sub_graph->input_num - 1; j++)
    // {
    //     int tensor_id = sub_graph->input_tensor_list[j];
    //     TLOG_INFO("%d, ", tensor_id);
    // }
    // TLOG_INFO("%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);

    // TLOG_INFO("\tSub output tensors: [ ");
    // for (int j = 0; j < sub_graph->output_num - 1; j++)
    // {
    //     int tensor_id = sub_graph->output_tensor_list[j];
    //     TLOG_INFO("%d, ", tensor_id);
    // }
    // TLOG_INFO("%d ].\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
}

