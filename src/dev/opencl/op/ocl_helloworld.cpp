
#include "ocl_helper.hpp"
#include "ocl_executor.hpp"

bool OCLEngine::AddHelloWorldNode()
{
    TLOG_INFO("Tengine OpenCL: Support OP_HelloWorld.\n");

    char abs_path_buff[50];

    //获取文件路径, 填充到abs_path_buff
    //realpath函数返回: null表示获取失败; 否则返回指向abs_path_buff的指针
    if(realpath(".", abs_path_buff))
        fprintf(stderr,"Log realpath: %s %s\n", ".", abs_path_buff);

    char curPath[100];
    getcwd(curPath, 100);
    std::cout<<"Log getcwd: "<<curPath<<std::endl;

//    struct ir_graph* ir_graph = ir_node->graph;
//
//    /**Step 9: Sets Kernel arguments.*/
//    for (int i = 0; i < ir_node->input_num; i++)
//    {
//        struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
//        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, input_tensor->idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->idx]) );
//    }
//    for (int i = 0; i < ir_node->output_num; i++)
//    {
//        struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
//        CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, input_tensor->idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->idx]) );
//    }

    /** Test Hello World */
//    this->build_kernel("/tmp/tmp.4s46ZdOirg/src/dev/opencl/cl/HelloWorld_Kernel.cl", "helloworld");
    char* cl_env = getenv("ROOT_PATH");
    strcat(cl_env, "/src/dev/opencl/cl/HelloWorld_Kernel.cl");
//    fprintf(stderr,"Log cl kernel path: %s\n",cl_env);
    this->build_kernel(cl_env, "helloworld");

    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&this->ocl_tensor_map[0]) );
    CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&this->ocl_tensor_map[1]) );

    size_t global_work_size[1] = {NUM};
    size_t local_work_size[1] = {64};

    struct OCLqueue HelloWorld;
    HelloWorld.queue_kernel = this->kernel;
    HelloWorld.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
    HelloWorld.queue_global_work_size[0] = NUM;
    HelloWorld.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
    HelloWorld.queue_local_work_size[0] = 64;
    queue_list.push_back(HelloWorld);

    return true;
}












