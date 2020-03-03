#include <iostream>
#include <unistd.h>
#include <string>
#include <sys/time.h>

/* the sample code to create a convolution and do calculation */

#include "tengine_c_api.h"

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad,
                     int in_c, int out_c, int group)
{
    node_t conv_node = create_graph_node(graph, node_name, "Deconvolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(conv_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {out_c, in_c / group, k_size, k_size};

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {out_c};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    /* attr */
    int pad1 = pad;
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h0", &pad);
    set_node_attr_int(conv_node, "pad_w0", &pad);
    set_node_attr_int(conv_node, "pad_h1", &pad1);
    set_node_attr_int(conv_node, "pad_w1", &pad1);
    set_node_attr_int(conv_node, "num_output", &out_c);
    set_node_attr_int(conv_node, "group", &group);

    release_graph_node(conv_node);

    return 0;
}

graph_t create_conv_graph(int c, int h, int w, int o_c, int k, int s, int p, int g)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    //set_graph_layout(graph, TENGINE_LAYOUT_NHWC);

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_conv_node(graph, conv_name, input_name, k, s, p, c, o_c, g) < 0)
    {
        std::cerr << "create conv node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

int main(int argc, char* argv[])
{
    int c, h, w, k, s, p, o_c, g;

    int rep = 1;
    c = 32;
    w = 20;
    k = 4;
    s = 2;
    p = 1;
    o_c = 64;
    g = 1;


    int res;
    while((res = getopt(argc, argv, "c:w:k:s:r:h")) != -1)
    {
        switch(res)
        {
            case 'c':
                c = atoi(optarg);
                break;
            case 'w':
                w = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 's':
                s = atoi(optarg);
                break;
            case 'r':
                rep = atoi(optarg);
                break;
            case 'p':
                p = atoi(optarg);
                break;
            case 'o':
                o_c = atoi(optarg);
                break;
            case 'g':
                g = atoi(optarg);
                break;
            case 'h':
                std::cout   << "[Usage]: " << argv[0] << " [-h]\n"
                            << "    [-c input_c] [-w input_w] [-k kernel] [-s stride] [-r repeat]\n";
                return 0;
            default:
                break;
        }
    }
    h = w;

    init_tengine();

    graph_t graph = create_conv_graph(c, h, w, o_c, k, s, p, g);
    graph_t graph1 = create_conv_graph(c, h, w, o_c, k, s, p, g);

    if(graph == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph1, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);
    float* i_buf1 = ( float* )malloc(buf_size);
    int input_size = buf_size /sizeof(float);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        i_buf[i] = i % 10;
        i_buf1[i] = i % 10;
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);
    set_tensor_buffer(input_tensor1, i_buf1, buf_size);
    release_graph_tensor(input_tensor1);

    /* set weight */
    node_t conv_node = get_graph_node(graph, "conv");
    node_t conv_node1 = get_graph_node(graph1, "conv");

    tensor_t weight_tensor = get_node_input_tensor(conv_node, 1);
    tensor_t weight_tensor1 = get_node_input_tensor(conv_node1, 1);

    buf_size = get_tensor_buffer_size(weight_tensor);
    float* w_buf = ( float* )malloc(buf_size);
    float* w_buf1 = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        w_buf[i] = i%40;
        w_buf1[i] = i % 40;
    }

    set_tensor_buffer(weight_tensor, w_buf, buf_size);
    release_graph_tensor(weight_tensor);
    set_tensor_buffer(weight_tensor1, w_buf1, buf_size);
    release_graph_tensor(weight_tensor1);

    /* set bias */

    int input_num = get_node_input_number(conv_node);
    float* b_buf = nullptr;
    float* b_buf1 = nullptr;

    if(input_num > 2)
    {
        tensor_t bias_tensor = get_node_input_tensor(conv_node, 2);
        tensor_t bias_tensor1 = get_node_input_tensor(conv_node1, 2);

        buf_size = get_tensor_buffer_size(bias_tensor);
        b_buf = ( float* )malloc(buf_size);
        b_buf1 = ( float* )malloc(buf_size);

        for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        {
            b_buf[i] = 1;
            b_buf1[i] = 1;
        }

        set_tensor_buffer(bias_tensor, b_buf, buf_size);
        release_graph_tensor(bias_tensor);
        set_tensor_buffer(bias_tensor1, b_buf1, buf_size);
        release_graph_tensor(bias_tensor1);
    }

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    setenv("OPS_REGISTRY", "reference", 1);
    setenv("OP_NAME", "Convolution", 1);
    prerun_graph(graph1);

    // dump_graph(graph);

    /* test save graph */
/*
    if(save_graph(graph, "tengine", "/tmp/test.tm") < 0)
    {
        std::cout << "save graph failed\n";
        return 1;
    }
*/
    const char* dev = get_node_device(conv_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }
    float total = 0.f;
    struct timeval tm1,tm2;
    for(int i =0;i<rep;i++)
    {
        for(int j =0;j<input_size;j++)
        {
            i_buf[i] = i % 10;
        }
        gettimeofday(&tm1,0);
        run_graph(graph, 1);
        gettimeofday(&tm2,0);
        float time = (tm2.tv_usec - tm1.tv_usec)*0.001 + (tm2.tv_sec-tm1.tv_sec)*1000;
        total += time;
        //printf("current run %f ( %f ) ms\n",time, total/(i+1));
    }
    printf("repeat time = %d , avg per time = %f  ms\n",rep, total/rep);
    run_graph(graph1,1);

    tensor_t output_tensor = get_node_output_tensor(conv_node, 0);
    tensor_t output_tensor1 = get_node_output_tensor(conv_node1, 0);

    float* buf = ( float* )get_tensor_buffer(output_tensor);
    float* buf1 = ( float* )get_tensor_buffer(output_tensor1);

    int flag = 0;
    int pad = k/2;
    h = (h - k + pad * 2)/s + 1;
    w = (w - k + pad * 2)/s + 1;
    for(int k = 0;k<c;k++)
    {
        for(int i = 0; i < h; i++)
        {
            for(int j = 0; j < w; j++)
            {
                
                float a = buf[k *h * w +i * w +j];
                float b = buf1[k*h*w + i * w +j];
                float tmp = a - b;
                if(tmp < 0) tmp = -tmp;
                if(tmp > 0.0001)
                {
                    printf("error: %d, %d, %d \n", k+1, i+1, j+1);
                    std::cout<<"-----------------------\ntest fail\n-------------------------->"<<a<< " , "<<b<<"\n";
                    flag = 1;
                    break;
                }
            }
            if(flag)
                break;
        }
        if(flag)
            break;
    }
    if(flag == 0)
        std::cout <<"=================================\n test pass\n==================================\n";
    //        }}}

    release_graph_tensor(output_tensor);
    release_graph_node(conv_node);
    release_graph_tensor(output_tensor1);
    release_graph_node(conv_node1);

    postrun_graph(graph);
    postrun_graph(graph1);

    destroy_graph(graph);
    destroy_graph(graph1);

    free(i_buf);
    free(w_buf);
    free(i_buf1);
    free(w_buf1);

    if(b_buf)
    {
        free(b_buf);
        free(b_buf1);
    }

    release_tengine();
    return 0;
}
