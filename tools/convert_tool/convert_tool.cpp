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
 * Author: xlchen@openailab.com
 */

#include <iostream>
#include <stdlib.h>
#ifdef _MSC_VER
#include <io.h>
#include <msc_getopt.h>
#else
#include <unistd.h>
#endif

#include "api/c_api.h"
#include "save_graph/save_graph.hpp"
#include "onnx/onnx2tengine.hpp"
#include "caffe/caffe2tengine.hpp"
#include "ncnn/ncnn2tengine.hpp"
#include "tensorflow/tf2tengine.hpp"
#include "mxnet/mxnet2tengine.hpp"
#include "tflite/tflite2tengine.hpp"
#include "darknet/darknet2tengine.hpp"
#include "utils/graph_optimizer/graph_opt.hpp"

const char* help_params = "[Convert Tools Info]: optional arguments:\n"
                          "\t-h    help            show this help message and exit\n"
                          "\t-f    input type      path to input float32 tmfile\n"
                          "\t-p    input structure path to the network structure of input model(*.param, *.prototxt, *.symbol, *.cfg, *.pdmodel)\n"
                          "\t-m    input params    path to the network params of input model(*.bin, *.caffemodel, *.params, *.weight, *.pb, *.onnx, *.tflite, *.pdiparams)\n"
                          "\t-o    output model    path to output fp32 tmfile\n";

const char* example_params = "[Convert Tools Info]: example arguments:\n"
                             "\t./convert_tool -f onnx -m ./mobilenet.onnx -o ./mobilenet.tmfile\n"
                             "\t./convert_tool -f caffe -p ./mobilenet.prototxt -m ./mobilenet.caffemodel -o ./mobilenet.tmfile\n"
                             "\t./convert_tool -f mxnet -p ./mobilenet.params -m ./mobilenet.json -o ./mobilenet.tmfile\n"
                             "\t./convert_tool -f darknet -p ./yolov3.weights -m ./yolov3.cfg -o yolov3.tmfile\n";
void show_usage()
{
    fprintf(stderr, "%s\n", help_params);
    fprintf(stderr, "%s\n", example_params);
}

int main(int argc, char* argv[])
{
    std::string file_format;
    std::string proto_file;
    std::string model_file;
    std::string output_tmfile;
    bool proto_file_needed = false;
    bool model_file_needed = false;
    int input_file_number = 0;

    int res;
    while ((res = getopt(argc, argv, "f:p:m:o:h")) != -1)
    {
        switch (res)
        {
        case 'f':
            file_format = optarg;
            break;
        case 'p':
            proto_file = optarg;
            break;
        case 'm':
            model_file = optarg;
            break;
        case 'o':
            output_tmfile = optarg;
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            show_usage();
            break;
        }
    }

    /* version */
    fprintf(stderr, "\n---- Tengine Convert Tool ---- \n");
    fprintf(stderr, "\nVersion     : v1.0, %s %s\n", __TIME__, __DATE__);
    fprintf(stderr, "Status      : float32\n\n");

    // Check the input parameters

    if (file_format.empty())
    {
        show_usage();
        return -1;
    }
    else
    {
        if (file_format == "caffe" || file_format == "mxnet" || file_format == "darknet" || file_format == "ncnn" || file_format == "oneflow" || file_format == "paddle")
        {
            proto_file_needed = true;
            model_file_needed = true;
            input_file_number = 2;
        }
        else if (file_format == "caffe_single" || file_format == "onnx" || file_format == "tensorflow" || file_format == "tflite")
        {
            model_file_needed = true;
            input_file_number = 1;
        }
        else
        {
            std::cout << "Allowed input file format: caffe, caffe_single, onnx, oneflow, mxnet, tensorflow, darknet, ncnn, paddle\n";
            return -1;
        }
    }

    if (proto_file_needed)
    {
        if (proto_file.empty())
        {
            std::cout << "Please specify the -p option to indicate the input proto file.\n";
            return -1;
        }
        if (access(proto_file.c_str(), 0) == -1)
        {
            std::cout << "Proto file does not exist: " << proto_file << "\n";
            return -1;
        }
    }

    if (model_file_needed)
    {
        if (model_file.empty())
        {
            std::cout << "Please specify the -m option to indicate the input model file.\n";
            return -1;
        }
        if (access(model_file.c_str(), 0) == -1)
        {
            std::cout << "Model file does not exist: " << model_file << "\n";
            return -1;
        }
    }

    if (output_tmfile.empty())
    {
        std::cout << "Please specify the -o option to indicate the output tengine model file.\n";
        return -1;
    }
    if (output_tmfile.rfind("/") != std::string::npos)
    {
        std::string output_dir = output_tmfile.substr(0, output_tmfile.rfind("/"));
        if (access(output_dir.c_str(), 0) == -1)
        {
            std::cout << "The dir of output file does not exist: " << output_dir << "\n";
            return -1;
        }
    }

    init_tengine();
    set_log_level(LOG_INFO);
    graph_t graph = NULL;
    if (file_format == "onnx")
    {
        onnx_serializer o2t;
        graph = o2t.onnx2tengine(model_file);
    }
    else if (file_format == "caffe")
    {
        caffe_serializer c2t;
        graph = c2t.caffe2tengine(model_file, proto_file);
    }
    else if (file_format == "ncnn")
    {
        ncnn_serializer n2t;
        graph = n2t.ncnn2tengine(model_file, proto_file);
    }
    else if (file_format == "tensorflow")
    {
        tensorflow_serializer tf2t;
        graph = tf2t.tensorflow2tengine(model_file);
    }
    else if (file_format == "mxnet")
    {
        mxnet_serializer m2t;
        graph = m2t.mxnet2tengine(model_file, proto_file);
    }
    else if (file_format == "tflite")
    {
        tflite_serializer l2t;
        graph = l2t.tflite2tengine(model_file);
    }
    else if (file_format == "darknet")
    {
        darknet_serializer d2t;
        graph = d2t.darknet2tengine(model_file, proto_file);
    }
    else
    {
        fprintf(stderr, "Convert model failed: unsupport model format.\n");
        return -1;
    }

    if (graph == NULL)
    {
        fprintf(stderr, "Convert model failed.\n");
        return -1;
    }

    if (graph_opt(graph) < 0)
    {
        fprintf(stderr, "optimize graph failed! \n");
        return -1;
    }

    if (save_graph(graph, output_tmfile.c_str()) < 0)
    {
        fprintf(stderr, "save graph failed! \n");
        return -1;
    }

    fprintf(stderr, "Convert model success. %s -----> %s \n", model_file.c_str(), output_tmfile.c_str());

    return 0;
}
