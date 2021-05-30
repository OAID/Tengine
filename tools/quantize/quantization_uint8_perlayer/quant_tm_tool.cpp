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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#include "quant_tool.hpp"
#include "quant_save_graph.hpp"


const char* help_params = "[Quant Tools Info]: optional arguments:\n"
                          "\t-h    help            show this help message and exit\n"
                          "\t-m    input model     path to input float32 tmfile\n"
                          "\t-i    image dir       path to calibration images folder\n"
                          "\t-f    scale file      path to calibration scale file\n"
                          "\t-o    output model    path to output uint8 tmfile\n"
                          "\t-a    algorithm       the type of quant algorithm(0:min-max, 1:kl, default is 0)\n"
                          "\t-g    size            the size of input image(using the resize the original image,default is 3,224,224)\n"
                          "\t-w    mean            value of mean (mean value, default is 104.0,117.0,123.0)\n"
                          "\t-s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)\n"
                          "\t-b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)\n"
                          "\t-c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)\n"
                          "\t-y    letter box      flag which indicates that letter box process image is necessary(maybe using for YOLOv3/v4, 0:OFF, 1:ON, default is 0)\n"
                          "\t-k    focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)\n"
                          "\t-t    num thread      count of processing threads(default is 1)\n";

const char* example_params = "[Quant Tools Info]: example arguments:\n"
                             "\t./quant_tool_uint8 -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_uint8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017\n";

void show_usage()
{
    fprintf(stderr, "%s\n", help_params);
    fprintf(stderr, "%s\n", example_params);
}

int main(int argc, char* argv[])
{
    QuantTool quant_tool;

    int res;
    while ((res = getopt(argc, argv, "m:a:f:o:i:g:s:w:b:c:y:k:t:h")) != -1)
    {
        switch (res)
        {
            case 'm':
                quant_tool.model_file = optarg;
                break;
            case 'a':
                quant_tool.algorithm_type = atoi(optarg);
                break;
            case 'f':
                quant_tool.scale_file = optarg;
                break;
            case 'o':
                quant_tool.output_file = optarg;
                break;
            case 'i':
                quant_tool.image_dir = optarg;
                break;
            case 'g':
                split(quant_tool.img_chw, optarg, ",");
                quant_tool.img_c = ( int )quant_tool.img_chw[0];
                quant_tool.img_h = ( int )quant_tool.img_chw[1];
                quant_tool.img_w = ( int )quant_tool.img_chw[2];
                break;
            case 'w':
                split(quant_tool.mean, optarg, ",");
                break;
            case 's':
                split(quant_tool.scale, optarg, ",");
                break;
            case 'b':
                quant_tool.sw_RGB = atoi(optarg);
                break;
            case 'c':
                quant_tool.center_crop = atoi(optarg);
                break;
            case 'y':
                split(quant_tool.letterboxs, optarg, ",");
                quant_tool.letterbox_rows = ( int )quant_tool.letterboxs[0];
                quant_tool.letterbox_cols = ( int )quant_tool.letterboxs[1];
                break;
            case 'k':
                quant_tool.focus = atoi(optarg);
                break;                
            case 't':
                quant_tool.num_thread = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* version */
    fprintf(stderr, "\n---- Tengine Post Training Quantization Tool ---- \n");
    fprintf(stderr, "\nVersion     : v1.1, %s %s\n", __TIME__, __DATE__);
    fprintf(stderr, "Status      : uint8, per-layer, asymmetric\n");

    /* check input params */
    if (quant_tool.model_file.empty())
    {
        fprintf(stderr,"[Quant Tools Info]: The input file of Float32 tmfile file not specified!\n");
        show_usage();
        return -1;
    }

    if (quant_tool.image_dir.empty())
    {
        fprintf(stderr,"[Quant Tools Info]: The input dir of Calibration image not specified!\n");
        show_usage();
        return -1;
    }

    if (quant_tool.output_file.empty())
    {
        fprintf(stderr,"[Quant Tools Info]: The output file of Int8 tmfile not specified!\n");
        show_usage();
        return -1;
    }

    /* debug info : input params */
    fprintf(stderr, "Input model : %s\n", quant_tool.model_file.c_str());
    fprintf(stderr, "Output model: %s\n", quant_tool.output_file.c_str());
    fprintf(stderr, "Calib images: %s\n", quant_tool.image_dir.c_str());
    fprintf(stderr, "Scale file  : %s\n", quant_tool.scale_file.empty()?"NULL":quant_tool.scale_file.c_str());
    fprintf(stderr, "Algorithm   : %s\n", quant_tool.algorithm_type?"KL":"MIN MAX");
    fprintf(stderr, "Dims        : %d %d %d\n", quant_tool.img_c, quant_tool.img_h, quant_tool.img_w);
    fprintf(stderr, "Mean        : %.3f %.3f %.3f\n", quant_tool.mean[0], quant_tool.mean[1], quant_tool.mean[2]);
    fprintf(stderr, "Scale       : %.3f %.3f %.3f\n", quant_tool.scale[0], quant_tool.scale[1], quant_tool.scale[2]);
    fprintf(stderr, "BGR2RGB     : %s\n", quant_tool.sw_RGB?"ON":"OFF");
    fprintf(stderr, "Center crop : %s\n", quant_tool.center_crop?"ON":"OFF");
    fprintf(stderr, "Letter box  : %.0f %.0f\n", quant_tool.letterboxs[0], quant_tool.letterboxs[1]);
    fprintf(stderr, "YOLOv5 focus: %s\n", quant_tool.focus?"ON":"OFF");
    fprintf(stderr, "Thread num  : %d\n\n", quant_tool.num_thread);

    /* quantize activation */
    quant_tool.activation_quant_tool(quant_tool.model_file.c_str(), quant_tool.image_dir.c_str(), quant_tool.img_c, quant_tool.img_h, quant_tool.img_w, quant_tool.mean, quant_tool.scale,
                                     quant_tool.num_thread, quant_tool.sw_RGB, quant_tool.center_crop, quant_tool.letterbox_rows, quant_tool.letterbox_cols, quant_tool.focus);

    /* using 3rd calibration table file */
    if (quant_tool.scale_file.empty())
    {
        /* select algorithm */
        if (quant_tool.algorithm_type == ALGORITHM_MIN_MAX)
            quant_tool.scale_file = "table_minmax.scale";
        else if  (quant_tool.algorithm_type == ALGORITHM_KL)
            quant_tool.scale_file = "table_kl.scale";
        else
        {
            fprintf(stderr,"[Quant Tools Info]: algorithm not specified, using default type MIN MAX\n");
            quant_tool.scale_file = "table_kl.scale";
        }
    }

    /* quantize weight/bias and save into uint8 tmfile */
    fprintf(stderr,"[Quant Tools Info]: Calibration file is using %s\n", quant_tool.scale_file.c_str());
    save_graph_u8_perlayer(quant_tool.model_file.c_str(), quant_tool.scale_file.c_str(), quant_tool.output_file, quant_tool.inplace, quant_tool.num_thread, false);

    fprintf(stderr, "\n---- Tengine Int8 tmfile create success, best wish for your INT8 inference has a low accuracy loss...\\(^0^)/ ----\n");

    return 0;
}
