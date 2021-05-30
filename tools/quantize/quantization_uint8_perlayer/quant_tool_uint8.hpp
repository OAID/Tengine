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

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/stat.h>
#include <dirent.h>
#include <tr1/unordered_map>

#include <fstream>
#include <string>
#include <cmath>
#include <vector>




#include "tengine/c_api.h"

extern "C" {
    #include "graph/graph.h"
    #include "graph/subgraph.h"
    #include "graph/node.h"
    #include "graph/tensor.h"
    #include "utility/sys_port.h"
    #include "utility/utils.h"
}

#include "operator/prototype/convolution_param.h"
#include "operator/prototype/fc_param.h"
#include "operator/prototype/pooling_param.h"
#include "operator/prototype/relu_param.h"

//#include "quant_utils.hpp"
//#include "save_graph.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define ALGORITHM_MIN_MAX 0
#define ALGORITHM_KL      1

class QuantTool
{
public:
    QuantTool();
    ~QuantTool();

    int activation_quant_tool(const char* model_file, const char* image_dir,
                              int img_c, int img_h, int img_w, const float* mean, const float* scale,
                              int num_thread, int sw_RGB, int center_crop, int letterbox_rows, int letterbox_cols, int focus);
public: 
    struct options opt;

    std::string model_file;
    std::string scale_file;
    std::string output_file;
    std::string image_dir;

    int num_thread;

    float img_chw[3];
    float letterboxs[2];
    int sw_RGB;
    int img_c;
    int img_h;
    int img_w;
    float mean[3];
    float scale[3];
    int center_crop;
    int letterbox_rows;
    int letterbox_cols;
    int focus;
    int inplace; // process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip
    int algorithm_type;
};
