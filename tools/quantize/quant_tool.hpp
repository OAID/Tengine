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


#include <string>
#include <vector>
#include <unordered_map>

extern "C"
{
#include "tengine/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "utility/sys_port.h"
#include "utility/utils.h"
}

#define ALGORITHM_MIN_MAX 0
#define ALGORITHM_KL      1


class QuantTool {
public:
    QuantTool();
    ~QuantTool();

    int activation_quant_tool();

public:
    struct options opt;

    std::string model_file;     // path to input float32 tmfile
    std::string scale_file;     // path to calibration scale file
    std::string output_file;    // path to output int8/uint8 tmfile
    std::string image_dir;      // path to calibration images folder

    int num_thread;

    int   img_c;
    int   img_h;
    int   img_w;
    float mean[3];        // value of mean (mean value, default is 104.0,117.0,123.0)
    float scale[3];       // value of normalize (scale value, default is 1.0,1.0,1.0)
    int   center_crop;    // flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
    int   letterbox_rows;
    int   letterbox_cols;
    int   sw_RGB;    // flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)
    int   focus;    // flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)
    int   inplace;    // process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip
    int   algorithm_type;    // the type of quant algorithm(0:min-max, 1:kl, default is 0)
};
