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
 * Copyright (c) 2021
 * Author: tpoisonooo
 */
#pragma once
#include "../graph/node.h"
#include <opencv2/opencv.hpp>

namespace pipe {

class DrawVideo : public Node<Param<cv::Mat>, Param<void> >
{
public:
    void exec() override
    {
        cv::Mat mat;
        while (true)
        {
            auto suc = input<0>()->pop(mat);
            if (not suc)
            {
                continue;
            }
            // fprintf(stdout, "show\n");
            cv::imshow("camera", mat);
            cv::waitKey(25);
        }
    }

    ~DrawVideo()
    {
        cv::destroyAllWindows();
    }
};

} // namespace pipe