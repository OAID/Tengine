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
#include "pipeline/graph/node.h"
#include <opencv2/opencv.hpp>

namespace pipeline {

class DrawVideo : public Node<Param<cv::Mat>, Param<void> >
{
public:
    DrawVideo(const std::string& name = "window")
        : m_window_name(name)
    {
    }

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
            cv::imshow(m_window_name, mat);
            cv::waitKey(1);
        }
    }

    ~DrawVideo()
    {
        cv::destroyAllWindows();
    }

private:
    std::string m_window_name;
};

} // namespace pipeline
