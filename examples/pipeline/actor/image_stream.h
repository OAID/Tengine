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
#include <thread>
#include <mutex>
#include "pipeline/graph/node.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <dirent.h>
#include <unistd.h>

namespace pipeline {

class ImageStream : public Node<Param<void>, Param<cv::Mat> >
{
public:
    ImageStream(const std::string file_path = "")
        : m_path(file_path)
    {
    }

    std::vector<std::string> list_files()
    {
        std::vector<std::string> out;

        auto is_image = [](const std::string& dname) -> bool {
            if (dname.find(".jpg") != std::string::npos || dname.find(".png") != std::string::npos || dname.find(".jpeg") != std::string::npos)
            {
                return true;
            }
            return false;
        };

        if (is_image(m_path))
        {
            out.emplace_back(m_path);
            return out;
        }

        struct dirent* ptr;
        DIR* dir;
        if ((dir = opendir(m_path.c_str())) == NULL)
        {
            fprintf(stdout, "cannot open %s\n", m_path.c_str());
            return out;
        }
        while ((ptr = readdir(dir)) != NULL)
        {
            if (ptr->d_type == 8)
            {
                std::string dname;
                if (m_path.rfind('/') == m_path.length() - 1)
                {
                    dname = m_path + ptr->d_name;
                }
                else
                {
                    dname = m_path + "/" + ptr->d_name;
                }
                if (is_image(dname))
                {
                    out.push_back(dname);
                }
            }
        }
        closedir(dir);
        return out;
    }

    void exec() override
    {
        std::call_once(flag, [&]() {
            std::vector<std::string> files = list_files();
            if (files.empty())
            {
                return;
            }
            for (const auto& file : files)
            {
                cv::Mat m = cv::imread(file);
                if (not output<0>()->try_push(m.clone()))
                {
                    fprintf(stdout, "drop " __FILE__ "\n");
                }
                // wait 40ms in case of drop
                cv::waitKey(40);
            }
        });
    }

private:
    std::string m_path;
    std::once_flag flag;
};

} // namespace pipeline
