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

#include <vector>
#include <string>
#include <sys/time.h>
#include "text_table.h"

namespace pipeline {
class Profiler
{
public:
    Profiler() = delete;
    Profiler(const std::string& name)
    {
        m_table = TextTable(name);
        m_table.padding(1);
        m_table.align(TextTable::Align::Mid)
            .add("preproc")
            .add("inference")
            .add("postproc")
            .eor();

        m_data.reserve(4);
    }

    void dot()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        m_data.emplace_back(tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0);
    }

    ~Profiler()
    {
        if (m_data.size() > 4)
        {
            m_data.resize(4);
        }
        const int size = m_data.size();
        switch (size)
        {
        case 0:
        case 1:
            break;
        case 2:
            m_table.align(TextTable::Align::Mid);
            m_table.add(std::to_string(m_data[1] - m_data[0]));
            m_table.eor();
            break;
        case 3:
            m_table.align(TextTable::Align::Mid);
            m_table.add(std::to_string(m_data[1] - m_data[0]));
            m_table.add(std::to_string(m_data[2] - m_data[1]));
            m_table.eor();
            break;
        case 4:
            m_table.align(TextTable::Align::Mid);
            m_table.add(std::to_string(m_data[1] - m_data[0]));
            m_table.add(std::to_string(m_data[2] - m_data[1]));
            m_table.add(std::to_string(m_data[3] - m_data[2]));
            m_table.eor();
            break;
        default:
            break;
        }
        std::stringstream ss;
        ss << m_table;
        fprintf(stdout, "%s\n", ss.str().c_str());
        return;
    }

    void reset()
    {
        m_data.clear();
    }

private:
    std::vector<double> m_data;
    TextTable m_table;
};

} // namespace pipeline
