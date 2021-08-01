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

namespace pipe {
class Profiler
{
public:
    Profiler() = delete;
    Profiler(std::vector<std::string> tags): m_tags(tags) {}

    void dot() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        m_data.emplace_back(tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0);
    }

    void show() noexcept {
        if (m_data.size() <= m_tags.size()) {
            fprintf(stderr, "profile dot not enough\n");
        }

        for (size_t i = 0; i < m_tags.size(); ++i) {
            fprintf(stdout, "%s: %.2f \t", m_tags[i].c_str(), m_data[i+1] - m_data[i]);
        }
        fprintf(stdout, "\n");
    }

    void clear() {
        m_data.clear();
    }

private:
    std::vector<std::string> m_tags;
    std::vector<double> m_data;

};

} // namespace pipe
