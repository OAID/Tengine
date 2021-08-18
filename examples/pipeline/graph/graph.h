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

#include "edge.h"
#include "node.h"
#include <atomic>
#include <chrono>
#include <memory>

namespace pipeline {

class Graph
{
public:
    explicit Graph()
    {
    }
    virtual ~Graph() = default;

    template<typename T, typename... Args>
    T* add_node(Args&&... args)
    {
        T* ptr = new T(std::forward<Args>(args)...);
        m_nodes.emplace_back(std::unique_ptr<T>(ptr));
        return ptr;
    }

    template<typename T>
    T* add_edge(size_t cap)
    {
        T* ptr = new T(cap);
        m_edges.emplace_back(std::unique_ptr<T>(ptr));
        return ptr;
    }

    void start()
    {
        m_running = true;
        for (auto&& node : m_nodes)
        {
            m_threads.emplace_back(std::thread([&]() -> void {
                while (m_running)
                {
                    node.get()->exec();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }));
        }
    }

    void finish()
    {
        m_running = false;
        for (auto&& thread : m_threads)
        {
            thread.join();
        }
    }

private:
    std::vector<std::unique_ptr<BaseNode> > m_nodes;
    std::vector<std::unique_ptr<BaseEdge> > m_edges;
    std::atomic<bool> m_running;
    std::vector<std::thread> m_threads;
};
} // namespace pipeline
