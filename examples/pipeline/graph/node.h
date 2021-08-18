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
#include <tuple>
#include <vector>

namespace pipeline {

template<typename... Args>
class Param
{
public:
    using DataTypes = std::tuple<Args...>;
    using EdgePtrTypes = std::tuple<InstantEdge<Args>*...>;
};

class BaseNode
{
public:
    BaseNode() = default;
    virtual ~BaseNode()
    {
    }
    virtual void exec()
    {
        fprintf(stderr, "do not exec this function!\n");
        assert(0);
    };
};

template<typename IN, typename OUT>
class Node : public BaseNode
{
    template<size_t I>
    using IN_Edge =
        typename std::tuple_element<I, typename IN::EdgePtrTypes>::type;

    template<size_t I>
    using OUT_Edge =
        typename std::tuple_element<I, typename OUT::EdgePtrTypes>::type;

public:
    template<size_t I>
    IN_Edge<I> input()
    {
        return std::get<I>(m_inputs);
    }

    template<size_t I>
    OUT_Edge<I> output()
    {
        return std::get<I>(m_outputs);
    }

    template<size_t I>
    void set_input(IN_Edge<I> edge)
    {
        std::get<I>(m_inputs) = edge;
        edge->set_out_node(this, m_name);
    }

    template<size_t I>
    void set_output(OUT_Edge<I> edge)
    {
        std::get<I>(m_outputs) = edge;
        edge->set_in_node(this, m_name);
    }

    void exec()
    {
    }

protected:
    std::string m_name;
    typename IN::EdgePtrTypes m_inputs;
    typename OUT::EdgePtrTypes m_outputs;
};

} // namespace pipeline
