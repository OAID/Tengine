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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <iostream>
#include <string>

#include "data_type.hpp"
#include "data_layout.hpp"

using namespace TEngine;

struct A
{
    A()
    {
        val = "none";
    }

    A(const std::string& str)
    {
        val = str;
    }

    A(A&& a)
    {
        val = std::move(a.val);
    }

    A& operator=(A&& a)
    {
        val = std::move(a.val);

        return *this;
    }

    ~A() {}

    std::string val;
};

namespace TEngine {

template <> A DataType::Convert<A>(const std::string& str) const
{
    A a(str);
    return a;
}
}    // namespace TEngine

int main(void)
{
    const DataType* p_type;

    p_type = DataType::GetType("int");

    std::cout << p_type->GetTypeName() << std::endl;

    std::cout << p_type->Convert<int>("100") << std::endl;

    std::cout << p_type->Convert<int>("0x100") << std::endl;

    p_type = DataType::GetType("float8");

    if(p_type != nullptr)
        std::cout << "ERROR, p_type should be nullptr" << std::endl;

    p_type = DataType::GetType("float32");

    std::cout << p_type->Convert<float>("1.05") << std::endl;
    std::cout << p_type->Convert<float>("-1.999") << std::endl;

    /* New type */

    DataType newA("newA", sizeof(A));

    p_type = DataType::GetType("newA");

    std::cout << p_type->GetTypeName() << std::endl;

    std::cout << p_type->Convert<A>("1.05").val << std::endl;

    return 0;
}
