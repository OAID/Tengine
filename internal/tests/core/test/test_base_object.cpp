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

#include "base_object.hpp"

using namespace TEngine;

class A0 : public BaseObject
{
public:
    A0()
    {
        (*this)["a0_obj"] = 1;
    }

    REGISTER_CLASS_ATTR_OPS(BaseObject);
};

class A1 : public A0
{
public:
    A1()
    {
        (*this)["a1_obj"] = 1;
    }
    REGISTER_CLASS_ATTR_OPS(A0);
};

class B0 : public BaseObject
{
public:
    B0()
    {
        (*this)["b0_obj"] = 1;
    }
};

class B1 : public B0
{
public:
    B1()
    {
        (*this)["b1_obj"] = 1;
    }
    REGISTER_CLASS_ATTR_OPS(B0);
};

void show_string_vector(const std::vector<std::string>& vec)
{
    for(unsigned int i = 0; i < vec.size(); i++)
    {
        std::cout << i << ": " << vec[i] << std::endl;
    }
}

int main(void)
{
    BaseObject root;

    root["obj_r0"] = "root";

    root.SetClassAttr("r0", 1);

    A0 a0;
    a0.SetClassAttr("a0", 100);
    a0["obj_a0"] = 100;

    A1 a1;
    a1.SetClassAttr("a1", 101);
    a1["obj_a1"] = 101;

    B0 b0;
    b0.SetClassAttr("b0", 200);
    b0["obj_b0"] = 200;

    B1 b1;
    b1.SetClassAttr("b1", 201);
    b1["obj_b1"] = 201;

    std::cout << "Attributes of root" << std::endl;

    show_string_vector(root.ListAttr());

    std::cout << "Attributes of a0" << std::endl;

    show_string_vector(a0.ListAttr());

    std::cout << "Attributes of a1" << std::endl;

    show_string_vector(a1.ListAttr());

    std::cout << "Attributes of b0" << std::endl;

    show_string_vector(b0.ListAttr());

    std::cout << "Attributes of b1" << std::endl;

    show_string_vector(b1.ListAttr());

    BaseObject* p_base = new B1;

    show_string_vector(p_base->ListAttr());

    std::cout << "b1: " << any_cast<int>((*p_base)["b1"]) << std::endl;

    return 0;
}
