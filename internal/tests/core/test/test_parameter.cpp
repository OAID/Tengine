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

#include "parameter.hpp"
using namespace TEngine;

struct test
{
    int w;
    float b;
    std::string s;

    test() = default;
    test(int n, float m, const std::string& str)
    {
        w = n;
        b = m;
        s = str;
    }
};

struct MyParam
{
    int n;
    int i;
    float f;
    test t;
    std::string doc;

    DECLARE_PARSER_STRUCTURE(MyParam)
    {
        DECLARE_PARSER_ENTRY(n);
        DECLARE_PARSER_ENTRY(f);
        DECLARE_PARSER_ENTRY(i);
        DECLARE_PARSER_ENTRY(doc);
        DECLARE_CUSTOM_PARSER_ENTRY(t);
    }
};

int main(void)
{
    BaseObject obj;

    obj["n"] = 100;
    obj["f"] = -1.02;
    obj["doc"] = "this is a test";
    obj["i"] = "print error please";
    obj["t"] = test(101, 0.99, "a complicated arg");

    MyParam param;

    MyParam::Parse(param, &obj);

    std::cout << param.n << std::endl;
    std::cout << param.doc << std::endl;
    std::cout << param.f << std::endl;
    test t = param.t;

    std::cout << t.w << std::endl;
    std::cout << t.b << std::endl;
    std::cout << t.s << std::endl;

    BaseObject Another(obj);

    std::cout << any_cast<int>(Another["n"]) << std::endl;

    return 0;
}
