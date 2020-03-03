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
#include <vector>
#include <memory>
#include <typeinfo>
#include "type_name.hpp"
#include "any.hpp"
#include "attribute.hpp"

using namespace TEngine;

class A
{
public:
    A()
    {
        val_ = 10;
    }

    A(int n)
    {
        val_ = n;
    }

    A(const A& other)
    {
        val_ = other.val_;
    };

    int Get()
    {
        return val_;
    }

    A& operator=(int n)
    {
        val_ = n;
        return *this;
    }

    ~A()
    {
        std::cout << "A destructed: " << val_ << std::endl;
    }

private:
    int val_;
};

class B
{
public:
    B() {}

    B(const std::string& d)
    {
        str_ = d;
    };

    B& operator=(const std::string& s)
    {
        str_ = s;
        return *this;
    };

    ~B()
    {
        std::cout << "B destructed: " << str_ << std::endl;
    };

private:
    std::string str_;
    char n[256];
};

class T_TEST
{
public:
    T_TEST()
    {
        ;
    }
    ~T_TEST() {}
    static void Add(int a, int b = 1000)
    {
        _x = a + b;
    }

private:
    static int _x;
};

using BaseObject = Attribute;

int main(int argc, char* argv[])
{
    B b("class b");
    A a;
    BaseObject obj = {{"name", "Tom"}, {"Age", 13}};

    a = 200;
    b = "new value";

    obj["my0"] = b;
    obj["my1"] = a;
    a = 100;

    obj["my3"] = a;
    obj["my2"] = b;
    obj.RemoveAttr("my0");

    A* p_a = new A(1000);

    obj["ptr"] = std::shared_ptr<A>(p_a);

    obj["int"] = 101;

    std::vector<std::string> attr = obj.ListAttr();

    std::vector<std::string>::iterator ir = attr.begin();

    std::cout << "List attr names: " << std::endl;
    while(ir != attr.end())
    {
        std::cout << *ir << "::type<" << GetTypeName(obj[*ir].type().name()) << ">" << std::endl;
        ++ir;
    }
    std::cout << "name: " << any_cast<char const*>(obj["name"]) << std::endl;
    std::cout << "Age:  " << any_cast<int>(obj["Age"]) << std::endl;
    std::cout << "my1:  " << (any_cast<A>(obj["my1"])).Get() << std::endl;
    std::cout << "my3:  " << (any_cast<A>(obj["my3"])).Get() << std::endl;

    OutputTypeName(typeid(T_TEST::Add).name());

    std::cout << std::endl;
    OutputTypeName("_ZN2A2C1Ei");
    std::cout << std::endl;
    OutputTypeName("_ZZN7TEngine12ClassFactory12RegisterTypeI2A2IEEEvRKNSt7__cxx1112basic_stringIcSt11char_"
                   "traitsIcESaIcEEEENKUlvE_clEv");
    std::cout << std::endl;

    return 0;
}
