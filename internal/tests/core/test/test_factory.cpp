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
#include <string>

#include "generic_factory.hpp"

struct base
{
    base() {}

    virtual void f()
    {
        std::cout << "base\n";
    }
    virtual void g() {}
};

struct A : public base
{
    A()
    {
        std::cout << "A constructed: no parameter\n";
        val = -1;
    }
    A(int n)
    {
        val = n;
        std::cout << "A constructed: val is: " << val << "\n";
    }

    void f()
    {
        std::cout << "A val is: " << val << "\n";
    }

    int val;
};

struct B : public base
{
    B() {}

    B(int n, const std::string& v)
    {
        val = v;
        number = n;
        std::cout << "B construct with const std::string&\n";
    }

    B(int n, std::string&& v)
    {
        val = v;
        number = n;
        std::cout << "B construct with std::string&&\n";
    }

    B(int n)
    {
        number = n;
    }

    void f()
    {
        std::cout << "B val: " << val << "\n";
    }
    void g()
    {
        std::cout << "number: " << number << "\n";
    }

    std::string val;
    int number;
};

struct C : public base
{
};

using namespace TEngine;

base* create_a(void)
{
    return new A();
}

base* create_b(const std::string& v)
{
    B* b = new B();

    b->val = v;

    return b;
}

base* create_b2(int n)
{
    B* b = new B();

    b->number = n;

    return b;
}

void test_creator(void)
{
    SpecificFactory<base> fc;

    /* creator test */

    fc.RegisterCreator("A", std::function<decltype(create_a)>(create_a));
    fc.RegisterCreator("B", std::function<base*()>([]() { return new B(); }));
    fc.RegisterCreator("C", create_a);

    base* p = fc.Create("A");

    p->f();

    p = fc.Create("B");

    p->f();

    p = fc.Create("C");

    p->f();

    /* creator with argument */

    fc.RegisterCreator("D", create_b);
    fc.RegisterCreator("E", std::function<base*(const std::string&)>(create_b));

    const std::string& hello = "hello";

    p = fc.Create<const std::string&>("D", hello);
    p->f();

    p = fc.Create<const std::string&>("E", "world");
    p->f();

    fc.RegisterCreator("E", create_b2);
    p = fc.Create("E", 1000);

    p->g();

    std::cout << "retrieval creator type info: \n";

    std::vector<std::string> info = fc.CreatorInfo("E");

    for(auto str : info)
    {
        std::cout << str << std::endl;
    }
}

struct G
{
    G()
    {
        std::cout << "No argument called\n";
    }

    G(int a)
    {
        std::cout << "int a =" << a << "\n";
    }

    G(const char* s)
    {
        std::cout << "const char * is: " << s << "\n";
    }

    G(const std::string& h)
    {
        std::cout << "const string is: " << h << "\n";
    }

    G(std::string&& r)
    {
        std::cout << "rval str is: " << r << "\n";
    }

    G(int a, int b, int c = 10)
    {
        std::cout << "three int: a " << a;
        std::cout << "b " << b;
        std::cout << "c " << c << "\n";
    }
};

void test_constructor(void)
{
    SpecificFactory<G> fc;

    fc.RegisterConstructor<int>("D");
    fc.RegisterConstructor<int, int, int>("D");
    fc.RegisterConstructor<const char*>("D");
    fc.RegisterConstructor<const std::string&>("D");
    fc.RegisterConstructor<std::string&&>("D");
    fc.RegisterConstructor("D");

    std::cout << "all register constructors for D \n";

    std::vector<std::string> info = fc.CreatorInfo("D");

    for(auto str : info)
    {
        std::cout << str << std::endl;
    }

    fc.Create("D");

    fc.Create("D", 1000);

    std::string str = "I'm string";

    fc.Create<const char*>("D", "i'm here");
    fc.Create<std::string&&>("D", std::move(str));
    fc.Create<std::string&&>("D", "who are you");
    fc.Create<const std::string&>("D", "hello, world");

    fc.Create("D", 1, 0, 1);
}

void test_interface()
{
    SpecificFactory<base>* p_fc = SpecificFactory<base>::GetFactory();

    p_fc->RegisterInterface<A>("A");
    p_fc->RegisterInterface<A, int>("A");
    p_fc->RegisterInterface<B>("B");
    p_fc->RegisterInterface<B, int>("B");
    p_fc->RegisterInterface<B, int, const std::string&>("B");
    p_fc->RegisterInterface<B, int, std::string&&>("B");

    std::cout << "Now, test create...\n";

    base* p_base;

    p_base = p_fc->Create("A");
    p_base->f();

    p_base = p_fc->Create("A", 102);
    p_base->f();

    const std::string& str = "lval";
    p_base = p_fc->Create("B", 100, str);

    p_base->f();
    p_base->g();

    p_base = p_fc->Create<int, std::string&&>("B", 200, "rval");

    p_base->f();
    p_base->g();
}

int main(void)
{
    std::cout << "test creator interface ....\n";
    test_creator();

    std::cout << "test class constructor ....\n";
    test_constructor();

    std::cout << "test class interface ....\n";
    test_interface();

    std::cout << "test done\n";

    return 0;
}
