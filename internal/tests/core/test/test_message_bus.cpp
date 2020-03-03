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
#include <string>
#include <iostream>
#include <chrono>

#include <future>
#include <thread>

#include "mesage_bus.hpp"
#include "async_message_bus.hpp"
#include "timer.hpp"
//测试“message”

using namespace TEngine;
using namespace std;
void show1(int i)
{
    cout << "show<int>:" << i << endl;
}

void show2(string& x)
{
    cout << "show<string>:" << x << endl;
}

struct Test_1
{
    void show(int i)
    {
        cout << "Test_1::show<int>:" << i << endl;
    }
};

class Test_2
{
public:
    Test_2() {}
    ~Test_2() {}
    void show(string& x)
    {
        cout << "Test_2::show<string>:" << x << endl;
    }
};

void TestMessageBus_1()
{
    MessageBus msgbus;
    // book message
    msgbus.Book([](int a) { cout << "no_ref: " << a << endl; });
    msgbus.Book([](int& a) { cout << "lv_ref: " << a << endl; });
    msgbus.Book([](int&& a) { cout << "rv_ref: " << a << endl; });
    msgbus.Book([](const int& a) { cout << "const_lv_ref: " << a << endl; });
    msgbus.Book(
        [](int a) {
            cout << "no_ref(ret_int & key): " << a << endl;
            return a;
        },
        "a_topic");
    msgbus.Book(&show1);
    msgbus.Book(&show2);
    msgbus.Book([]() { cout << "void(void)" << endl; });
    msgbus.Book([]() { cout << "void(void) b_topic" << endl; }, "b_topic");
    Test_1 t1;
    msgbus.Book([&t1](int a) { t1.show(a); });
    Test_2 t2;
    msgbus.Book([&t2](string& x) { t2.show(x); });

    int i = 2;
    cout << "---------------------------------------------------" << endl;
    // send message
    msgbus.Send<void, int>(2);
    msgbus.Send<int, int>(2, "a_topic");
    msgbus.Send<void, int&>(i);
    msgbus.Send<void, const int&>(2);
    msgbus.Send<void, int&&>(2);
    msgbus.Send<void>();
    msgbus.Send<void>("b_topic");
    string st = "test";
    msgbus.Send<void, string&>(st);
    cout << "---------------------------------------------------" << endl;

    // remove message
    msgbus.Remove<void>();
    msgbus.Remove<void>("b_topic");
    msgbus.Remove<void, int>();
    msgbus.Remove<int, int>("a_topic");
    msgbus.Remove<void, int&>();
    msgbus.Remove<void, const int&>();
    msgbus.Remove<void, int&&>();
    msgbus
        .Remove<void, string>();    //这儿移除string，而不是string&，所以msgbus.Send<void, string&>(st)还会导致消息输出

    // Send message again
    msgbus.Send<void>();
    msgbus.Send<void>("b_topic");
    msgbus.Send<void, int>(2);
    msgbus.Send<int, int>(2, "a_topic");
    msgbus.Send<void, int&>(i);
    msgbus.Send<void, const int&>(2);
    msgbus.Send<void, int&&>(2);
    msgbus.Send<void, string&>(st);
    cout << "---------------------------------------------------" << endl;
}

//演示不同Class之间的message传递
MessageBus g_msgbus;    //或采用MessageBus的sigleton类型smb_t来定义
AsyncMessageBus g_amsgbus;
const string lunch_ready = "LunchReady";
const string rating = "Rating";

class Mama
{
public:
    Mama(bool _async)
    {
        async = _async;
        if(async)
        {
            g_amsgbus.Book([this](string* name, string* rate) { GetFeedback(name, rate); }, rating);
        }
        else
        {
            g_msgbus.Book([this](string* name, string* rate) { GetFeedback(name, rate); }, rating);
        }
    }
    ~Mama() {}
    void Cooking()
    {
        cout << "Mama is cooking ..." << endl;
        cout << "Luch is ready!" << endl;
        if(async)
        {
            g_amsgbus.Send<void>(lunch_ready);
        }
        else
        {
            g_msgbus.Send<void>(lunch_ready);
        }
    }
    void GetFeedback(string* name, string* rate)
    {
        cout << *name << ": " << *rate << endl;
    }

private:
    bool async;
};

class Child
{
public:
    Child(bool _async, string _name, string rate)
    {
        async = _async;
        name = _name;
        words = rate;
        if(async)
        {
            g_amsgbus.Book([this]() { HaveLunch(); }, lunch_ready);
        }
        else
        {
            g_msgbus.Book([this]() { HaveLunch(); }, lunch_ready);
        }
    }
    ~Child() {}
    void HaveLunch()
    {
        cout << name << " is eating ..." << endl;
        cout << "... Done!" << endl;
        if(async)
        {
            g_amsgbus.Send<void, string*, string*>(&name, &words, rating);
        }
        else
        {
            g_msgbus.Send<void, string*, string*>(&name, &words, rating);
        }
    }

private:
    string words;
    string name;
    bool async;
};

void test_async(Mama* ma, Child* tom, Child* rose, Child* jean)
{
    int64_t i = 0;
    while(i++ < 100000L)
    {
        ma->Cooking();
    }
}

void TestMessageBus_2()
{
    Mama ma(false);
    Child tom(false, string("Tom"), string("Delicious!"));
    Child rose(false, string("Rose"), string("Just so so!"));
    Child jean(false, string("Jean"), string("Terrible!"));
    ma.Cooking();
    {
        Timer tm;
#if 0        
        std::shared_future<void> task = std::async(std::launch::async/*deferred*/,  //std::launch::async
            test_async, &ma, &tom, &rose, &jean
        );
        task.get();
#else
        int64_t i = 0;
        while(i++ < 100000L)
        {
            ma.Cooking();
        }
#endif
        cerr << 100000L * 1000 / tm.elapsed_ms() << "times/s\n";
    }
}

void test_async_1(Mama* ma /*, Child* tom, Child* rose, Child* jean*/)
{
    int64_t i = 0;
    while(i++ < 100000L)
    {
        ma->Cooking();
        g_amsgbus.ProcessAllMessage();
        std::cout << i << " pass\n";
    }
}

void TestMessageBus_3()
{
    Mama ma(true);
    Child tom(true, string("Tom"), string("Delicious!"));
    Child rose(true, string("Rose"), string("Just so so!"));
    Child jean(true, string("Jean"), string("Terrible!"));
    cout << "0==================================================" << endl;
    ma.Cooking();
    cout << "1==================================================" << endl;
    while(!g_amsgbus.MessageEmpty())
    {
        g_amsgbus.ProcessOneMessage();
    }
    cout << "2==================================================" << endl;
    ma.Cooking();
    cout << "3==================================================" << endl;
    g_amsgbus.ProcessAllMessage();
    cout << "4==================================================" << endl;
    {
        Timer tm;
#if 1
        int64_t i = 0;
        while(i++ < 100000L)
        {
            ma.Cooking();
            g_amsgbus.ProcessAllMessage();
        }
#else
        std::shared_future<void> task = std::async(std::launch::async /*deferred*/,    // std::launch::async
                                                   test_async_1, &ma /*, &tom, &rose, &jean*/
        );
        task.get();
#endif
        cerr << 100000L * 1000 / tm.elapsed_ms() << "times/s\n";
    }
}

int main()
{
    cout << "==================================================" << endl;
    TestMessageBus_2();
    cout << "==================================================" << endl;
    TestMessageBus_3();
    cout << "==================================================" << endl;
    TestMessageBus_1();
    cout << "==================================================" << endl;
}