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
#include <thread>
#include <mutex>
#include <cstdlib>

#include <unistd.h>

#include "logger.hpp"
#include "notify_instance.hpp"

using namespace TEngine;

volatile int active_thread;    // should use atomic_t

void EventSender(NotifyBus* bus, int event, int number)
{
    EventData arg;

    arg.event = event;

    for(int i = number - 1; i >= 0; i--)
    {
        arg.data = i;
        arg.args = bus;
        bus->Send(event, &arg, i);
    }
}

bool Eventhandler0(int event, EventData* data, any* argument)
{
    std::cout << "0: event: " << event << " data: " << data->data << "\n";

    if(data->data == 0)
    {
        NotifyBus* bus;

        bus = any_cast<NotifyBus*>(data->args);

        EventData arg = *data;
        arg.event = 0;
        arg.data = 10000;
        bus->Send(arg.event, &arg, 0);
    }

    return true;
}

bool Eventhandler1(int event, EventData* data, any* argument)
{
    std::cout << "1: event: " << event << " data: " << data->data << "\n";

    return true;
}

int test_single(void)
{
    NotifyBus bus("test", 1024, 256);

    const std::string event_name = "TEST";

    int event_id = bus.CreateEvent(event_name);

    std::cout << "second event: " << bus.CreateEvent("TEST1") << "\n";

    EventHandle handle;

    handle.func = Eventhandler0;

    bus.Book(event_id, handle);
    bus.Book(1, handle);

    handle.func = Eventhandler1;

    bus.Book(event_name, handle);

    EventStats stats;

    EventSender(&bus, event_id, 10);

    std::cout << "Sender done\n";

    bus.GetStats(event_id, stats);

    stats.Dump(std::cout);

    bus.ProcessAllEvents();

    std::cout << "Process Done\n";
    std::cout << "sync send\n";

    {
        EventData arg;

        arg.event = 0;
        arg.data = 0;
        arg.args = &bus;

        bus.SyncSend(event_id, &arg);
    }

    bus.GetStats(event_id, stats);

    stats.Dump(std::cout);

    bus.GetStats(1, stats);

    stats.Dump(std::cout);

    std::cout << "Remove Event: \n";
    bus.RemoveEvent(0);

    std::cout << "Remove Done \n";
    return 0;
}

struct SyncData
{
    std::mutex sync;
};

bool handle_event(int event, EventData* data, any* argument)
{
    SyncData* p_data = any_cast<SyncData*>(*argument);

    p_data->sync.unlock();

    return true;
}

void echo_thread(int count, NotifyBus* bus, int my_event, int peer_event)
{
    int i = 0;

    EventHandle handle;
    SyncData sync_data;

    sync_data.sync.lock();

    handle.func = handle_event;
    handle.argument = &sync_data;

    bus->Book(my_event, handle);

    EventData data;

    data.event = peer_event;

    if(my_event == 0)
    {
        // make sure other threads have initialized already
        sleep(1);
        data.data = -1;
        bus->Send(peer_event, &data);
    }

    std::cout << "My event: " << my_event << "\n";

    while(true)
    {
        sync_data.sync.lock();

        std::cout << "send event to: " << peer_event << " # " << i << "\n";

        data.data = i;

        bus->Send(peer_event, &data);

        if(i++ == count)
            break;
    }

    std::cout << "Total received: " << count << " events\n";
}

void test_threads(void)
{
    NotifyBus bus("thread", 1024, 256);

    int e0 = bus.CreateEvent("thread0");
    int e1 = bus.CreateEvent("thread1");
    int e2 = bus.CreateEvent("thread2");
    int e3 = bus.CreateEvent("thread3");
    int count = 10;

    std::thread t0(echo_thread, count, &bus, e0, e1);
    std::thread t1(echo_thread, count, &bus, e1, e2);
    std::thread t2(echo_thread, count, &bus, e2, e3);
    std::thread t3(echo_thread, count, &bus, e3, e0);

    // using furture and promise to get the  quit of threads

    while(true)
        bus.ProcessAllEvents();

    t0.join();
    t1.join();
    t2.join();
    t3.join();
}

int main(void)
{
    test_single();
    test_threads();

    return 0;
}
