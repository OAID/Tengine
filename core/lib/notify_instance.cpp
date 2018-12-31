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
#include "notify_instance.hpp"

namespace TEngine {

static void ResetEventStats(NotifyBus::EventNode* node)
{
    node->sync_send_count = 0;
    node->send_count = 0;
    node->drop_count = 0;
    node->process_count = 0;
    node->send_fail_count = 0;
    node->active = 0;
}

NotifyBus::NotifyBus(const std::string& name, int max_event_type_num, int max_event_pending_num)
{
    bus_name_ = name;
    max_event_type_ = max_event_type_num;
    max_pending_num_ = max_event_pending_num;
    batch_budget_ = 128;

    /* create the event node array */
    event_array_ = new EventNode[max_event_type_];

    for(int i = 0; i < max_event_type_; i++)
        event_array_[i].event = -1;
}

NotifyBus::~NotifyBus(void)
{
    DropPendingEvent();
    delete[] event_array_;
}

int NotifyBus::CreateEvent(const std::string& name)
{
    if(name.size() == 0)
        return -1;

    LockNameMap();

    if(name_map_.count(name))
    {
        UnLockNameMap();
        return -1;
    }

    int i = 0;

    for(i = 0; i < max_event_type_; i++)
        if(event_array_[i].event < 0)
            break;

    if(i == max_event_type_)
    {
        UnLockNameMap();
        return -1;
    }

    /* it is safe to work on the event, since no one should touch it */

    name_map_[name] = i;
    event_array_[i].event = i;
    event_array_[i].name = name;
    ResetEventStats(&event_array_[i]);

    UnLockNameMap();

    return i;
}

int NotifyBus::GetPendingCount(EventNode* node)
{
    event_tree_t::iterator ir = node->event_tree.begin();

    int count = 0;

    while(ir != node->event_tree.end())
    {
        count += ir->second.size();
        ir++;
    }

    return count;
}

int NotifyBus::RealDropPendingEvent(EventNode* node)
{
    int count = GetPendingCount(node);

    node->drop_count += count;

    node->event_tree.clear();

    return count;
}

bool NotifyBus::RemoveEvent(int event)
{
    LockNameMap();

    bool found = false;

    name_map_t::iterator ir = name_map_.begin();

    while(ir != name_map_.end())
    {
        if(ir->second == event)
        {
            found = true;
            break;
        }
        ir++;
    }

    if(!found)
    {
        UnLockNameMap();
        return false;
    }

    name_map_.erase(ir);

    EventNode* node = GetNode(event);

    RealDropPendingEvent(node);

    node->handle_list.clear();
    node->event = -1;
    node->name = "unused";

    PutNode(node);

    UnLockNameMap();

    return true;
}

bool NotifyBus::RemoveEvent(const std::string& name)
{
    int event = EventNum(name);

    if(event < 0)
        return false;

    return RemoveEvent(event);
}

std::string NotifyBus::EventName(int event)
{
    std::string result;

    LockNameMap();

    name_map_t::iterator ir = name_map_.begin();

    while(ir != name_map_.end())
    {
        if(ir->second == event)
        {
            result = ir->first;
            break;
        }
    }

    UnLockNameMap();

    return result;
}

int NotifyBus::EventNum(const std::string& name)
{
    int id = -1;

    LockNameMap();

    name_map_t::iterator ir = name_map_.find(name);

    if(ir != name_map_.end())
        id = ir->second;

    UnLockNameMap();

    return id;
}

std::string NotifyBus::BusName()
{
    return bus_name_;
}

int NotifyBus::DropPendingEvent(int event)
{
    EventNode* node = GetNode(event);

    if(node == nullptr)
        return 0;

    int count = RealDropPendingEvent(node);

    PutNode(node);

    return count;
}

int NotifyBus::DropPendingEvent(void)
{
    int count = 0;

    for(int i = 0; i < max_event_type_; i++)
        count += DropPendingEvent(i);

    return count;
}

bool NotifyBus::Send(int event, EventData* data, int priority)
{
    EventNode* node = GetNode(event);

    if(node == nullptr)
        return false;

    int count = GetPendingCount(node);

    if(count >= max_pending_num_)
    {
        node->send_fail_count++;
        PutNode(node);
        return false;
    }

    node->event_tree[priority].emplace(*data);

    node->send_count++;

    if(!node->active)
    {
        node->active = true;

        pending_queue_mutex.lock();

        pending_queue.push(node);

        pending_queue_mutex.unlock();
    }

    PutNode(node);

    return true;
}

bool NotifyBus::Send(const std::string& event_name, EventData* data, int priority)
{
    int event = EventNum(event_name);

    if(event < 0)
        return false;

    return Send(event, data, priority);
}

bool NotifyBus::SyncSend(int event, EventData* data)
{
    EventNode* node = GetNode(event);

    if(node == nullptr)
    {
        return false;
    }

    HandleSingleEvent(node, data);

    node->sync_send_count++;

    PutNode(node);

    return true;
}

bool NotifyBus::SyncSend(const std::string& event_name, EventData* data)
{
    int event = EventNum(event_name);

    if(event < 0)
        return false;

    return SyncSend(event, data);
}

bool NotifyBus::Book(int event, const EventHandle& handle)
{
    EventNode* node = GetNode(event);

    if(node == nullptr)
        return false;

    node->handle_list.emplace_back(handle);

    PutNode(node);

    return true;
}

bool NotifyBus::Book(const std::string& event_name, const EventHandle& handle)
{
    int event = EventNum(event_name);

    if(event < 0)
        return false;

    return Book(event, handle);
}

void NotifyBus::HandleSingleEvent(EventNode* node, EventData* data)
{
    for(unsigned int i = 0; i < node->handle_list.size(); i++)
    {
        EventHandle& handle = node->handle_list[i];

        /* if do not want other handler to process this event */
        if(!handle.func(node->event, data, &handle.argument))
            break;
    }

    node->process_count++;
}

int NotifyBus::RealHandlePendingEvent(EventNode* node)
{
    int count = 0;

    event_tree_t::iterator ir = node->event_tree.begin();

    while(ir != node->event_tree.end())
    {
        std::queue<EventData>& event_queue = ir->second;

        while(!event_queue.empty())
        {
            EventData data = event_queue.front();

            event_queue.pop();

            HandleSingleEvent(node, &data);
            count++;

            batch_count_++;

            if(batch_count_ > batch_budget_)
                return count;
        }

        ir = node->event_tree.erase(ir);
    }

    return count;
}

int NotifyBus::HandlePendingEvent(EventNode* node)
{
    int count = 0;

    while(node->event_tree.size())
    {
        count += RealHandlePendingEvent(node);

        if(batch_count_ >= batch_budget_)
            break;
    }

    return count;
}

int NotifyBus::ProcessAllEvents(void)
{
    batch_count_ = 0;

    /* first, move Node from pending queue to active queue */
    pending_queue_mutex.lock();

    while(!pending_queue.empty())
    {
        EventNode* node = pending_queue.front();
        active_queue.push(node);
        pending_queue.pop();
    }

    pending_queue_mutex.unlock();

    while(!active_queue.empty())
    {
        EventNode* node = active_queue.front();

        active_queue.pop();

        if(!GetNode(node))
            continue;

        HandlePendingEvent(node);

        if(node->event_tree.size())
        {
            active_queue.push(node);
        }
        else
        {
            node->active = false;
        }

        PutNode(node);

        if(batch_count_ >= batch_budget_)
            break;
    }

    return batch_count_;
}

void NotifyBus::SetBatchNum(int budget)
{
    batch_budget_ = budget;
}

int NotifyBus::GetBatchNum(void)
{
    return batch_budget_;
}

bool NotifyBus::GetStats(int event, EventStats& stats)
{
    EventNode* node = GetNode(event);

    if(node == nullptr)
        return false;

    stats.event = node->event;
    stats.name = node->name;
    stats.book_count = node->handle_list.size();
    stats.pending_count = GetPendingCount(node);
    stats.send_count = node->send_count;
    stats.sync_send_count = node->sync_send_count;
    stats.process_count = node->process_count;
    stats.drop_count = node->drop_count;
    stats.send_fail_count = node->send_fail_count;

    PutNode(node);

    return true;
}

void NotifyBus::LockNameMap()
{
    name_map_mutex.lock();
}

void NotifyBus::UnLockNameMap()
{
    name_map_mutex.unlock();
}

bool NotifyBus::GetNode(NotifyBus::EventNode* node)
{
    node->event_mutex.lock();
    /* if it is a valid event node */

    if(node->event >= 0)
        return true;

    node->event_mutex.unlock();

    return false;
}

NotifyBus::EventNode* NotifyBus::GetNode(int event)
{
    EventNode* node = &event_array_[event];

    if(GetNode(node))
        return node;

    return nullptr;
}

void NotifyBus::PutNode(EventNode* node)
{
    node->event_mutex.unlock();
}

}    // namespace TEngine
