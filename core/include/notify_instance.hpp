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
#ifndef __NOTIFY_INSTANCE_HPP__
#define __NOTIFY_INSTANCE_HPP__

#include <map>
#include <unordered_map>
#include <mutex>
#include <queue>

#include "notify.hpp"

namespace TEngine {

class NotifyBus: public EventBusInterface {

public:

	using name_map_t=std::unordered_map<std::string, int>;
	using event_tree_t=std::map<int, std::queue<EventData> >;

	struct EventNode 
	{
		int event;
                bool active; //in active queue or not
		uint64_t drop_count;
		uint64_t send_count;
		uint64_t process_count;
		uint64_t sync_send_count;
		uint64_t send_fail_count;
             

		std::string name;
		event_tree_t event_tree; //priority and event data
		std::vector<EventHandle> handle_list;
		std::recursive_mutex event_mutex;
	};



	int  CreateEvent(const std::string& name);
	bool RemoveEvent(int event);
	bool RemoveEvent(const std::string& name);

	std::string EventName(int event);
	int  EventNum(const std::string& name);

	std::string BusName(void);

	int  DropPendingEvent(int event); 
	int  DropPendingEvent();

	bool Send(int event, EventData * data, int priority=10);
	bool Send(const std::string& event_name, EventData * data, int priority=10);

	bool SyncSend(int event,EventData *data);
	bool SyncSend(const std::string& event_name,EventData *data );

	bool Book(int event,  const EventHandle& handle);
        bool Book(const std::string& event_name, const EventHandle& handle);

	int ProcessAllEvents(void);
        void SetBatchNum(int budget);
        int  GetBatchNum(void);

	bool GetStats(int event, EventStats& stats);

	~NotifyBus();

	NotifyBus(const std::string& name, int max_event_type_num=1024, int max_event_pending_num=128);


protected:
	void HandleSingleEvent(EventNode * node, EventData * data);
	int HandlePendingEvent(EventNode * node);
        int GetPendingCount(EventNode * node);


	void LockNameMap();
	void UnLockNameMap();

	EventNode *  GetNode(int event);
        bool         GetNode(EventNode * node);
	void PutNode(EventNode *);

	name_map_t name_map_;
	std::string bus_name_;
	EventNode  * event_array_;

	std::recursive_mutex name_map_mutex;

        std::mutex              pending_queue_mutex;
        std::queue<EventNode *> pending_queue;
        std::queue<EventNode *> active_queue;

private:
        int RealHandlePendingEvent(EventNode * node);
	int RealDropPendingEvent(EventNode * node);

	int max_event_type_;
	int max_pending_num_;
        int batch_budget_;
        int batch_count_;

};

} //namespace TEngine


#endif
