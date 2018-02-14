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
#ifndef __NOTIFY_HPP__
#define __NOTIFY_HPP__

#include <string>
#include <vector>
#include <iostream>
#include <functional>

#include "any.hpp"

namespace TEngine {

struct EventBusInterface {

	struct EventData {
	   int event;
           int data;
           any args; 
//           ~EventData(){  std::cout<<"RELEASE: data is "<<data<<"\n"; }

	};


	using event_handle_t=std::function<bool(int,EventData *, any *)>;

        struct EventHandle {
	    event_handle_t func;
            any argument;
        };

	
	struct EventStats {
	   int event;
	   int book_count;
	   std::string name;
	   uint32_t pending_count;
	   uint64_t send_count;
	   uint64_t sync_send_count;
	   uint64_t drop_count;
	   uint64_t process_count;
	   uint64_t send_fail_count;

           void Dump(std::ostream& os)
           {
              os<<"event: "<<event<<" name: "<<name<<" booked handler: "<<book_count<<std::endl;
              os<<"\tsend "<<send_count<<" sync_send "<<sync_send_count<<" send fail "<<send_fail_count<<"\n";
              os<<"\tpending "<<pending_count<<" process "<<process_count<<" drop "<<drop_count<<"\n";

           }
	};
	 



	//Create/Register a new event on the bus, and return the event id
	virtual int  CreateEvent(const std::string& name)=0;
	virtual bool RemoveEvent(int event)=0;
	virtual bool RemoveEvent(const std::string& name)=0;

	virtual std::string EventName(int event)=0;
	virtual int  EventNum(const std::string& name)=0;

	virtual std::string BusName()=0;

	//Drop target pending events for this type of event, return dropped number
	virtual int  DropPendingEvent(int event) { return 0;} 

	//Drop all pending events, return dropped number
	virtual int  DropPendingEvent()          { return 0;}  

	virtual bool Send(const std::string& event_name, EventData * data, int priority=10)=0;
	virtual bool SyncSend(const std::string& event_name,EventData * data)=0;
	virtual bool Book(const std::string& event_name, const EventHandle& handle)=0;

	virtual bool Send(int event, EventData * data, int priority=10)=0;
	virtual bool SyncSend(int event,EventData * data)=0;
	virtual bool Book(int event,  const EventHandle& handle)=0;

	virtual int ProcessAllEvents(void)=0;
        virtual void SetBatchNum(int budget)=0;
        virtual int  GetBatchNum(void)=0;

	virtual bool GetStats(int event, EventStats& stats)=0;

	virtual ~EventBusInterface(){};

};


	using EventData=EventBusInterface::EventData;
	using EventStats=EventBusInterface::EventStats;
        using EventHandle=EventBusInterface::EventHandle;
} //namespace TEngine


#endif
