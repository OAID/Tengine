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
#define NON_INSTANCE_DEFINE
#include "event.hpp"
#undef NON_INSTANCE_DEFINE

namespace TEngine{
//for data_stream.hpp ==>
    //event for DataStream
    //   CallBack_DataStream_t & topic_DataStream
    DEFINE_EVENT_TOPIC(DataStream);
//for data_stream.hpp <==
        
//for workshop_hal.hpp ==>
    //处理Events (topic: "WorkshopXxx")
    // 1. start -- 开始执行
    //   CallBack_WorkshopStart_t & topic_WorkshopStart
    DEFINE_EVENT_TOPIC(WorkshopStart);

    // 2. stop  -- 停止执行
    //   CallBack_WorkshopStop_t & topic_WorkshopStop
    DEFINE_EVENT_TOPIC(WorkshopStop);

    // 3. pause -- 暂停执行
    //   CallBack_WorkshopPause_t & topic_WorkshopPause
    DEFINE_EVENT_TOPIC(WorkshopPause);

    // 4. in_out_ready -- wake up from the ready events of input/output 
    //   CallBack_WorkshopIoReady_t & topic_WorkshopIoReady
    DEFINE_EVENT_TOPIC(WorkshopIoReady);

    // 5. watch_dog_timer
    //   CallBack_WorkshopWDTimer_t & topic_WorkshopWDTimer
    DEFINE_EVENT_TOPIC(WorkshopWDTimer);
        
//for workshop_hal.hpp <==
}