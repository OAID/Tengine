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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */

#ifndef __Caffe_DRIVER_HPP__
#define __Caffe_DRIVER_HPP__

#include <atomic>
#include <condition_variable>
#include <queue>
#include <thread>

#include "caffe_executor.hpp"
#include "device_driver.hpp"
#include "node_dev_driver.hpp"

namespace TEngine {

using CaffeNodeDevice = NodeDevice;

class CaffeNodeDriver : public NodeDriver {
 public:
  CaffeNodeDriver();
  ~CaffeNodeDriver();

  bool Prerun(Device* dev, void* node_handle, Node* node);
  bool Run(Device* dev, void* node_handle, Node* node);
  bool SyncRun(Device* dev, void* node_handle, Node* node);
  bool Postrun(Device* dev, void* node_handle, Node* node);

  bool ProbeDevice(const dev_id_t& dev_id) override;
  bool DestroyDevice(Device* device) override;

  bool InitDev(NodeDevice* device) override;
};

}  // namespace TEngine

#endif
