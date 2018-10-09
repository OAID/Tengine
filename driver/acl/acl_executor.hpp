
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

#ifndef __ACL_EXECUTOR_HPP__
#define __ACL_EXECUTOR_HPP__

#include "node_dev_executor.hpp"

namespace TEngine {

class ACLExecutor : public NodeExecutor {
 public:
  /* by disable NonblockRun(),
        driver->SyncRun() will be called instead of driver->Run()
  */

  ACLExecutor(const dev_id_t& dev_id) : NodeExecutor(dev_id) {
    DisableNonblockRun();
  }
};

}  // namespace TEngine

#endif
