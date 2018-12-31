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
#ifndef __DEV_PROPOSAL_HPP__
#define __DEV_PROPOSAL_HPP__

#include <string>

namespace TEngine {

#define DEV_PROPOSAL_ATTR "dev_proposal"

#define DEV_PROPOSAL_UNSPPORT 0
#define DEV_PROPOSAL_CAN_DO 1
#define DEV_PROPOSAL_GOODAT 2
#define DEV_PROPOSAL_BEST 3
#define DEV_PROPOSAL_STATIC 4

struct DevProposal
{
    std::string dev_id;
    int level;

    DevProposal() : level(DEV_PROPOSAL_UNSPPORT){};
};

}    // namespace TEngine

#endif
