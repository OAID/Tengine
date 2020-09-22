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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __EXEC_SCHEDULER_H__
#define __EXEC_SCHEDULER_H__

struct ir_graph;

struct exec_scheduler
{
    char* name;

    int (*prerun)(struct exec_scheduler*, struct ir_graph*, int num_thread, int cpu_affinity);
    int (*run)(struct exec_scheduler*, struct ir_graph*, int block);
    int (*wait)(struct exec_scheduler*, struct ir_graph*);
    int (*postrun)(struct exec_scheduler*, struct ir_graph*);
    void (*release)(struct exec_scheduler*);
};

#endif
