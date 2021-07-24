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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#pragma once

struct node_ops;
struct exec_graph;

int init_cpu_node_ops_registry(void);
void release_cpu_node_ops_registry(void);

int register_builtin_node_ops(int op_type, struct node_ops* node_ops);
int unregister_builtin_node_ops(int op_type, struct node_ops* node_ops);

int register_custom_node_ops(int op_type, struct node_ops* node_ops);
int unregister_custom_node_ops(int op_type, struct node_ops* node_ops);

struct node_ops* find_node_ops(struct exec_graph* exec_graph, struct node* ir_node);
