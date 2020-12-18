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

#ifndef __TENGINE_UTILS_H__
#define __TENGINE_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

const char* tensor_type_string(int tensor_type);
const char* layout_string(int dayout);
const char* model_format_string(int model_format);

int init_op_name_map(void);
void release_op_name_map(void);

int register_op_map(int op_type, const char* name);
int unregister_op_map(int op_type);
int get_op_type(const char* name);
const char* get_op_name(int op_type);

int data_type_size(int data_type);
const char* data_type_string(int data_type);
const char* data_type_typeinfo_name(int data_type);

int param_entry_type_mapping(const char* type_name);

void dump_float(const char* fname, float* data, int number);

#ifdef __cplusplus
}
#endif

#endif
