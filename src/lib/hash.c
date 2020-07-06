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

#include "hash.h"

extern struct hash* create_hash_impl(void);

struct hash* create_hash(int bucket_size, hash_key_t hash_func, int cpy_key, free_data_t free_func, int mt_safe)
{
    struct hash* h = create_hash_impl();

    h->init(h, bucket_size, hash_func);
    h->config(h, cpy_key, free_func, mt_safe, -1);

    return h;
}

void destroy_hash(struct hash* h)
{
    h->release(h);
}
