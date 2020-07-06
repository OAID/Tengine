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

#include <stdio.h>

#include "vector.h"
#include "tengine_errno.h"
#include "nn_device.h"
#include "tengine_c_api.h"
#include "tengine_log.h"

static struct vector* dev_list = NULL;
static struct nn_device* def_dev = NULL;

int init_nn_dev_registry(void)
{
    dev_list = create_vector(sizeof(struct nn_device*), NULL);

    if (dev_list == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    return 0;
}

void release_nn_dev_registry(void)
{
    int dev_num = get_vector_num(dev_list);

    for (int i = 0; i < dev_num; i++)
    {
        struct nn_device* dev = *( struct nn_device** )get_vector_data(dev_list, i);

        if (dev->release)
            dev->release(dev);
    }

    release_vector(dev_list);
}

struct nn_device* get_nn_device_by_name(const char* name)
{
    int dev_num = get_vector_num(dev_list);

    for (int i = 0; i < dev_num; i++)
    {
        struct nn_device* dev = *( struct nn_device** )get_vector_data(dev_list, i);

        if (!strcmp(dev->name, name))
            return dev;
    }

    return NULL;
}

struct nn_device* get_nn_device(int idx)
{
    if (idx < 0 || idx >= get_vector_num(dev_list))
    {
        set_tengine_errno(EINVAL);
        return NULL;
    }

    return *( struct nn_device** )get_vector_data(dev_list, idx);
}

int get_nn_device_number(void)
{
    return get_vector_num(dev_list);
}

struct nn_device* get_default_nn_device(void)
{
    if (def_dev)
        return def_dev;

    /* try to use cpu_device as default device */

    def_dev = get_nn_device_by_name("cpu_dev");

    return def_dev;
}

int set_default_device(const char* device)
{
    struct nn_device* dev = get_nn_device_by_name(device);

    if (dev)
    {
        def_dev = dev;
        return 0;
    }

    TLOG_ERR("no nn device's name is %s\n", device);

    set_tengine_errno(ENOENT);
    return -1;
}

const char* get_default_device(void)
{
    struct nn_device* dev = get_default_nn_device();
    return dev->name;
}

int register_nn_device(struct nn_device* dev)
{
    if (get_nn_device_by_name(dev->name) != NULL)
    {
        TLOG_ERR("dev %s name duplicated\n", dev->name);
        set_tengine_errno(EEXIST);
        return -1;
    }

    if (dev->init && dev->init(dev) < 0)
    {
        TLOG_ERR("dev %s initialize failed\n", dev->name);
        return -1;
    }

    push_vector_data(dev_list, &dev);

    return 0;
}

void release_nn_dev_exec_graph(struct nn_device* dev, void* exec_graph)
{
    if (dev->release_exec_graph == NULL || exec_graph == NULL)
        return;

    dev->release_exec_graph(dev, exec_graph);
}
