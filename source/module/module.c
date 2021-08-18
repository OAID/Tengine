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
 * Author: lswang@openailab.com
 */

#include "module/module.h"

#include "serializer/serializer.h"
#include "operator/op.h"
#include "graph/node.h"
#include "device/device.h"

#include "utility/vector.h"
#include "utility/log.h"

#include <stddef.h>
#include <string.h>

static vector_t* internal_serializer_registry = NULL; //!< registry of model serializer
static vector_t* internal_device_registry = NULL;     //!< registry of runnable neural network device
static vector_t* internal_op_method_registry = NULL;  //!< registry of operators
static vector_t* internal_op_name_registry = NULL;    //!< registry of operators name

/*!
 * @struct ir_op_map_t
 * @brief  Operator type and name map
 */
typedef struct op_name_entry
{
    int type;         //!< the type of a operator
    const char* name; //!< the name of a operator
} ir_op_name_entry_t;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int initialize_serializer_registry(const char* name)
{
    if (NULL == internal_serializer_registry)
    {
        internal_serializer_registry = create_vector(sizeof(serializer_t*), NULL);
        if (NULL == internal_serializer_registry)
        {
            TLOG_CRIT("Tengine: Can not init serializer %s, create the vector failed.\n", name);
            return -1;
        }
    }

    return 0;
}

int register_serializer(serializer_t* serializer)
{
    initialize_serializer_registry(serializer->get_name(serializer));
    if (NULL == internal_serializer_registry)
    {
        TLOG_CRIT("Tengine: Can not register %s, module was not be inited.\n", serializer->get_name(serializer));
        return -1;
    }

    if (NULL != serializer->init)
    {
        int ret = serializer->init(serializer);
        if (0 != ret)
        {
            TLOG_CRIT("Tengine: Can not register %s, module init was failed(%d).\n", serializer->get_name(serializer), ret);
            return -1;
        }
    }

    int ret = push_vector_data(internal_serializer_registry, &serializer);
    if (0 != ret)
    {
        TLOG_CRIT("Tengine: Can not register %s, module cannot not be inserted.\n", serializer->get_name(serializer));
        return ret;
    }

    return 0;
}

serializer_t* find_serializer_via_name(const char* name)
{
    if (NULL == internal_serializer_registry)
    {
        TLOG_CRIT("Tengine: Can not find any serializer, module was not inited.\n");
        return NULL;
    }

    int count = get_vector_num(internal_serializer_registry);
    if (0 == count)
    {
        TLOG_CRIT("Tengine: Can not find any serializer, module was empty.\n");
        return NULL;
    }

    for (int i = 0; i < count; i++)
    {
        serializer_t* serializer = *(serializer_t**)get_vector_data(internal_serializer_registry, i);

        if (0 == strcmp(serializer->get_name(serializer), name))
        {
            return serializer;
        }
    }

    TLOG_CRIT("Tengine: Can not find serializer %s, module was empty.\n", name);
    return NULL;
}

serializer_t* find_serializer_via_index(int index)
{
    int count = get_serializer_count();

    if (0 <= index && index < count)
    {
        serializer_t* serializer = *(serializer_t**)get_vector_data(internal_serializer_registry, index);
        return serializer;
    }
    else
    {
        return NULL;
    }
}

int get_serializer_count()
{
    if (NULL == internal_serializer_registry)
    {
        return 0;
    }
    else
    {
        return get_vector_num(internal_serializer_registry);
    }
}

int unregister_serializer(serializer_t* serializer)
{
    if (NULL == serializer)
    {
        TLOG_CRIT("Tengine: Can not unregister serializer, pointer is null.\n");
        return -1;
    }

    serializer_t* result = find_serializer_via_name(serializer->get_name(serializer));
    if (NULL == result)
    {
        const char* name = serializer->get_name(serializer);
        TLOG_CRIT("Tengine: Can not find serializer %s, unregister failed.\n", name);
        return -1;
    }
    if (result != serializer)
    {
        const char* name = serializer->get_name(serializer);
        TLOG_CRIT("Tengine: Can not find serializer %s, pointer mismatched(%p vs. %p), .\n", name, serializer, result);
        return -1;
    }

    int ret = serializer->release(serializer);
    if (0 != ret)
    {
        const char* name = serializer->get_name(serializer);
        TLOG_CRIT("Tengine: Unregister serializer %s failed(%d).\n", name, ret);
        return -1;
    }

    return remove_vector_via_pointer(internal_serializer_registry, &serializer);
}

int release_serializer_registry()
{
    while (get_vector_num(internal_serializer_registry) > 0)
    {
        serializer_t* serializer = (serializer_t*)get_vector_data(internal_serializer_registry, 0);
        unregister_serializer(serializer);
    }

    release_vector(internal_serializer_registry);
    internal_serializer_registry = NULL;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int initialize_device_registry(const char* name)
{
    if (NULL == internal_device_registry)
    {
        internal_device_registry = create_vector(sizeof(ir_device_t*), NULL);
        if (NULL == internal_device_registry)
        {
            TLOG_CRIT("Tengine: Can not init device %s, create the vector failed.\n", name);
            return -1;
        }
    }

    return 0;
}

ir_device_t* find_device_via_name(const char* name)
{
    if (NULL == internal_device_registry)
    {
        TLOG_CRIT("Tengine: Can not find any device, module was not inited.\n");
        return NULL;
    }

    int count = get_vector_num(internal_device_registry);
    if (0 == count)
    {
        TLOG_CRIT("Tengine: Can not find any device, module was empty.\n");
        return NULL;
    }

    for (int i = 0; i < count; i++)
    {
        ir_device_t* device = *(ir_device_t**)get_vector_data(internal_device_registry, i);

        if (0 == strcmp(device->name, name))
        {
            return device;
        }
    }

    TLOG_CRIT("Tengine: Can not find device %s, module was empty.\n", name);
    return NULL;
}

struct device* find_default_device()
{
    return find_device_via_name("CPU");
}

ir_device_t* find_device_via_index(int index)
{
    int count = get_device_count();

    if (0 <= index && index < count)
    {
        ir_device_t* device = (ir_device_t*)get_vector_data(internal_device_registry, index);
        return device;
    }
    else
    {
        return NULL;
    }
}

int get_device_count()
{
    if (NULL == internal_device_registry)
    {
        return 0;
    }
    else
    {
        return get_vector_num(internal_device_registry);
    }
}

int register_device(ir_device_t* device)
{
    initialize_device_registry(device->name);
    if (NULL == internal_device_registry)
    {
        TLOG_CRIT("Tengine: Can not register %s, module was not be inited.\n", device->name);
        return -1;
    }

    if (NULL != device->interface && NULL != device->interface->init)
    {
        int ret = device->interface->init(device);
        if (0 != ret)
        {
            TLOG_CRIT("Tengine: Can not register %s, module init was failed(%d).\n", device->name, ret);
            return -1;
        }
    }

    int ret = push_vector_data(internal_device_registry, &device);
    if (0 != ret)
    {
        TLOG_CRIT("Tengine: Can not register %s, module cannot not be inserted.\n", device->name);
        return ret;
    }

    return 0;
}

int unregister_device(ir_device_t* device)
{
    if (NULL == find_device_via_name(device->name))
    {
        return -1;
    }

    if (NULL != device->interface && NULL != device->interface->release_device)
    {
        device->interface->release_device(device);
    }

    return remove_vector_via_pointer(internal_device_registry, &device);
}

int release_device_registry()
{
    while (get_vector_num(internal_device_registry) > 0)
    {
        ir_device_t* device = (ir_device_t*)get_vector_data(internal_device_registry, 0);
        unregister_device(device);
    }

    release_vector(internal_device_registry);
    internal_device_registry = NULL;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int initialize_op_name_registry(const char* name)
{
    if (NULL == internal_op_name_registry)
    {
        internal_op_name_registry = create_vector(sizeof(ir_op_name_entry_t), NULL);
    }

    if (NULL == internal_op_name_registry)
    {
        TLOG_CRIT("Tengine: Can not init op name map for %s, create the vector failed.\n", name);
        return -1;
    }

    return 0;
}

int register_op_name(int type, const char* name)
{
    initialize_op_name_registry(name);
    if (NULL == internal_op_name_registry)
    {
        TLOG_CRIT("Tengine: Can not register op name %s, module was not be inited.\n", name);
        return -1;
    }

    ir_op_name_entry_t op_map;

    op_map.type = type;
    op_map.name = name;

    return push_vector_data(internal_op_name_registry, &op_map);
}

int unregister_op_name(int type)
{
    int i;
    const int op_name_count = get_vector_num(internal_op_name_registry);
    for (i = 0; i < op_name_count; i++)
    {
        const ir_op_name_entry_t* op_map_entry = (ir_op_name_entry_t*)get_vector_data(internal_op_name_registry, i);
        if (op_map_entry->type == type)
        {
            break;
        }
    }

    if (op_name_count == i)
    {
        return -1;
    }

    remove_vector_via_index(internal_op_name_registry, i);
    return 0;
}

int release_op_name_registry()
{
    while (get_vector_num(internal_op_name_registry) > 0)
    {
        ir_op_name_entry_t* op_map_entry = (ir_op_name_entry_t*)get_vector_data(internal_op_name_registry, 0);
        unregister_op_name(op_map_entry->type);
    }

    release_vector(internal_op_name_registry);
    internal_op_name_registry = NULL;

    return 0;
}

static int initialize_op_registry(const char* name)
{
    if (NULL == internal_op_method_registry)
    {
        internal_op_method_registry = create_vector(sizeof(ir_method_t), NULL);
        if (NULL == internal_op_method_registry)
        {
            TLOG_CRIT("Tengine: Can not init op %s, create the vector failed.\n", name);
            return -1;
        }
    }

    return 0;
}

static int register_op_registry(ir_method_t* method)
{
    if (find_op_method(method->type, method->version))
    {
        return -1;
    }

    return push_vector_data(internal_op_method_registry, method);
}

int register_op(int type, const char* name, ir_method_t* method)
{
    initialize_op_registry(name);
    if (NULL == internal_op_method_registry)
    {
        TLOG_CRIT("Tengine: Can not register op %s, module was not be inited.\n", name);
        return -1;
    }

    if (NULL != name)
    {
        if (0 != register_op_name(type, name))
        {
            TLOG_CRIT("Tengine: Can not register op name %s.\n", name);
            return -1;
        }
    }

    if (NULL != method)
    {
        method->type = type;
        return register_op_registry(method);
    }

    return 0;
}

ir_method_t* find_op_method(int type, int version)
{
    int op_count = get_vector_num(internal_op_method_registry);
    for (int i = 0; i < op_count; i++)
    {
        ir_method_t* op = (ir_method_t*)get_vector_data(internal_op_method_registry, i);

        /* TODO, check the version */
        if (op->type == type)
        {
            return op;
        }
    }

    return NULL;
}

ir_method_t* find_op_method_via_index(int index)
{
    int count = get_op_method_count();

    if (0 <= index && index < count)
    {
        ir_method_t* method = (ir_method_t*)get_vector_data(internal_op_method_registry, index);
        return method;
    }
    else
    {
        return NULL;
    }
}

const char* find_op_name(int type)
{
    int count = get_vector_num(internal_op_name_registry);
    for (int i = 0; i < count; i++)
    {
        const ir_op_name_entry_t* op_name = (const ir_op_name_entry_t*)get_vector_data(internal_op_name_registry, i);
        if (op_name->type == type)
        {
            return op_name->name;
        }
    }

    return NULL;
}

int get_op_method_count()
{
    if (NULL == internal_op_method_registry)
    {
        return 0;
    }
    else
    {
        return get_vector_num(internal_op_method_registry);
    }
}

int unregister_op(int type, int version)
{
    int matched_count = 0;

    for (int i = 0; i < get_vector_num(internal_op_method_registry); i++)
    {
        const ir_method_t* method = (ir_method_t*)get_vector_data(internal_op_method_registry, i);
        if (method->type == type)
        {
            matched_count++;
        }
    }

    for (int i = 0; i < matched_count; i++)
    {
        for (int j = 0; j < get_vector_num(internal_op_method_registry); j++)
        {
            const ir_method_t* method = (ir_method_t*)get_vector_data(internal_op_method_registry, j);
            if (method->type == type)
            {
                if (0 == version || (0 < version && method->version == version))
                {
                    remove_vector_via_index(internal_op_method_registry, j);
                }

                if (0 < version)
                {
                    return 0;
                }
            }
        }
    }

    unregister_op_name(type);

    return 0;
}

int release_op_registry(void)
{
    while (get_vector_num(internal_op_method_registry) > 0)
    {
        ir_method_t* method = (ir_method_t*)get_vector_data(internal_op_method_registry, 0);
        unregister_op(method->type, method->version);
    }

    release_vector(internal_op_method_registry);
    internal_op_method_registry = NULL;

    return 0;
}
