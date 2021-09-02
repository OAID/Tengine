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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct serializer;
struct op;
struct method;
struct device;

/*!
 * @brief Register a serializer.
 *
 * @param [in]  serializer: The pointer to a struct of serializer.
 *
 * @return statue value, 0 success, other value failure.
 */
int register_serializer(struct serializer* serializer);

/*!
 * @brief Find the serializer via its name.
 *
 * @param [in]  name: The name of serializer.
 *
 * @return  The pointer of the serializer.
 */
struct serializer* find_serializer_via_name(const char* name);

/*!
 * @brief Find the serializer via its registered index.
 *
 * @param [in]  index: The index of serializer.
 *
 * @return  The pointer of the serializer.
 */
struct serializer* find_serializer_via_index(int index);

/*!
 * @brief Get count of all registered serializer.
 *
 * @return  The count of registered serializer.
 */
int get_serializer_count();

/*!
 * @brief Unregister a serializer.
 *
 * @param [in]  serializer: The pointer to a struct of serializer.
 *
 * @return statue value, 0 success, other value failure.
 */
int unregister_serializer(struct serializer* serializer);

/*!
 * @brief Release all serializer.
 *
 * @return statue value, 0 success, other value failure.
 */
int release_serializer_registry();

/*!
 * @brief Register a device.
 *
 * @param [in]  device: The pointer to a struct of device.
 *
 * @return statue value, 0 success, other value failure.
 */
int register_device(struct device* device);

/*!
 * @brief Find the device via its name.
 *
 * @param [in]  name: The name of device.
 *
 * @return  The pointer of the device.
 */
struct device* find_device_via_name(const char* name);

/*!
 * @brief Find the default device.
 *
 * @return  The pointer of the device.
 */
struct device* find_default_device();

/*!
 * @brief Find the device via its registered index.
 *
 * @param [in]  name: The index of device.
 *
 * @return  The pointer of the device.
 */
struct device* find_device_via_index(int index);

/*!
 * @brief Get count of all registered device.
 *
 * @return  The count of registered device.
 */
int get_device_count();

/*!
 * @brief Register a device.
 *
 * @param [in]  device: The pointer to a struct of device.
 *
 * @return statue value, 0 success, other value failure.
 */
int unregister_device(struct device* device);

/*!
 * @brief Release all device.
 *
 * @return statue value, 0 success, other value failure.
 */
int release_device_registry();

/*!
 * @brief Register an operator method.
 *
 * @param [in]  type: The type of an operator.
 * @param [in]  type: The name of an operator.
 * @param [in]  type: The method of an operator.
 *
 * @return statue value, 0 success, other value failure.
 */
int register_op(int type, const char* name, struct method* method);

/*!
 * @brief Find an operator method.
 *
 * @param [in]  type: The type of an operator method.
 * @param [in]  version: The version of an operator method.
 *
 * @return  The pointer of the method.
 */
struct method* find_op_method(int type, int version);

/*!
 * @brief Find an operator method via its registered index.
 *
 * @param [in]  index: The index of operator method.
 *
 * @return  The pointer of the operator method.
 */
struct method* find_op_method_via_index(int index);

/*!
 * @brief Find an operator name.
 *
 * @param [in]  type: The type of an operator method.
 *
 * @return  The char array of the method.
 */
const char* find_op_name(int type);

/*!
 * @brief Get count of all registered operator method.
 *
 * @return  The count of registered operator method.
 */
int get_op_method_count();

/*!
 * @brief Register an operator.
 *
 * @param [in]  type: The type of an operator method.
 * @param [in]  version: The version of an operator method.
 *
 * @return statue value, 0 success, other value failure.
 */
int unregister_op(int type, int version);

/*!
 * @brief Release all operator.
 *
 * @return statue value, 0 success, other value failure.
 */
int release_op_registry();

#ifdef __cplusplus
}
#endif /* __cplusplus */
