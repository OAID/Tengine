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
 * Revised: lswang@openailab.com
 */

#pragma once

/*!
 * @brief  Solve min value
 *
 * @param [in]  a: number a
 * @param [in]  b: number b
 *
 * @return  The solved min value
 */
int imin(int a, int b);

/*!
 * @brief  Solve max value
 *
 * @param [in]  a: number a
 * @param [in]  b: number b
 *
 * @return  The solved max value
 */
int imax(int a, int b);

/*!
 * @brief  Solve min absolute value
 *
 * @param [in]  a: number a
 * @param [in]  b: number b
 *
 * @return  The solved min absolute value
 */
int min_abs(int a, int b);

/*!
 * @brief  Solve max absolute value
 *
 * @param [in]  a: number a
 * @param [in]  b: number b
 *
 * @return  The solved max absolute value
 */
int max_abs(int a, int b);

/*!
 * @brief  Solve greatest common divisor
 *
 * @param [in]  a: number a
 * @param [in]  b: number b
 *
 * @return  The solved GCD
 */
int gcd(int a, int b);

/*!
 * @brief  Solve lowest common multiple
 *
 * @param [in]  a: number a
 * @param [in]  b: number b
 *
 * @return  The solved LCM
 */
int lcm(int a, int b);

/*!
 * @brief  Solve min aligned value with the step length
 *
 * @param [in]  value: number which may not be aligned
 * @param [in]  step: align step
 *
 * @return  The solved aligned value
 */
int align(int value, int step);

/*!
 * @brief  Get aligned pointer
 *
 * @param [in]  value: pointer which may not be aligned
 * @param [in]  step: align step
 *
 * @return  The solved aligned pointer
 */
void* align_address(void* address, int step);
