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

#ifndef __VA_ARG_UTIL_H__
#define __VA_ARG_UTIL_H__

/*support at most 20 variable var args */

/* get arg count */
#define COUNT_VA_ARG2(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, n, \
                      ...)                                                                                         \
    n

#define COUNT_VA_ARG(...) \
    COUNT_VA_ARG2(__VA_ARGS__, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define EVAL_FUNC(a) a

/* get single arg */
#define GET_VA_ARG_EX(a0) a0
#define GET_VA_ARG_0(a0) GET_VA_ARG_EX(a0)
#define GET_VA_ARG_1(a0, ...) GET_VA_ARG_0(__VA_ARGS__)
#define GET_VA_ARG_2(a0, ...) GET_VA_ARG_2(__VA_ARGS__)
#define GET_VA_ARG_3(a0, ...) GET_VA_ARG_2(__VA_ARGS__)
#define GET_VA_ARG_4(a0, ...) GET_VA_ARG_3(__VA_ARGS__)
#define GET_VA_ARG_5(a0, ...) GET_VA_ARG_4(__VA_ARGS__)
#define GET_VA_ARG_6(a0, ...) GET_VA_ARG_5(__VA_ARGS__)
#define GET_VA_ARG_7(a0, ...) GET_VA_ARG_6(__VA_ARGS__)
#define GET_VA_ARG_8(a0, ...) GET_VA_ARG_7(__VA_ARGS__)
#define GET_VA_ARG_9(a0, ...) GET_VA_ARG_8(__VA_ARGS__)
#define GET_VA_ARG_10(a0, ...) GET_VA_ARG_9(__VA_ARGS__)
#define GET_VA_ARG_11(a0, ...) GET_VA_ARG_10(__VA_ARGS__)
#define GET_VA_ARG_12(a0, ...) GET_VA_ARG_11(__VA_ARGS__)
#define GET_VA_ARG_13(a0, ...) GET_VA_ARG_12(__VA_ARGS__)
#define GET_VA_ARG_14(a0, ...) GET_VA_ARG_13(__VA_ARGS__)
#define GET_VA_ARG_15(a0, ...) GET_VA_ARG_14(__VA_ARGS__)
#define GET_VA_ARG_16(a0, ...) GET_VA_ARG_15(__VA_ARGS__)
#define GET_VA_ARG_17(a0, ...) GET_VA_ARG_16(__VA_ARGS__)
#define GET_VA_ARG_18(a0, ...) GET_VA_ARG_17(__VA_ARGS__)
#define GET_VA_ARG_19(a0, ...) GET_VA_ARG_18(__VA_ARGS__)

#define WALK_ARG_WITH_0_0(func, a0) EVAL_FUNC(func(a0))
#define WALK_ARG_WITH_0_1(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_0(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_2(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_1(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_3(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_2(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_4(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_3(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_5(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_4(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_6(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_5(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_7(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_6(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_8(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_7(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_9(func, a0, ...) \
    EVAL_FUNC(func(a0));                 \
    WALK_ARG_WITH_0_8(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_10(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_9(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_11(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_10(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_12(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_11(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_13(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_12(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_14(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_13(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_15(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_14(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_16(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_15(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_17(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_16(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_18(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_17(func, __VA_ARGS__)
#define WALK_ARG_WITH_0_19(func, a0, ...) \
    EVAL_FUNC(func(a0));                  \
    WALK_ARG_WITH_0_18(func, __VA_ARGS__)

#define WALK_ARG_WITH_0_EX(n, func, ...) WALK_ARG_WITH_0_##n(func, __VA_ARGS__)

#define WALK_ARG_WITH_0(func, ...) WALK_ARG_WITH_0_EX(COUNT_VA_ARG(__VA_ARGS__), func, __VA_ARGS__)

/* 1 fixed */

#define WALK_ARG_WITH_1_0(func, f0, a0, ...) EVAL_FUNC(func(f0, a0));
#define WALK_ARG_WITH_1_1(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_0(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_2(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_1(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_3(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_2(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_4(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_3(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_5(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_4(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_6(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_5(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_7(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_6(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_8(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_7(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_9(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                 \
    WALK_ARG_WITH_1_8(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_10(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_9(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_11(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_10(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_12(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_11(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_13(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_12(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_14(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_13(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_15(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_14(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_16(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_15(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_17(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_16(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_18(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_17(func, f0, __VA_ARGS__)
#define WALK_ARG_WITH_1_19(func, f0, a0, ...) \
    EVAL_FUNC(func(f0, a0));                  \
    WALK_ARG_WITH_1_18(func, f0, __VA_ARGS__)

#define WALK_ARG_WITH_1_EX(n, f0, func, ...) WALK_ARG_WITH_1_##n(func, f0, __VA_ARGS__)

#define WALK_ARG_WITH_1(func, f0, ...) WALK_ARG_WITH_1_EX(COUNT_VA_ARG(__VA_ARGS__), func, f0, __VA_ARGS__)

/* 2 fixed */

#define WALK_ARG_WITH_2_0(func, f0, f1, a0, ...) EVAL_FUNC(func(f0, f1, a0));
#define WALK_ARG_WITH_2_1(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_0(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_2(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_1(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_3(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_2(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_4(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_3(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_5(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_4(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_6(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_5(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_7(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_6(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_8(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_7(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_9(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                 \
    WALK_ARG_WITH_2_8(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_10(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_9(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_11(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_10(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_12(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_11(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_13(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_12(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_14(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_13(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_15(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_14(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_16(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_15(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_17(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_16(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_18(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_17(func, f0, f1, __VA_ARGS__)
#define WALK_ARG_WITH_2_19(func, f0, f1, a0, ...) \
    EVAL_FUNC(func(f0, f1, a0));                  \
    WALK_ARG_WITH_2_18(func, f0, f1, __VA_ARGS__)

#define WALK_ARG_WITH_2_EX(n, f0, f1, func, ...) WALK_ARG_WITH_2_##n(func, f0, f1, __VA_ARGS__)

#define WALK_ARG_WITH_2(func, f0, f1, ...) WALK_ARG_WITH_2_EX(COUNT_VA_ARG(__VA_ARGS__), func, f0, f1, __VA_ARGS__)

/* 3 fixed */
#define WALK_ARG_WITH_3_0(func, f0, f1, f2, a0, ...) EVAL_FUNC(func(f0, f1, f2, a0));
#define WALK_ARG_WITH_3_1(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_0(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_2(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_1(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_3(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_2(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_4(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_3(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_5(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_4(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_6(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_5(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_7(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_6(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_8(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_7(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_9(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                 \
    WALK_ARG_WITH_3_8(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_10(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_9(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_11(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_10(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_12(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_11(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_13(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_12(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_14(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_13(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_15(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_14(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_16(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_15(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_17(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_16(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_18(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_17(func, f0, f1, f2, __VA_ARGS__)
#define WALK_ARG_WITH_3_19(func, f0, f1, f2, a0, ...) \
    EVAL_FUNC(func(f0, f1, f2, a0));                  \
    WALK_ARG_WITH_3_18(func, f0, f1, f2, __VA_ARGS__)

#define WALK_ARG_WITH_3_EX2(n, func, f0, f1, f2, ...) WALK_ARG_WITH_3_##n(func, f0, f1, f2, __VA_ARGS__)

#define WALK_ARG_WITH_3_EX(n, func, f0, f1, f2, ...) WALK_ARG_WITH_3_EX2(n, func, f0, f1, f2, __VA_ARGS__)

#define WALK_ARG_WITH_3(func, f0, f1, f2, ...) \
    WALK_ARG_WITH_3_EX(COUNT_VA_ARG(__VA_ARGS__), func, f0, f1, f2, __VA_ARGS__)

/***********************************************************************************/

#define WALK_VA_ARG2(func, func_arg_num, ...) WALK_ARG_WITH_##func_arg_num(func, __VA_ARGS__)

#define WALK_VA_ARG(func, func_arg_num, ...) WALK_VA_ARG2(func, func_arg_num, __VA_ARGS__)

#endif
