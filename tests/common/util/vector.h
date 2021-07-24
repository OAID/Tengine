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
 * @struct vector_t
 * @brief  C style vector for consecutive storage.
 */
typedef struct vector
{
    int elem_size; //!< elements size which will be pushed into vector
    int elem_num;  //!< current counter of inserted elements

    int entry_size;           //!< size of inside vector header entry
    int space_num;            //!< the allocated elements counter, which should greater equal to 'elem_num'
    int ahead_num;            //!< allocated step when vector is full
    void* real_mem;           //!< real aligned memory address which point to vector entry
    void* mem;                //!< visual aligned address which point to the very begging of elements
    void (*free_func)(void*); //!< elements free function, will be called when release elements or vector
} vector_t;

/*!
 * @brief  Create a vector for a struct(or something else).
 *
 *            This function is not a STL-like version.
 *
 * @param [in]  elem_size: Size of the elements which will be pushed in.
 * @param [in]  free_func: Free function pointer of elements.
 *
 * @return  The pointer of the vector.
 */
vector_t* create_vector(int elem_size, void (*free_func)(void*));

/*!
 * @brief  Release a vector.
 *
 * @param [in]  v: The vector which will be released.
 */
void release_vector(vector_t* v);

/*!
 * @brief Get the count of elements.
 *
 * @param [in]  v: The vector.
 *
 * @return  The count of pushed elements.
 */
int get_vector_num(vector_t* v);

/*!
 * @brief  Resize a vector.
 *
 * @param [in]  v: The vector which will be resized.
 * @param [in]  new_size: The new vector elements count.
 *
 * @return statue value, 0 success, other value failure.
 */
int resize_vector(vector_t* v, int new_size);

/*!
 * @brief Push a element into vector from its pointer.
 *
 * @param [in]  v: The vector which will be pushed a new element.
 * @param [in]  data: The pointer of new element.
 *
 * @return statue value, 0 success, other value failure.
 */
int push_vector_data(vector_t* v, void* data);

/*!
 * @brief Set a element via its index.
 *
 * @param [in]  v: The vector.
 * @param [in]  index: The index of the element.
 * @param [in]  data: The pointer of new element.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_vector_data(vector_t* v, int index, void* data);

/*!
 * @brief Get a element via its index.
 *
 * @param [in]  v: The vector.
 * @param [in]  index: The index of the element.
 *
 * @return  The pointer of the elements.
 */
void* get_vector_data(vector_t* v, int index);

/*!
 * @brief Remove a element via its pointer.
 *
 * @param [in]  v: The vector.
 * @param [in]  data: The pointer of the elements.
 *
 * @return statue value, 0 success, other value failure.
 */
int remove_vector_via_pointer(vector_t* v, void* data);

/*!
 * @brief Remove a element via its index.
 *
 * @param [in]  v: The vector.
 * @param [in]  index: The index of the element.
 */
void remove_vector_via_index(vector_t* v, int index);
