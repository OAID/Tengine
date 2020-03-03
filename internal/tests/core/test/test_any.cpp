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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
// Very simplist test, could be better.

#include "any.hpp"
#include <memory>
#include <cstdio>

#define CHECK(x)         \
    ((x) ? ( void )(0) : \
           (void(fprintf(stdout, "Failed at %d:%s: %s\n", __LINE__, __FILE__, #x)), std::exit(EXIT_FAILURE)))

template <size_t N> struct words
{
    void* w[N];
};

struct big_type
{
    char i_wanna_be_big[256];
    std::string value;

    big_type() : value(std::string(300, 'b'))
    {
        i_wanna_be_big[0] = i_wanna_be_big[50] = 'k';
    }

    bool check()
    {
        CHECK(value.size() == 300);
        CHECK(value.front() == 'b' && value.back() == 'b');
        CHECK(i_wanna_be_big[0] == 'k' && i_wanna_be_big[50] == 'k');
        return true;
    }
};

// small type which has nothrow move ctor but throw copy ctor
struct regression1_type
{
    const void* confuse_stack_storage = ( void* )(0);
    regression1_type() {}
    regression1_type(const regression1_type&) {}
    regression1_type(regression1_type&&) noexcept {}
    regression1_type& operator=(const regression1_type&)
    {
        return *this;
    }
    regression1_type& operator=(regression1_type&&)
    {
        return *this;
    }
};

int main()
{
    using TEngine::any;
    using TEngine::any_cast;
    using TEngine::bad_any_cast;

    {
        any x = 4;
        any y = big_type();
        any z = 6;

        CHECK(any().empty());
        CHECK(!any(1).empty());
        CHECK(!any(big_type()).empty());

        CHECK(!x.empty() && !y.empty() && !z.empty());
        y.clear();
        CHECK(!x.empty() && y.empty() && !z.empty());
        x = y;
        CHECK(x.empty() && y.empty() && !z.empty());
        z = any();
        CHECK(x.empty() && y.empty() && z.empty());
    }

    {
        CHECK(any().type() == typeid(void));
        CHECK(any(4).type() == typeid(int));
        CHECK(any(big_type()).type() == typeid(big_type));
        CHECK(any(1.5f).type() == typeid(float));
    }

    {
        bool except0 = false;
        bool except1 = false, except2 = false;
        bool except3 = false, except4 = false;

        try
        {
            any_cast<int>(any());
        }
        catch(const bad_any_cast&)
        {
            except0 = true;
        }

        try
        {
            any_cast<int>(any(4.0f));
        }
        catch(const bad_any_cast&)
        {
            except1 = true;
        }

        try
        {
            any_cast<float>(any(4.0f));
        }
        catch(const bad_any_cast&)
        {
            except2 = true;
        }

        try
        {
            any_cast<float>(any(big_type()));
        }
        catch(const bad_any_cast&)
        {
            except3 = true;
        }

        try
        {
            any_cast<big_type>(any(big_type()));
        }
        catch(const bad_any_cast&)
        {
            except4 = true;
        }

        CHECK(except0 == true);
        CHECK(except1 == true && except2 == false);
        CHECK(except3 == true && except4 == false);
    }

    {
        any i4 = 4;
        any i5 = 5;
        any f6 = 6.0f;
        any big1 = big_type();
        any big2 = big_type();
        any big3 = big_type();

        CHECK(any_cast<int>(&i4) != nullptr);
        CHECK(any_cast<float>(&i4) == nullptr);
        CHECK(any_cast<int>(i5) == 5);
        CHECK(any_cast<float>(f6) == 6.0f);
        CHECK(any_cast<big_type>(big1).check() && any_cast<big_type>(big2).check() && any_cast<big_type>(big3).check());
    }

    {
        std::shared_ptr<int> ptr_count(new int);
        std::weak_ptr<int> weak = ptr_count;
        any p0 = 0;

        CHECK(weak.use_count() == 1);
        any p1 = ptr_count;
        CHECK(weak.use_count() == 2);
        any p2 = p1;
        CHECK(weak.use_count() == 3);
        p0 = p1;
        CHECK(weak.use_count() == 4);
        p0 = 0;
        CHECK(weak.use_count() == 3);
        p0 = std::move(p1);
        CHECK(weak.use_count() == 3);
        p0.swap(p1);
        CHECK(weak.use_count() == 3);
        p0 = 0;
        CHECK(weak.use_count() == 3);
        p1.clear();
        CHECK(weak.use_count() == 2);
        p2 = any(big_type());
        CHECK(weak.use_count() == 1);
        p1 = ptr_count;
        CHECK(weak.use_count() == 2);
        ptr_count = nullptr;
        CHECK(weak.use_count() == 1);
        p1 = any();
        CHECK(weak.use_count() == 0);
    }

    {
        auto is_stack_allocated = [](const any& a, const void* obj1) {
            uintptr_t a_ptr = (uintptr_t)(&a);
            uintptr_t obj = (uintptr_t)(obj1);
            return (obj >= a_ptr && obj < a_ptr + sizeof(any));
        };

        // static_assert(sizeof(std::unique_ptr<big_type>) <= sizeof(void*) * 1, "unique_ptr too big");
        static_assert(sizeof(std::shared_ptr<big_type>) <= sizeof(void*) * 2, "shared_ptr too big");

        any i = 400;
        any f = 400.0f;
        // any unique = std::unique_ptr<big_type>(); -- must be copy constructible
        any shared = std::shared_ptr<big_type>();
        any rawptr = ( void* )(nullptr);
        any big = big_type();
        any w2 = words<2>();
        any w3 = words<3>();

        CHECK(is_stack_allocated(i, any_cast<int>(&i)));
        CHECK(is_stack_allocated(f, any_cast<float>(&f)));
        CHECK(is_stack_allocated(rawptr, any_cast<void*>(&rawptr)));
        // CHECK(is_stack_allocated(unique, any_cast<std::unique_ptr<big_type>>(&unique)));
        CHECK(is_stack_allocated(shared, any_cast<std::shared_ptr<big_type>>(&shared)));
        CHECK(!is_stack_allocated(big, any_cast<big_type>(&big)));
        CHECK(is_stack_allocated(w2, any_cast<words<2>>(&w2)));
        CHECK(!is_stack_allocated(w3, any_cast<words<3>>(&w3)));

        // Regression test for GitHub Issue #1
        any r1 = regression1_type();
        CHECK(is_stack_allocated(r1, any_cast<const regression1_type>(&r1)));
    }
    fprintf(stdout, "Done!\n");
}