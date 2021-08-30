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
 * Copyright (c) 2021
 * Author: tpoisonooo
 */

#include <string>
#include <vector>
#include <fstream>
#include <cassert>

namespace pipeline {

template<class T, class = typename std::enable_if<std::is_pod<std::remove_reference<T> >::value>::type>
void pack(std::string& buf, const T& t)
{
    auto ptr = reinterpret_cast<const char*>(&t);
    buf.append(ptr, sizeof(T));
}

template<class T, class = typename std::enable_if<std::is_pod<std::remove_reference<T> >::value>::type>
void unpack(const char*& buf, T& t)
{
    t = *(reinterpret_cast<const T*>(buf));
    buf += sizeof(T);
}

template<>
void pack<std::string>(std::string& buf, const std::string& t)
{
    int32_t len = static_cast<int32_t>(t.size());
    buf.append(reinterpret_cast<const char*>(&len), sizeof(len));
    buf.append(t);
}

template<>
void unpack<std::string>(const char*& buf, std::string& t)
{
    int32_t len = 0;
    unpack(buf, len);
    assert(len > 0);
    t.append(buf, len);
    buf += len;
}

template<class T>
void pack(std::string& buf, const std::vector<T>& vec)
{
    int32_t len = static_cast<int32_t>(vec.size());
    buf.append(reinterpret_cast<const char*>(&len), sizeof(len));
    for (const auto& val : vec)
    {
        pack(buf, val);
    }
}

template<class T>
void unpack(const char*& buf, std::vector<T>& vec)
{
    int32_t len = 0;
    unpack(buf, len);
    assert(len > 0);
    for (int i = 0; i < len; ++i)
    {
        T t;
        unpack(buf, t);
        vec.emplace_back(t);
    }
}

template<typename T, typename... Args>
void pack(std::string& buf, const T& t, Args&... args)
{
    pack(buf, t);
    pack(buf, args...);
}

template<typename T, typename... Args>
void unpack(const char*& buf, T& t, Args&... args)
{
    unpack(buf, t);
    unpack(buf, args...);
}

// get data pack size
template<class T, class = typename std::enable_if<std::is_pod<std::remove_reference<T> >::value>::type>
size_t pack_size(const T& t)
{
    return sizeof(t);
}

template<>
size_t pack_size(const std::string& t)
{
    return sizeof(int32_t) + t.size();
}

template<class T>
size_t pack_size(const std::vector<T>& val)
{
    size_t ret = sizeof(int32_t);
    for (const auto& v : val)
    {
        ret += pack_size(v);
    }
    return ret;
}

template<typename T, typename... Args>
size_t pack_size(const T& t, Args&... args)
{
    return pack_size(t) + pack_size(args...);
}

template<typename... Args>
void save(const std::string& path, Args&... args)
{
    std::string buf;
    pack(buf, args...);

    std::ofstream out(path, std::ios::binary | std::ios::out);

    size_t sz = pack_size(args...);
    out.write(reinterpret_cast<const char*>(&sz), sizeof(int32_t));
    out.write(buf.data(), buf.size());

    out.flush();
    out.close();
}

template<typename... Args>
void load(const std::string& path, Args&... args)
{
    const int MAX_BUF_SIZE = 8192;

    std::ifstream in(path, std::ios::binary | std::ios::in);
    char len_str[sizeof(int32_t)] = {0};
    in.read(len_str, sizeof(int32_t));

    int32_t len = *(reinterpret_cast<int32_t*>(len_str));
    assert(len > 0);
    assert(len <= MAX_BUF_SIZE);

    char buf[MAX_BUF_SIZE] = {0};

    in.read(buf, len);

    const char* ptr = buf;

    unpack(ptr, args...);
    in.close();
}

} // namespace pipeline
