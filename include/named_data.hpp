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
#ifndef __NAMED_DATA_HPP__
#define __NAMED_DATA_HPP__

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

namespace TEngine {

template <typename T>
struct NamedData {

        using map_t=std::unordered_map<std::string, T*>;

	static map_t& GetMap(void)
	{
                static map_t internal_map;
		return internal_map;
	}

	static T * GetDefaultData(void)
	{
		return GetData("default");
	}

	static void SetDefaultData( T * data)
	{
		SetData("default",data);
	}

	static  T * GetData( const std::string& name)
	{
		map_t& map=GetMap();

		if(map.count(name)==0)
			return nullptr;

		return map[name];
	}

	static void SetData( const std::string& name,  T * data)
	{
		map_t&  map=GetMap();

		map[name]=data;
	}

        static void InitPredefinedData();


};

} //namespace TEngine

#endif
