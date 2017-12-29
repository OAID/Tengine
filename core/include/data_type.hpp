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
#ifndef __DATA_TYPE_HPP__
#define __DATA_TYPE_HPP__

#include "named_data.hpp"

namespace TEngine {

struct  DataType: public NamedData<DataType> {

	DataType(const std::string& str, int size, bool as_default=false)
	{
		dtype_name=str;
		dtype_size=size;
		SetData(dtype_name,this);

                if(as_default)
                   SetDefaultData(this);
	}

	DataType(std::string&& str,int size, bool as_default=false)
	{
		dtype_size=size;
		dtype_name=std::move(str);
		SetData(dtype_name,this);

                if(as_default)
                   SetDefaultData(this);
	}

	static  const DataType * GetType( const std::string& name)
	{
		return GetData(name);
	}

	const std::string& GetTypeName(void) const
	{
		return dtype_name;
	}

	int GetTypeSize(void) const
	{
		return dtype_size;
	}

	template <typename T>
	T Convert(const std::string& str) const;


	std::string dtype_name;
	int dtype_size;
};

} //namespace TEngine

#endif

