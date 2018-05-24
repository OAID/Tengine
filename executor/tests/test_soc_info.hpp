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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __TEST_SOC_INFO_HPP__
#define __TEST_SOC_INFO_HPP__

#include "soc_runner.hpp"

namespace TEngine {

SocInfo *  TestGetSocInfo(void)
{
	static SocInfo soc_info;
	static bool inited=false;

        if(inited)
           return &soc_info;

         inited=true;

	soc_info.cpu_number=8;
	soc_info.master_cpu=4;
	soc_info.soc_name="HIKEY960";

	CPUInfo cpu_info;

	for(int i=0;i<4;i++)
	{
		cpu_info.cpu_id=i;
		cpu_info.cpu_type="A53";
		cpu_info.cpu_arch="arm64";
		cpu_info.l1_size=32*1024;
		cpu_info.l2_slice=256*1024;

		soc_info.cpu_info.push_back(cpu_info);
                soc_info.cpu_list.push_back(i);
	}

	for(int i=4;i<8;i++)
	{
		cpu_info.cpu_id=i;
		cpu_info.cpu_type="A73";
		cpu_info.cpu_arch="arm64";
		cpu_info.l1_size=32*1024;
		cpu_info.l2_slice=512*1024;

		soc_info.cpu_info.push_back(cpu_info);
                soc_info.cpu_list.push_back(i);
	}

	return &soc_info;
}



} //namespace TEngine

#endif
