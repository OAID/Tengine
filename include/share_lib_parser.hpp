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
 * Author: honggui@openailab.com
 */
#ifndef __SHARE_LIB_PARSER_HPP__
#define __SHARE_LIB_PARSER_HPP__

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <dlfcn.h>
#include <stdexcept>
#include <unordered_map>
#include <functional>
#include "te_error.hpp"

namespace TEngine{
	class ShareLibParser{
    public:
        ShareLibParser() : sl(nullptr){}
        ShareLibParser(const std::string& so_path)
        {
            sl = nullptr;
            Load(so_path);
        }
		int Load(const std::string& so_path){

                 //test if the so is already loaded
                 sl=::dlopen(so_path.c_str(),RTLD_NOLOAD|RTLD_NOW);

                 if(sl)
                     return 0;
                 sl = ::dlopen(so_path.c_str(), RTLD_LAZY|RTLD_GLOBAL);
			if(!sl){
                std::printf("%s\n", dlerror());
                throw te_error_unable_to_load_library(so_path);
                return -1;             
            }
            return 0; //ok
		};

		template<typename F>
		std::function<F> GetFunction(const std::string func_name){
            auto it = func_map.find(func_name);
            if (it == func_map.end()){
                auto f = dlsym(sl, func_name.c_str());
                if(!f){
                    throw te_error_shared_function_not_found(func_name);
                    return nullptr;            
                }
                func_map.emplace(func_name, (func*)f);
                it = func_map.find(func_name);
            }
			return std::function<F>((F*)(it->second));
        }

        template <typename F, typename... Args>
        typename std::result_of<std::function<F>(Args...)>::type ExecuteFunc(const std::string& func_name, Args&&... args)
        {
                auto f = GetFunction<F>(func_name);
                return f(std::forward<Args>(args)...);
        }
        
		bool IsValid(){ //check if it is ready
			return sl != nullptr;
		}

		~ShareLibParser(){
            if(sl) ::dlclose(sl);
		};
    private:
        using func = void();
        void* sl; //shared library
        std::unordered_map<std::string, func*> func_map;
    };
}

#endif
