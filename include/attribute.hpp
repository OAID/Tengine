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
#ifndef __ATTRIBUTE_HPP__
#define __ATTRIBUTE_HPP__

#include <unordered_map>
#include <vector>

#include "any.hpp"

namespace TEngine {
	class Attribute
	{
		using attribute_map_t = std::unordered_map<std::string, any>;
		using value_t = attribute_map_t::value_type;
		public:

			Attribute()=default;
			Attribute(const Attribute&)=default;

			Attribute(std::initializer_list<value_t> __l){
				dict_map_ = __l;
			}

			any&  operator[](const std::string& name) 
			{
                             if(ExistAttr(name))
				return GetAttr(name);
                                                          
                              SetAttr(name,any());

                              return dict_map_.at(name);
			}

                        virtual bool ExistAttr(const std::string& name) const
                        {
                              if(dict_map_.count(name))
                                  return true;
                              else
                                  return false;
                        }

			template<typename T>
			void GetAttr(const std::string& name, T* v, T default_v)
			{
				if (ExistAttr(name)){
                    *v = any_cast<T>(GetAttr(name));
				}
				else{
					*v = default_v;
				}
			}

			template<typename T>
			void GetAttr(const std::string& name, T* v)
			{
				if (ExistAttr(name)){
                    *v = any_cast<T>(GetAttr(name));
				}
			}

                        
                        virtual any& GetAttr(const std::string& name) 
                        {
                               static any dummy;

                               if(dict_map_.count(name))
                                   return dict_map_.at(name);

                               return dummy;
                        }

                        virtual const any& GetAttr(const std::string& name) const
                        {
                               static any dummy;

                               if(dict_map_.count(name))
                                   return dict_map_.at(name);

                               return dummy;
                        }

			void SetAttr(const std::string& name, const any& val)
			{
				dict_map_[name]=val;
			}

			void RemoveAttr(const std::string& name)
			{
				if(dict_map_.count(name))
					dict_map_.erase(name);
			}

			virtual std::vector<std::string> ListAttr(void) const
			{
				std::vector<std::string>  result;

				std::unordered_map<std::string, any>::const_iterator begin=dict_map_.cbegin();

				while(begin!=dict_map_.cend())
				{
					result.push_back(begin->first);	
					begin++;
				}

				return result;
			}

			virtual ~Attribute() {}

		protected:
			attribute_map_t dict_map_;
	};


#define INSERT_NEW_ATTRIBUTE(obj,name,val)  obj[name]=val

} //namespace TEngine
#endif


