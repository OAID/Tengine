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
#ifndef __BASE_OBJECT_HPP__
#define __BASE_OBJECT_HPP__

#include "attribute.hpp"

namespace TEngine {

class BaseObject: public Attribute {
public:
        bool RealExistClassAttr(const std::string& name) const
        {
                return ClassAttr()->ExistAttr(name);
        }

        any& RealGetClassAttr(const std::string& name) 
        {
                return ClassAttr()->GetAttr(name);
        }

        const any& RealGetClassAttr(const std::string& name) const
        {
                return ClassAttr()->GetAttr(name);
        }

        std::vector<std::string> RealListClassAttr(void)  const
        {
              std::vector<std::string> result;

              result=ClassAttr()->ListAttr();

              return result;
        }

	virtual bool ExistClassAttr(const std::string& name)  const
	{ 
		return RealExistClassAttr(name);
	}

	virtual const any& GetClassAttr(const std::string& name) const
	{ 
		return RealGetClassAttr(name);
	}

	virtual any& GetClassAttr(const std::string& name) 
	{ 
		return RealGetClassAttr(name);
	}

	virtual void SetClassAttr(const std::string& name, const any& val)
	{
		ClassAttr()->SetAttr(name,val);
	}

        virtual std::vector<std::string> ListClassAttr(void)  const
        {
              return RealListClassAttr();
        }

	bool ExistAttr(const std::string& name) const override
	{
		if(dict_map_.count(name))
			return true;

		if(ExistClassAttr(name))
			return true;

		return false;
	}

        const any& GetAttr(const std::string& name) const override
        {
                if(dict_map_.count(name))
                        return dict_map_.at(name);

                if(ExistClassAttr(name))
                        return GetClassAttr(name);

                return Attribute::GetAttr(name);
        }

	any& GetAttr(const std::string& name) override
	{
		if(dict_map_.count(name))
			return dict_map_.at(name);

		if(ExistClassAttr(name))
			return GetClassAttr(name);

		SetAttr(name,any());

		return dict_map_[name];
	}

	std::vector<std::string> ListAttr(void) const override
	{
		std::unordered_map<std::string, any>::const_iterator begin;
		std::vector<std::string>  result;

		begin=dict_map_.cbegin();

		while(begin!=dict_map_.cend())
		{
			result.push_back(begin->first);
			begin++;
		}


                std::vector<std::string> class_attr=ListClassAttr();

                result.insert(result.end(),class_attr.begin(),class_attr.end());

		return result;
	}

        BaseObject()=default;
        BaseObject(const BaseObject&)=default;

        virtual ~BaseObject(){}

private:

	Attribute * ClassAttr() const
	{
		static  Attribute class_attr;
		return &class_attr;
	}

};


#define REGISTER_CLASS_ATTR_OPS(parent) \
        bool RealExistClassAttr(const std::string& name) const \
        {\
                if(ClassAttr()->ExistAttr(name))\
                         return true;\
						\
                if(parent::RealExistClassAttr(name))\
                        return true;\
						\
                return false;\
        }\
        any& RealGetClassAttr(const std::string& name)  \
        {\
                if(ClassAttr()->ExistAttr(name))\
                     return ClassAttr()->GetAttr(name);\
							\
                return parent::RealGetClassAttr(name);\
        }\
							\
        const any& RealGetClassAttr(const std::string& name) const \
        {\
                if(ClassAttr()->ExistAttr(name))\
                     return ClassAttr()->GetAttr(name);\
							\
                return parent::RealGetClassAttr(name);\
        }\
							\
        virtual void SetClassAttr(const std::string& name, const any& val)\
        {\
                ClassAttr()->SetAttr(name,val);\
        }\
							\
        std::vector<std::string> RealListClassAttr(void) const\
        {\
              std::vector<std::string> result;\
							\
              result=parent::RealListClassAttr();\
							\
              std::vector<std::string> my=ClassAttr()->ListAttr();\
							\
              result.insert(result.end(),my.begin(),my.end());\
						\
              return result;\
        }\
        const Attribute * ClassAttr(void) const\
        {\
            static Attribute class_attr;\
            return &class_attr;\
        }\
        Attribute * ClassAttr(void)\
        {\
            static Attribute class_attr;\
            return &class_attr;\
        }\
        virtual bool ExistClassAttr(const std::string& name) \
        { return RealExistClassAttr(name);}\
        virtual any& GetClassAttr(const std::string& name)\
        { return RealGetClassAttr(name);}\
        virtual std::vector<std::string> ListClassAttr(void) \
        {  return RealListClassAttr(); }
         
        
        

} //namespace TEngine

#endif
