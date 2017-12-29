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
#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <utility>
#include <mutex>
#include <typeindex>
#include <type_traits>
#include "any.hpp"
#include "utilities/non_copyable.hpp"
//#include "utilities/type_name.hpp"  //debug

using namespace std;
namespace TEngine{
	class ClassFactory : NonCopyable
	{
	public:
		ClassFactory(void){}
		~ClassFactory(void){}

		//通过new Class来创建类的实例，支持继承类的创建（创建子类，返回夫类指针）
		template<class T, typename... Args>
		void RegisterType(const std::string& name = "")
		{
			RegisterType_Helper<1+sizeof...(Args), T, Args...>(name);
		}
        //复杂的Creator（带参数）
		template <typename  T, typename ... Args>
		typename std::enable_if<std::is_class<T>::value, void>::type
		RegisterType(T* creator(Args...), const std::string& name = "")
		{
			//using funt_t = T*(Args...);
			RegisterType_Helper<1+sizeof...(Args), T, Args...>(creator, name);
		}

		//为继承Class创建实例
		template<class T, class U, typename ...Args>
		typename std::enable_if<std::is_base_of<T, U>::value, T*>::type
		Create(Args... args)
		{
			any constructor = GetDerivedFunction<T, U, Args...>();
			std::function<T* (Args&&...)> function = any_cast<std::function<T* (Args&&...)>>(constructor);
			return function(std::forward<Args>(args)...);
		}
		//为简单Class创建实例,或Function Creator
		template<class T, typename... Args>
		T* Create(Args... args)
		{
			any constructor = GetFunction<T, Args...>();
			std::function<T* (Args&&...)> function = any_cast<std::function<T* (Args&&...)>>(constructor);
			return function(std::forward<Args>(args)...);
		}
		//为继承Class创建实例  (with Name)
		template<class T, class U, typename ...Args>
		typename std::enable_if<std::is_base_of<T, U>::value, T*>::type
		CreateWithName(const std::string& name, Args... args)
		{
			any constructor = GetDerivedFunction<T, U, Args...>(name);
			std::function<T* (Args&&...)> function = any_cast<std::function<T* (Args&&...)>>(constructor);
			return function(std::forward<Args>(args)...);
		}
		//为简单Class创建实例,或Function Creator  (with Name)
		template<class T, typename... Args>
		T* CreateWithName(const std::string& name, Args... args)
		{
			any constructor = GetFunction<T, Args...>(name);
			std::function<T* (Args&&...)> function = any_cast<std::function<T* (Args&&...)>>(constructor);
			return function(std::forward<Args>(args)...);
		}

		//Create Shared Pointer for the instance
		template<class T, class U, typename ...Args>
		typename std::enable_if<std::is_base_of<T, U>::value, std::shared_ptr<T>>::type
		CreateShared(Args... args)
		{
			T* t = Create<T, U, Args...>(std::forward<Args>(args)...);
			return std::shared_ptr<T>(t);
		}
		template<class T, typename... Args>
		std::shared_ptr<T> CreateShared(Args... args)
		{
			T* t = Create<T, Args...>(std::forward<Args>(args)...);
			return std::shared_ptr<T>(t);
		}
		//Create Shared Pointer for the instance  (with Name)
		template<class T, class U, typename ...Args>
		typename std::enable_if<std::is_base_of<T, U>::value, std::shared_ptr<T>>::type
		CreateSharedWithName(const std::string& name, Args... args)
		{
			T* t = CreateWithName<T, U, Args...>(name, std::forward<Args>(args)...);
			return std::shared_ptr<T>(t);
		}
		template<class T, typename... Args>
		std::shared_ptr<T> CreateSharedWithName(const std::string& name, Args... args)
		{
			T* t = CreateWithName<T, Args...>(name, std::forward<Args>(args)...);
			return std::shared_ptr<T>(t);
		}

		//Create Unique Pointer for the instance
		template<class T, class U, typename ...Args>
		typename std::enable_if<std::is_base_of<T, U>::value, std::unique_ptr<T>>::type
		CreateUnique(Args... args)
		{
			T* t = Create<T, U, Args...>(std::forward<Args>(args)...);
			return std::unique_ptr<T>(t);
		}
		template<class T, typename... Args>
		std::unique_ptr<T> CreateUnique(Args... args)
		{
			T* t = Create<T, Args...>(std::forward<Args>(args)...);
			return std::unique_ptr<T>(t);
		}
		//Create Unique Pointer for the instance  (with Name)
		template<class T, class U, typename ...Args>
		typename std::enable_if<std::is_base_of<T, U>::value, std::unique_ptr<T>>::type
		CreateUniqueWithName(const std::string& name, Args... args)
		{
			T* t = CreateWithName<T, U, Args...>(name, std::forward<Args>(args)...);
			return std::unique_ptr<T>(t);
		}
		template<class T, typename... Args>
		std::unique_ptr<T> CreateUniqueWithName(const std::string& name, Args... args)
		{
			T* t = CreateWithName<T, Args...>(name, std::forward<Args>(args)...);
			return std::unique_ptr<T>(t);
		}

	private:
		void RegisterType(const size_t key, any constructor)
		{
            std::lock_guard<std::mutex> lg(mut_mapList);
			if (creatorMapList.find(key) != creatorMapList.end())
				throw std::invalid_argument("this key has already exist!");
            //inseat the constructor to creatorMapList
			creatorMapList.emplace(key, constructor);
		}

		//U is directed from T, with 1 or more parameters
		template<std::size_t N, class T, class U, typename ...Args>
		typename std::enable_if<((N >= 2) && !std::is_function<typename std::remove_pointer<T>::type>::value && std::is_base_of<T, U>::value), void>::type
		RegisterType_Helper(const std::string& name = "")
		{
			using func_t = T*(Args&&...);
			size_t key = typeid(func_t).hash_code();
			if(name.length() == 0)	key += typeid(U).hash_code();
			else 				    key += hash_string_fn{}(name);
			std::function<T*(Args&&...)> function=[](Args ... args){ return (T*)(new U(std::forward<Args>(args)...)); };
			RegisterType(key, function);
		}
		//T is a class whose constructor has 1 or more parameter 
		template<std::size_t N, class T, class U, typename ...Args>
		typename std::enable_if<((N >= 2) && !std::is_function<typename std::remove_pointer<T>::type>::value && !std::is_base_of<T, U>::value), void>::type
		RegisterType_Helper(const std::string& name = "")
		{
            using func_t = T*(U&& u, Args&&...);
			size_t key = typeid(func_t).hash_code();
			if(name.length() != 0) key += hash_string_fn{}(name);
			std::function<T* (U&& u, Args&&...)> function = [](U&& u, Args&&... args){ return new T(std::forward<U>(u), std::forward<Args>(args)...); };
			RegisterType(key, function);
		}
		//simple class
		template<std::size_t N, class T>
		typename std::enable_if<(N == 1) && !std::is_function<typename std::remove_pointer<T>::type>::value, void>::type
		RegisterType_Helper(const std::string& name = "")
		{
            using func_t = T*();
			size_t key = typeid(func_t).hash_code();
			if(name.length() != 0) key += hash_string_fn{}(name);
			std::function<T* ()> function = [](){ return new T(); };
			RegisterType(key, function);
		}
		//function
		// this function is diferent from simple function, simple function do not use move/forward(&&)
		template<std::size_t N, class T, typename ...Args>
		typename std::enable_if<(N >= 1) && std::is_class<T>::value, void>::type
		RegisterType_Helper(T* creator(Args...), const std::string& name = "")
		{
            using func_t = T*(Args&&...);
//std::cout << "RegisterType_Helper<func&&>: " << type_name<func_t>() << std::endl;			
			size_t key = typeid(func_t).hash_code();
			if(name.length() != 0) key += hash_string_fn{}(name);
			std::function<T*(Args&&...)> function = [creator](Args&&... args){ return creator(std::forward<Args>(args)...); };
			RegisterType(key, std::function<func_t>(function));
		}
		
		//Function为继承Class创建实例(T*) new (U(Args&&...))
		template<class T, class U, typename ...Args>
		inline any GetDerivedFunction(const std::string& name = "")
		{
            using func_t = T*(Args&&...);
			size_t key = typeid(func_t).hash_code();
			if(name.length() == 0)	key += typeid(U).hash_code();
			else 				    key += hash_string_fn{}(name);
			return GetFunction(key);
		}
		//Function为简单Class创建实例(new T*(Args&&...)),或Function Creator(T*(Args...) 或 T*(Args&&...))
		template<class T, typename ...Args>
		inline any GetFunction(const std::string& name = "")
		{
            using func_t = T*(Args&&...);
//std::cout << "GetFunction<func&&>: " << type_name<func_t>() << std::endl;			
			size_t key = typeid(func_t).hash_code();
			if(name.length() != 0) key += hash_string_fn{}(name);
			return GetFunction(key);
		}
		//base
		inline any GetFunction(const size_t key)
		{
			std::lock_guard<std::mutex> lg(mut_mapList);
			if (creatorMapList.find(key) == creatorMapList.end()){
				std::cout << "error: can not got the constructor\n";
				return nullptr; //the constructor for the class was not found 
			}
			return creatorMapList[key];
		}
		
	private:
		unordered_map<size_t, any> creatorMapList;
		std::mutex mut_mapList;    
		using hash_string_fn = std::hash<std::string>;  
	};
};
