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
#ifndef __PROF_RECORD_HPP__
#define __PROF_RECORD_HPP__

#include <vector>
#include <unordered_map>
#include <functional>

namespace TEngine {

struct ProfRecord {

#define PROF_DUMP_SEQ 0
#define PROF_DUMP_DECREASE 1
#define PROF_DUMP_INCREASE 2
   
   virtual bool Start(int idx, void * ident)=0;
   virtual bool Stop(int idx)=0;
   virtual void Dump(int method=PROF_DUMP_DECREASE) =0;
   virtual void Reset(void)=0;

   virtual ~ProfRecord() {};
};

/* 
   void * -- ident 
   int -- repeat_count
   unsigned long -- used time
*/ 
using prof_parser_t=std::function<void(void*,int,unsigned long)>;

struct ProfTime: public ProfRecord {
    struct TimeRecord {
         unsigned int count;
         uint64_t     total_used_time;
         uint64_t     start_time;
         uint64_t     end_time;
         uint64_t     min_time;
         uint64_t     max_time;
         void *       ident;

         TimeRecord(){
            Reset();
         }
         void Reset(){
            count=0;
            start_time=end_time=max_time=total_used_time=0;
            min_time=~1UL;
         }
    };


    ProfTime(int size, prof_parser_t func )
    {
          record.resize(size);
          parser=func;
    }
   
    bool Start(int idx, void * ident) override;
    bool Stop(int idx) override;
    void Dump(int method=PROF_DUMP_DECREASE) override;
    void Reset(void) override;
    const TimeRecord * GetRecord(int idx) const;
    int GetRecordNum(void) const;

    ~ProfTime(){}


prof_parser_t parser;
std::vector<TimeRecord> record;

};



class ProfRecordManager: public std::unordered_map<std::string, ProfRecord *> {

public:

    static ProfRecordManager * GetInstance(void)
    {
          static ProfRecordManager * ptr=new ProfRecordManager();
          return ptr;
    }

    static ProfRecord * Create(const std::string& name, int size, prof_parser_t func)
    {
         ProfTime * ptr=new ProfTime(size,func);

         ProfRecordManager * manager=GetInstance();

         (*manager)[name]=ptr;

         return ptr;
    }

    static ProfRecord * Get(const std::string& name)
    {
         ProfRecordManager * manager=GetInstance();

         if(manager->count(name)==0)
                 return nullptr;

          return manager->at(name);
    }



};

} //namespace TEngine


#endif
